import glob
import cv2
import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image
from torchmetrics import MetricCollection, Dice, AveragePrecision, AUROC
from pytorch_lightning import LightningModule

from .utils.utils import to_numpy, split_tensor, rebuild_tensor
from .utils.utils import find_best_dice_th, sharpening, poly_lr, make_loss_function, DistDW, DiffDW
from .train_ours import LM
from .utils.losses import WeightedCrossEntropyLoss

import random
np.random.seed(1024)
torch.manual_seed(1024)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConfidenceBank:
    def __init__(self, confs=None, num_cls=5, momentum=0.99, update_iter=1):
        self.num_cls = num_cls
        self.momentum = momentum
        self.update_iter = update_iter
        self.tot_iter = 0
        if confs is None:
            self.confs = torch.rand(self.num_cls).float()#.cuda()
        else:
            self.confs = torch.clone(confs)
        self.confs_all = torch.zeros(self.num_cls).float()#.cuda()
        self.confs_cnt = torch.zeros(self.num_cls).float()#.cuda()

    def record_data(self, output_scores, label):
        # scores = F.softmax(output, dim=1)
        
        for ind in range(self.num_cls):
            cat_mask_sup_gt = (label[:, ind] == 1).detach().cpu() # we have classes as channel not intensity
            conf_map_sup = output_scores[:, ind, ...].detach().cpu()
            self.confs_all[ind] = self.confs_all[ind] + torch.sum(conf_map_sup * cat_mask_sup_gt)
            self.confs_cnt[ind] = self.confs_cnt[ind] + torch.sum(cat_mask_sup_gt).float()
        
        del conf_map_sup, cat_mask_sup_gt
        
        self.tot_iter += 1
        if self.tot_iter % self.update_iter == 0:
            new_confs = self.confs_all / (self.confs_cnt + 1e-12)
            if self.tot_iter <= self.update_iter: # first update
                self.confs = new_confs
            else:
                self.confs = self.confs * self.momentum + new_confs * (1 - self.momentum)

            # del self.confs_all, self.confs_cnt
            self.confs_all = torch.zeros(self.num_cls).float()#.cuda()
            self.confs_cnt = torch.zeros(self.num_cls).float()#.cuda()

    def get_confs(self):
        return self.confs

class LM_CLD(LM):    
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 1e-3,
        batch_size:int = 1,
        version: str = 'multi',
        num_cls: int = 4,
        minibatch_size: int = 2,
        vis_test: int = 0,
        cps_w: float = 0.1,
        sup_loss: str = 'w_ce+dice',
        consistency_loss: str = 'wce'
    ):
        super(LM_CLD, self).__init__(
            net=net, 
            lr=lr, 
            batch_size=batch_size, 
            version=version, 
            num_cls=num_cls,
            minibatch_size=minibatch_size,
            vis_test=vis_test,
            cps_w=cps_w,
            sup_loss=sup_loss,
            consistency_loss=consistency_loss
            )
        
        print("model weight init to k and x")
        self.model.generator1 = kaiming_normal_init_weight(self.model.generator1)
        self.model.generator2 = xavier_normal_init_weight(self.model.generator2)
        
    def cal_confidence(self, output_score, label):
        new_conf = torch.zeros(self.num_cls).float().cuda()
        for ind in range(self.num_cls):
            cat_mask_sup_gt = (label[:, ind] == 1).detach().cpu() #.squeeze(1)
            conf_map_sup = output_score[:, ind, ...].detach().cpu()
            value = torch.sum(conf_map_sup * cat_mask_sup_gt) / (torch.sum(cat_mask_sup_gt) + 1e-12)
            new_conf[ind] = value
        del conf_map_sup, output_score, cat_mask_sup_gt, value
        return new_conf

    def cal_sampling_rate(self, confs, gamma=0.5):
        sam_rate = (1 - confs)
        sam_rate = sam_rate / (torch.max(sam_rate) + 1e-12)
        sam_rate = sam_rate ** gamma
        return sam_rate

    def cal_sampling_mask(self, output, sam_rate, min_sr=0.0):
        # pred_map = torch.argmax(output, dim=1).float()
        pred_map = (sharpening(output.detach()) >= 0.5).float().detach().cpu()
        sample_map = torch.zeros_like(pred_map).float()
        vol_shape = pred_map.shape
        for idx in range(self.num_cls):
            prob = 1 - sam_rate[idx]
            if idx >= 1 and prob > (1 - min_sr):
                prob = (1 - min_sr)
            rand_map = torch.rand(vol_shape) * (pred_map[:,idx:idx+1] == 1)
            rand_map = (rand_map > prob) * 1.0
            sample_map += rand_map
        del pred_map, rand_map
        return sample_map
             
    def patch_step(self, split, x_data, x_data2, y_data): 
        if split == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        total_batch = len(x_data) // self.minibatch_size 
        losses, supervised_losses, consistency_losses = [], [], []
        output_image = []
        for j in range(total_batch):
            # print(f'{j}th / total_batch : {total_batch} process ... ')
            if (j + 1) * self.minibatch_size > len(x_data):
                # continue # drop_last
                patch_images = x_data[j * self.minibatch_size:]
                patch_images2 = x_data2[j * self.minibatch_size:]
                patch_masks = y_data[j * self.minibatch_size:]
            else:
                patch_images = x_data[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                patch_images2 = x_data2[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                patch_masks = y_data[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                
    
            if split == 'train':
                    
                self.v_opt.zero_grad()
                self.r_opt.zero_grad()

                output_A_l, output_B_l, output_A_u, output_B_u = self.model(patch_images, patch_images2)
                
                # sup loss
                self.confs_A.record_data(output_A_l, patch_masks)
                self.confs_B.record_data(output_B_l, patch_masks)
                loss1, _, _ = self.loss_func(output_A_l, patch_masks) 
                loss2, _, _ = self.loss_func(output_B_l, patch_masks)
                loss_sup = loss1 + loss2

                # cps loss
                output_A = (torch.cat([output_A_l, output_A_u], dim=0))
                output_B = (torch.cat([output_B_l, output_B_u], dim=0))
                max_A = sharpening(output_A.detach()) >= 0.5 #torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                max_B = sharpening(output_B.detach()) >= 0.5 #torch.argmax(output_B.detach(), dim=1, keepdim=True).long()
                
                sam_rate_A = self.cal_sampling_rate(self.confs_A.get_confs())
                sam_rate_B = self.cal_sampling_rate(self.confs_B.get_confs())
                sample_map_A = self.cal_sampling_mask(output_A.detach(), sam_rate_A)
                sample_map_B = self.cal_sampling_mask(output_B.detach(), sam_rate_B)
                    
                loss_cps = self.cps_loss_func(output_A, max_B, sample_map_A.cuda()) + self.cps_loss_func(output_B, max_A, sample_map_B.cuda())
                loss = loss_sup + (self.cps_w * loss_cps)
                    
                losses.append(loss.item())
                supervised_losses.append(loss_sup.item())
                consistency_losses.append(loss_cps.item())
                
                self.manual_backward(loss) 
                self.v_opt.step()
                self.r_opt.step()
                del sample_map_A, sample_map_B, sam_rate_A, sam_rate_B
                
            else:
                # We only calc dice and prc in val
                output_A_l, output_B_l, output_A_u, output_B_u = self.model(patch_images, patch_images2)
                                
                output_image.append((output_A_l + output_B_l) / 2.)

            
            pred = (output_A_l + output_B_l) / 2.
            for idx, lesion in enumerate(self.lesion): # total, MA, HE, EX, SE, Sal
                if idx != 0:
                    if patch_masks[:,idx-1].any():
                        self.metrics[f'metric_{split}_{lesion}'].update(pred[:,idx-1], (patch_masks[:,idx-1]).type(torch.int64)) 
                        
        total_loss = {'losses':torch.mean(torch.Tensor(losses)),\
                    'supervised_losses':torch.mean(torch.Tensor(supervised_losses)), \
                    'consistency_losses':torch.mean(torch.Tensor(consistency_losses))}
        
        del patch_images, patch_images2, patch_masks, pred
        
        return total_loss, output_image
    
    
    def configure_optimizers(self):
        optimizer_v = torch.optim.AdamW(self.model.generator1.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        optimizer_r = torch.optim.AdamW(self.model.generator2.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        return [optimizer_v, optimizer_r]
    
    def init_weight(self, labeled_dataset):
        weight = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list: # labeled data idx list
            _, _, label = labeled_dataset._get_data(data_id)
            tmp = label.sum(axis=0).sum(axis=0) 
            weight += tmp
        weight = weight.astype(np.float32)
        cls_weight = np.power(np.amax(weight) / weight, 1/3)
        print('cld init weight', weight, '>>', cls_weight)
        return cls_weight, weight
    
    def on_train_epoch_start(self):
        if self.current_epoch == 0:        
            # make loss function
            # weight, _ = self.trainer.datamodule.train_dataset.weight()
            weight, _ = self.init_weight(self.trainer.datamodule.train_dataset)
            print('Weight initialization in LM init', weight)
            
            self.loss_func = make_loss_function(self.sup_loss, weight=weight) # w_ce+dice
            self.cps_loss_func = WeightedCrossEntropyLoss(weight=weight)
            del weight

            # confidence bank
            self.confs_iter = 8
            self.confs_A = ConfidenceBank(num_cls=self.num_cls, momentum=0.999, update_iter=self.confs_iter)
            self.confs_B = ConfidenceBank(confs=self.confs_A.get_confs(), num_cls=self.num_cls, momentum=0.999, update_iter=self.confs_iter) 

    
    def training_step(self, batch, batch_idx): 
        self.v_opt, self.r_opt = self.optimizers()
        split = 'train'
        self.cps_w = self.get_current_consistency_weight(self.current_epoch) 
        total_loss, _ = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0])
        
        self.log(f'TotalLoss{split}', total_loss['losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalVSupervisedLoss{split}', total_loss['supervised_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalConsistencyLoss{split}', total_loss['consistency_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
            
        self.v_opt.param_groups[0]['lr'] = poly_lr(self.current_epoch, self.trainer.max_epochs, self.hparams.lr, 0.9)
        self.r_opt.param_groups[0]['lr'] = poly_lr(self.current_epoch, self.trainer.max_epochs, self.hparams.lr, 0.9)
        self.compute_metrics(split)
        return {'loss': total_loss['losses']}
            
    def validation_step(self, batch, batch_idx):
        split = 'val'
        st_time = time.time()
        total_loss, output_image = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0])
        ptime = time.time()-st_time
        self.val_time.append(ptime)
        print("Time for step", ptime)
        
        if 'fgadr' not in self.version:
            y_data = batch[2][0]
            y_data = rebuild_tensor(y_data.detach().cpu(), self.mask_t, self.base_tensor, self.t_size, tile_size=self.patch_size, dim=self.num_cls)[0] # 1,4,2848,4288
            output_image = torch.cat(output_image, dim=0)
            output_image = rebuild_tensor(output_image.detach().cpu(), self.mask_t, self.base_tensor, self.t_size, tile_size=self.patch_size, dim=self.num_cls)[0] # 1,4,2848,4288
        
            for idx, lesion in enumerate(self.lesion): # total, MA, HE, EX, SE, Sal
                if idx != 0 and lesion != 'SE': 
                    self.metrics_val_prc[f'metric_val_{lesion}'].update(output_image[idx-1], (y_data[idx-1]).type(torch.int64)) 
                elif lesion == 'SE':
                    if batch[-1] in self.se_idx:
                        self.metrics_val_prc[f'metric_val_{lesion}'].update(output_image[idx-1], (y_data[idx-1]).type(torch.int64))
        if batch_idx == 26:
            print("Validation Step Done,, time : ",np.mean(self.val_time))
            self.compute_metrics(split)
            
        return {'loss': total_loss['losses']}   

    def compute_metrics(self, split):
        # st = time.time()
        for idx, lesion in enumerate(self.lesion): # total, MA, HE, EX, SE, Sal
            if idx == 0:
                total_prc = []
                total_mean = []
                total_mean_unlabeled = []
            else:
                    
                score = self.metrics[f'metric_{split}_{lesion}'].compute()
                self.metrics[f'metric_{split}_{lesion}'].reset()
                for k, v in score.items():
                    self.log(f'{split}/{k}_{lesion}', v.item()) #, batch_size=1
                    total_mean.append(v.item())                      
                
                if split == 'val':
                    if 'fgadr' not in self.version:
                        score = self.metrics_val_prc[f'metric_val_{lesion}'].compute()
                        self.metrics_val_prc[f'metric_val_{lesion}'].reset()
                        for k, v in score.items():
                            self.log(f'{split}/{k}_{lesion}', v.item())
                            total_prc.append(v.item())
                            
    
        if split == 'val':
            if 'fgadr' not in self.version:
                self.log(f'val/PRC_total', torch.mean(torch.tensor(total_prc, dtype=torch.float16))) #, batch_size=1
    
        self.log(f'{split}/Dice_total', torch.mean(torch.tensor(total_mean, dtype=torch.float16))) #, batch_size=1 
        self.log(f'{split}/Dice_total_unlabeled', torch.mean(torch.tensor(total_mean_unlabeled, dtype=torch.float16))) #, batch_size=1
        