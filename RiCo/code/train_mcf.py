import glob
import cv2
import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image
from torchmetrics import MetricCollection, Dice, AveragePrecision, AUROC
from pytorch_lightning import LightningModule

from .utils.utils import to_numpy, split_tensor, rebuild_tensor
from .utils.utils import find_best_dice_th, sharpening, poly_lr, make_loss_function, DistDW, DiffDW
from .train_ours import LM
np.random.seed(1024)
torch.manual_seed(1024)
    
class LM_MCF(LM):    
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
        super(LM_MCF, self).__init__(
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
        
        self.val_time = []
        self.seg_criterion = make_loss_function('w_ce+dice')
        self.mse_loss = torch.nn.MSELoss(reduction='none') # (a-b)**2 for rec
        self.rectification_weight = 0.5

    def patch_step(self, split, x_data, x_data2, y_data): 
        if split == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        Good_student = 0
        total_batch = len(x_data) // self.minibatch_size ## 961//64=15, +1
        v_losses, r_losses, v_seg_losses, r_seg_losses, v_dice_losses, r_dice_losses, v_mse_losses, r_mse_losses, consistency_losses, output_image = [], [], [], [], [], [], [], [], [], []
        for j in range(total_batch):
            if (j + 1) * self.minibatch_size > len(x_data):
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
                v_label, r_label = patch_masks.clone(), patch_masks.clone()
                
                ## Supervised loss
                v_loss_seg, v_loss_seg_dice, _ = self.seg_criterion(output_A_l, v_label) 
                r_loss_seg, r_loss_seg_dice, _ = self.seg_criterion(output_B_l, r_label) 
                                                                                                                                                                                                            
                if v_loss_seg_dice < r_loss_seg_dice: # v: eff, r: swin
                    Good_student = 0 # v is good student
                else:
                    Good_student = 1 # r is good student
                
                # Rectification loss
                v_predict = sharpening(output_A_l) >= 0.5
                r_predict = sharpening(output_B_l) >= 0.5
                diff_mask = ((v_predict == 1) ^ (r_predict == 1)).to(torch.int32).clone().detach()
                v_mse_dist = self.mse_loss(output_A_l, v_label) # MSE
                r_mse_dist = self.mse_loss(output_B_l, r_label) # MSE
                v_mse      = torch.sum(diff_mask * v_mse_dist) / (torch.sum(diff_mask) + 1e-16)
                r_mse      = torch.sum(diff_mask * r_mse_dist) / (torch.sum(diff_mask) + 1e-16)
                
                v_supervised_loss = v_loss_seg + (self.rectification_weight * v_mse)
                r_supervised_loss = r_loss_seg + (self.rectification_weight * r_mse)
                
                ## Unsupervised loss
                if Good_student == 0:  # v is good student
                    v_outputs_clone = output_A_u.clone().detach()
                    Plabel = sharpening(v_outputs_clone)
                    
                    r_consistency_dist = self.mse_loss(output_B_u, Plabel)
                    b, c, w, h = r_consistency_dist.shape
                    r_consistency_loss = torch.sum(r_consistency_dist) / (b * c * w * h)

                    v_loss = v_supervised_loss
                    r_loss = r_supervised_loss + (self.cps_w * r_consistency_loss)
                    consistency_loss = r_consistency_loss
                    
                if Good_student == 1: # r is good student
                    r_outputs_clone = output_B_u.clone().detach()
                    Plabel = sharpening(r_outputs_clone)

                    v_consistency_dist = self.mse_loss(output_A_u, Plabel)
                    b, c, w, h = v_consistency_dist.shape
                    v_consistency_loss = torch.sum(v_consistency_dist) / (b * c * w * h)

                    v_loss = v_supervised_loss + (self.cps_w * v_consistency_loss)
                    r_loss = r_supervised_loss
                    consistency_loss = v_consistency_loss 
                
                
                v_losses.append(v_loss)
                r_losses.append(r_loss)
                v_seg_losses.append(v_loss_seg)
                r_seg_losses.append(v_loss_seg)
                v_dice_losses.append(v_loss_seg_dice)
                r_dice_losses.append(r_loss_seg_dice)
                v_mse_losses.append(v_mse)
                r_mse_losses.append(r_mse)
                consistency_losses.append(consistency_loss)
                
                self.manual_backward(v_loss)
                self.manual_backward(r_loss) 
                self.v_opt.step()
                self.r_opt.step()
                    
            else:
                output_A_l, output_B_l, output_A_u, output_B_u = self.model(patch_images, patch_images2)
                
                output_image.append((output_A_l + output_B_l) / 2.)
                
            pred = (output_A_l + output_B_l) / 2.
            for idx, lesion in enumerate(self.lesion): # total, MA, HE, EX, SE, Sal
                if idx != 0:
                    self.metrics[f'metric_{split}_{lesion}'].update(pred[:,idx-1], (patch_masks[:,idx-1]).type(torch.int64)) 
            
            del pred, patch_masks, output_A_l, output_B_l, output_A_u, output_B_u 
        total_loss = {'v_losses':torch.mean(torch.Tensor(v_losses)),\
                        'r_losses':torch.mean(torch.Tensor(r_losses)), \
                        'v_seg_losses':torch.mean(torch.Tensor(v_seg_losses)), \
                        'r_seg_losses':torch.mean(torch.Tensor(r_seg_losses)), \
                        'v_dice_losses':torch.mean(torch.Tensor(v_dice_losses)), \
                        'r_dice_losses':torch.mean(torch.Tensor(r_dice_losses)), \
                        'v_mse_losses':torch.mean(torch.Tensor(v_mse_losses)), \
                        'r_mse_losses':torch.mean(torch.Tensor(r_mse_losses)), \
                        'consistency_losses':torch.mean(torch.Tensor(consistency_losses))}
                        
        return total_loss, output_image
    
    def on_train_epoch_start(self):
        if self.current_epoch == 0 :
            print("MCF LM EPOCH Train Start", self.consistency_weight )
    
    def training_step(self, batch, batch_idx): 
        self.v_opt, self.r_opt = self.optimizers()
        split = 'train'
        self.cps_w = self.get_current_consistency_weight(self.current_epoch) 
        total_loss, _ = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0])
        
        self.log(f'TotalLoss{split}', total_loss['v_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalRLoss{split}', total_loss['r_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalVSegLoss{split}', total_loss['v_seg_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalRSegLoss{split}', total_loss['r_seg_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalVdiceLoss{split}', total_loss['v_dice_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalRdiceLoss{split}', total_loss['r_dice_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalVMSELoss{split}', total_loss['v_mse_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalRMSELoss{split}', total_loss['r_mse_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalConsistencyLoss{split}', total_loss['consistency_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        
        self.v_opt.param_groups[0]['lr'] = poly_lr(self.current_epoch, self.trainer.max_epochs, self.hparams.lr, 0.9)
        self.r_opt.param_groups[0]['lr'] = poly_lr(self.current_epoch, self.trainer.max_epochs, self.hparams.lr, 0.9)
        
        self.compute_metrics(split='train')
    
        return {'loss': total_loss['v_losses']}
            
    def validation_step(self, batch, batch_idx):
        split = 'val'

        st_time = time.time()
        total_loss, output_image= self.patch_step(split, batch[0][0], batch[1][0], batch[2][0])
        ptime = time.time()-st_time
        self.val_time.append(ptime)
        print("Time for step", ptime)
        
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
            print("Validation Step End", np.mean(self.val_time))
            self.compute_metrics(split='val')
        return {'loss': total_loss['v_losses']}
    
    
    def configure_optimizers(self):
        optimizer_v = torch.optim.AdamW(self.model.generator1.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        optimizer_r = torch.optim.AdamW(self.model.generator2.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        return [optimizer_v, optimizer_r]
    
    
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
                    score = self.metrics_val_prc[f'metric_val_{lesion}'].compute()
                    self.metrics_val_prc[f'metric_val_{lesion}'].reset()
                    for k, v in score.items():
                        self.log(f'{split}/{k}_{lesion}', v.item())
                        total_prc.append(v.item())
                
        if split == 'val':
            self.log(f'val/PRC_total', torch.mean(torch.tensor(total_prc, dtype=torch.float16))) #, batch_size=1
    
        self.log(f'{split}/Dice_total', torch.mean(torch.tensor(total_mean, dtype=torch.float16))) #, batch_size=1 
        self.log(f'{split}/Dice_total_unlabeled', torch.mean(torch.tensor(total_mean_unlabeled, dtype=torch.float16))) #, batch_size=1
        