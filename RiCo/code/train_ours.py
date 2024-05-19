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

np.random.seed(1024)
torch.manual_seed(1024)

class LM(LightningModule):
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
        rec_w: float = 0.5, 
        sup_loss: str = 'w_ce+dice',
        consistency_loss: str = 'wmse'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"], logger=False)
        self.model = net
        self.version = version
        self.vis_test = vis_test
        self.num_cls = num_cls
        self.c_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (127, 0, 127), (0, 127, 127)]
        self.automatic_optimization = False
        self.minibatch_size = minibatch_size
        self.val_time = []
        
        
        self.rec_w = rec_w
        self.consistency_weight = cps_w
        
        
        self.sup_loss = sup_loss
        self.consistency_loss = consistency_loss
        self.mse_loss = torch.nn.MSELoss(reduction='none') # (a-b)**2
        
        self.dice_th = [0.4,0.4,0.4,0.4]
        self.lesion = ['total', 'MA' , 'HE', 'EX', 'SE']
        
        ver = ['train', 'val', 'test']
        split_list = [f'metric_{i}_{j}' for i in ver for j in self.lesion] # + [f'metric_{i}_{j}_unlabeled' for i in ver for j in self.lesion] # 3*6=18
        
        self.metrics = nn.ModuleDict({
                split: MetricCollection([
                    Dice(average='micro'), # cannot use same name for both metrics
                ])
                for split in split_list
            })
        
        self.se_idx = [0, 1, 4, 5, 6, 9, 12, 13, 15, 16, 17, 18, 19, 20]
        self.test_input_size_x, self.test_input_size_y = 2848, 4288
        self.patch_size = 512
        
        temp_tensor = torch.ones([1, self.num_cls, self.test_input_size_x, self.test_input_size_y], dtype=torch.float16)
        _, self.mask_t, self.base_tensor, self.t_size = split_tensor(temp_tensor, tile_size=self.patch_size, dim=self.num_cls)
        
        self.metrics_val_prc = nn.ModuleDict({
                split: MetricCollection([
                    AveragePrecision(task="binary"),
                    ])
                for split in [f'metric_val_{j}' for j in self.lesion] # total, MA, HE, EX, SE
            })
        
    def forward(self, x1, x2):  
        return self.model(x1, x2) 
    
    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    
    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency_weight * self.sigmoid_rampup(epoch, self.trainer.max_epochs)
    
    
    def smooth2avg(self, input_list, adjustment_factor=0.99):
        avg = sum(input_list) / len(input_list)
        smoothed_list = []
        for item in input_list:
            adjusted_item = item + (avg - item) * adjustment_factor
            smoothed_list.append(adjusted_item)
        return smoothed_list
        
    def patch_step(self, split, x_data, x_data2, y_data, y_data_sal_un): 
        if split == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        total_batch = len(x_data) // self.minibatch_size ## 961//64=15, +1
        vis_images, vis_masks, vis_preds = None, None, None
        vis_images2_1, vis_pseudo = None, None
        vis_images2_2, vis_preds2 = None, None
        v_losses, r_losses, v_seg_losses, r_seg_losses, v_dice_losses, r_dice_losses, v_mse_losses, r_mse_losses, consistency_losses = [], [], [], [], [], [], [], [], []
        output_image = []
        for j in range(total_batch):
            # print(f'{j}th / total_batch : {total_batch} process ... ')
            if (j + 1) * self.minibatch_size > len(x_data):
                # continue # drop_last
                patch_images = x_data[j * self.minibatch_size:]
                patch_images2 = x_data2[j * self.minibatch_size:]
                patch_masks = y_data[j * self.minibatch_size:]
                # patch_saliency = y_data_sal[j * self.minibatch_size:]
                patch_saliency_un = y_data_sal_un[j * self.minibatch_size:]

            else:
                patch_images = x_data[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                patch_images2 = x_data2[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                patch_masks = y_data[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                # patch_saliency = y_data_sal[j * self.minibatch_size: (j + 1) * self.minibatch_size] # 1ch
                patch_saliency_un = y_data_sal_un[j * self.minibatch_size: (j + 1) * self.minibatch_size] # 1ch
    
            if split == 'train':
                    
                self.v_opt.zero_grad()
                self.r_opt.zero_grad()

                output_A_l, output_B_l, output_A_u, output_B_u = self.model(patch_images, patch_images2)
                v_label, r_label = patch_masks.clone(), patch_masks.clone()

                # sup weight
                weight_A = self.diffdw.cal_weights(sharpening(output_A_l.detach())>=0.5, patch_masks.detach(), ver_smooth=True) # using Dice
                weight_B = self.distdw.get_ema_weights(sharpening(output_B_l.detach())>=0.5) # pseudo label > class probability

                # sup (ce+dice)
                af = np.clip(0.9 - (self.current_epoch/self.trainer.max_epochs), 0.1, 0.9) # 0.9~0.1
                weight_A = self.smooth2avg(weight_A, adjustment_factor=af) # 0.9~0.1
                weight_B = self.smooth2avg(weight_B, adjustment_factor=af) # 0.9~0.1
                
                self.loss_func_A.update_weight(weight_A) # w_bce+dice
                self.loss_func_B.update_weight(weight_B) # w_bce+dice
                v_loss_seg, v_loss_seg_dice, _, v_loss_seg_dice_nw = self.loss_func_A(output_A_l, patch_masks, ours=True) # output must be passed through sigmoid
                r_loss_seg, r_loss_seg_dice, _, r_loss_seg_dice_nw = self.loss_func_B(output_B_l, patch_masks, ours=True) # output must be passed through sigmoid
                            
                max_A_l = sharpening(output_A_l.detach()) >= 0.5 #torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                max_B_l = sharpening(output_B_l.detach()) >= 0.5 #torch.argmax(output_B.detach(), dim=1, keepdim=True).long()
                                                              
                # Rectification loss 
                diff_mask = ((max_A_l == 1) ^ (max_B_l == 1)).to(torch.int32).clone().detach() # ^ means XOR, & means AND, | means OR
                # diff_mask = torch.clip(diff_mask + patch_saliency, 0, 1).to(torch.int32)
                v_mse_dist = self.mse_loss(output_A_l, v_label) # MSE
                r_mse_dist = self.mse_loss(output_B_l, r_label) # MSE
                v_mse      = torch.sum(diff_mask * v_mse_dist) / (torch.sum(diff_mask) + 1e-16)
                r_mse      = torch.sum(diff_mask * r_mse_dist) / (torch.sum(diff_mask) + 1e-16)
                
                v_supervised_loss = v_loss_seg + (self.rec_w * v_mse)
                r_supervised_loss = r_loss_seg + (self.rec_w * r_mse)

                ## Unsupervised loss                                                                                                                          
                if v_loss_seg_dice_nw < r_loss_seg_dice_nw: # v: eff, r: u2net # not weighted dice
                    Good_student = 0 # v is good student
                else:
                    Good_student = 1 # r is good student
                
                if Good_student == 0:  
                    weightB_winnerA = self.distdw_winner.get_ema_weights(sharpening(output_A_u.detach()) >= 0.5) # pseudo label > class probability
                    self.cps_loss_func_B.update_weight(weightB_winnerA) # w_cez
                    
                    max_A = sharpening(output_A_u.detach()) >= 0.5 if 'ce' in self.consistency_loss else sharpening(output_A_u.detach())
                    consistency_loss = self.cps_loss_func_B(output_B_u, max_A) # output must be logits                
                    consistency_loss = torch.sum(patch_saliency_un * consistency_loss.cuda()) / (self.num_cls * torch.sum(patch_saliency_un) + 1e-16)

                    v_loss = v_supervised_loss
                    r_loss = r_supervised_loss + (self.cps_w * consistency_loss)

                if Good_student == 1: 
                    max_B = sharpening(output_B_u.detach()) >= 0.5 if 'ce' in self.consistency_loss else sharpening(output_B_u.detach())
                    weightA_winnerB = self.diffdw_winner.cal_weights(sharpening(output_B_l.detach())>=0.5, patch_masks.detach(), ver_perf=True) # using Dice
                    self.cps_loss_func_A.update_weight(weightA_winnerB) # w_cez
                    
                    consistency_loss = self.cps_loss_func_A(output_A_u, max_B) # output must be logits
                    consistency_loss = torch.sum(patch_saliency_un * consistency_loss.cuda()) / (self.num_cls * torch.sum(patch_saliency_un) + 1e-16)
                    v_loss = v_supervised_loss + (self.cps_w * consistency_loss)
                    r_loss = r_supervised_loss

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
                    if patch_masks[:,idx-1].any():
                        self.metrics[f'metric_{split}_{lesion}'].update(pred[:,idx-1], (patch_masks[:,idx-1]).type(torch.int64)) 
                    # if Plabel[:,idx-1].any():
                    #     self.metrics[f'metric_{split}_{lesion}_unlabeled'].update(Poutput[:,idx-1], (Plabel[:,idx-1]).type(torch.int64))  
    
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
            self.v_opt, self.r_opt = self.optimizers()
            print("define the losses ")

            # make loss function
            self.diffdw = DiffDW(self.num_cls, accumulate_iters=50)
            self.distdw = DistDW(self.num_cls, momentum=0.99)
            weight_A = self.diffdw.init_weights() # model difficulty
            weight_B = self.distdw.init_weights(self.trainer.datamodule.train_dataset) #labeled_loader.dataset) # data aware
            self.loss_func_A     = make_loss_function(self.sup_loss, weight_A) 
            self.loss_func_B     = make_loss_function(self.sup_loss, weight_B)
                       
            
            self.diffdw_winner = DiffDW(self.num_cls, accumulate_iters=50)
            self.distdw_winner = DistDW(self.num_cls, momentum=0.99)
            weight_A = self.diffdw_winner.init_weights() # model difficulty
            weight_B = self.distdw_winner.init_weights(self.trainer.datamodule.train_dataset) #labeled_loader.dataset) # data aware
            self.cps_loss_func_A = make_loss_function(self.consistency_loss, weight_A) 
            self.cps_loss_func_B = make_loss_function(self.consistency_loss, weight_B) 
    
        else:
            if not hasattr(self, 'diffdw'):
                # make loss function
                self.diffdw = DiffDW(self.num_cls, accumulate_iters=50)
                self.distdw = DistDW(self.num_cls, momentum=0.99)
                weight_A = self.diffdw.init_weights() # model difficulty
                weight_B = self.distdw.init_weights(self.trainer.datamodule.train_dataset) #labeled_loader.dataset) # data aware
                self.loss_func_A     = make_loss_function(self.sup_loss, weight_A) 
                self.loss_func_B     = make_loss_function(self.sup_loss, weight_B)
                        
                
                self.diffdw_winner = DiffDW(self.num_cls, accumulate_iters=50)
                self.distdw_winner = DistDW(self.num_cls, momentum=0.99)
                weight_A = self.diffdw_winner.init_weights() # model difficulty
                weight_B = self.distdw_winner.init_weights(self.trainer.datamodule.train_dataset) #labeled_loader.dataset) # data aware
                self.cps_loss_func_A = make_loss_function(self.consistency_loss, weight_A) 
                self.cps_loss_func_B = make_loss_function(self.consistency_loss, weight_B) 
                
                
                
    def training_step(self, batch, batch_idx): 
        self.v_opt, self.r_opt = self.optimizers()
        split = 'train'
        self.cps_w = self.get_current_consistency_weight(self.current_epoch) 
        total_loss, _ = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0], batch[3][0])
        
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
        total_loss, output_image = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0], batch[2][0])
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
                if batch[3] in self.se_idx:
                    self.metrics_val_prc[f'metric_val_{lesion}'].update(output_image[idx-1], (y_data[idx-1]).type(torch.int64))
        print("Validation Step End", batch_idx)
        if batch_idx == 26:
            print("Validation Step End,,, AVG TIME is", np.mean(self.val_time))
            self.val_time = []
            self.compute_metrics(split='val')
        return {'loss': total_loss['v_losses']}

    def compute_metrics(self, split):
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
        
    
    def configure_optimizers(self):
        optimizer_v = torch.optim.AdamW(self.model.generator1.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        optimizer_r = torch.optim.AdamW(self.model.generator2.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        return [optimizer_v, optimizer_r]
    
