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
np.random.seed(1024)
torch.manual_seed(1024)

class LM_CPS(LM):    
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
        super(LM_CPS, self).__init__(
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
        self.loss_func = make_loss_function(sup_loss)
        self.cps_loss_func = make_loss_function(consistency_loss)
        
     
    
    def on_train_epoch_start(self):
        if self.current_epoch == 0 :
            self.v_opt, self.r_opt = self.optimizers()   
    
    def patch_step(self, split, x_data, x_data2, y_data): 
        if split == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        total_batch = len(x_data) // self.minibatch_size
        v_losses, r_losses, consistency_losses, output_image = [], [], [], []
        for j in range(total_batch):
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
                
                # sup seg (w_ce+dice)
                v_loss_seg, _, _ = self.loss_func(output_A_l, patch_masks) # w_ce+dice outputs total, dice, ce loss
                r_loss_seg, _, _ = self.loss_func(output_B_l, patch_masks) # output must be passed through sigmoid
                loss_sup = v_loss_seg + r_loss_seg
                
                # cps (wce)
                max_A = (sharpening(output_A_u) >= 0.5).clone().detach()
                max_B = (sharpening(output_B_u) >= 0.5).clone().detach()
                loss_cps = self.cps_loss_func(output_A_u, max_B) + self.cps_loss_func(output_B_u, max_A)
                
                loss = loss_sup + (self.cps_w * loss_cps)
                                
                v_losses.append(v_loss_seg)
                r_losses.append(r_loss_seg)
                consistency_losses.append(loss_cps)
                
                self.manual_backward(loss)
                self.v_opt.step()
                self.r_opt.step()
                    
            else:
                output_A_l, output_B_l, output_A_u, output_B_u = self.model(patch_images, patch_images2)
                max_B = (sharpening(output_B_u) >= 0.5).clone().detach()
                output_image.append((output_A_l + output_B_l) / 2.)

            pred = (output_A_l + output_B_l) / 2.
            
            for idx, lesion in enumerate(self.lesion): # total, MA, HE, EX, SE, Sal
                if idx != 0:
                    self.metrics[f'metric_{split}_{lesion}'].update(pred[:,idx-1], (patch_masks[:,idx-1]).type(torch.int64)) 
            
            del output_A_l, output_B_l, output_A_u, output_B_u
            
            
        total_loss = {'v_losses':torch.mean(torch.Tensor(v_losses)),\
                        'r_losses':torch.mean(torch.Tensor(r_losses)), \
                        'consistency_losses':torch.mean(torch.Tensor(consistency_losses))}
                        
        return total_loss, output_image
    
    def configure_optimizers(self):
        optimizer_v = torch.optim.AdamW(self.model.generator1.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        optimizer_r = torch.optim.AdamW(self.model.generator2.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        return [optimizer_v, optimizer_r]
        
    def training_step(self, batch, batch_idx): 
        self.v_opt, self.r_opt = self.optimizers()
        split = 'train'
        self.cps_w = self.get_current_consistency_weight(self.current_epoch) 

        total_loss, _ = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0])
        
        self.log(f'TotalLoss{split}', total_loss['v_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalRLoss{split}', total_loss['r_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'TotalConsistencyLoss{split}', total_loss['consistency_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
            
            
        self.v_opt.param_groups[0]['lr'] = poly_lr(self.current_epoch, self.trainer.max_epochs, self.hparams.lr, 0.9)
        self.r_opt.param_groups[0]['lr'] = poly_lr(self.current_epoch, self.trainer.max_epochs, self.hparams.lr, 0.9)
        
        return {'loss': total_loss['v_losses']}
    
    def validation_step(self, batch, batch_idx):
        split = 'val'
        total_loss, output_image = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0])
        
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
    
            
        return {'loss': total_loss['v_losses']}   

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
                        
                # score = self.metrics[f'metric_{split}_{lesion}_unlabeled'].compute()
                # self.metrics[f'metric_{split}_{lesion}_unlabeled'].reset()
                # for k, v in score.items():
                #     self.log(f'{split}/{k}_{lesion}_unlabeled', v.item()) #, batch_size=1
                #     total_mean_unlabeled.append(v.item())            
                
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
        # print("Compute Metrics (sec) : ", time.time()-st)
        
       