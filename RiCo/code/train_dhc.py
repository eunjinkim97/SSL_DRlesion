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
    
class LM_DHC(LM):    
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
        super(LM_DHC, self).__init__(
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
        
        # make loss function
        self.diffdw = DiffDW(self.num_cls, accumulate_iters=50)
        self.distdw = DistDW(self.num_cls, momentum=0.99)
        self.sup_loss = sup_loss
        self.consistency_loss = consistency_loss

        
    def patch_step(self, split, x_data, x_data2, y_data): 
        if split == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        total_batch = len(x_data) // self.minibatch_size ## 961//64=15, +1
        vis_images, vis_masks, vis_preds = None, None, None
        vis_images2_1, vis_pseudo = None, None
        vis_images2_2, vis_preds2 = None, None
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

                # output must be logits 
                output_A_l, output_B_l, output_A_u, output_B_u = self.model(patch_images, patch_images2)
                
                # cps (ce only)
                output_A = (torch.cat([output_A_l, output_A_u], dim=0))
                output_B = (torch.cat([output_B_l, output_B_u], dim=0))
                max_A = sharpening(output_A.detach()) >= 0.5 #torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                max_B = sharpening(output_B.detach()) >= 0.5 #torch.argmax(output_B.detach(), dim=1, keepdim=True).long()

                weight_A = self.diffdw.cal_weights(output_A_l.detach(), patch_masks.detach()) # using Dice
                weight_B = self.distdw.get_ema_weights(output_B_u.detach()) # pseudo label > class probability

                self.loss_func_A.update_weight(weight_A) # w_bce+dice
                self.loss_func_B.update_weight(weight_B) # w_bce+dice
                self.cps_loss_func_A.update_weight(weight_A) # w_ce
                self.cps_loss_func_B.update_weight(weight_B) # w_ce

                s1, _, _ = self.loss_func_A(output_A_l, patch_masks) 
                s2, _, _ = self.loss_func_B(output_B_l, patch_masks) # output must be logits
                loss_sup = s1 + s2
                loss_cps = self.cps_loss_func_A(output_A, max_B) + self.cps_loss_func_B(output_B, max_A) # output must be logits
                loss = loss_sup + (self.cps_w * loss_cps)
                    
                losses.append(loss)
                supervised_losses.append(loss_sup)
                consistency_losses.append(loss_cps)
                
                self.manual_backward(loss) 
                self.v_opt.step()
                self.r_opt.step()
                    
            else:
                # output must be logits 
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
        del output_A_l, output_B_l, output_A_u, output_B_u 
        return total_loss, output_image
    
    
    def configure_optimizers(self):
        optimizer_v = torch.optim.AdamW(self.model.generator1.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        optimizer_r = torch.optim.AdamW(self.model.generator2.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        return [optimizer_v, optimizer_r]
     
    def on_train_epoch_start(self):
        if self.current_epoch == 0 :
            print("define the losses ")
            weight_A = self.diffdw.init_weights() # model difficulty
            weight_B = self.distdw.init_weights(self.trainer.datamodule.train_dataset, version='dhc') #labeled_loader.dataset) # data aware

            self.loss_func_A     = make_loss_function(self.sup_loss, weight_A) 
            self.loss_func_B     = make_loss_function(self.sup_loss, weight_B)
            self.cps_loss_func_A = make_loss_function(self.consistency_loss, weight_A) 
            self.cps_loss_func_B = make_loss_function(self.consistency_loss, weight_B) 
                      
    
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
        total_loss, output_image = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0])

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
            self.compute_metrics(split)
            
        return {'loss': total_loss['losses']}
