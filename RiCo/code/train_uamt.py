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

class LM_UAMT(LM):    
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
        super(LM_UAMT, self).__init__(
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
        
        
    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha) # more slowly update EMA(before ema_model update)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
    def softmax_mse_loss(self, input_logits, target_logits, sigmoid=False):
        """Takes softmax on both sides and returns MSE loss
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
        if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        if sigmoid:
            input_softmax = torch.sigmoid(input_logits)
            target_softmax = torch.sigmoid(target_logits)
        else:
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)

        mse_loss = (input_softmax-target_softmax)**2
        return mse_loss

    def patch_step(self, split, x_data, x_data2, y_data, batch_idx=1): 
        if split == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        total_batch = len(x_data) // self.minibatch_size ## 961//64=15, +1
        g_losses, seg_losses, consistency_losses, output_image = [], [], [], []
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
                self.opt.zero_grad()
                outputs = self.model(im_labeled=patch_images, im_unlabeled=patch_images2)
                #{'q_output_labeled':pred_labeled, 'uncertainty':uncertainty, 'noisy_ema_output':noisy_ema_output}
                
                ## Labeled
                rec_output = torch.sigmoid(outputs['q_output_labeled'])
                seg_loss, _, _ = self.loss_func(rec_output, patch_masks) # bce + dice
                
                ## Unlabeled
                consistency_dist = self.softmax_mse_loss(outputs['q_output_unlabeled'], outputs['noisy_ema_output'], sigmoid=True)
                threshold = (0.75+0.25*self.sigmoid_rampup(self.current_epoch, self.trainer.max_epochs))*np.log(2)
                mask = (outputs['uncertainty'] < threshold).float()
                consistency_loss = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
                g_loss = (seg_loss/2.) + (self.cps_w * consistency_loss)

                g_losses.append(g_loss)
                seg_losses.append(seg_loss)
                consistency_losses.append(consistency_loss)
                
                self.manual_backward(g_loss)# clip gradients
                self.opt.step()
                self.update_ema_variables(self.model.encoder_q, self.model.encoder_k, 0.999, self.current_epoch)
                    
            else:
                outputs = self.model(im_labeled=patch_images, im_unlabeled=patch_images2)                
                rec_output = torch.sigmoid(outputs['q_output_labeled'])

            output_image.append(rec_output)
            for idx, lesion in enumerate(self.lesion): # total, MA, HE, EX, SE, Sal
                if idx != 0:
                    self.metrics[f'metric_{split}_{lesion}'].update(rec_output[:,idx-1], (patch_masks[:,idx-1]).type(torch.int64)) 
            
            del rec_output, patch_masks,outputs
            
        total_loss = {'g_losses':torch.mean(torch.Tensor(g_losses)),\
                    'seg_losses':torch.mean(torch.Tensor(seg_losses)), \
                    'consistency_losses':torch.mean(torch.Tensor(consistency_losses))}
        return total_loss, output_image

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.encoder_q.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        return optimizer

    def on_train_epoch_start(self):
        self.opt = self.optimizers()
    
    def training_step(self, batch, batch_idx): 
        split = 'train'
        self.cps_w = self.get_current_consistency_weight(self.current_epoch) 
        total_loss, _ = self.patch_step(split, batch[0][0], batch[1][0], batch[2][0], batch_idx)
                
        self.log(f'TotalLoss{split}', total_loss['g_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'SegLoss{split}', total_loss['seg_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
        self.log(f'ConsistencyLoss{split}', total_loss['consistency_losses'], batch_size=self.hparams.batch_size, on_step=False, on_epoch=True)
            
        self.opt.param_groups[0]['lr'] = poly_lr(self.current_epoch, self.trainer.max_epochs, self.hparams.lr, 0.9)
        
        return {'loss': total_loss['g_losses']}
    
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
    
        return {'loss': total_loss['g_losses']}
 