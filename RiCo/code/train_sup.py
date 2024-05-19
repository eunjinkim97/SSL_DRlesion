import cv2
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

import wandb
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image
from torchmetrics import MetricCollection, Dice, AveragePrecision, AUROC
from pytorch_lightning import LightningModule

from .utils.losses import BinaryDiceLoss, MultiBCELoss
from .utils.utils import to_numpy, split_tensor, rebuild_tensor, mean_dice_coeff, save_seg_result
from .utils.utils import calc_sensitivity_specificity, find_best_dice_th

np.random.seed(1024)
torch.manual_seed(1024)

class LM(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.005,
        batch_size: int = 0,
        version: str = 'multi',
        num_cls: int = 5,
        vis_test: int = 0,
        minibatch_size: int = 16,
        weight_type: str = 'MA'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"], logger=False)
        self.automatic_optimization = False
        self.model = net
        self.version = version
        self.vis_test = vis_test
        self.minibatch_size = minibatch_size
        
        self.weight_type = weight_type
        self.num_cls = self.hparams.num_cls
        self.c_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (127, 0, 127), (0, 127, 127)]
                
        self.seg_weight = 1
        self.lesion = ['total', 'MA' , 'HE', 'EX', 'SE']
        self.seg_criterion = MultiBCELoss(classes=self.num_cls) 
        self.dice_loss = BinaryDiceLoss(self.num_cls)

        ver = ['train', 'val', 'test']
        split_list = [f'metric_{i}_{j}' for i in ver for j in self.lesion]
        
        self.metrics = nn.ModuleDict({
                split: MetricCollection([
                    Dice(average='micro'), 
                ])
                for split in split_list
            })
        
    def forward(self, x1):  
        return self.model(x1) 
    
    
    def patch_step(self, split, x_data, y_data): # x_data2, x_data2_strong,
        if split == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        total_batch = len(x_data) // self.minibatch_size ## 961//64=15, +1
        vis_images, vis_masks, vis_preds = None, None, None
        g_losses, seg_losses, dice_losses = [], [], []
        for j in range(total_batch):
            # print(f'{j}th / total_batch : {total_batch} process ... ')
            if (j + 1) * self.minibatch_size > len(x_data):
                patch_images = x_data[j * self.minibatch_size:]
                patch_masks = y_data[j * self.minibatch_size:]
            else:
                patch_images = x_data[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                patch_masks = y_data[j * self.minibatch_size: (j + 1) * self.minibatch_size]
                
                
            if split == 'train':
                
                self.opt.zero_grad()
                rec_output = self.model(im_q=patch_images)
                seg_loss = self.seg_criterion(rec_output, patch_masks) 
                dice_loss = self.dice_loss(rec_output, patch_masks)
                
                g_loss = (seg_loss + dice_loss) / 2
                g_losses.append(g_loss.item())
                seg_losses.append(seg_loss.item())
                dice_losses.append(dice_loss.item())
                
               
                self.manual_backward(g_loss)# clip gradients
                self.clip_gradients(self.opt, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
                self.opt.step()
                    
            else:
                rec_output = self.model(im_q=patch_images)
                seg_loss = self.seg_criterion(rec_output, patch_masks) 
                dice_loss = self.dice_loss(rec_output, patch_masks)
                
                
                g_loss = (seg_loss + dice_loss) / 2
                g_losses.append(g_loss.item())
                seg_losses.append(seg_loss.item())
                dice_losses.append(dice_loss.item())
                
            
            if vis_images == None and patch_masks.max() != 0:
                vis_images, vis_masks, vis_preds = patch_images, patch_masks, rec_output
    
            for idx, lesion in enumerate(self.lesion): # total, MA, HE, EX, SE, Sal
                if idx != 0:
                    self.metrics[f'metric_{split}_{lesion}'].update(rec_output[:,idx-1], (patch_masks[:,idx-1]).type(torch.int64)) 
    
    
        total_loss = {'g_losses':torch.mean(torch.Tensor(g_losses)),\
                    'seg_losses':torch.mean(torch.Tensor(seg_losses)), \
                    'dice_losses':torch.mean(torch.Tensor(dice_losses))}
        return total_loss, vis_images, vis_masks, vis_preds
    
    def on_train_epoch_start(self):
        self.scheduler = self.lr_schedulers()
        self.opt = self.optimizers()
    
    def training_step(self, batch, batch_idx): 
        split = 'train'
        total_loss, x1_data, x1_target, rec_output = self.patch_step(split,  batch[0][0], batch[1][0])
        self.log(f'TotalLoss{split}', total_loss['g_losses'], batch_size=1)
        self.log(f'SegLoss{split}', total_loss['seg_losses'], batch_size=1)
        self.log(f'DiceLoss{split}', total_loss['dice_losses'], batch_size=1)
        losses = total_loss['g_losses']
        
            
        # if self.current_epoch % 10 == 0 and batch_idx % 100 == 0: # bs, 4, 224, 224, 3 
        #     figs = self.save_figure(x1_data, x1_target, rec_output, x1_data, x1_data, x1_target, rec_output, done_sig=True) #(16, 5, 224, 224, 3)
        #     pair_imgs = torch.clip(torch.tensor(figs), 0, 255).to(torch.uint8).permute((0,1,4,2,3))
        #     grid = make_grid(pair_imgs.reshape(pair_imgs.size(0)*7,3,512,512), nrow=7)
        #     self.logger.experiment.log({f'{split}_output_Epoch:{self.current_epoch}':wandb.Image(np.transpose(grid.numpy(),[1,2,0]))})

        return {'loss': losses}
    
    def on_train_epoch_end(self):           
        self.compute_metrics(split='train')
            
    def validation_step(self, batch, batch_idx):
        split = 'val'
        total_loss, x1_data, x1_target, rec_output = self.patch_step(split,  batch[0][0], batch[1][0])
        self.log(f'TotalLoss{split}', total_loss['g_losses'], batch_size=1)
        self.log(f'SegLoss{split}', total_loss['seg_losses'], batch_size=1)
        self.log(f'DiceLoss{split}', total_loss['dice_losses'], batch_size=1)
        losses = total_loss['g_losses']
            
        return {'loss': losses}

    def on_validation_epoch_end(self):
        self.compute_metrics(split='val')

    def compute_metrics(self, split):
        # st = time.time()
        for idx, lesion in enumerate(self.lesion):
            if idx == 0:
                total_mean = []
            elif idx == 5:
                continue
            else:
                score = self.metrics[f'metric_{split}_{lesion}'].compute()
                self.metrics[f'metric_{split}_{lesion}'].reset()
                for k, v in score.items():
                    self.log(f'{split}/{k}_{lesion}', v.item()) #, batch_size=1
                    total_mean.append(v.item())       
                         
        self.log(f'{split}/Dice_total', torch.mean(torch.tensor(total_mean, dtype=torch.float16))) #, batch_size=1 
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs//2, eta_min=1e-6)
        return [optimizer], [scheduler]
    