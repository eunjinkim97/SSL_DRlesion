import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class RobustCrossEntropyLoss(nn.Module): # nn.CrossEntropyLoss
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def __init__(self, n_classes=4, weight=None, loss_type='BCE'):
        super().__init__()
        self.loss = torch.nn.BCELoss() 
        self.weight = torch.FloatTensor(weight).cuda() if weight is not None else [1.] * n_classes     
    
    def forward(self, input, target):
        assert len(target.shape) == len(input.shape)
        loss = 0
        for ch in range(target.shape[1]):
            target_ch = target[:,ch]
            input_ch = input[:,ch] 
            loss += self.loss(input_ch, target_ch.float()) * self.weight[ch] 
        return loss / target.shape[1]

    def update_weight(self, weight):
        self.weight = weight
        

class DC_and_CE_loss(nn.Module):
    def __init__(self, w_dc=None, w_ce=None, aggregate="sum", weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None, n_classes=4):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()

        ce_kwargs = {'weight': w_ce if w_ce is not None else [1.]*n_classes ,'n_classes':n_classes}
        if ignore_label is not None:
            ce_kwargs['reduction'] = 'none'
        
        self.log_dice = log_dice
        self.weight_dice = weight_dice # 1
        self.weight_ce = weight_ce # 1
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ignore_label = ignore_label        
        self.dc = DiceLoss(n_classes, weight=w_dc)

    def forward(self, net_output, target, ours=False, classwise=False):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if ours:
            dc_loss, dc_losses_nw = self.dc(net_output, target, ours=ours, classwise=classwise) 

            ce_loss = self.ce(net_output, target) 

            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            
            return result, dc_loss, ce_loss, dc_losses_nw

        
        else:            
            dc_loss = self.dc(net_output, target) 

            ce_loss = self.ce(net_output, target) 

            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            
            return result, dc_loss, ce_loss
        
    def update_weight(self, weight):
        self.dc.weight = weight
        self.ce.weight = weight


class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes       
        self.weight = weight if weight is not None else [1.,] * self.n_classes
        
    def _dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.clamp(pred, smooth, 1.0 - smooth)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        # dice coefficient
        dice = 2.0 * (intersection + smooth) / (union + smooth)

        # dice loss
        dice_loss = 1.0 - dice

        return dice_loss
    
    def forward(self, inputs, target, training=True, ours=False, classwise=False):
            
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        cls_loss = []
        loss = 0

        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * self.weight[i]
            cls_loss.append(dice)
        if training:
            if ours:
                if classwise:
                    return loss / self.n_classes, torch.Tensor(cls_loss)
                
                else:
                    
                    return loss / self.n_classes, torch.sum(torch.Tensor(cls_loss)) / self.n_classes
            else:
                return loss / self.n_classes
        
        else: # when cal_weight
            return cls_loss

# For supervised loss
class BinaryDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    
    def _dice_loss(self, pred, target, smooth=1e-10):
        pred = torch.clamp(pred, smooth, 1.0 - smooth)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        # dice coefficient
        dice = 2.0 * (intersection + smooth) / (union + smooth)

        # dice loss
        dice_loss = 1.0 - dice

        return dice_loss

    def forward(self, inputs, target, weight=None, softmax=False, one_hot=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        
        cnt_cls = 0
        loss = 0.0
        for i in range(0, self.n_classes):
            cnt_cls += 1
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]

        return loss / cnt_cls if cnt_cls > 0 else loss # 0


class MultiBCELoss(nn.Module):
    def __init__(self,classes:int =5):
        super(MultiBCELoss, self).__init__()
        self.classes = classes
        self.smooth = 1e-6  # set '1e-4' when train with FP16

    def forward(self, output, target):
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)

        losses = 0.
        for clss in range(self.classes):
            output_ch = prob[:,clss,...]
            pos_mask = (target[:,clss,...] != 0).float()

            bce = F.binary_cross_entropy(output_ch, pos_mask) ## it calculated Ygt*Log(Ypred) + (1-Ygt)*Log(1-Ypred)
            losses += bce
        
        return losses / self.classes
    

class WeightedCrossEntropyLoss(nn.Module):    
    def __init__(self, n_classes=4, weight=None, loss_type='BCE'):
        super().__init__()
        self.loss = torch.nn.BCELoss(reduction='none') 
        self.weight = torch.FloatTensor(weight).cuda() if weight is not None else [1.] * n_classes     
    
    def forward(self, input, target, weight_map=None):
        assert len(target.shape) == len(input.shape)
        b = input.shape[0]
        loss_ch = 0
        loss_total = 0
        for ch in range(target.shape[1]):
            target_ch = target[:,ch]
            input_ch = input[:,ch] 
            loss_ch = self.loss(input_ch, target_ch.float())
            loss_ch = loss_ch.view(b, -1)
                
            if weight_map is not None:
                weight = weight_map[:,ch].view(b, -1).detach()
                loss_ch = loss_ch * weight
                
            loss_total += torch.mean(loss_ch)
        return loss_total / target.shape[1]

    def update_weight(self, weight):
        self.weight = weight