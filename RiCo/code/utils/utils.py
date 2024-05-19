import torch
import numpy as np
from torch import nn
from .losses import DiceLoss, DC_and_CE_loss, RobustCrossEntropyLoss

def make_loss_function(name, weight=None):
    if name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)

class DistDW:
    def __init__(self, num_cls, version=None, do_bg=False, momentum=0.95):
        self.num_cls = num_cls
        self.do_bg = do_bg
        self.momentum = momentum
        self.weights = torch.ones(num_cls).float().cuda()
        self.version = version
        
    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        if self.version is not None and 'dhc' in self.version:
            P = ((num_each_class.max())+1e-8) / (num_each_class+1e-8)
        else: #ours,ablation
            P = ((num_each_class.max()+num_each_class.min())+1e-8) / (num_each_class+1e-8)
        
            
        P_log = torch.log(P)
        weight = P_log / P_log.max()
        return weight

    def init_weights(self, labeled_dataset, version=None):
        self.version = version
        print("DistDW version", version)

        num_each_class = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list:
            _, _, label = labeled_dataset._get_data(data_id) # h, w, ch
            tmp = label.sum(axis=0).sum(axis=0) 
            num_each_class += tmp
        weights = self._cal_weights(num_each_class)
        self.weights = weights * self.num_cls
        print('Cal num each class', num_each_class)
        print('Initial weights from labeled dataset', self.weights)
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label):
        label_numpy = pseudo_label.data.cpu().numpy()
        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i]
            tmp = label.sum(axis=-1).sum(axis=-1) # bs, ch, h, w
            num_each_class += tmp
            
        if num_each_class.max() == 0:
            cur_weights = torch.FloatTensor([1,1,1,1]).cuda() * self.num_cls # not to be nan
        else:
            cur_weights = self._cal_weights(num_each_class) * self.num_cls
            
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights



class DiffDW:
    def __init__(self, num_cls, accumulate_iters=20): 
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = DiceLoss(num_cls, [1,1,1,1])
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        print('DiffDW init weight : ', self.weights)
        return weights

    def cal_weights(self, pred, label, ver_perf=False, ver_smooth=False):
        zero_idx = np.zeros(self.num_cls)
        for ch in range(label.shape[1]):
            if not label[:,ch,...].any():
                zero_idx[ch] = 1
        cur_dice_loss = self.dice_func((pred).float(), label, training=False) # dice loss per class 
        cur_dice = []
        mul_delta = []
        for ch, i in enumerate(cur_dice_loss):
            if i != -1:
                cur_dice.append((i-1) * -1.)
            else:
                cur_dice.append(self.last_dice[ch]) # means dice coef 0 but 1 for denominator
                
            if cur_dice[ch] == 0:
                if cur_dice[ch] - self.last_dice[ch] > 0:
                    mul_delta.append(1)
                else:
                    mul_delta.append(-1)
            else:
                mul_delta.append(torch.log(cur_dice[ch] / (self.last_dice[ch]+1e-10)))                
        cur_dice = torch.Tensor(cur_dice).cuda()
        mul_delta = torch.Tensor(mul_delta).cuda()
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, torch.tensor(0).to(torch.float).cuda())  * mul_delta
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, torch.tensor(0).to(torch.float).cuda()) * mul_delta
        
        self.last_dice = cur_dice 
        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)

        if ver_perf:
            cur_diff = (self.cls_learn + 1e-8) / (self.cls_unlearn + 1e-8)
            cur_diff = torch.pow(cur_diff, 1/5)
            self.dice_weight = EMA(cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)            weights = cur_diff * self.dice_weight
            weights = weights / weights.max() if weights.max() != 0 else torch.Tensor([0.,0.,0.,0.]).cuda()
            return weights * self.num_cls
        elif ver_smooth:
              
            pow_prm = 1/2 
            cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
            cur_diff = torch.pow(cur_diff, pow_prm)
            
            self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)            
            weights = cur_diff * self.dice_weight 
            weights = weights / weights.max() if weights.max() != 0 else torch.Tensor([0.,0.,0.,0.]).cuda()
            return weights * self.num_cls

            
        else:        
            cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
            cur_diff = torch.pow(cur_diff, 1/5)
            self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
            
            weights = cur_diff * self.dice_weight # cur_diff : delta, dice_weight : dice loss
            weights = weights / weights.max() if weights.max() != 0 else torch.Tensor([0.,0.,0.,0.]).cuda()
            return weights * self.num_cls

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight

def sharpening(pseudo_mask_k, T=0.1):
    numerator = pseudo_mask_k**(1/T)
    denominator = numerator + (1-pseudo_mask_k)**(1/T)
    pl = numerator / denominator
    return pl

def to_numpy(x):
    return x.detach().cpu().numpy()

def split_tensor(tensor, tile_size=256, dim=3, stride=256):
    mask = torch.ones_like(tensor)
    # use torch.nn.Unfold
    # stride = tile_size // 2
    unfold = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    mask_p = unfold(mask)  # [1, 3, 2048, 2048] > [1, 49152(3*128*128), 961]
    patches = unfold(tensor)

    patches = patches.reshape(dim, tile_size, tile_size, -1).permute(3, 0, 1, 2)  # [3, 128, 128, 961] > [961, 3, 128, 128]
    patches_base = torch.zeros(patches.size())

    tiles = []
    for t in range(patches.size(0)):
        tiles.append(patches[[t], :, :, :])
    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))

# mask_t: ones_patches,torch.Size([1, 50176, 289]), base_tensor: zeros_patches, torch.Size([289, 1, 224, 224])
def rebuild_tensor(tensor_list, mask_t, base_tensor, t_size, tile_size=256, dim=3, stride=256):
    # stride = tile_size // 2
    # base_tensor here is used as a container

    for t, tile in enumerate(tensor_list):
        base_tensor[[t], :, :] = tile

    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(dim * tile_size * tile_size, base_tensor.size(0)).unsqueeze(0)
    fold = nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
    # https://discuss.pytorch.org/t/seemlessly-blending-tensors-together/65235/2?u=bowenroom
    overlap_mask_t = fold(mask_t)
    if 0 in np.unique(overlap_mask_t):
        overlap_mask_t[overlap_mask_t==0] = 1
    output_tensor = fold(base_tensor) / overlap_mask_t
    # output_tensor = fold(base_tensor)
    return output_tensor


def find_best_dice_th(pred, target,  interval=20, smooth=1e-5):
    ths = np.linspace(0, 1, interval)
    dices = []
    for i, th in enumerate(ths):
        i_pred = np.uint8(pred.flatten() > th)
        i_true = target.flatten()

        intersection = (i_pred * i_true).sum()
        union = i_pred.sum() + i_true.sum()

        # dice coefficient
        dice = 2.0 * (intersection + smooth) / (union + smooth)
        dices.append(dice)
    max_idx = np.argmax(dices)
    best_dice = dices[max_idx]
    best_th = ths[max_idx]
    return best_dice, best_th

