import os
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A

from torch.utils.data import Dataset
from .utils import split_tensor
from .augmentation import augment_imgs

random.seed(1024)
class Base(Dataset):
    def __init__(self, train_or_test: str, mask_paths: list, image_paths: list, img_size: int=2048, patch_size:int=512, num_cls: int=5, version:str='multi', stride:int=256):
        self.mask_paths = mask_paths
        self.image_paths = image_paths

        self.train_or_test = train_or_test
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_cls = num_cls
        self.wanted_size = (self.img_size, self.img_size)
        
        self.version = version # none, clahe, blur, g_ch
        self.stride = stride
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[..., ::-1]
        mask = np.load(self.mask_paths[idx], allow_pickle=True)
        mask = mask[...,:self.num_cls]
        
        ## Aug & Resize & Normliazation of mask
        if self.train_or_test == 'train':
            image, mask_aug = augment_imgs(image, mask) #image_aug[~255], mask_aug[~1]
            mask_aug = np.uint8(mask_aug) # 0~1
            mask_input = cv2.threshold(mask_aug, 0.2, 1, cv2.THRESH_BINARY)[1]
            
        else: ############### TEST AND VALIDATION ################     
            if mask.max() == 255:
                mask_input = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]
            else:
                mask_input = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)[1]
            
        ## Image Normalization
        mask_input = np.transpose(mask_input , [2, 0, 1]) 
        mask_tensor = torch.FloatTensor(mask_input)
        image_tensor = torch.FloatTensor(np.transpose(image / 255., [2, 0, 1]))
        
        if self.patch_size != 1280:
            # image to patches
            tile_images, mask_t, base_tensor, t_size = split_tensor(image_tensor.unsqueeze(0), tile_size=self.patch_size, stride=self.stride)  # [961, 3, 128, 128]
            tile_patch_images = torch.cat(tile_images, dim=0)
            tile_masks, mask_t, base_tensor, t_size = split_tensor(mask_tensor.unsqueeze(0), tile_size=self.patch_size, dim=self.num_cls, stride=self.stride)
            tile_patch_masks = torch.cat(tile_masks, dim=0) ## output binary mask : [0, 1]
    
            
            if self.train_or_test == 'train':
                new_idx = torch.randperm(tile_patch_images.size()[0])
                tile_patch_images = tile_patch_images[new_idx]
                tile_patch_masks = tile_patch_masks[new_idx]
                return tile_patch_images, tile_patch_masks
            else:
                return tile_patch_images, tile_patch_masks, self.image_paths[idx]
        else:
            if self.train_or_test == 'train':
                return image_tensor, mask_tensor
            else:
                return image_tensor, mask_tensor, self.image_paths[idx]

        
        
class Base_Unlabeled(Dataset):
    def __init__(self, train_or_test: str, mask_paths: list, image_paths: list, unlabeled_image_paths: list, img_size: int=2048, patch_size:int=512, num_cls: int=5, version:str='multi', stride:int=256):
        self.mask_paths = mask_paths
        self.image_paths = image_paths
        self.unlabeled_image_paths = unlabeled_image_paths
        self.train_or_test = train_or_test
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_cls = num_cls
        self.wanted_size = (self.img_size, self.img_size)
        self.ids_list = [i for i in range(len(mask_paths))]
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2),
            ])
        
        self.version = version # none, clahe, blur, g_ch
        self.stride = stride
        
    def _get_data(self, data_id):
        # image, label = read_data(data_id, task=self.task)
        image = cv2.imread(self.image_paths[data_id])[..., ::-1]
        mask = np.load(self.mask_paths[data_id], allow_pickle=True)
        mask = mask[...,:self.num_cls]
        
        return data_id, image, mask

    def mask_transform(self, mask_np):
        mask_tensor = torch.FloatTensor(np.transpose(mask_np, [2, 0, 1]))
        tile_masks, _, _, _ = split_tensor(mask_tensor.unsqueeze(0), tile_size=self.patch_size, dim=self.num_cls, stride=self.stride)
        return torch.cat(tile_masks, dim=0)

    def img_transform(self, img_np):
        image_tensor = torch.FloatTensor(np.transpose(img_np/255., [2, 0, 1]))
        tile_images, _, _, _ = split_tensor(image_tensor.unsqueeze(0), tile_size=self.patch_size, stride=self.stride)  # [961, 3, 128, 128]
        return torch.cat(tile_images, dim=0)

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[..., ::-1]
        mask = np.load(self.mask_paths[idx], allow_pickle=True)
        mask = mask[...,:self.num_cls]
        
        rand_idx_unlabel = random.sample(range(len(self.unlabeled_image_paths)),1)[0]
        image2 = cv2.imread(self.unlabeled_image_paths[rand_idx_unlabel])[..., ::-1]
                
        ## Aug & Resize & Normliazation of mask
        if self.train_or_test == 'train':
            image, mask_aug = augment_imgs(image, mask) #image_aug[~255], mask_aug[~1]
            mask_aug = np.uint8(mask_aug) # 0~1
            mask_input = cv2.threshold(mask_aug, 0.2, 1, cv2.THRESH_BINARY)[1]
            
            image2_aug = self.transform(image=image2)
            image2_weak = image2_aug['image']
            
            tile_patch_masks = self.mask_transform(mask_input)
            tile_patch_images = self.img_transform(image)
            
            tile_patch_images2 = self.img_transform(image2_weak)
            
            new_idx = torch.randperm(tile_patch_images.size()[0])
            tile_patch_images = tile_patch_images[new_idx]
            tile_patch_masks = tile_patch_masks[new_idx]
            tile_patch_images2 = tile_patch_images2[new_idx]
            
            return tile_patch_images, tile_patch_images2, tile_patch_masks
    
        elif self.train_or_test == 'val':
            if mask.max() == 255:
                mask_input = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]
            else:
                mask_input = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)[1]
                
            image2_aug = self.transform(image=image2)
            image2_weak = image2_aug['image']
                        
            ### Image Normalization and Patch
            tile_patch_masks = self.mask_transform(mask_input)
            tile_patch_images = self.img_transform(image)
            
            tile_patch_images2 = self.img_transform(image2_weak)
            
            return tile_patch_images, tile_patch_images2, tile_patch_masks, int(self.mask_paths[idx].split('/')[-1].split('.')[0].split('_')[-1])-55 #, un_mask_tensor
        elif self.train_or_test == 'test': 
            ############### TEST AND VALIDATION ################     
            if mask.max() == 255:
                mask_input = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]
            else:
                mask_input = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)[1]
            
            ## Image Normalization
            mask_input = np.transpose(mask_input , [2, 0, 1]) 
            mask_tensor = torch.FloatTensor(mask_input)
            image_tensor = torch.FloatTensor(np.transpose(image / 255., [2, 0, 1]))
            
            # image to patches
            tile_images, _, _, _ = split_tensor(image_tensor.unsqueeze(0), tile_size=self.patch_size, stride=self.stride)  # [961, 3, 128, 128]
            tile_patch_images = torch.cat(tile_images, dim=0)
            tile_masks, _, _, _ = split_tensor(mask_tensor.unsqueeze(0), tile_size=self.patch_size, dim=self.num_cls, stride=self.stride)
            tile_patch_masks = torch.cat(tile_masks, dim=0) ## output binary mask : [0, 1]
            
            return tile_patch_images, tile_patch_masks, self.image_paths[idx]


class Base_Unlabeled_EMA(Dataset):
    def __init__(self, train_or_test: str, mask_paths: list, image_paths: list, unlabeled_image_paths: list, img_size: int=2048, patch_size:int=512, num_cls: int=5, version:str='multi', stride:int=256):
        self.mask_paths = mask_paths
        self.image_paths = image_paths
        self.unlabeled_image_paths = unlabeled_image_paths
        self.train_or_test = train_or_test
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_cls = num_cls
        self.wanted_size = (self.img_size, self.img_size)
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2),
            ])
    
        self.version = version # none, clahe, blur, g_ch
        self.stride = stride
    
    def mask_transform(self, mask_np):
        mask_tensor = torch.FloatTensor(np.transpose(mask_np, [2, 0, 1]))
        tile_masks, _, _, _ = split_tensor(mask_tensor.unsqueeze(0), tile_size=self.patch_size, dim=self.num_cls, stride=self.stride)
        return torch.cat(tile_masks, dim=0)

    def img_transform(self, img_np):
        image_tensor = torch.FloatTensor(np.transpose(img_np/255., [2, 0, 1]))
        tile_images, _, _, _ = split_tensor(image_tensor.unsqueeze(0), tile_size=self.patch_size, stride=self.stride)  # [961, 3, 128, 128]
        return torch.cat(tile_images, dim=0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[..., ::-1]
        mask = np.load(self.mask_paths[idx], allow_pickle=True)
        mask = mask[...,:self.num_cls]
        
        rand_idx_unlabel = random.sample(range(len(self.unlabeled_image_paths)),1)[0]
        image2 = cv2.imread(self.unlabeled_image_paths[rand_idx_unlabel])[..., ::-1]
                
        ## Aug & Resize & Normliazation of mask
        if self.train_or_test == 'train':
            # print("DATASET PAIR CONFIRMATION", self.image_paths[idx], self.unlabeled_image_paths[rand_idx_unlabel])
            image, mask_aug = augment_imgs(image, mask) #image_aug[~255], mask_aug[~1]
            mask_aug = np.uint8(mask_aug) # 0~1
            mask_input = cv2.threshold(mask_aug, 0.2, 1, cv2.THRESH_BINARY)[1]
                
            image2_aug = self.transform(image=image2)
            image2_weak = image2_aug['image']
            
            ### Image Normalization and Patch
            tile_patch_masks = self.mask_transform(mask_input)
            
            tile_patch_images = self.img_transform(image)
            
            tile_patch_images2 = self.img_transform(image2_weak)
            
            new_idx = torch.randperm(tile_patch_images.size()[0])
            tile_patch_images = tile_patch_images[new_idx]
            tile_patch_masks = tile_patch_masks[new_idx]
            tile_patch_images2 = tile_patch_images2[new_idx]
            
            return tile_patch_images, tile_patch_images2, tile_patch_masks
    
        elif self.train_or_test == 'val':
            if mask.max() == 255:
                mask_input = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]
            else:
                mask_input = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)[1]
                
            image2_aug = self.transform(image=image2)
            image2_weak = image2_aug['image']
            
            ### Image Normalization and Patch
            tile_patch_masks = self.mask_transform(mask_input)
            tile_patch_images = self.img_transform(image)
            tile_patch_images2 = self.img_transform(image2_weak)
            
            return tile_patch_images, tile_patch_images2, tile_patch_masks, int(self.mask_paths[idx].split('/')[-1].split('.')[0].split('_')[-1])-55 #, un_mask_tensor
            
        
class Base_Unlabeled_cld(Base_Unlabeled):
    def weight(self):
        weight = np.zeros(self.num_cls)
        for data_id in self.ids_list:
            _, _, label = self._get_data(data_id)
            tmp = label.sum(axis=0).sum(axis=0) 
            weight += tmp
        weight = weight.astype(np.float32)
        self._weight = np.power(np.amax(weight) / weight, 1/3)
        print('cld weight', weight, '>>', self._weight)
        return self._weight, weight


        
class Base_Unlabeled_Saliency_Un(Dataset):
    def __init__(self, train_or_test: str, mask_paths: list, image_paths: list, unlabeled_image_paths: list, unlabeled_sal_paths: list, img_size: int=2048, patch_size:int=512, num_cls: int=5, version:str='multi', stride:int=256):
        self.mask_paths = mask_paths
        self.image_paths = image_paths
        self.unlabeled_image_paths = unlabeled_image_paths
        self.unlabeled_sal_paths = unlabeled_sal_paths
        self.train_or_test = train_or_test
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_cls = num_cls
        self.wanted_size = (self.img_size, self.img_size)
        self.ids_list = [i for i in range(len(mask_paths))]
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2),
            ])
        
        self.version = version # none, clahe, blur, g_ch
        self.stride = stride
        
    def mask_transform(self, mask_np):
        mask_tensor = torch.FloatTensor(np.transpose(mask_np, [2, 0, 1]))
        tile_masks, _, _, _ = split_tensor(mask_tensor.unsqueeze(0), tile_size=self.patch_size, dim=mask_np.shape[-1], stride=self.stride)
        return torch.cat(tile_masks, dim=0)

    def img_transform(self, img_np):
        image_tensor = torch.FloatTensor(np.transpose(img_np/255., [2, 0, 1]))
        tile_images, _, _, _ = split_tensor(image_tensor.unsqueeze(0), tile_size=self.patch_size, stride=self.stride)  # [961, 3, 128, 128]
        return torch.cat(tile_images, dim=0)
    
    def _get_data(self, data_id):
        # image, label = read_data(data_id, task=self.task)
        image = cv2.imread(self.image_paths[data_id])[..., ::-1]
        mask = np.load(self.mask_paths[data_id], allow_pickle=True)[...,:self.num_cls]
        return data_id, image, mask

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[..., ::-1]
        mask = np.load(self.mask_paths[idx], allow_pickle=True)
        
        rand_idx_unlabel = random.sample(range(len(self.unlabeled_image_paths)),1)[0]
        image2 = cv2.imread(self.unlabeled_image_paths[rand_idx_unlabel])[..., ::-1]
        image2_sal = np.load(self.unlabeled_sal_paths[rand_idx_unlabel], allow_pickle=True)[...,-1:]
        
        ## Aug & Resize & Normliazation of mask
        if self.train_or_test == 'train':
            image, mask_aug = augment_imgs(image, mask) #image_aug[~255], mask_aug[~1]
            mask_aug = np.uint8(mask_aug) # 0~1
            mask_input = cv2.threshold(mask_aug, 0.2, 1, cv2.THRESH_BINARY)[1]
            
            image2_aug = self.transform(image=image2, mask=image2_sal)
            image2_weak = image2_aug['image'] # 0~255
            image2_mask = image2_aug['mask'] > 0.5 # org 0.5
            
            tile_patch_masks = self.mask_transform(mask_input)
            tile_patch_images = self.img_transform(image)
            tile_patch_images2 = self.img_transform(image2_weak)
            tile_patch_masks2 = self.mask_transform(image2_mask)
            
            new_idx = torch.randperm(tile_patch_images.size()[0])
            tile_patch_images = tile_patch_images[new_idx]
            tile_patch_masks = tile_patch_masks[new_idx]
            tile_patch_images2 = tile_patch_images2[new_idx]
            tile_patch_masks2 = tile_patch_masks2[new_idx]         
            return tile_patch_images, tile_patch_images2, tile_patch_masks[:,:-1,...], tile_patch_masks2
            
        elif self.train_or_test == 'val':
            if mask.max() == 255:
                mask_input = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]
            else:
                mask_input = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)[1]
                
            image2_aug = self.transform(image=image2)
            image2_weak = image2_aug['image']
                        
            ### Image Normalization and Patch
            tile_patch_masks = self.mask_transform(mask_input)
            tile_patch_images = self.img_transform(image)
            
            tile_patch_images2 = self.img_transform(image2_weak)
            return tile_patch_images, tile_patch_images2, tile_patch_masks[:,:-1,...], int(self.mask_paths[idx].split('/')[-1].split('.')[0].split('_')[-1])-55 #, un_mask_tensor
            
