import os
import glob
import random
import numpy as np

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .utils.dataset import Base, Base_Unlabeled, Base_Unlabeled_cld, Base_Unlabeled_EMA, Base_Unlabeled_Saliency_Un


class DM(LightningDataModule):
    def __init__(
        self,
        unlabeled_tr: str,
        data_dir_tr: str,
        data_dir_te: str,
        batch_size: int = 1,
        patch_size: int = 512,
        num_workers: int = 4,
        num_cls: int = 5,
        version:str = 'multi',
        sample_prop: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        print("VERSION", version, version[4:6], flush=True)
        random.seed(int(version[4:6]))

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.num_cls = num_cls
        
        # HE 43 gt doesnt exist
        self.se_idx_list_tr = [int(i)-1 for i in ['03', '08', '13', '14', '17', '18', '19', '22', '23', '25', '30', '31', '32', '33', '35', '38', '39', '46', '47', '48', '49', '50', '51', '52', '53', '54']]
        self.se_idx_list_te =[int(i)-55 for i in ['55', '56', '59', '60', '61', '64', '67', '68', '70', '71', '72', '73', '74', '75']]

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage=None):
        train_image_paths_raw = sorted(glob.glob(f'{self.hparams.data_dir_te}/A. Segmentation/1. Original Images/a. Training Set/IDRiD_*.jpg'))
        train_mask_paths_raw = sorted(glob.glob(f'{self.hparams.data_dir_tr}/A. Segmentation/train_mask/*.npy')) 
        test_image_paths = sorted(glob.glob(f'{self.hparams.data_dir_tr}/A. Segmentation/1. Original Images/b. Testing Set/IDRiD_*.jpg'))
        test_mask_paths = sorted(glob.glob(f'{self.hparams.data_dir_te}/A. Segmentation/test_mask/*.npy'))
    
        if self.hparams.sample_prop != 0: # 1/8, 1/4, 1/2, 1
            sample_num = int(len(train_image_paths_raw) * self.hparams.sample_prop)
            if self.hparams.sample_prop == 1:
                sample_idx = range(len(train_mask_paths_raw))
            else:
                sample_idx = random.sample(self.se_idx_list_tr, sample_num)                            
            print(f"SAMPLE Labeled image for TR", self.hparams.sample_prop, len(sample_idx), sample_idx)
            
            train_mask_paths = [i for idx, i in enumerate(train_mask_paths_raw) if idx in sample_idx]
            train_image_paths = [i for idx, i in enumerate(train_image_paths_raw) if idx in sample_idx]
            
            if self.hparams.sample_prop != 1:
                unlabeled_image_paths = [i for idx, i in enumerate(train_image_paths_raw) if idx not in sample_idx]
                unlabeled_mask_paths = [i for idx, i in enumerate(train_mask_paths_raw) if idx not in sample_idx]

        print("*******DATASET INFO*******")
        print(f"<<{self.hparams.version}>>")
        print("train_mask_paths : ", len(train_mask_paths),'\n',flush=True)
        print("train_image_paths : ", len(train_image_paths),'\n',flush=True)
        print(train_image_paths[-1], "\n", train_mask_paths[-1])
        
        print("test_mask_paths : ", len(test_mask_paths),'\n',flush=True)
        print("test_image_paths : ", len(test_image_paths),'\n',flush=True)
        print(test_image_paths[-1], "\n", test_mask_paths[-1])
    
        if stage == 'fit' or stage is None:

            if 'unlabeled' in self.hparams.version:
                if 'uamt' in self.hparams.version:
                    self.train_dataset = Base_Unlabeled_EMA(train_or_test='train', mask_paths=train_mask_paths,
                                                image_paths=train_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                patch_size=self.hparams.patch_size, num_cls=self.num_cls, version=self.hparams.version)          
                    self.val_dataset = Base_Unlabeled_EMA(train_or_test='val', mask_paths=test_mask_paths, 
                                                image_paths=test_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                patch_size=self.hparams.patch_size, num_cls=self.num_cls, version=self.hparams.version)
                        

                elif 'cld' in self.hparams.version:
                    self.train_dataset = Base_Unlabeled_cld(train_or_test='train', mask_paths=train_mask_paths,
                                                patch_size=self.hparams.patch_size,
                                                image_paths=train_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                num_cls=self.num_cls, version=self.hparams.version)          
                    self.val_dataset = Base_Unlabeled_cld(train_or_test='val', mask_paths=test_mask_paths, 
                                                    patch_size=self.hparams.patch_size,
                                                    image_paths=test_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                    num_cls=self.num_cls, version=self.hparams.version)
                
                elif 'ours' in self.hparams.version:
                    print("OURS for DATASET LOADING... Base_Unlabeled_Saliency_Un")
                    self.train_dataset = Base_Unlabeled_Saliency_Un(train_or_test='train', mask_paths=train_mask_paths,
                                                patch_size=self.hparams.patch_size, unlabeled_sal_paths=unlabeled_mask_paths,
                                                image_paths=train_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                num_cls=self.num_cls, version=self.hparams.version)          
                    self.val_dataset = Base_Unlabeled_Saliency_Un(train_or_test='val', mask_paths=test_mask_paths, 
                                                patch_size=self.hparams.patch_size, unlabeled_sal_paths=unlabeled_mask_paths,
                                                image_paths=test_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                num_cls=self.num_cls, version=self.hparams.version)
                
                else:  # CPS, DHC, MCF
                    print("Base_Unlabeled for DATASET LOADING...")
                    self.train_dataset = Base_Unlabeled(train_or_test='train', mask_paths=train_mask_paths,
                                                patch_size=self.hparams.patch_size,
                                                image_paths=train_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                num_cls=self.num_cls, version=self.hparams.version)          
                    self.val_dataset = Base_Unlabeled(train_or_test='val', mask_paths=test_mask_paths, 
                                                    patch_size=self.hparams.patch_size,
                                                    image_paths=test_image_paths, unlabeled_image_paths=unlabeled_image_paths,
                                                    num_cls=self.num_cls, version=self.hparams.version)
                       
                
                
            else: 
                self.train_dataset = Base(train_or_test='train', mask_paths=train_mask_paths,
                                                image_paths=train_image_paths, patch_size=self.hparams.patch_size,
                                                num_cls=self.num_cls,version=self.hparams.version)          
                self.val_dataset = Base(train_or_test='val', mask_paths=test_mask_paths, 
                                                    image_paths=test_image_paths, patch_size=self.hparams.patch_size,
                                                    num_cls=self.num_cls, version=self.hparams.version)            
            

        if stage == 'test':
            if 'unlabeled' in self.hparams.version:
                self.test_dataset = Base_Unlabeled(train_or_test='test',mask_paths=test_mask_paths, 
                                            unlabeled_image_paths=unlabeled_image_paths, patch_size=self.hparams.patch_size,
                                            image_paths=test_image_paths, num_cls=self.num_cls,version=self.hparams.version)

            else:        
                self.test_dataset = Base(train_or_test='test', mask_paths=test_mask_paths, 
                                                    image_paths=test_image_paths, patch_size=self.hparams.patch_size,
                                                    num_cls=self.num_cls, version=self.hparams.version)  

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, shuffle=True)  

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.hparams.num_workers)
        
