import glob, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class Network_Unet(nn.Module):
    def __init__(self, out_dim, version, pretrained_path="imagenet", arch="efficientnet-b3"):    
        super(Network_Unet, self).__init__() 
        activate = 'sigmoid'
        self.version = version
        self.generator = UNet(arch, pretrained_path, activate, output_classes=out_dim)
        
    def forward(self, im_q):
        _, pred = self.generator(im_q)
        return pred

class Network_UnetPP(nn.Module):
    def __init__(self, out_dim, version, pretrained_path="imagenet", arch="efficientnet-b3"):    
        super(Network_UnetPP, self).__init__() 
        self.version = version
        activate = 'sigmoid'        
        self.generator = UNetPlusPlus(arch, pretrained_path, activate, output_classes=out_dim)
        
    def forward(self, im_q):
        _, pred = self.generator(im_q)
        return pred
    
class Network_UAMT(nn.Module): # uamt, acmt
    def __init__(self, out_dim, version, pretrained_path="imagenet", arch="efficientnet-b3"):    
        super(Network_UAMT, self).__init__() 
        
        activate = None 
        self.m = 0.999
        self.version = version
        
        from_ckpt = load_best_ckpt(version, pretrained_path, arch, arch_version="EMA") if pretrained_path != "imagenet" else "imagenet"
        self.encoder_q = UNet(arch, from_ckpt, activate, output_classes=out_dim)  
        self.encoder_k = UNet(arch, None, activate, output_classes=out_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    def forward(self, im_labeled, im_unlabeled):
        image = torch.cat([im_labeled, im_unlabeled], dim=0)
        _, pred = self.encoder_q(image) 
        pred_labeled, pred_unlabeled = pred[:im_labeled.size(0)], pred[im_labeled.size(0):]
        
        noise = torch.clamp(torch.randn_like(im_unlabeled) * 0.1, -0.2, 0.2)
        noisy_ema_inputs = im_unlabeled + noise
        with torch.no_grad():
            _, noisy_ema_output = self.encoder_k(noisy_ema_inputs)

        T = 8
        num_cls = pred.size(1)
        _, c, w, h = im_unlabeled.shape
        volume_batch_r = im_unlabeled.repeat(2, 1, 1, 1)
        stride = volume_batch_r.shape[0] // 2
        preds = torch.zeros([stride * T, num_cls, w, h]).cuda()
        for i in range(T//2): 
            ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                _, preds[2 * stride * i:2 * stride * (i + 1)] = self.encoder_k(ema_inputs)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(T, stride, num_cls, w, h)
        preds = torch.mean(preds, dim=0)
        uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
        return {'q_output_labeled':pred_labeled, 'q_output_unlabeled':pred_unlabeled, 'uncertainty':uncertainty, 'noisy_ema_output':noisy_ema_output}
  
class Network_Dual(nn.Module): # CPS, CLD, DHC
    def __init__(self, out_dim, version, pretrained_path1="imagenet", pretrained_path2="imagenet", arch1="efficientnet-b3", arch2="efficientnet-b3"):    
        super(Network_Dual, self).__init__() 
        activate = 'sigmoid'
        self.version = version  

        from_ckpt = load_best_ckpt(version, pretrained_path1, arch1, arch_version="Dual1") if pretrained_path1 != "imagenet" else "imagenet"
        self.generator1 = UNet(arch1, from_ckpt, activate, output_classes=out_dim)  
        
        from_ckpt = load_best_ckpt(version, pretrained_path2, arch2, arch_version="Dual2") if pretrained_path2 != "imagenet" else "imagenet"
        self.generator2 = UNet(arch2, from_ckpt, activate, output_classes=out_dim)
        
    
    def forward(self, im_q, im_k):
        assert im_q.size(0) == im_k.size(0)
        
        bs = im_q.size(0)
        im_set = torch.cat([im_q, im_k], dim=0)
        
        _, pred1 = self.generator1(im_set)
        _, pred2 = self.generator2(im_set)
        pred1, pred1_un = pred1[:bs], pred1[bs:]
        pred2, pred2_un = pred2[:bs], pred2[bs:]
        
        return pred1, pred2, pred1_un, pred2_un
    
class Network_Dual_Diff(nn.Module):
    def __init__(self, out_dim, version, pretrained_path1="imagenet", pretrained_path2="imagenet", arch1="efficientnet-b3", arch2="efficientnet-b3"):    
        super(Network_Dual_Diff, self).__init__() 
        self.version = version
        activate = 'sigmoid'
        
        from_ckpt = load_best_ckpt(version, pretrained_path1, arch1, arch_version='Dual1') if pretrained_path1 != 'imagenet' else 'imagenet'
        self.generator1 = UNet(arch1, from_ckpt, activate, output_classes=out_dim)  
        
        from_ckpt = load_best_ckpt(version, pretrained_path2, arch2, arch_version='Dual2') if pretrained_path2 != 'imagenet' else 'imagenet'
        self.generator2 = UNetPlusPlus(arch2, from_ckpt, activate, output_classes=out_dim)
        
    def forward(self, im_q, im_k):
        assert im_q.size(0) == im_k.size(0)
        
        bs = im_q.size(0)
        im_set = torch.cat([im_q, im_k], dim=0)
        
        _, pred1 = self.generator1(im_set)
        _, pred2 = self.generator2(im_set)

        pred1, pred1_un = pred1[:bs], pred1[bs:]
        pred2, pred2_un = pred2[:bs], pred2[bs:]       
         
        return pred1, pred2, pred1_un, pred2_un
    
    
class UNet(nn.Module):
    def __init__(self, backbone, encoder_weights=None, activation=None, output_classes=3):
        super(UNet, self).__init__()
        
        if encoder_weights is None:
            self.segmentor = smp.Unet(backbone, encoder_weights=None, classes=output_classes, activation=activation)#, aux_params=params)
            
        elif '.ckpt' in encoder_weights:
            if 'EMA' in encoder_weights:
                encoder_weights = encoder_weights.replace('EMA','')
                self.segmentor = smp.Unet(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                            'model.encoder_q.segmentor.', # instead of k
                            ]
                
                for k, v in load_weight.items():
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v
                print('UNET EMA model weight LOAD!')
            elif 'Dual1' in encoder_weights:
                self.segmentor = smp.Unet(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                
                encoder_weights = encoder_weights.replace('Dual1','')
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                                'model.generator1.segmentor.'
                                ] # ,'model.segmentor.'
                for k, v in load_weight.items():
                    
                    if "_fc.bias" in k or "_fc.weight" in k:
                        continue
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v
                print("UNET DUAL model (1)of(2) Loded!!")

            elif 'Dual2' in encoder_weights:
                self.segmentor = smp.Unet(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                
                encoder_weights = encoder_weights.replace('Dual2','')
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                                'model.generator2.segmentor.'
                                ] # ,'model.segmentor.'
                for k, v in load_weight.items():
                    
                    if "_fc.bias" in k or "_fc.weight" in k:
                        continue
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v
                print("UNET DUAL model (2)of(2) Loded!!")
            else:
                self.segmentor = smp.Unet(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                                'model.generator.segmentor.'
                                ] # ,'model.segmentor.'
                for k, v in load_weight.items():
                    
                    if "_fc.bias" in k or "_fc.weight" in k:
                        continue
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v

            self.segmentor.load_state_dict(weight)
            print("LOAD UNET weight in DiRAUNET!!")

        else:
            self.segmentor = smp.Unet(backbone, encoder_weights='imagenet', classes=output_classes, activation=activation)#, aux_params=params)
                
        
    def forward(self, x):
        features = self.segmentor.encoder(x) # [8, 2048, 16, 16], [8, 1024, 32, 32], [8, 512, 64, 64], [8, 256, 128, 128], [8, 64, 256, 256], [8, 3, 512, 512]
        decoder_output = self.segmentor.decoder(*features) # decoder_output : [8, 16, 512, 512]
        masks = self.segmentor.segmentation_head(decoder_output) # segm output : ([8, 5, 512, 512])

        return decoder_output, masks

class UNetPlusPlus(nn.Module):
    def __init__(self, backbone, encoder_weights=None, activation=None, output_classes=3):
        super(UNetPlusPlus, self).__init__()
        # self.segmentor = smp.UnetPlusPlus(backbone, classes=output_classes, activation=activation)#, aux_params=params)
                
        if encoder_weights is None:
            self.segmentor = smp.UnetPlusPlus(backbone, encoder_weights=None, classes=output_classes, activation=activation)#, aux_params=params)
            
        elif '.ckpt' in encoder_weights:
            if 'EMA' in encoder_weights:
                encoder_weights = encoder_weights.replace('EMA','')
                self.segmentor = smp.UnetPlusPlus(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                            'model.encoder_q.segmentor.', # instead of k
                            ]
                
                for k, v in load_weight.items():
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v
                print('U++ EMA model weight LOAD!')
            elif 'Dual1' in encoder_weights:
                self.segmentor = smp.UnetPlusPlus(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                
                encoder_weights = encoder_weights.replace('Dual1','')
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                                'model.generator1.segmentor.'
                                ] # ,'model.segmentor.'
                for k, v in load_weight.items():
                    
                    if "_fc.bias" in k or "_fc.weight" in k:
                        continue
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v
                print("U++ DUAL model (1)of(2) Loded!!")

            elif 'Dual2' in encoder_weights:
                self.segmentor = smp.UnetPlusPlus(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                
                encoder_weights = encoder_weights.replace('Dual2','')
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                                'model.generator2.segmentor.'
                                ] # ,'model.segmentor.'
                for k, v in load_weight.items():
                    
                    if "_fc.bias" in k or "_fc.weight" in k:
                        continue
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v
                print("U++ DUAL model (2)of(2) Loded!!")
            else:
                self.segmentor = smp.UnetPlusPlus(backbone, encoder_weights=None, classes=output_classes, activation=activation)
                load_weight = torch.load(encoder_weights)['state_dict']
                weight = {} 
                except_list = [
                                'model.generator.segmentor.'
                                ] # ,'model.segmentor.'
                for k, v in load_weight.items():
                    
                    if "_fc.bias" in k or "_fc.weight" in k:
                        continue
                    
                    for exp in except_list:
                        if exp in k:
                            weight[k.replace(exp,'')] = v

            self.segmentor.load_state_dict(weight)
            print("LOAD U++  weight in DiRAUNET!!")

        else:
            self.segmentor = smp.UnetPlusPlus(backbone, encoder_weights='imagenet', classes=output_classes, activation=activation)#, aux_params=params)
                
        
             
    def forward(self, x):
        features = self.segmentor.encoder(x) # [8, 2048, 16, 16], [8, 1024, 32, 32], [8, 512, 64, 64], [8, 256, 128, 128], [8, 64, 256, 256], [8, 3, 512, 512]
        decoder_output = self.segmentor.decoder(*features) # decoder_output : [8, 16, 512, 512]
        masks = self.segmentor.segmentation_head(decoder_output) # segm output : ([8, 5, 512, 512])
        return decoder_output, masks


def load_best_ckpt(version, path, arch, arch_version='Dual1'):
    if '.ckpt' in path:
        load_ckpt = path
    elif 'from_sup' in version:
        from_version = '_'.join(version.split('_')[0:2]) + f'_{arch}'
        if 'fgadr' in version:
            from_version = from_version + '_fgadr'
        load_ckpt = find_best_weight(path, from_version)
    elif 'from_same_version' in version:
        from_version = version.replace('from_same_version','from_sup').replace('_last', '')
        if arch_version is not None:
            load_ckpt = find_best_weight(path, from_version) + arch_version if 'last' not in version else find_best_weight(path, from_version, last=True) + arch_version
        else:
            load_ckpt = find_best_weight(path, from_version)  if 'last' not in version else find_best_weight(path, from_version, last=True) 

    else:
        load_ckpt = None
    return load_ckpt
    
def find_best_weight(path, version, last=False):
    print("Find version : ", version)
    if last:
        ckpts = [i for i in sorted(glob.glob(path+f'/*/{version}/checkpoints/last.ckpt'))]
        best_ckpt_path = ckpts[-1]
        print("Just load Last weight,,,")
    else:
        ckpts = [i for i in sorted(glob.glob(path+f'/*/{version}/checkpoints/*.ckpt')) if 'vdice' in i]
        if len(ckpts) == 0:
            ckpts = [i for i in sorted(glob.glob(path+f'/*/{version}/checkpoints/last.ckpt'))] 
            best_ckpt_path = ckpts[-1] # the latest weight
            print("Fail to find the best weight,,, Just load Last weight,,,")
        else:
            latest_time = [i.split('/')[-4] for i in ckpts][-1]
            latest_ckpt = [i for i in ckpts if latest_time in i]
            best_ckpt_idx = np.argmax([float(i.split('/')[-1].split('vdice')[-1][:6]) for i in latest_ckpt]) # 0.xxxx
            best_ckpt_path = latest_ckpt[best_ckpt_idx] # the latest time and best weight
            print("Last time & Best weight load,,,")
    print(best_ckpt_path)
    return best_ckpt_path