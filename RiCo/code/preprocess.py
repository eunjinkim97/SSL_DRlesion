import os, glob, cv2
from tqdm import tqdm
import numpy as np

th = 0.25
saliency_fine = cv2.saliency.StaticSaliencyFineGrained_create()
def get_saliency(img):
    fov = (img[..., 0:1] >= 20 * 1).astype(np.uint8) # R channel
    image_org = img * fov
    image_pre = saliency_preprocess(image_org, fov)
    
    circle = fov.copy()
    kernel = np.ones((5, 5), np.uint8)
    circle = cv2.erode(circle, kernel, iterations=50)

    (_, saliencyMap_fine) = saliency_fine.computeSaliency(image_pre)
    saliencyMap_fine = saliencyMap_fine * circle
    saliencyMap_fine = (saliencyMap_fine >= th).astype(np.uint8)
    return saliencyMap_fine


def saliency_preprocess(img, fov=None):
    scale = int(img.shape[0])        
    mask = fov.copy()
    
    w = 4
    weighted_img = cv2.addWeighted(img, w, cv2.GaussianBlur(img, (0, 0), scale/30), -1*w, 128) #  scale/30
    processed_img = weighted_img * mask + 128 * (1 - mask)
    return processed_img.astype(np.uint8)

if __name__=='__main__':
    
    root = '/data1/eunjkinkim/SSL_DRlesion/IDRiD/'
    image_path = f'{root}/A. Segmentation/1. Original Images'
    gt_path = f'{root}/A. Segmentation/2. All Segmentation Groundtruths'

    images = sorted(glob.glob(image_path+'/*/*.jpg'))
    gts = sorted(glob.glob(gt_path+'/*/*/*.tif'))
    gts_ma = [gt for gt in gts if 'MA' in gt]
    gts_he = [gt for gt in gts if 'HE' in gt]
    gts_ex = [gt for gt in gts if 'EX' in gt]
    gts_se = [gt for gt in gts if 'SE' in gt]
    
    print("Images:", len(images))
    print("MA:", len(gts_ma))
    print("HE:", len(gts_he))
    print("EX:", len(gts_ex))
    print("SE:", len(gts_se))
    print("SE idx", [gt.split('/')[-1].split('.')[0] for gt in gts_se])
    
    save_path_train = f'{root}/A. Segmentation/train_mask'
    save_path_test = f'{root}/A. Segmentation/test_mask'
    os.makedirs(save_path_train, exist_ok=True)
    os.makedirs(save_path_test, exist_ok=True)
    for img_path in tqdm(images):
        img_name = img_path.split('/')[-1][:-4]
        img_np = cv2.imread(img_path)[..., ::-1] # BGR to RGB
        saliency_mask = get_saliency(img_np)
        
        masks_path = []
        for gt_ma in gts_ma:
            if img_name in gt_ma:
                mask_ma = cv2.imread(gt_ma, cv2.IMREAD_GRAYSCALE)
                mask_ma = (mask_ma > 0).astype(np.uint8)
                masks_path.append(mask_ma)
                break
        else:
            print("Zero mask in MA", img_name)
            masks_path.append(np.zeros_like(saliency_mask))
        for gt_he in gts_he:
            if img_name in gt_he:
                mask_he = cv2.imread(gt_he, cv2.IMREAD_GRAYSCALE)
                mask_he = (mask_he > 0).astype(np.uint8)
                masks_path.append(mask_he)
                break
        else:
            print("Zero mask in HE", img_name)
            masks_path.append(np.zeros_like(saliency_mask))
        for gt_ex in gts_ex:
            if img_name in gt_ex:
                mask_ex = cv2.imread(gt_ex, cv2.IMREAD_GRAYSCALE)
                mask_ex = (mask_ex > 0).astype(np.uint8)
                masks_path.append(mask_ex)
                break
        else:
            print("Zero mask in EX", img_name)
            masks_path.append(np.zeros_like(saliency_mask))
        for gt_se in gts_se:
            if img_name in gt_se:
                mask_se = cv2.imread(gt_se, cv2.IMREAD_GRAYSCALE)
                mask_se = (mask_se > 0).astype(np.uint8)
                masks_path.append(mask_se)
                break
        else:
            print("Zero mask in SE", img_name)
            masks_path.append(np.zeros_like(saliency_mask))
            
        masks_path.append(saliency_mask)
        if len(masks_path) == 5:
            if 'a. Training Set' in img_path:
                masks_gt_path = f'{save_path_train}/{img_name}.npy'
            elif 'b. Testing Set' in img_path:
                masks_gt_path = f'{save_path_test}/{img_name}.npy'
            masks_path = np.stack(masks_path).transpose(1, 2, 0)
            np.save(masks_gt_path, masks_path)
            print("Saved", masks_gt_path)
            
