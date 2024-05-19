import numpy as np
import random
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.augmentations import transforms

from skimage.transform import warp, AffineTransform

PROB_AUG = 0.1


def random_perturbation(img, rescale_factor=1.5):
    im = Image.fromarray(img.astype(np.uint8))
    en_color = ImageEnhance.Color(im)
    # im = en_color.enhance(random.uniform(1. / rescale_factor, rescale_factor))
    # en_cont = ImageEnhance.Contrast(im)
    # im = en_cont.enhance(random.uniform(1. / rescale_factor, rescale_factor))
    # en_bright = ImageEnhance.Brightness(im)
    # im = en_bright.enhance(random.uniform(1. / rescale_factor, rescale_factor))
    en_sharpness = ImageEnhance.Sharpness(im)
    im = en_sharpness.enhance(random.uniform(1. / rescale_factor, rescale_factor))
    return np.asarray(im).astype(np.uint8)


def augment_imgs(img, mask):
    assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
    h, w, d = img.shape

    aug_img = random_perturbation(img, rescale_factor=1.5)
    shift_y, shift_x = np.array(aug_img.shape[:2]) / 2.
    shift_to_ori = AffineTransform(translation=[-shift_x, -shift_y])
    shift_to_img_center = AffineTransform(translation=[shift_x, shift_y])
    r_angle = random.randint(0, 359)

    scale_factor = 1.15
    scale = random.uniform(1. / scale_factor, scale_factor)
    tform = AffineTransform(scale=(scale, scale), rotation=np.deg2rad(r_angle))
    
    aug_img = warp(aug_img, (shift_to_ori + (tform + shift_to_img_center)).inverse, output_shape=(aug_img.shape[0], aug_img.shape[1]))
    aug_img = (aug_img * 255).astype(np.uint8)
    mask_cls = sorted(np.unique(mask))
    
    if mask_cls[-1] != 255:
        mask = (mask*255).astype(np.uint8)
    aug_mask = warp(mask, (shift_to_ori + (tform + shift_to_img_center)).inverse, output_shape=(aug_img.shape[0], aug_img.shape[1]))
    aug_mask = np.round(aug_mask)
    

    # # GridDistortion
    # if random.uniform(0, 1) < PROB_AUG:
    #     img = A.GridDistortion(num_steps=5, distort_limit=0.5, value=None, mask_value=None, always_apply=True, p=1)(image=img)["image"].astype(np.uint8)

    # # blur
    # blur_limit = 10
    # if random.uniform(0, 1) < PROB_AUG:
    #     aug_img = A.Blur(blur_limit=blur_limit, p=1)(image=aug_img)["image"].astype(np.uint8)

    # # GaussNoise
    # GaussNoise_val_limit = 80
    # if random.uniform(0, 1) < PROB_AUG:
    #     aug_img = transforms.GaussNoise(var_limit=GaussNoise_val_limit, p=1)(image=aug_img)["image"].astype(np.uint8)

    # # ISONoise
    # ISONoise_color_shift_max = 0.01
    # if random.uniform(0, 1) < PROB_AUG:
    #     aug_img = transforms.ISONoise(color_shift=(0.01, ISONoise_color_shift_max), p=1)(image=aug_img)["image"].astype(np.uint8)

    # # JpegCompression
    # JpegCompression_quality_lower = 20
    # if random.uniform(0, 1) < PROB_AUG:
    #     aug_img = transforms.ImageCompression(quality_lower=JpegCompression_quality_lower, quality_upper=100, p=1)(image=aug_img)["image"].astype(np.uint8)

    # # RGBShift
    # RGBShift_val = 10
    # if random.uniform(0, 1) < PROB_AUG:
    #     aug_img = transforms.RGBShift(r_shift_limit=RGBShift_val, g_shift_limit=RGBShift_val, b_shift_limit=RGBShift_val, p=1)(image=aug_img)["image"].astype(np.uint8)

    # # RandomGamma
    # RandomGamma_min = 95
    # RandomGamma_max = 105
    # if random.uniform(0, 1) < PROB_AUG:
    #     aug_img = transforms.RandomGamma(gamma_limit=(RandomGamma_min, RandomGamma_max), p=1)(image=aug_img)["image"].astype(np.uint8)


    # # ElasticTransform
    # ElasticTransform_alpha = int(random.uniform(512, 1024))
    # ElasticTransform_sigma = ElasticTransform_alpha * random.uniform(0.03, 0.05)
    # if random.uniform(0, 1) < PROB_AUG:
    #     aug_img = A.ElasticTransform(alpha=ElasticTransform_alpha, sigma=ElasticTransform_sigma, alpha_affine=0, p=1)(image=aug_img)["image"].astype(np.uint8)

    return aug_img, aug_mask