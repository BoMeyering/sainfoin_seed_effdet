"""
transforms.py
Definitions for all image augmentations and transformations applied duing training and inference
BoMeyering 2025
"""

# Standard imports
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def train_transforms(img_size: int=512, rgb_means: tuple[float, ...]=None, rgb_std: tuple[float, ...]=None) -> A.Compose:
    """
    Training transformations for image augmentation and preparation

    Args:
        img_size (int, optional): Integer value of the desired output image size as (int, int). 
            Defaults to 512.
        rgb_means (tuple[float, ...], optional): A tuple of three float values between 0 and 1
            for the RGB channel normalization means. Defaults to None.
        rgb_std (tuple[float, ...], optional): A tuple of three float values between 0 and 1
            for the RGB channel normalization standard deviations. Defaults to None.

    Returns:
        A.Compose: A callable Albumentations Compose function
    """
    
    # Check for defined means and std
    if rgb_means is None or rgb_std is None:
        print("no means passed")
        rgb_means = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]

    return A.Compose(
        [   
            A.Normalize(mean=rgb_means, std=rgb_std),
            A.Resize(height=img_size, width=img_size, p=1),
            A.HorizontalFlip(),
            A.GaussianBlur(),
            A.SafeRotate(p=.75),
            A.ChannelShuffle(),
            A.GridDistortion(),
            A.PlasmaShadow([0.0, 0.2], roughness=1),
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def val_transforms(img_size: int=512, rgb_means: tuple[float, ...]=None, rgb_std: tuple[float, ...]=None) -> A.Compose:
    """
    Validation transformations for image augmentation and preparation

    Args:
        img_size (int, optional): Integer value of the desired output image size as (int, int). 
            Defaults to 512.
        rgb_means (tuple[float, ...], optional): A tuple of three float values between 0 and 1
            for the RGB channel normalization means. Defaults to None.
        rgb_std (tuple[float, ...], optional): A tuple of three float values between 0 and 1
            for the RGB channel normalization standard deviations. Defaults to None.

    Returns:
        A.Compose: A callable Albumentations Compose function
    """

    # Check for defined means and std
    if rgb_means is None or rgb_std is None:
        rgb_means = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]

    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, p=1),
            A.Normalize(mean=rgb_means, std=rgb_std),
            ToTensorV2(p=1)
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format="pascal_voc", 
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )



