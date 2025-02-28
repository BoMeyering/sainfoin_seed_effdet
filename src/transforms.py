"""
src.transforms.py

Definitions for all image augmentations and transformations applied duing training and inference

BoMeyering 2025
"""

# Standard imports
import torch
import argparse
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def train_transforms(args: argparse.Namespace) -> A.Compose:
    """Training transformations for image augmentation and preparation

    Args:
        args (argparse.Namespace): A parsed args.Namespace from a config.yaml file.

    Returns:
        A.Compose: A callable Albumentations Compose function
    """
    
    # Check for defined means and std
    if ('rgb_means' not in vars(args)) or ('rgb_std' not in vars(args)):
        print("no means passed. Defaulting to COCO standards")
        args.rgb_means = [0.485, 0.456, 0.406]
        args.rgb_std = [0.229, 0.224, 0.225]
    elif args.rgb_means is None or args.rgb_std is None:
        print("no means passed. Defaulting to COCO standards")
        args.rgb_means = [0.485, 0.456, 0.406]
        args.rgb_std = [0.229, 0.224, 0.225]
    
    if 'img_size' not in vars(args):
        args.img_size = 512

    

    return A.Compose(
        [   
            A.Normalize(mean=args.rgb_means, std=args.rgb_std),
            A.Resize(height=args.img_size, width=args.img_size, p=1),
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

def val_transforms(args: argparse.Namespace) -> A.Compose:
    """Validation transformations for image augmentation and preparation

    Args:
        args (argparse.Namespace): A parsed args.Namespace from a config.yaml file.

    Returns:
        A.Compose: A callable Albumentations Compose function
    """

    # Check for defined means and std
    if ('rgb_means' not in vars(args)) or ('rgb_std' not in vars(args)):
        print("No means passed. Defaulting to COCO standards")
        args.rgb_means = [0.485, 0.456, 0.406]
        args.rgb_std = [0.229, 0.224, 0.225]
    elif args.rgb_means is None or args.rgb_std is None:
        print("No means passed. Defaulting to COCO standards")
        args.rgb_means = [0.485, 0.456, 0.406]
        args.rgb_std = [0.229, 0.224, 0.225]
    
    if 'img_size' not in vars(args):
        args.img_size = 512

    return A.Compose(
        [
            A.Resize(height=args.img_size, width=args.img_size, p=1),
            A.Normalize(mean=args.rgb_means, std=args.rgb_std),
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



