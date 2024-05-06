import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def train_transforms(img_size: int=512):
    """
    Takes one argument, img_size
    Returns an albumentations compose function with random transformations for training images
    """
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.GaussianBlur(),
            A.SafeRotate(),
            A.Resize(height=img_size, width=img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

def val_transforms(img_size: int=512):
    """
    Takes one argument, img_size
    Returns an albumentations compose function that resizes and normalizes validation images
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

tms = val_transforms()
print(type(tms))