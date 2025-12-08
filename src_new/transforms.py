from torch.utils.data import Dataset
from typing import List, Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(target_img_size=512, means: Optional[tuple]=None, stds: Optional[tuple]=None):
    if means is None:
        means = (0.485, 0.456, 0.406)
    if stds is None:
        stds = (0.229, 0.224, 0.225)

    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.GaussianBlur(),
            A.SafeRotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(means, stds),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

def get_valid_transforms(target_img_size=512, means: Optional[tuple]=None, stds: Optional[tuple]=None):
    if means is None:
        means = (0.485, 0.456, 0.406)
    if stds is None:
        stds = (0.229, 0.224, 0.225)

    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(means, stds),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
