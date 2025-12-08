"""
datasets.py
Contains all of the code for creating training, validation, and testing datasets
BoMeyering 2025
"""

import torch
import os
import cv2
import random
import argparse

import numpy as np
import polars as pl
import albumentations as A
import torch.multiprocessing as mp

from pathlib import Path
from glob import glob
from torch.utils.data import Dataset, DataLoader

from src.transforms import train_transforms, val_transforms
# from transforms import train_transforms, val_transforms

class EffDetDataset(Dataset):
    """Efficient Det Dataset
    
    Subclasses torch.utils.data.Dataset
    """

    def __init__(self, img_dir: Path, label_path: Path, transforms: A.Compose, label_map: dict):
        """
        Initialize the training dataset
        """
        self.img_dir = img_dir
        self.label_path = label_path
        self.transforms = transforms
        self.label_map = label_map

        self.img_names = []
        self.img_labels = pl.read_csv(self.label_path)

        # Build out image names in img_dir
        extensions = ['JPG', 'PNG', 'JPEG', 'jpg', 'png', 'jpeg']
        for ext in extensions:
            names = glob("*" + ext, root_dir=self.img_dir)
            self.img_names.extend(names)
        
        # Validate image names in directory and label dataframe
        if len(self.img_names) != len(self.img_labels['external_id'].unique()):
            raise AssertionError(
                f"Number of training images, {len(self.img_names)}, "\
                    "and unique target labels, "\
                    f"{len(self.img_labels['external_id'].unique())}, "\
                    "are not the same."
                )
        
        for i in self.img_labels['external_id'].unique():
            assert i in self.img_names

        # Create numeric class for feature classes
        self.img_labels = self.img_labels.with_columns(
            pl.col('feature_class').replace_strict(self.label_map).alias('class')
        )

    def __len__(self):
        """
        Return the length (image number) of the dataset
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Get one sample from the dataset using index
        """
        
        # Construct path to image
        img_path = os.path.join(self.img_dir, self.img_names[index])

        # Read in image if path exists
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise FileExistsError(f"Could not find image file at path {img_path}")
        
        # Select bboxes in Pascal format
        pascal_bboxes = self.img_labels.filter(
            pl.col('external_id') ==  self.img_names[index]
        )["xmin", "ymin", "xmax", "ymax"].to_numpy()

        # Grab obect labels
        labels = self.img_labels.filter(pl.col('external_id') == self.img_names[index])['class'].to_numpy()

        # Contruct sample dict for transforms
        sample = {
            "image": img,
            "bboxes": pascal_bboxes,
            "labels": labels,
        }

        # Augment sample
        sample = self.transforms(**sample)
        _, new_h, new_w = sample['image'].shape

        # Convert to ymin, xmin, ymax, xmax
        # EffDet requires [ymin, xmin, ymax, xmax] order
        # If nothing present, advance to next sample
        try:
            sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]
        except IndexError as e:
            print('Transformed image contained no bounding boxes.\nAdvancing to next index.')
            return self.__getitem__((index + 1) % len(self))
        
        aug_img = sample['image']

        target = {
            "bbox": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "cls": torch.as_tensor(sample['labels']),
            "img_name": self.img_names[index],
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
            "raw_img_size": img.shape[:2]
        }

        # Return the image and the target data
        return aug_img, target
        

def collate_fn(batch):
    images, targets = tuple(zip(*batch))

    img_stack = torch.stack(images).float()
    boxes = [target['bbox'] for target in targets]
    labels = [target['cls'] for target in targets]
    img_names = [target['img_name'] for target in targets]
    raw_img_size = [target['raw_img_size'] for target in targets]
    img_size = torch.tensor([target['img_size'] for target in targets]).float()
    img_scale = torch.tensor([target['img_scale'] for target in targets]).float()

    targets = {
        'bbox': boxes,
        'cls': labels,
        'img_size': img_size,
        'img_scale': img_scale
    }

    img_stack_metadata = {
        'img_names': img_names,
        'raw_img_size': raw_img_size
    }
        
    return img_stack, targets, img_stack_metadata

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    from torch.utils.data import DataLoader
    from create_model import create_model

    mapping = {
        "split": 1, 
        "seed": 2,
        "pod": 3
    }

    img_dir = './data/images/train'
    label_path = './data/annotations/train_annotations.csv'

    args = argparse.Namespace
    setattr(args, 'architecture', 'tf_efficientdet_d1')
    setattr(args, 'num_classes', 3)
    setattr(args, 'img_size', 512)
    setattr(args, 'max_det_per_image', 300)

    train_tfms = train_transforms(args)

    # Instantiate dataset
    train_dataset = EffDetDataset(img_dir=img_dir, label_path=label_path, transforms=train_tfms, label_map=mapping)

    model = create_model(args)
    model.eval()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    iter_loader = iter(train_dataloader)

    for batch in iter_loader:
        images, targets, img_stack_metadata = batch
        # print(images)
        # print(annotations)
        # print(img_names)
        print(img_stack_metadata)

        # output = model(images, targets)

        # print(output['detections'].shape)

    

    

    # # for i in range(10):
    # #     image, target = train_dataset[i]

    # #     print(image.shape)
    # #     print(target)
    # #     print(type(target))

    # # for batch_index in range(len(iter_loader)):
    # #     print(batch_index)
    # #     images, targets = next(iter_loader)

    # #     print(targets)

    # # model.eval()

    # for i in range(10):
    #     images, targets = tuple(zip(*(train_dataset[i], train_dataset[i])))
    #     # print(images, targets)
    #     img1, target1 = train_dataset[i]
    #     img2, target2 = train_dataset[i]

    #     img = torch.stack([img1, img2])
    #     # print(img.shape)

    #     annotations = {
    #         'bbox': [target1['bbox'], target2['bbox']],
    #         'cls': [target1['cls'], target2['cls']],
    #         'img_scale': torch.tensor([target1['img_scale'], target2['img_scale']]).float(),
    #         'img_size': torch.tensor([target1['img_size'], target2['img_size']]).float()
    #     }
    #     # print(annotations)



    #     loss = model(img, annotations)

    #     print(loss)

    # #     loss['loss'].backward()
