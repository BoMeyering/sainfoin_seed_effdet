# from src.datasets import EffDetDataset
import unittest
import sys
from pathlib import Path
import numpy as np
import torch
import string
import random
import pandas
import json

path_root = Path(__file__).parents[1]
print(path_root)
sys.path.append(str())

from src.datasets import EffDetDataset
from src.transforms import train_transforms, val_transforms

with open("./config/class_mapping.json", 'r') as f:
        classes = json.load(f)

with open('./config/model_config.json', 'r') as f:
        config = json.load(f)

class TestDataset(unittest.TestCase):
    """
    Test the functionality of the EffDetDataset
    """

    def test_train_dataset_init(self):
        EDDS = EffDetDataset(
            img_dir='./data/images/train',
            an_path='./data/annotations/train_annotations.csv',
            transforms=train_transforms(img_size=512), 
            class_mapping=classes,
            config_dict=config
        )

        self.assertIsInstance(EDDS.an_df, pandas.core.frame.DataFrame)
        self.assertIsInstance(EDDS.img_names, list)
        self.assertGreater(len(EDDS.img_names), 0)

    def test_val_dataset_init(self):
        EDDS = EffDetDataset(
            img_dir='./data/images/val',
            an_path='./data/annotations/val_annotations.csv',
            transforms=val_transforms(img_size=512), 
            class_mapping=classes,
            config_dict=config
        )

        self.assertIsInstance(EDDS.an_df, pandas.core.frame.DataFrame)
        self.assertIsInstance(EDDS.img_names, list)
        self.assertGreater(len(EDDS.img_names), 0)

    def test_dataset_len(self):
        EDDS = EffDetDataset(
            img_dir='./data/images/val',
            an_path='./data/annotations/val_annotations.csv',
            transforms=val_transforms(img_size=512), 
            class_mapping=classes,
            config_dict=config
        )
        self.assertEqual(EDDS.__len__(), len(EDDS))
        self.assertEqual(EDDS.__len__(), len(EDDS.img_names))

    def test_dataset_idx(self):
        EDDS = EffDetDataset(
            img_dir='./data/images/train',
            an_path='./data/annotations/train_annotations.csv',
            transforms=train_transforms(img_size=512), 
            class_mapping=classes,
            config_dict=config
        )
        for i in range(len(EDDS)):
             EDDS[i]

if __name__ == '__main__':
    unittest.main()
