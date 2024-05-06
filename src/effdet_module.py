import torch
import lightning
from lightning import LightningModule
from objdetecteval.metrics.coco_metrics import get_coco_stats
from transforms import get_train_transforms, get_valid_transforms
# from create_model import create_model
# from fastcore.dispatch import typedispatch
# from typing import List
# import numpy as np
# from ensemble_boxes import ensemble_boxes_wbf


# class EffDetModel(LightningModule