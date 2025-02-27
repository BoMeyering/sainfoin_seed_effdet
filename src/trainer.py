"""
trainer.py
Instantiate all trainer classes
BoMeyering 2025
"""

import uuid
import logging
import argparse
from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import torch.distributed as dist
import numpy as np

# from src.eval import AverageMeterSet
# from src.fixmatch import get_pseudo_labels
# from src.callbacks import ModelCheckpoint
# from src.metrics import MetricLogger
from src.transforms import get_strong_transforms


class Trainer(ABC):
    """Abstract Trainer Class"""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def _train_step(self, batch):
        """Implement the train step for one batch"""

    @abstractmethod
    def _val_step(self, batch):
        """Implement the val step for one batch"""

    @abstractmethod
    def _train_epoch(self, epoch):
        """Implement the training method for one epoch"""

    @abstractmethod
    def _val_epoch(self, epoch):
        """Implement the validation method for one epoch"""

    @abstractmethod
    def train(self):
        """Implement the whole training loop"""

class EffDetTrainer(Trainer):
    """ Standard trainer for an effdet model """
    def __init__(
            self, 
            name, 
            args: argparse.Namespace,
            model: torch.nn.Module
    ):
        super().__init__(name=name)

    def _train_step(self, batch: Tuple):
        """ Train one batch of images """
        img, target = batch
        img = img.to_device(self.args.device)
        target = target.to_device(self.args.device)

        self.model(img, target)
        pass

    def _train_epoch(self, epoch):
        pass

    @torch.no_grad()
    def _val_step(self, batch: Tuple):
        pass

    @torch.no_grad()
    def _val_epoch(self, epoch):
        pass

    def train(self):
        for epoch in range(self.args.epochs):
