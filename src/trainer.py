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

from src.eval import AverageMeterSet
# from src.fixmatch import get_pseudo_labels
# from src.callbacks import ModelCheckpoint
# from src.metrics import MetricLogger
from src.transforms import train_transforms, val_transforms


class Trainer(ABC):
    """Abstract Trainer Class"""

    def __init__(self):
        super().__init__()
        self.meters = AverageMeterSet()

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
            args: argparse.Namespace,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            optimizer, 
            scheduler
    ):
        super().__init__()

        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _train_step(self, batch: Tuple):
        """ Train one batch of images """
        # Unpack the batch
        img_stack, targets, img_names = batch
        # Move image stack and targets to device
        img_stack = img_stack.to(self.args.device)
        targets = move_target_to_device(targets, device=self.args.device)

        # Make forward pass over data
        loss_dict = self.model(img_stack, targets)

        return loss_dict
        
    def _train_epoch(self, epoch: int):
        """
        Train one epoch
        """

        # Reset all meters and put model in train mode
        self.model.train()
        self.meters.reset()
        # self.train_metrics.reset()
        # self.val_metrics.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.train_loader)))

        for batch_idx, batch in enumerate(self.train_loader):
            # Zero the optimizer
            self.optimizer.zero_grad()
            
            # Send batch to _train_step and backpropagate
            loss_dict = self._train_step(batch)
            loss_dict['loss'].backward()

            # Update loss meters
            self.meters.update("total_loss", loss_dict['loss'].item(), 1)
            self.meters.update("box_loss", loss_dict['box_loss'].item(), 1)
            self.meters.update("class_loss", loss_dict['class_loss'].item(), 1)

            # Step optimizer
            self.optimizer.step()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch,
                    epochs=self.args.epochs,
                    batch=batch_idx,
                    iter=len(self.train_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss_dict['loss'].item()
                )
            )
            p_bar.update()

        # Step LR scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Compute epoch metrics and loss
        return self.meters.averages()

    @torch.no_grad()
    def _val_step(self, batch: Tuple):
        """ Validate one batch of images """
        # Unpack the batch
        img_stack, targets, img_names = batch
        # Move image stack and targets to device
        img_stack = img_stack.to(self.args.device)
        targets = move_target_to_device(targets, device=self.args.device)

        # Make forward pass over data
        loss_dict = self.model(img_stack, targets)

        return loss_dict

    @torch.no_grad()
    def _val_epoch(self, epoch):
        self.model.eval()
        self.meters.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.val_loader)))

        for batch_idx, batch in enumerate(self.val_loader):
            # Send batch to _train_step and backpropagate
            loss_dict = self._val_step(batch)

            predictions = loss_dict['detections']

            # Update loss meters
            self.meters.update("total_loss", loss_dict['loss'].item(), 1)
            self.meters.update("box_loss", loss_dict['box_loss'].item(), 1)
            self.meters.update("class_loss", loss_dict['class_loss'].item(), 1)

            # Update progress bar
            p_bar.set_description(
                "Val Epoch: {epoch}/{epochs:4}. Iter: {batch}/{iter:4}. Loss: {loss:.6f}".format(
                    epoch=epoch,
                    epochs=self.args.epochs,
                    batch=batch_idx,
                    iter=len(self.val_loader),
                    loss=loss_dict['loss'].item()
                )
            )
            p_bar.update()
        
        # Compute metrics and loss
        return self.meters.averages()


    def train(self):
        for epoch in range(1, self.args.epochs+1):
            # train_loss = self._train_epoch(epoch)
            # print(train_loss)
            val_loss = self._val_epoch(epoch)
            print(val_loss)

def move_target_to_device(target: dict, device: str):
    """Recursively moves all tensors in a nested dictionary to the specified device.

    Args:
        target (dict): A target dictionary with keys 'bbox', 'cls', 'img_size', 'img_scale'
        device (str): Either 'cpu' or 'gpu'

    Returns:
        target (dict): The target dictionary with keys moved to torch.device
    """
    
    if isinstance(target, dict):
        return {key: move_target_to_device(value, device) for key, value in target.items()}
    elif isinstance(target, list):
        return [move_target_to_device(item, device) for item in target]
    elif isinstance(target, torch.Tensor):
        return target.to(device)
    else:
        return target