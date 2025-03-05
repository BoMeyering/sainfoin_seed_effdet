"""
train.py

Main sainfoin seed detection training script
BoMeyering 2025
"""

import os
import torch
import yaml
import sys
import json
import logging
import logging.config
import datetime
import argparse

import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.datasets import EffDetDataset, collate_fn
from src.create_model import create_model, create_inference_model
from src.trainer import EffDetTrainer
from src.transforms import train_transforms, val_transforms
from src.optim import ConfigOptim, get_params

from src.utils.config import YamlConfigLoader, ArgsAttributes, setup_loggers, read_label_map, read_rgb_norm

parser = ArgumentParser(
        description="Pytorch EfficientDet model for sainfoin (Onobrychis viciifolia) detection and classification"
    )

parser.add_argument("-c", "--config", nargs='?', default='config/model_config.yaml')
parser.add_argument("-n", "--run-name", default="test")
args = parser.parse_args()

logger = logging.getLogger()

def main(args: argparse.Namespace):
    """Model training entrypoint

    Args:
        args (argparse.Namespace): args namespace containing the model config
    """
    # torch.multiprocessing settings
    mp.set_start_method("spawn", force=True)

    # Load config yaml file
    config_loader = YamlConfigLoader(args.config)
    config = config_loader.load_config()
    
    # Instantiate args namespace with config file
    arg_setter = ArgsAttributes(args, config)
    arg_setter.set_args_attr()

    # Set class label mapping
    read_label_map(args)

    # Set image normalization
    read_rgb_norm(args)

    # Create model and send to device
    model = create_model(args)
    model.to(torch.device(args.device))

    # Grab model parameters, filter, and apply weight decay
    model_parameters = get_params(args=args, model=model)
    if args.optimizer.filter_bias_and_bn:
        logger.info(f"Applied decay rate to non bias and batch norm parameters.")
    else:
        logger.info(f"Applied decay rate to all parameters.")

    # Get optimizer and lr scheduler
    opt_stuff = ConfigOptim(args, model_parameters)
    optimizer = opt_stuff.get_optimizer()

    if 'scheduler' in vars(args):
        scheduler = opt_stuff.get_scheduler()
    else: 
        scheduler = None

    

    logger.info(f"Initialized optimizer {args.optimizer.name}")
    logger.info(f"Initialized scheduler {args.scheduler.name}")

    # Build Datasets and Dataloaders
    # logger.info(f"Building datasets from {[v for _, v in vars(args.directories).items() if v.startswith('data')]}")

    train_tfms = train_transforms(args=args)
    val_tfms = val_transforms(args=args)

    train_ds = EffDetDataset(
        img_dir=args.train_dir, 
        label_path=args.train_label_path, 
        transforms=train_tfms, 
        label_map=args.label_mapping
    )

    val_ds = EffDetDataset(
        img_dir=args.val_dir,
        label_path=args.val_label_path,
        transforms=val_tfms,
        label_map=args.label_mapping
    )

    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        pin_memory=True, 
        drop_last=True, 
        num_workers=2
    )

    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        pin_memory=True, 
        drop_last=True, 
        num_workers=2
    )

    trainer = EffDetTrainer(
        args=args, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )

    trainer.train()

if __name__ == '__main__':
    main(args)