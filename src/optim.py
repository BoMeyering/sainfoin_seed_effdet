"""
src.optim.py

Optimizer and lr scheduler classes and functions

BoMeyering 2025
"""

import torch
import inspect
import argparse
import logging

from typing import List, Generator, Tuple
import segmentation_models_pytorch as smp

logger = logging.getLogger()

def add_weight_decay(model: torch.nn.Module, weight_decay: float, skip_list: List=[]) -> List:
    """Apply weight decay to model parameters
    by filtering out bias and batch norm layers

    Args:
        model (torch.nn.Module): The model whose parameters will be filtered
        weight_decay (float): The weight decay value
        skip_list (List, optional): A list of any known parameters that need to be filtered. Defaults to [].

    Returns:
        List: A list of two dictionaries, one with the weight decay applied, the other with no weight decay applied.
    """

    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    params_list = [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]

    return params_list

def get_params(args: argparse.Namespace, model: torch.nn.Module) -> Tuple:
    """
    If args.filter_bias_and_bn is True, applies the weight decay only to non bias and bn parameters
    Else, applies weight decay to all parameters.

    Args:
        args (argparse.Namespace): The args parsed from the config yaml file.
        model (torch.nn.module): A torch.nn.Module model

    Returns:
        Tuple: A tuple containing the model parameters and the udpated weight decay value for the optimizer
    """

    if args.optimizer.weight_decay and args.optimizer.filter_bias_and_bn:
        logger.info("Filtering bias and norm parameters from weight decay parameter.")
        logger
        parameters = add_weight_decay(model, args.optimizer.weight_decay)

        # Set new weight decay as 0 so the weight decay already set does not get overwritten
        weight_decay = 0
        setattr(args.optimizer, 'original_weight_decay', args.optimizer.weight_decay)
        setattr(args.optimizer, 'weight_decay', weight_decay)
    else:
        logger.info("Applying weight decay to all parameters.")
        parameters = model.parameters()
    
    return parameters

class ConfigOptim:
    """
    Initialize optimization config
    """
    def __init__(self, args, model_parameters):
        self.args = args
        self.optim_hyperparams = vars(self.args.optimizer).copy()
        self.scheduler_hyperparams = vars(self.args.scheduler).copy()
        self.model_params = model_parameters

    def get_optimizer(self):
        """Instantiate a new optimizer with desired hyperparameters

        Returns:
            torch.optim optimizer: An instantiated optimizer
        """
        # Search for the optimzer class in torch.optim, else revert to SGD
        try:
            OptimClass = getattr(torch.optim, self.optim_hyperparams['name'])
        except AttributeError:
            print(f"The optimizer {self.optim_hyperparams['name']} is not in torch.nn. Defaulting to torch.optim.SGD.")
            OptimClass = torch.optim.SGD
            
        # Filter for the valid hyperparameters in args.optimizer, eg 'lr', 'beta', 'eps', etc.
        valid_hyperparams = inspect.signature(OptimClass).parameters

        # Construct a new hyperparam dict
        filtered_hyperparams = {k: v for k, v in self.optim_hyperparams.items() if k in valid_hyperparams}

        # Add model params to dictionary and update with filtered hyperparameters
        optim_params = {'params': self.model_params}
        optim_params.update(filtered_hyperparams)

        # Construct a new optimizer instance with the param dictionary
        optimizer = OptimClass(**optim_params)
        self.optimizer = optimizer

        return self.optimizer

    def get_scheduler(self):
        """Instantiate a new learning rate scheduler with the desired hyperparameters

        Returns:
            torch.optim.lr_scheduler: An instantiated lr scheduler
        """

        # Search for the scheduler class in torch.optim,.lr_scheduler else revert to LinearLR
        try:
            SchedClass = getattr(torch.optim.lr_scheduler, self.scheduler_hyperparams['name'])
        except AttributeError:
            print(f"The scheduler {self.args.scheduler.name} is not in torch.optim.lr_scheduler. Defaulting to torch.optim..")
            SchedClass = torch.optim.lr_scheduler.LinearLR

        # Filter for the valid hyperparameters in args.scheduler
        valid_params = inspect.signature(SchedClass).parameters

        # Construct a new hyperparam dictionary
        filtered_params = {k: v for k, v in self.scheduler_hyperparams.items() if k in valid_params}

        # Add optimizer to dictionary and update with filtered hyperparameters
        scheduler_params = {'optimizer': self.optimizer}
        scheduler_params.update(filtered_params)

        # Construct a new optimizer instance with the param dictionary
        scheduler = SchedClass(**scheduler_params)
        self.scheduler = scheduler

        return self.scheduler