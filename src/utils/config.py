"""
src.utils.config.py

Sets up all of the config YAML file parsing

BoMeyering 2025
"""

import yaml
import json
import os
import pathlib
import argparse
import sys
import logging

import numpy as np

from pathlib import Path
from typing import Union
from datetime import datetime
from typing import Tuple, Any


    
class YamlConfigLoader:
    def __init__(self, path: Union[Path, str]) -> None:
        """ Initialize yaml loader """
        if not isinstance(path, (str, pathlib.Path)):
            raise TypeError(f"'path' argument should be either type 'str' or 'pathlib.Path', not type {type(path)}.")
        if not str(path).endswith(('.yml', '.yaml')):
            raise ValueError(f"path should be a Yaml file ending with either '.yaml' or '.yml'.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File path at {path} does not exist. Please specify a different path")
        
        self.path = path

    def load_config(self) -> dict:
        """Reads a yaml config file at path and returns a dictionary of config arguments.

        Returns:
            dict: A dictionary of key/value pairs for the arguments in the config file.
        """
        with open(self.path, 'r') as file:
            return yaml.safe_load(file)
        
class ArgsAttributes:
    def __init__(self, args: argparse.Namespace, config: dict) -> None:
        if not isinstance(args, argparse.Namespace):
            raise TypeError(f"'args' should be an argparse.Namespace object.")
        if not isinstance(config, dict):
            raise TypeError(f"'config' should be an dict object.")
        self.args = args
        self.config = config
    
    def append_timestamp_to_run(self):
        now = datetime.now().isoformat(timespec='seconds', sep='_').replace(":", ".")
        try:
            if hasattr(self.args, 'run_name'):
                self.args.run_name = "_".join((self.args.run_name, now))
            else:
                self.args.run_name = "_".join(('default_run', now))
        except AttributeError as e:
            print(e)
            print(f"Setting default run_name to 'default_run_{now}'")
            setattr(self.args, 'run_name', "_".join(('default_run', now)))

    def set_args_attr(self, check_run_name=True) -> argparse.Namespace:
        """Takes a parsed yaml config file as a dict and adds the arguments to the args namespace.

        Returns:
            argparse.Namespace: The args namespace updated with the configuration parameters.
        """
        for k, v in self.config.items():
            if isinstance(v, dict):
                setattr(self.args, k, argparse.Namespace(**v))
            else:
                setattr(self.args, k, v)
        if check_run_name:
            self.append_timestamp_to_run()

    def validate(self):
        pass

def setup_loggers(args):
    """
    Configures a simple logger to log outputs to the console and the output file.

    Args:
        args (argparse.Namespace): arguments object from the configuration file.
    """
    filename = args.run_name + '.log'
    filepath = Path(args.directories.log_dir) / filename

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler(filepath, 'a', 1000000, 3)
    stream_handler = logging.StreamHandler(sys.stdout,)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.DEBUG)


def read_label_map(args: argparse.Namespace):
    """Read in and validate the label map JSON

    Args:
        args (argparse.Namespace): The parsed args.Namespace from a config yaml

    Raises:
        ValueError: If 'args.label_map' does not point to a JSON file
        ValueError: If 'args.label_map' is not set
        ValueError: If the number of key-value pairs in the map differs from 'args.num_classes'
    """
    if args.label_map:
        if args.label_map.endswith('json'):
            with open(args.label_map, 'r') as f:
                mapping = json.load(f)
        else:
            raise ValueError(f"'label_map' path in {args.config} must point to a valid JSON file.")
    else:
        raise ValueError("No path associated with key 'label_map' in config.yaml")
    
    # Validate number of keys
    if len(mapping.keys()) != args.num_classes:
        raise ValueError(
            f"The number of key-value pairs in {args.label_map} ({len(mapping.keys())}) "\
            f"differs from the number of target classes specified in {args.config} ({args.num_classes})"
        )
    
    args.__setattr__('label_mapping', mapping)


def read_rgb_norm(args: argparse.Namespace):
    """Read in the rgb norm json file

    Args:
        args (argparse.Namespace): The parsed args.Namespace from a config yaml

    Raises:
        ValueError: If 'args.rgb_norm' is not a valid JSON
        KeyError: If 'args.rgb_norm' JSON contains incorrect keys
        ValueError: If values in 'args.rgb_norm' are not lists
        ValueError: If values in 'args.rgb_norm' do not have length 3
        ValueError: If values in 'args.rgb_norm['rgb_means']' are not in the interval [0, 1]
        ValueError: If values in 'args.rgb_norm['rgb_std']' are not positive
    """

    if 'rgb_norm' in vars(args):
        if args.rgb_norm.endswith('json'):
            with open(args.rgb_norm, 'r') as f:
                rgb_norm = json.load(f)
        else:
            raise ValueError(f"'rgb_norm' path in {args.config} must point to a valid JSON file")
        
        # Validate rgb means
        for key in ['rgb_means', 'rgb_std']:
            if key not in rgb_norm.keys():
                raise KeyError(f"{args.rgb_norm} file must contain the key {key}")
            
            if type(rgb_norm[key]) is not list:
                raise ValueError(f"Value for key {key} must be a list of floats")
            
            if len(rgb_norm[key]) != 3:
                raise ValueError(f"Value for key {key} must be a list of 3 float values, one for each rgb channel.")
            
            if key == 'rgb_means':
                if np.any(np.array(rgb_norm[key]) < 0) or np.any(np.array(rgb_norm[key]) > 1):
                    raise ValueError(f"Values for {key} must all be in the interval [0, 1].")
            elif key == 'rgb_std':
                if np.any(np.array(rgb_norm[key]) < 0):
                    raise ValueError(f"Values for {key} must all be positive.")

        args.rgb_means = rgb_norm['rgb_means']
        args.rgb_std = rgb_norm['rgb_std']

    else:
        print(f"Key 'rgb_norm' not set in {args.config}. Defaulting to standard COCO dataset rgb means and std.")
        args.rgb_means = None
        args.rgb_std = None