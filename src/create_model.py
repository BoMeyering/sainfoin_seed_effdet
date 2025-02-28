"""
create_model.py
Stand up training and inference models

BoMeyering 2024
"""

import argparse
import torch
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

def create_model(args: argparse.Namespace):
    try:
       config = get_efficientdet_config(args.architecture)
    except KeyError:
        efficientdet_model_param_dict[args.architecture] = dict(
            name=args.architecture,
            backbone_name=args.architecture,
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=args.num_classes,
            url='', )
        config = get_efficientdet_config(args.architecture)

    config.update({'num_classes': args.num_classes})
    config.update({'image_size': (args.img_size, args.img_size)})
    if 'max_det_per_image' in vars(args):
        config.update({'max_det_per_image': args.max_det_per_image})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)

def create_inference_model(args: argparse.Namespace):
    try:
        config = get_efficientdet_config(args.architecture)
    except KeyError:
        efficientdet_model_param_dict[args.architecture] = dict(
            name=args.architecture,
            backbone_name=args.architecture,
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=args.num_classes,
            url='', )
        config = get_efficientdet_config(args.architecture)

    config.update({'num_classes': args.num_classes})
    config.update({'image_size': (args.img_size, args.img_size)})

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes
    )
    return DetBenchPredict(net)