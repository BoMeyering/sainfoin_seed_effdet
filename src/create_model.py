"""
create_model.py
Stand up training and inference models

BoMeyering 2024
"""

import torch
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

def create_model(num_classes=2, image_size=512, architecture="tf_efficientdet_d1"):
    try:
       config = get_efficientdet_config(architecture)
    except KeyError:
        efficientdet_model_param_dict[architecture] = dict(
            name=architecture,
            backbone_name=architecture,
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=num_classes,
            url='', )
        config = get_efficientdet_config(architecture)

    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)

def create_inference_model(num_classes=2, image_size=512, architecture="tf_efficientdet_d1"):
    try:
        config = get_efficientdet_config(architecture)
    except KeyError:
        efficientdet_model_param_dict[architecture] = dict(
            name=architecture,
            backbone_name=architecture,
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=num_classes,
            url='', )
        config = get_efficientdet_config(architecture)

    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes
    )
    return DetBenchPredict(net)


if __name__ == '__main__':
    train_model = create_model(3, 640)

    inf_model = create_inference_model(3, 640)


    X = torch.randn(1, 3, 640, 640)

    target = {
        'bbox': [[]],
        'cls': []
    }

    train_model(X, target)