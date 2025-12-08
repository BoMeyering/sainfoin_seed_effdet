"""
src.eval.py

Loss value meter functions

BoMeyering 2025
"""

import torch
import torchmetrics
import warnings

from typing import List

from torchmetrics import MetricCollection
from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class AverageMeter:
    """
    AverageMeter implements a class which can be used to track a metric over the entire training process.
    (see https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all class variables to default values
        """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates class variables with new value and weight
        """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        """
        Implements format method for printing of current AverageMeter state
        """
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )

class AverageMeterSet:
    """
    AverageMeterSet implements a class which can be used to track a set of metrics over the entire training process
    based on AverageMeters (Source: https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.vals for name, meter in self.meters.items()}

    def averages(self, postfix="_avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="_sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="_count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
    

class EffDetEvalMetrics(MetricCollection):
    """ """
    def __init__(self, class_metrics: bool=False):
        metrics = {
            'giou': GeneralizedIntersectionOverUnion(class_metrics=class_metrics),
            'map': MeanAveragePrecision(class_metrics=class_metrics, average='micro')
        }
        super().__init__(metrics)
    @staticmethod
    def _format_preds(preds: dict):
        formatted_preds = []
        for i in range(len(preds['boxes'])):
            sample = {
                'boxes': torch.tensor(preds['boxes'][i]),
                'scores': torch.tensor(preds['scores'][i]),
                'labels': torch.tensor(preds['labels'][i], dtype=torch.int)
            }
            formatted_preds.append(sample)
            print("PREDS", sample)
        
        return formatted_preds
    
    @staticmethod
    def _format_target(target: dict):
        formatted_target = []
        for i in range(len(target['bbox'])):
            # Reorder boxes from (ymin, xmin, ymax, xmax) (required for model) to (xmin, ymin, xmax, ymax)
            try:
                reordered_boxes = target['bbox'][i][:, [1, 0, 3, 2]]
                sample = {
                    'boxes': reordered_boxes,
                    'labels': target['cls'][i]
                }
            except IndexError as e:
                sample = {
                    'boxes': [[]],
                    'labels': []
                }
            formatted_target.append(sample)
            print("TARGET", sample)
        
        return formatted_target

    def update(self, preds, target):
        preds = self._format_preds(preds)
        target = self._format_target(target)
        
        super().update(preds, target)

    def forward(self, preds, target):
        preds = self._format_preds(preds)
        target = self._format_target(target)
        
        metrics = super().forward(preds, target)

        return metrics

    def compute(self):
        """ """
        try:
            return super().compute()
        except RuntimeError:
            warnings.warn("Torchmetrics MetricCollection compute() method - torch.cat(): expected a non-empty list of Tensors. Returning an empty dictionary.", RuntimeWarning)

            return {}
        