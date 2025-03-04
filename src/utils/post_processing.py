"""
src.utils.post_processing.py

Post detection processing functionality

BoMeyering 2025
"""

import torch
import argparse
import random

import numpy
import numpy as np

from ensemble_boxes import ensemble_boxes_wbf
from typing import Tuple, List

class WbfDetector:
    """ Run Weighted Box Fusion on bboxes post inference """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _format_sample_detections(self, detections: torch.Tensor) -> dict:
        """ Format detections for one sample of predictions """

        # Check that detections type is a torch.Tensor
        if type(detections) != torch.Tensor:
            raise ValueError(f"'detections' must be of type torch.Tensor.")
        
        # Check that only one sample is present
        if (len(detections.shape) == 3) and (detections.shape[0] > 1):
            raise ValueError(f"Detections for more than one image are present.")
        
        # Remove batch dimension if only one sample
        elif (len(detections.shape) == 3) and (detections.shape[0] == 1):
            detections = detections.squeeze(0)

        # Check that there are 6 columns in detections
        if detections.shape[1] != 6:
            raise ValueError(f"'detections' with no batch dimension should have shape (num_detections, 6).")
        
        # Detach from the graph and move to numpy array
        detections = detections.detach().cpu().numpy()
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5].astype(int)

        # Get index above confidence threshold
        idx = np.where(scores > self.args.pred_conf_threshold)[0]

        # Return formatted detections
        return {'boxes': boxes[idx], 'scores': scores[idx], 'classes': classes[idx]}, idx
    
    def _format_batch_detections(self, batch_detections: torch.Tensor) -> List[dict]:
        """ Format detections for a batch of samples """

        # Check if batch_detections is a torch.Tensor
        if type(batch_detections) != torch.Tensor:
            raise ValueError(f"'detections' must be of type torch.Tensor.")
        
        formatted_predictions = []

        # Format each sample detections and append to formatted predictions
        for det in batch_detections:
            formatted_det, _ = self._format_sample_detections(det)
            formatted_predictions.append(formatted_det)

        return formatted_predictions
    
    def _run_wbf(self, formatted_predictions, img_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None) -> Tuple[List, ...]:
        """ Run Weighted Box Fusion on a list of formatted prediction dictionaries """

        # Check that predictions is a list of dictionaries
        if type(formatted_predictions) != list:
            raise ValueError(f"_run_wbf() method'formatted_predictions' should be a list of dictionaries.")
        
        # Check that each sample in predictions has correct keys and values
        for i, prediction in enumerate(formatted_predictions):
            for key in ['boxes', 'scores', 'classes']:
                if key not in prediction.keys():
                    raise KeyError(f"key {key} not found in sample {i} in 'formatted_predictions'.")

                if type(prediction[key]) != numpy.ndarray:
                    raise ValueError(f"Value for key {key} in each samples in 'formatted_predictions' must be of type numpy.ndarray.")
        
        # Instantiate wbf results lists
        bboxes = []
        confidences = []
        class_labels = []

        for prediction in formatted_predictions:
            boxes = [(prediction["boxes"] / img_size).tolist()]
            scores = [prediction["scores"].tolist()]
            labels = [prediction["classes"].tolist()]

            boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
                boxes,
                scores,
                labels,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )

            # Scale back up to the inference img_size
            boxes = boxes * img_size

            # Append individual sample results to results lists
            bboxes.append(boxes.tolist())
            confidences.append(scores.tolist())
            class_labels.append(labels.tolist())

        return bboxes, confidences, class_labels
    
    def _rescale_bboxes(self, unscaled_bboxes: List[List], img_sizes: List[Tuple]) -> List:
        """ Rescales the bounding boxes from _run_wbf to the original raw image size """

        # Initiate scaled bounding box list
        scaled_bboxes = []

        # Loop through bounding boxes and scale
        for bboxes, img_dims in zip(unscaled_bboxes, img_sizes):
            img_h, img_w = img_dims

            if len(bboxes) > 0:
                scale_factor = np.array([
                    img_w / self.args.img_size,
                    img_h / self.args.img_size,
                    img_w / self.args.img_size,
                    img_h / self.args.img_size
                ])

                scaled_bboxes.append((bboxes * scale_factor).tolist())
            else:
                # Append empty bounding boxes to scaled_bboxes if empty
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

    def process_batch(self, batch_detections: torch.Tensor, rescale_img_sizes: List[Tuple]=None) -> Tuple[List, ...]:
        """ process a whole batch of detections for n images """

        # Check that batch_detections is a torch.Tensor
        if type(batch_detections)  != torch.Tensor:
            raise ValueError(f"'batch_detections' must be of type torch.Tensor")
        
        # Check that batch_detections has 3 dimensions
        if len(batch_detections.shape) != 3:
            raise ValueError(f"Expected 'batch_detections' to have a batch dimension a index 0")
        
        # Format the batch_detections
        formatted_predictions = self._format_batch_detections(batch_detections)

        # Run Weighted Box Fusion
        bboxes, confidences, class_labels = self._run_wbf(
            formatted_predictions=formatted_predictions, 
            img_size=self.args.img_size,
            iou_thr=self.args.wbf_iou_threshold,
            skip_box_thr=self.args.skip_box_threshold
        )

        # Rescale bounding boxes if raw image dimensions are passed
        if rescale_img_sizes is not None:
            bboxes = self._rescale_bboxes(
                unscaled_bboxes=bboxes, 
                img_sizes=rescale_img_sizes
            )

        return bboxes, confidences, class_labels
