"""
test.test_post_processing.py
BoMeyering 2025
"""
from src.utils.post_processing import WbfDetector
import unittest
import argparse
from pathlib import Path
import numpy
import torch
import string
import random
import pandas
import json

from wonderwords import RandomWord

def create_dummy_detections(batch_size: int=3, num_det: int=100, num_classes: int=3, img_size: int=512) -> torch.Tensor:
    """
    Generates a batch of dummy detections

    Args:
        batch_size (int, optional): Number of samples in a batch. Defaults to 3.
        num_det (int, optional): Number of detections per sample. Defaults to 100.
        num_classes (int, optional): Number of classes to detect. Defaults to 3.
        img_size (int, optional): Image size. Defaults to 512.

    Returns:
        torch.Tensor: A torch.tensor of dummy bounding box detections
    """
    detections = torch.zeros((batch_size, num_det, 6))
    for i, sample in enumerate(detections):
        for j, row in enumerate(sample):
            bbox_size = random.randint(30, 100)
            xmin = random.randint(0, img_size-bbox_size)
            ymin = random.randint(0, img_size-bbox_size)
            xmax = xmin + bbox_size
            ymax = ymin + bbox_size
            score = random.uniform(0, 1)
            cls = random.randint(1, num_classes)
            row = torch.tensor([xmin, ymin, xmax, ymax, score, cls])
            sample[j] = row
        detections[i] = sample

    return detections

class TestWbfDetector(unittest.TestCase):
    """
    Test the functionality of the WbfDetector class
    """
    def setUp(self):
        """ Setup the TestWbfDetector class """
        self.args = argparse.Namespace()
        self.args.pred_conf_threshold = 0.2
        self.args.wbf_iou_threshold = 0.5
        self.args.skip_box_threshold = 0.001
        self.args.img_size = 512

        self.detections_list = [[1, 1, 1, 1, 1, 1], # Incorrect data type for detections
                                [2, 2, 2, 2, 2, 2],
                                [3, 3, 3, 3, 3, 3]]
        
        self.batch = create_dummy_detections(batch_size=3, num_det=100) # Create a dummy batch
        self.sample = create_dummy_detections(batch_size=1, num_det=100) # Create one sample with batch_size 1

        # Set up formatted_detections with bad keys
        self.bad_keys_formatted_predictions = []
        for _ in range(3):
            prediction = {
                RandomWord().word(): [],
                RandomWord().word(): [],
                RandomWord().word(): []
            }
            self.bad_keys_formatted_predictions.append(prediction)

        # Set up formatted detections with correct keys and bad values
        self.bad_values_formatted_predictions = []
        for _ in range(3):
            prediction = {
                'boxes': [],
                'scores': [],
                'classes': []
            }
            self.bad_values_formatted_predictions.append(prediction)

        # Instantiate new WbfDetector
        self.wbf_detector = WbfDetector(self.args)

    def test_wbf_detector_init(self):
        """ Test the __init__ method """
        self.assertEqual(type(self.wbf_detector.args), argparse.Namespace)

    def test__format_sample_detections_errors(self):  
        """ Test raise error for _format_sample_detections method """      
        # Check ValueError for wrong data type (list)
        with self.assertRaises(ValueError):
            self.wbf_detector._format_sample_detections(self.detections_list)
        
        # Check ValueError for batch dim > 1
        with self.assertRaises(ValueError):
            self.wbf_detector._format_sample_detections(self.batch)
    
    def test__format_sample_detections_data(self):
        """ Test data integrity for _format_sample_detections method """
        # Process correct dimensions
        detections, idx = self.wbf_detector._format_sample_detections(self.sample)


        # Check that wbf_detector returns a dictionary
        self.assertEqual(type(detections), dict)
        
        # Check for the correct keys and data types in detections
        for k in ['boxes', 'scores', 'classes']:
            self.assertIn(k, detections.keys())
            self.assertEqual(type(detections[k]), numpy.ndarray)
        
        # Check dimensions of each object in detections
        num_det = len(idx)
        self.assertEqual(detections['boxes'].shape, (num_det, 4)) # Check for dimensions of (num_det, 4)
        self.assertEqual(detections['scores'].shape, (num_det,)) # Check for dimensions of (num_det, )
        self.assertEqual(detections['classes'].shape, (num_det,))


        
    def test__format_batch_detections_errors(self):
        """ Test raise error for _format_batch_detections method """
        # Check ValueError for wrong data type
        with self.assertRaises(ValueError):
            self.wbf_detector._format_batch_detections(self.detections_list)

    def test__format_batch_detections_data(self):
        """ Test data integrity for _format_batch_detections method """
        for _ in range(100):
            # Set random batch size, num_det and create dummy detections
            batch_size = random.randint(1,10)
            num_det = random.randint(50, 300)
            batch_detections = create_dummy_detections(batch_size=batch_size, num_det=num_det)

            # Process the batch
            predictions = self.wbf_detector._format_batch_detections(batch_detections)

            # Check that predictions is a list
            self.assertEqual(type(predictions), list)
            
            # Check for correct size according to batch
            self.assertEqual(len(predictions), batch_size)
            
            # Check each sample is a formatted det
            self.assertTrue(
                all([type(formatted_det) == dict for formatted_det in predictions])
            )

    def test__run_wbf_errors(self):
        """ Test raise error for _run_wbf method """
        # Check ValueError for wrong data type
        with self.assertRaises(ValueError):
            self.wbf_detector._run_wbf(
                self.batch, 
                img_size=self.args.img_size, 
                iou_thr=self.args.wbf_iou_threshold, 
                skip_box_thr=self.args.skip_box_threshold
            )

        # Check KeyError for 'formatted_predictions'
        with self.assertRaises(KeyError):
            self.wbf_detector._run_wbf(
                self.bad_keys_formatted_predictions, 
                img_size=self.args.img_size, 
                iou_thr=self.args.wbf_iou_threshold, 
                skip_box_thr=self.args.skip_box_threshold
            )

        # Check ValueError for 'formatted_predictions'
        with self.assertRaises(ValueError):
            self.wbf_detector._run_wbf(
                self.bad_values_formatted_predictions, 
                img_size=self.args.img_size, 
                iou_thr=self.args.wbf_iou_threshold, 
                skip_box_thr=self.args.skip_box_threshold
            )

    def test__run_wbf_data(self):
        """ Test data integrity for _run_wbf method """
        # Create new batches, format, and send to _run_wbf method
        for _ in range(100):
            batch_size = random.randint(1, 8)
            num_det = random.randint(50, 100)
            new_batch = create_dummy_detections(batch_size=batch_size, num_det=num_det)
            formatted_predictions = self.wbf_detector._format_batch_detections(new_batch)
            bboxes, confidences, class_labels = self.wbf_detector._run_wbf(
                formatted_predictions, 
                img_size=self.args.img_size, 
                iou_thr=self.args.wbf_iou_threshold, 
                skip_box_thr=self.args.skip_box_threshold
            )

            # Check length of _run_wbf results
            self.assertEqual(len(bboxes), batch_size)
            self.assertEqual(len(confidences), batch_size)
            self.assertEqual(len(class_labels), batch_size)

            # Check that number of returned predictions is the <= num_det
            for i in range(batch_size):
                sample_box_list = bboxes[i]
                sample_confidence_list = confidences[i]
                sample_class_list = class_labels[i]

                self.assertLessEqual(len(sample_box_list), num_det)
                self.assertLessEqual(len(sample_confidence_list), num_det)
                self.assertLessEqual(len(sample_class_list), num_det)

    def test_process_batch_errors(self):
        """ Test raise error for process_batch method """
        # Check ValueError for wrong data type
        with self.assertRaises(ValueError):
            self.wbf_detector.process_batch(self.detections_list)
        
        # Check ValueError for batch_detections with no batch dimension
        with self.assertRaises(ValueError):
            self.wbf_detector.process_batch(self.sample.squeeze(0))

    def test_process_batch_data(self):
        """ Test data integrity for process_batch method """
        for _ in range(100):
            # Set random batch size, num_det and create dummy detections
            batch_size = random.randint(1,10)
            num_det = random.randint(50, 300)
            batch_detections = create_dummy_detections(batch_size=batch_size, num_det=num_det)

            bboxes, confidences, class_labels = self.wbf_detector.process_batch(batch_detections)

            # Check that bboxes, confidences, and class_labels all have the correct length
            self.assertEqual(len(bboxes), batch_size)
            self.assertEqual(len(confidences), batch_size)
            self.assertEqual(len(class_labels), batch_size)

if __name__ == "__main__":
    unittest.main(verbosity=2)
