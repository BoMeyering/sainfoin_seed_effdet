"""
calculate_dataset_means.py
BoMeyering 2025

This scripts loops through each image in the dataset, 
and uses the pixel values to calculate the channel means and standard deviations.
"""

import os
import argparse
import torch
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from utils.welford import WelfordCalculator

# Set up the terminal parser
parser = argparse.ArgumentParser()
parser.add_argument("directory", help="The path to the directory where the images are stored")
parser.add_argument("-d", "--device", default='cpu')
args = parser.parse_args()

# Initialize the torch device
device = torch.device(args.device)

# Initialize the Welford Calculator
welford = WelfordCalculator(device=device)

# Grab all image paths
img_names = glob('*', root_dir=args.directory)

# Main loop - update stats for each image
for img_name in tqdm(img_names):
    img_path = os.path.join(args.directory, img_name) # Create image path
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # Read and convert to RGB
    img = img / 255. # Normalize [0, 1]
    img_array = np.moveaxis(img, 2, 0) # Move channels to first dim
    img_array = torch.Tensor(img_array, device=device) # Convert to Tensor
    welford.update(img_array)

# Calculate final values
means, std = welford.compute()

print(means, std)

# Write results to json
out_path = './data/annotations/dataset_norm.json'
with open(out_path, 'w') as f:
    data = json.dump(
        {
            'rgb_means': means.cpu().numpy().tolist(), 
            'rgb_std': std.cpu().numpy().tolist()
        },
        f
    )