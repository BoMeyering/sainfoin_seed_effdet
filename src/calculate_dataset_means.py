import numpy as np
from PIL import Image
from glob import glob
import os
from tqdm import tqdm
import json

img_dir = './data/images/all_images'
img_paths = glob('*', root_dir=img_dir)

img_rgb = []
for img in tqdm(img_paths):
    img_array = np.array(Image.open(os.path.join(img_dir, img)).getdata()) / 255.
    img_rgb.append(img_array)

means = []
for img in img_rgb:
    means.append(np.mean(img, axis=0))
mu_rgb = np.mean(means, axis=0)

variances = []
for img in img_rgb:
    var = np.mean((img - mu_rgb) ** 2, axis=0)
    variances.append(var)
std_rgb = np.sqrt(np.mean(variances, axis=0))

out_path = './data/annotations/dataset_norm.json'

with open(out_path, 'w') as f:
    data = json.dump(
        {
            'means': mu_rgb.tolist(), 
            'std': std_rgb.tolist()
        },
        f
    )