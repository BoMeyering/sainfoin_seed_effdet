import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from src.models import EffDetWrapper
from src.transforms import get_inference_transforms
import random
from ensemble_boxes import weighted_boxes_fusion

state_dict = torch.load('model_checkpoints/effdet_d4_wd0.005_2026-02-16_16.37.20/effdet_d4_wd0.005_2026-02-16_16.37.20_epoch_12_vloss-0.305933.pth')

conf = OmegaConf.load('model_checkpoints/effdet_d4_wd0.005_2026-02-16_16.37.20/effdet_d4_wd0.005_2026-02-16_16.37.20_config.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EffDetWrapper(conf=conf, device=device)
model.load_state_dict(state_dict['ema_state_dict'])
model.eval_mode()

print("Model state dict loaded successfully.")

transforms = get_inference_transforms(rgb_means=conf.metadata.norm.means, rgb_stds=conf.metadata.norm.std, resize=conf.images.resize)


import json
import cv2

with open('data/val_labels.json', 'r') as f:
    val_labels = json.load(f)

BASE_DIR = 'data/raw'
OUTPUT_DIR = 'outputs/inference'
files = list(val_labels.keys())

for filename in tqdm(files, colour='green'):
    image = cv2.imread(f"{BASE_DIR}/{filename}", cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)
    h, w, _ = image.shape
    input_tensor = transforms(image=image)['image'].unsqueeze(0).to(device)
    outputs = model.predict(input_tensor)

    outputs = outputs.cpu().numpy()
    top = outputs[np.where(outputs[:, :, 4] > 0.6)]

    bboxes = top[:, :4].astype(np.int32)
    scores = top[:, 4]
    labels = top[:, 5].astype(np.int32)

    bboxes = bboxes / conf.images.resize # Scaled [0,1]

    print("BOXES SHAPE: ", bboxes.shape)
    print("SCORES SHAPE: ", scores.shape)
    print("LABELS SHAPE: ", labels.shape)

    iou_thr = 0.4
    thresh = 0.01
    method = 'wbf'
    bboxes, scores, labels = weighted_boxes_fusion([bboxes], [scores], [labels], weights=None, iou_thr=iou_thr, skip_box_thr=thresh)

    # Rescale boxes to original image dimensions
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    print(bboxes)
    out_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box.astype(int)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(out_image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out_image, f'{scores[i]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 3)
        cv2.putText(out_image, f'Class: {labels[i]}', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 3)
    cv2.imwrite(f'{OUTPUT_DIR}/{filename}_preds.jpg', out_image)