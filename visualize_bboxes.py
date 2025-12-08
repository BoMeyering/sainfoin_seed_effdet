import polars as pl
import torch
import albumentations as A
import cv2
import numpy as np
import os
from glob import glob
from pathlib import Path

from albumentations.pytorch.transforms import ToTensorV2

from src.create_model import create_model

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

from torch.utils.data import Dataset, DataLoader
from torch.optim.adamw import AdamW

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_SIZE = 1024
ARCH = "tf_efficientdet_d1"
config = get_efficientdet_config(ARCH)
config.update({'num_classes': 3})
config.update({'image_size': (IMG_SIZE, IMG_SIZE)})

cuustom_anchors = [
    [8, 8],
    [16, 16],
    [32, 32],
    [64, 64],
    [128, 128],
    [256, 256]
]

config.update({'anchor_sizes': cuustom_anchors})

net = EfficientDet(config, pretrained_backbone=True)
net.class_net = HeadNet(
    config,
    num_outputs=config.num_classes,
)

model = DetBenchTrain(net, config)
model = model.to(device)

# print(model)

def draw_bounding_boxes(img, bboxes, labels):
    unique_labels = np.unique(labels).tolist()

    np.random.seed(42)  # Fixed seed for reproducibility
    colors = {label: tuple(np.random.randint(0, 255, 3).tolist()) for label in unique_labels}
    
    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        color = colors[label]
        
        # Draw bounding box
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1)
        
        # Add label text
        label_text = f"Class {label}"
        cv2.putText(img, label_text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return img

def show_img(img):
    cv2.namedWindow('test', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def train_transforms(img_size) -> A.Compose:
    return A.Compose(
        [   
            A.Normalize(),
            A.Resize(height=img_size, width=img_size, p=1),
            # A.HorizontalFlip(),
            # A.GaussianBlur(),
            A.SafeRotate(p=.75),
            A.ChannelShuffle(p=.75),
            # A.GridDistortion(p=.75),
            # A.PlasmaShadow([0.0, 0.2], roughness=1)
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def val_transforms(img_size) -> A.Compose:
    return A.Compose(
        [   
            A.Normalize(),
            A.Resize(height=img_size, width=img_size, p=1),
            ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


img_names = [
    'cycle1_0003.JPG',
    'cycle1_0017.JPG',
    'cycle1_0018.JPG',
    'cycle1_0019.JPG',
    'cycle1_0035.JPG',
    'cycle1_0040.JPG'
]

def move_target_to_device(target: dict, device: str):
    """Recursively moves all tensors in a nested dictionary to the specified device.

    Args:
        target (dict): A target dictionary with keys 'bbox', 'cls', 'img_size', 'img_scale'
        device (str): Either 'cpu' or 'gpu'

    Returns:
        target (dict): The target dictionary with keys moved to torch.device
    """
    
    if isinstance(target, dict):
        return {key: move_target_to_device(value, device) for key, value in target.items()}
    elif isinstance(target, list):
        return [move_target_to_device(item, device) for item in target]
    elif isinstance(target, torch.Tensor):
        return target.to(device)
    else:
        return target

class EffDetDataset(Dataset):
    def __init__(self, img_names: str, label_path: Path, transforms: A.Compose):
        self.img_names = img_names
        self.label_path = label_path
        self.transforms = transforms

        self.img_labels = pl.read_csv(self.label_path)
        
    def __len__(self):
        """
        Return the length (image number) of the dataset
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Get one sample from the dataset using index
        """
        
        # Construct path to image
        img_path = Path('data/images/val') / self.img_names[index]

        # Read in image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Select bboxes in Pascal format
        pascal_bboxes = self.img_labels.filter(
            pl.col('external_id') ==  self.img_names[index]
        )["xmin", "ymin", "xmax", "ymax"].to_numpy()

        # Grab obect labels
        labels = self.img_labels.filter(pl.col('external_id') == self.img_names[index])['feature_class'].to_numpy()

        # Contruct sample dict for transforms
        sample = {
            "image": img,
            "bboxes": pascal_bboxes,
            "labels": labels,
        }

        # Augment sample
        sample = self.transforms(**sample)
        _, new_h, new_w = sample['image'].shape

        # Convert to ymin, xmin, ymax, xmax
        # EffDet requires [ymin, xmin, ymax, xmax] order
        # If nothing present, advance to next sample
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]


        aug_img = sample['image']

        target = {
            "bbox": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "cls": torch.as_tensor(sample['labels']),
            "img_name": self.img_names[index],
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
            "raw_img_size": img.shape[:2]
        }

        aug_img = aug_img.to(device)
        target = move_target_to_device(target, device)

        # Return the image and the target data
        return aug_img, target
        
def collate_fn(batch):
    images, targets = tuple(zip(*batch))

    img_stack = torch.stack(images).float()
    boxes = [target['bbox'] for target in targets]
    labels = [target['cls'] for target in targets]
    img_names = [target['img_name'] for target in targets]
    raw_img_size = [target['raw_img_size'] for target in targets]
    img_size = torch.tensor([target['img_size'] for target in targets]).float()
    img_scale = torch.tensor([target['img_scale'] for target in targets]).float()

    targets = {
        'bbox': boxes,
        'cls': labels,
        'img_size': img_size,
        'img_scale': img_scale
    }

    img_stack_metadata = {
        'img_names': img_names,
        'raw_img_size': raw_img_size
    }
        
    return img_stack, targets, img_stack_metadata

transforms = train_transforms(IMG_SIZE)
val_transforms = val_transforms(IMG_SIZE)
test_ds = EffDetDataset(img_names, 'data/annotations/val_annotations.csv', val_transforms)
val_ds = EffDetDataset(img_names, 'data/annotations/val_annotations.csv', val_transforms)
test_dataloader = DataLoader(
    test_ds, batch_size=2, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_ds, batch_size=2, collate_fn=collate_fn
)

optimizer = AdamW(model.parameters(), lr=0.001)

rgb_means = [[[0.485, 0.456, 0.406]]]
rgb_std = [[[0.229, 0.224, 0.225]]]

for epoch in range(100):
    print("TRAINING: ", epoch)
    model.train()
    for batch_idx, batch in enumerate(test_dataloader):
        img_stack, targets, metadata = batch

        optimizer.zero_grad()

        loss_dict = model(img_stack, targets)
        total_loss = loss_dict['loss']
        print(total_loss.item())
        total_loss.backward()
        optimizer.step()
    model.eval()
    print("VALIDATING: ", epoch)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            img_stack, targets, metadata = batch
            targets = move_target_to_device(targets, device)
            loss_dict = model(img_stack, targets)
            print(loss_dict['loss'].item())

            detections = loss_dict['detections']

            for img, box_list, img_name in zip(img_stack, detections, metadata['img_names']):
                img = img.squeeze(0).detach().cpu().numpy()
                box_list = box_list.squeeze(0).detach().cpu().numpy()
               
                img = np.moveaxis(img, source=0, destination=2)
                img = (img * rgb_std) + rgb_means
                img = np.clip(img * 255, a_min = 0, a_max=255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                boxes = box_list[:, :4].tolist()
                # print(boxes)
                labels = box_list[:, 5].tolist()
                out_img = draw_bounding_boxes(img.copy(), boxes, labels)

                annotation_dir = Path(f'output/inference/test_1024/{epoch}')
                if not os.path.exists(annotation_dir):
                    os.makedirs(annotation_dir)
                    
                out_path = annotation_dir / img_name

                cv2.imwrite(out_path, out_img)




        
    














    

# transforms = train_transforms(IMG_SIZE)

# annotations = pl.read_csv('data/annotations/val_annotations.csv')

# img_names = glob("*", root_dir='data/images/val')

# for img_name in img_names:
#     img_path = Path('data/images/val') / img_name

#     img = cv2.imread(img_path)

#     img_h, img_w = img.shape[:2]
#     img_annotations = annotations.filter(
#         pl.col('external_id') == img_name
#     )['xmin', 'ymin', 'xmax', 'ymax'].to_numpy()

#     labels = annotations.filter(
#         pl.col('external_id') == img_name
#     )['feature_class'].to_list()

#     bbox_img = draw_bounding_boxes(img.copy(), img_annotations, labels)

#     sample = {
#         "image": img,
#         "bboxes": img_annotations,
#         "labels": labels,
#     }

#     sample = transforms(**sample)
#     # print(sample)

#     trans_img = sample['image']
#     # show_img(trans_img)

#     bbox_img = draw_bounding_boxes(trans_img.copy(), sample['bboxes'].tolist(), labels)
#     # show_img(bbox_img)