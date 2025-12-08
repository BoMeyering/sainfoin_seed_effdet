from src_new.datasets import DatasetAdaptor
from src_new.datasets import DatasetAdaptor
from src_new.models import EfficientDetModel
from src_new.datasets import EfficientDetDataModule
from src_new.transforms import get_train_transforms, get_valid_transforms
import torch
from pytorch_lightning import Trainer
import pandas as pd

IMG_SIZE = 1024

train_df = pd.read_csv("./data/annotations/train_annotations.csv")
val_df = pd.read_csv("./data/annotations/val_annotations.csv")

train_ds_adaptor = DatasetAdaptor(images_dir_path="./data/images/train", annotations_dataframe=train_df)

val_ds_adaptor = DatasetAdaptor(images_dir_path="./data/images/val", annotations_dataframe=val_df)

train_transforms = get_train_transforms(target_img_size=IMG_SIZE)
val_transforms = get_valid_transforms(target_img_size=IMG_SIZE)

dm = EfficientDetDataModule(
    train_dataset_adaptor=train_ds_adaptor,
    validation_dataset_adaptor=val_ds_adaptor,
    batch_size=4,
    num_workers=4,
    train_transforms=train_transforms, 
    valid_transforms=val_transforms
)

model = EfficientDetModel(
    num_classes=3, 
    img_size=IMG_SIZE,
    prediction_confidence_threshold=0.2,
    learning_rate=0.0002,
    wbf_iou_threshold=0.44,
    inference_transforms=get_valid_transforms(target_img_size=IMG_SIZE)
)


trainer = Trainer(
    accelerator="gpu",
    devices=-1,
    max_epochs=10,
    num_sanity_val_steps=1,
)

trainer.fit(model, datamodule=dm)