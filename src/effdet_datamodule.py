from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transforms import train_transforms, val_transforms
from datasets import EffDetDataset

class EffDetDataModule(LightningDataModule):
    """
    
    """
    def __init__(self) -> None:
        super().__init__()
    
    def train_dataset(self):
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()
    
    def val_dataset(self):
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return super().val_dataloader()
    