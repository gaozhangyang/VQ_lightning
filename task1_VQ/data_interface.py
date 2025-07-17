from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, DistributedSampler
from src.interface.data_interface import DInterface_base
from src.dataset.build import build_dataset
from torchvision import transforms
from src.dataset.augmentation import random_crop_arr, center_crop_arr
from copy import deepcopy

class DInterface(DInterface_base):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def setup(self, stage=None):
        pass
    
    def data_setup(self):
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, self.hparams.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        args = deepcopy(self.hparams)
        self.train_set = build_dataset(args, transform=transform)
        args.data_path = args.val_data_path
        self.val_set = build_dataset(args, transform=transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True, prefetch_factor=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True, prefetch_factor=8)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True, prefetch_factor=8)

    