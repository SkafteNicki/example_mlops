"""Data module for MNIST dataset."""

import os
from dataclasses import dataclass

import torch
import torchvision.transforms.v2 as transforms
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from example_mlops.utils import HydraRichLogger

logger = HydraRichLogger(level=os.getenv("LOG_LEVEL", "INFO"))

unsqueeze_transform = transforms.Lambda(lambda x: x.unsqueeze(0))

default_img_transform = transforms.Compose(
    [
        unsqueeze_transform,
        transforms.RGB(),
        transforms.ToDtype(torch.float32),
    ]
)
default_target_transform = transforms.ToDtype(torch.int64)
train_img_transform = transforms.Compose(
    [
        unsqueeze_transform,
        transforms.RGB(),
        transforms.ToDtype(torch.float32),
        transforms.RandomHorizontalFlip(),
    ]
)


class MnistDataset(Dataset):
    """MNIST dataset for PyTorch.

    Args:
        data_folder: Path to the data folder.
        train: Whether to load training or test data.
        img_transform: Image transformation to apply.
        target_transform: Target transformation to apply.
    """

    def __init__(
        self,
        data_folder: str = "data",
        train: bool = True,
        img_transform: transforms.Transform | None = default_img_transform,
        target_transform: transforms.Transform | None = default_target_transform,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.load_data()

    def load_data(self) -> None:
        """Load images and targets from disk."""
        images, target = [], []
        if self.train:
            nb_files = len([f for f in os.listdir(self.data_folder) if f.startswith("train_images")])
            for i in range(nb_files):
                images.append(torch.load(f"{self.data_folder}/train_images_{i}.pt"))
                target.append(torch.load(f"{self.data_folder}/train_target_{i}.pt"))
        else:
            images.append(torch.load(f"{self.data_folder}/test_images.pt"))
            target.append(torch.load(f"{self.data_folder}/test_target.pt"))
        self.images = torch.cat(images, 0)
        self.target = torch.cat(target, 0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor."""
        img, target = self.images[idx], self.target[idx]
        if self.img_transform:
            img = self.img_transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        """Return the number of images in the dataset."""
        return self.images.shape[0]


@dataclass
class MnistDataModule(LightningDataModule):
    """Data module for MNIST dataset."""

    data_dir: str = "data"
    val_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    train_img_transform: transforms.Transform | None = train_img_transform
    train_target_transform: transforms.Transform | None = default_target_transform
    test_img_transform: transforms.Transform | None = default_img_transform
    test_target_transform: transforms.Transform | None = default_target_transform

    def __post_init__(self):
        """Initialize data module."""
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        """Setup data structures."""
        if stage == "fit":
            train_dataset = MnistDataset(
                self.data_dir,
                train=True,
                img_transform=self.train_img_transform,
                target_transform=self.train_target_transform,
            )
            n_train = len(train_dataset)
            n_val = int(n_train * self.val_split)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                train_dataset, [n_train - n_val, n_val]
            )

        if stage == "test":
            self.test_dataset = MnistDataset(
                self.data_dir,
                train=False,
                img_transform=self.test_img_transform,
                target_transform=self.test_target_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
