import os
from dataclasses import dataclass

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from example_mlops.utils import HydraRichLogger

logger = HydraRichLogger()


def normalize_img(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


@dataclass
class MnistDataModule(LightningDataModule):
    """Data module for MNIST dataset."""

    data_dir: str
    val_split: float
    batch_size: int
    num_workers: int
    pin_memory: bool

    def __post_init__(self):
        """Initialize data module."""
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        """Download data."""
        files = os.listdir(self.data_dir)
        if stage == "fit":
            if "train_images.pt" not in files or "train_target.pt" not in files:
                raise FileNotFoundError("Train data not found. Please download the data first.")

            train_imgs = torch.load(f"{self.data_dir}/train_images.pt")
            train_target = torch.load(f"{self.data_dir}/train_target.pt")
            train_dataset = torch.utils.data.TensorDataset(train_imgs, train_target)
            n_train = len(train_dataset)
            n_val = int(n_train * self.val_split)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                train_dataset, [n_train - n_val, n_val]
            )

        if stage == "test":
            if "test_images.pt" not in files or "test_target.pt" not in files:
                raise FileNotFoundError("Test data not found. Please download the data first.")

            test_imgs = torch.load(f"{self.data_dir}/test_images.pt")
            test_target = torch.load(f"{self.data_dir}/test_target.pt")

            self.test_dataset = torch.utils.data.TensorDataset(test_imgs, test_target)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return train dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Return test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="data")
def preprocess_data(cfg: DictConfig) -> None:
    """Process raw data and save it to processed directory."""
    raw_dir = cfg.raw_dir
    processed_dir = cfg.processed_dir
    num_train_files = cfg.num_train_files
    normalize = cfg.normalize
    convert_to_rgb = cfg.convert_to_rgb

    train_images, train_target = [], []
    for i in range(num_train_files):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    if normalize:
        logger.info("Normalizing images.")
        train_images = normalize_img(train_images)
        test_images = normalize_img(test_images)

    if convert_to_rgb:
        logger.info("Converting images to RGB.")
        train_images = train_images.repeat(1, 3, 1, 1)
        test_images = test_images.repeat(1, 3, 1, 1)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")
    logger.info("Data preprocessing completed.")


if __name__ == "__main__":
    preprocess_data()
