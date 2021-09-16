# -*- coding: utf-8 -*-
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import torch
import wget
from torch import Tensor
from torch.utils.data import Dataset


class CorruptMnist(Dataset):
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()

        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder

        if self.out_folder:  # try loading from proprocessed
            try:
                self.load_preprocessed()
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                pass

        self.download_data()

        if self.train:
            content = []
            for i in range(8):
                content.append(np.load(f"{in_folder}/train_{i}.npz", allow_pickle=True))
            data = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(
                -1, 1, 28, 28
            )
            targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
        else:
            content = np.load(f"{in_folder}/test.npz", allow_pickle=True)
            data = torch.tensor(content["images"]).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content["labels"])

        self.data = data
        self.targets = targets

        if self.out_folder:
            self.save_preprocessed()

    def save_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        torch.save([self.data, self.targets], f"{self.out_folder}/{split}_processed.pt")

    def load_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        try:
            self.data, self.targets = torch.load(f"{self.out_folder}/{split}_processed.pt")
        except:
            raise ValueError("No preprocessed files found")

    def download_data(self) -> None:
        files = os.listdir(self.in_folder)
        if self.train:
            for file_idx in range(8):
                if file_idx < 5 and f"train_{file_idx}.npz" not in files:
                    wget.download(
                        f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz"
                    )
                    shutil.move(f"train_{file_idx}.npz", f"{self.in_folder}/train_{file_idx}.npz")
                if file_idx > 4 and f"train_{file_idx}.npz" not in files:
                    wget.download(
                        f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist_v2/train_{file_idx}.npz"
                    )
                    shutil.move(f"train_{file_idx}.npz", f"{self.in_folder}/train_{file_idx}.npz")
        else:
            if "test.npz" not in files:
                wget.download(
                    "https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz"
                )
                shutil.move("test.npz", f"{self.in_folder}/test.npz")

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx].float(), self.targets[idx]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train = CorruptMnist(train=True, in_folder=input_filepath, out_folder=output_filepath)
    train.save_preprocessed()

    test = CorruptMnist(train=False, in_folder=input_filepath, out_folder=output_filepath)
    test.save_preprocessed()

    print(train.data.shape)
    print(train.targets.shape)
    print(test.data.shape)
    print(test.targets.shape)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
