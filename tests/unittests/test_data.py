import os

import pytest
import torch
from example_mlops.data import MnistDataModule, MnistDataset

from unittests import _PATH_DATA


@pytest.mark.skipif(not os.listdir(_PATH_DATA), reason="data not available")
class TestDataset:
    """Test dataset."""

    def test_dataset(self):
        """Test if dataset is loaded correctly without transforms."""
        train_dataset = MnistDataset(data_folder=_PATH_DATA, train=True, img_transform=None, target_transform=None)
        test_dataset = MnistDataset(data_folder=_PATH_DATA, train=False)

        assert len(train_dataset) == 50000
        assert len(test_dataset) == 5000

        x, y = train_dataset[0]  # check a single datapoint
        assert x.shape == (28, 28)
        assert y.numel() == 1

    def test_dataset_with_transform(self):
        """Test if dataset is loaded correctly with default transforms."""
        train_dataset = MnistDataset(data_folder=_PATH_DATA, train=True)

        x, y = train_dataset[0]  # check a single datapoint
        assert x.shape == (3, 28, 28)
        assert y.numel() == 1
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64


@pytest.mark.skipif(not os.listdir(_PATH_DATA), reason="data not available")
class TestDatamodule:
    """Test datamodule."""

    @pytest.mark.parametrize("val_split", [0.0, 0.1])
    def test_val_splitting(self, val_split):
        """Test if dataset is split correctly."""
        datamodule = MnistDataModule(val_split=val_split)
        datamodule.setup("fit")
        datamodule.setup("test")

        train_set = datamodule.train_dataset
        val_set = datamodule.val_dataset
        test_set = datamodule.test_dataset

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(val_set, torch.utils.data.Dataset)
        assert isinstance(test_set, torch.utils.data.Dataset)

        assert len(train_set) == int(50000 * (1 - val_split))
        assert len(val_set) == int(50000 * val_split)
        assert len(test_set) == 5000

        for dataset in [train_set, test_set] + ([val_set] if val_split else []):
            x, y = dataset[0]  # check a single datapoint
            assert x.shape == (3, 28, 28)
            assert y.numel() == 1

    @pytest.mark.parametrize("batch_size", [16, 32])
    def test_dataloaders(self, batch_size):
        """Test if dataloaders works as expected."""
        # drop last batch to make sure all batches are of same size
        datamodule = MnistDataModule(batch_size=batch_size)
        datamodule.setup("fit")
        datamodule.setup("test")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        for loader in [train_loader, val_loader, test_loader]:
            for x, y in loader:
                assert x.shape == (batch_size, 3, 28, 28)
                assert y.shape == (batch_size,)
                break  # check only first batch
