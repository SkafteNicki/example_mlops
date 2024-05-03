import os

import pytest
import torch
from example_mlops.data import MnistDataModule

from unittests import _PATH_PROCESSED

if os.path.exists(_PATH_PROCESSED):
    data_exist = all(
        [
            x in os.listdir(_PATH_PROCESSED)
            for x in ["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]
        ]
    )
else:
    data_exist = False


@pytest.mark.skipif(not data_exist, reason="data not available")
class TestDatamodule:
    """Test datamodule."""

    @pytest.mark.parametrize("val_split", [0.0, 0.1])
    def test_data_shape(self, val_split):
        """Test if data is loaded correctly."""
        datamodule = MnistDataModule(val_split=val_split)
        datamodule.setup("fit")
        datamodule.setup("test")

        train_set = datamodule.train_dataset
        val_set = datamodule.val_dataset
        test_set = datamodule.test_dataset

        assert isinstance(train_set, torch.utils.data.Dataset)
        assert isinstance(val_set, torch.utils.data.Dataset)
        assert isinstance(test_set, torch.utils.data.Dataset)

        assert len(train_set) == int(25000 * (1 - val_split))
        assert len(val_set) == int(25000 * val_split)
        assert len(test_set) == 5000

        for dataset in [train_set, test_set] + ([val_set] if val_split else []):
            x, y = dataset[0]  # check a single datapoint
            assert x.shape == (3, 28, 28)
            assert y.numel() == 1

    @pytest.mark.parametrize("batch_size", [16, 32])
    def test_dataloaders(self, batch_size):
        """Test if dataloaders works as expected."""
        # drop last batch to make sure all batches are of same size
        datamodule = MnistDataModule(batch_size=batch_size, drop_last=True)
        datamodule.setup("fit")
        datamodule.setup("test")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        for loader in [train_loader, val_loader, test_loader]:
            for x, y in loader:
                assert x.shape == (batch_size, 3, 28, 28)
                assert y.shape == (batch_size,)
