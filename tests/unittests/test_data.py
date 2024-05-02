import os

import pytest
import torch
from example_mlops.data import MnistDataModule

from unittests import _PATH_PROCESSED

data_exist = all(
    [
        x in os.listdir(_PATH_PROCESSED)
        for x in ["train_images.pt", "train_labels.pt", "test_images.pt", "test_labels.pt"]
    ]
)

# def init_hydra():
#     with initialize(version_base=None, config_path="../hydra_app/conf"):
#         # config is relative to a module
#         cfg = compose(config_name="config", overrides=["app.user=test_user"])
#         assert cfg == {
#             "app": {"user": "test_user", "num1": 10, "num2": 20},
#             "db": {"host": "localhost", "port": 3306},
#         }


@pytest.mark.skipif(data_exist, reason="data not available")
@pytest.mark.parametrize("val_split", [0.0, 0.1])
def test_data_shape(val_split):
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
