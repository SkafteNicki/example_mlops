import os
import tempfile

import pytest
from example_mlops.train import train_model
from hydra import compose, initialize

import wandb
from unittests import _PATH_DATA


class TestTraining:
    """Test training script."""

    def setup_method(self, test_method):
        """Setup wandb for testing purpose e.g. disable logging etc."""
        self.tempdir = tempfile.mkdtemp()
        wandb.setup(wandb.Settings(mode="disabled", program=__name__, program_relpath=__name__, disable_code=True))
        wandb.init(dir=self.tempdir)

    @pytest.mark.skipif(not os.listdir(_PATH_DATA), reason="data not available")
    def test_training(self):
        """Test training script with single training and validation batch."""
        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "trainer.accelerator='cpu'",
                    "+trainer.fast_dev_run=True",
                    "upload_model=False",
                    f"logdir='{self.tempdir}'",
                ],
            )
            train_model(cfg)
