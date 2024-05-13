import os
import tempfile

import pytorch_lightning as pl
import torch
from example_mlops.evaluate import evaluate_model
from example_mlops.model import MnistClassifier
from hydra import compose, initialize

import wandb


class TestEvaluate:
    """Test evaluation script."""

    def setup_method(self, test_method):
        """Setup wandb for testing purpose e.g. disable logging etc."""
        self.tempdir = tempfile.mkdtemp()
        wandb.setup(wandb.Settings(mode="disabled", program=__name__, program_relpath=__name__, disable_code=True))
        wandb.init(dir=self.tempdir)

    def test_evaluate(self):
        """Test evaluation script can complete."""
        model = MnistClassifier()
        temp_checkpoint = os.path.join(self.tempdir, "model.ckpt")
        state_dict = {  # dummy checkpoint
            "state_dict": model.state_dict(),
            "pytorch-lightning_version": pl.__version__,
        }
        torch.save(state_dict, temp_checkpoint)

        with initialize(version_base=None, config_path="../../configs"):
            cfg = compose(
                config_name="evaluate",
                overrides=[
                    f"model_checkpoint='{temp_checkpoint}'",
                    f"logdir='{self.tempdir}'",
                ],
            )
            evaluate_model(cfg)
