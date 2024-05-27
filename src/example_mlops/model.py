"""Model definition."""

import os

import onnxruntime as ort
import torch
import torchmetrics
import torchmetrics.classification
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models

import wandb
from example_mlops.utils import HydraRichLogger

logger = HydraRichLogger(level=os.getenv("LOG_LEVEL", "INFO"))


class MnistClassifier(LightningModule):
    """My awesome model."""

    def __init__(
        self,
        backbone: str = "resnet18",
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
    ) -> None:
        """Initialize model and metrics."""
        super().__init__()
        self.save_hyperparameters(logger=False)

        if backbone not in models.list_models():
            raise ValueError(f"Backbone {backbone} not available.")
        self.backbone = models.get_model(backbone, weights=None)
        self.fc = nn.Linear(1000, 10)
        self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.optimizer_class = torch.optim.Adam if optimizer == "adam" else torch.optim.SGD

        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=10, average="micro"),
                "precision": torchmetrics.classification.MulticlassPrecision(num_classes=10, average="micro"),
                "recall": torchmetrics.classification.MulticlassRecall(num_classes=10, average="micro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.input_sample = torch.randn(1, 3, 28, 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = nn.functional.leaky_relu(self.backbone(x))
        return self.fc(x)

    @torch.inference_mode()
    def inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform inference."""
        probs = nn.functional.softmax(self(x), dim=1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds

    def _shared_step(self, batch):
        """Shared step for training, validation, and test steps."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds

    def training_step(self, batch) -> torch.Tensor:
        """Training step."""
        loss, preds = self._shared_step(batch)
        batch_metrics = self.train_metrics(preds, batch[1])
        self.log_dict(batch_metrics)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch) -> None:
        """Validation step."""
        loss, preds = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True)
        self.val_metrics.update(preds, batch[1])

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at the end of the epoch."""
        epoch_metrics = self.val_metrics.compute()
        self.log_dict(epoch_metrics, prog_bar=True)

    def test_step(self, batch) -> None:
        """Test step."""
        loss, preds = self._shared_step(batch)
        self.log("test_loss", loss)
        self.test_metrics.update(preds, batch[1])

    def on_test_epoch_end(self) -> None:
        """Log test metrics at the end of the epoch."""
        epoch_metrics = self.test_metrics.compute()
        self.log_dict(epoch_metrics)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]


class MnistClassifierONNX(object):
    """Basic wrapper class for ONNX versions of the above model."""

    ort_session = None

    @classmethod
    def load_from_checkpoint(cls, path: str) -> "MnistClassifierONNX":
        """Load the ONNX model from a checkpoint."""
        ort_session = ort.InferenceSession(path)
        model_class = cls()
        model_class.ort_session = ort_session
        return model_class

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self.ort_session.run(None, {"image": x.numpy()})
        return torch.tensor(output[0])

    def inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform inference."""
        output = self.ort_session.run(None, {"image": x.numpy()})
        probs = nn.functional.softmax(torch.tensor(output[0]), 1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds


def load_from_checkpoint(model_checkpoint: str, logdir: str = "models", return_path: bool = False):
    """Load the model from a checkpoint.

    Accounts for model path either being local or a wandb artifact path.
    Also accounts if the model is ONNX or PyTorch.

    Args:
        model_checkpoint (str): Path to the model checkpoint.
        logdir (str): Directory to save the model to.
        return_path (bool): Whether to return the path of the model.
    """
    if os.path.exists(model_checkpoint):
        if model_checkpoint.endswith(".onnx"):
            model = MnistClassifierONNX.load_from_checkpoint(model_checkpoint)
        else:
            model = MnistClassifier.load_from_checkpoint(model_checkpoint, map_location="cpu").eval()
        path = model_checkpoint
    else:
        logger.debug("Loading model from wandb artifact...")
        logger.debug(f"WANDB_API_KEY: {os.getenv('WANDB_API_KEY')}")
        logger.debug(f"WANDB_ENTITY: {os.getenv('WANDB_ENTITY')}")
        logger.debug(f"WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")
        api = wandb.Api(
            api_key=os.getenv("WANDB_API_KEY"),
            overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
        )
        artifact = api.artifact(model_checkpoint)
        artifact.download(root=logdir)
        file_name = artifact.files()[0].name
        if file_name.endswith(".onnx"):
            model = MnistClassifierONNX.load_from_checkpoint(f"{logdir}/{file_name}")
        else:
            model = MnistClassifier.load_from_checkpoint(f"{logdir}/{file_name}", map_location="cpu").eval()
        path = f"{logdir}/{file_name}"
    if return_path:
        return model, path
    return model
