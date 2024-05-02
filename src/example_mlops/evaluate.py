import os

import hydra
import rich
import torchmetrics
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassROC,
)

from example_mlops.data import MnistDataModule
from example_mlops.model import MnistClassifier
from example_mlops.utils import HydraRichLogger

load_dotenv()
logger = HydraRichLogger()


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="evaluate")
def evaluate_model(cfg: DictConfig):
    """Evaluate model on test dataset."""
    logger.info("Evaluating model on test dataset.")
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger.info("Loading data.")
    datamodule = MnistDataModule(**cfg.datamodule)
    datamodule.setup("test")

    logger.info("Loading model.")

    if os.path.exists(cfg.model_checkpoint):  # local path to model checkpoint
        model = MnistClassifier.load_from_checkpoint(cfg.model_checkpoint, map_location="cpu")
    else:  # assume it is a wandb artifact path
        api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
        artifact = api.artifact(cfg.model_checkpoint)
        artifact.download(root=logdir)
        model = MnistClassifier.load_from_checkpoint(f"{logdir}/best.ckpt", map_location="cpu")

    # Defining metrics
    base_classification_metrics = lambda average: torchmetrics.MetricCollection(
        {
            f"{average}_accuracy": MulticlassAccuracy(num_classes=10, average=average),
            f"{average}_precision": MulticlassPrecision(num_classes=10, average=average),
            f"{average}_recall": MulticlassRecall(num_classes=10, average=average),
            f"{average}_f1": MulticlassF1Score(num_classes=10, average=average),
        }
    )

    metrics = torchmetrics.MetricCollection(
        {
            **base_classification_metrics("macro"),
            **base_classification_metrics("micro"),
            **base_classification_metrics("weighted"),
            "confusion_matrix": MulticlassConfusionMatrix(num_classes=10, normalize="true"),
            "auroc": MulticlassAUROC(num_classes=10),
            "roc_auc": MulticlassROC(num_classes=10),
        }
    )

    logger.info("Evaluating model.")
    for batch in datamodule.test_dataloader():
        x, y = batch
        y_hat = model(x)
        metrics.update(y_hat, y)
    results = metrics.compute()

    logger.info("Results")
    table = rich.table.Table(title="Evaluation Results")
    table.add_column("Metric")
    table.add_column("Value")

    for key, value in results.items():
        if key in ["confusion_matrix", "roc_auc"]:
            continue
        table.add_row(key, str(round(value.item(), 4)))

    rich.console.Console().print(table)


if __name__ == "__main__":
    evaluate_model()
