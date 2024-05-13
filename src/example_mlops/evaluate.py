import os

import hydra
import torch
import torchmetrics
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassROC,
)
from tqdm.rich import tqdm

import wandb
from example_mlops.model import MnistClassifier
from example_mlops.utils import HydraRichLogger, get_hydra_dir_and_job_name

load_dotenv()
logger = HydraRichLogger()


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="evaluate")
def evaluate_model(cfg: DictConfig):
    """Evaluate model on test dataset."""
    logger.info("Evaluating model on test dataset.")
    logdir = cfg.logdir or get_hydra_dir_and_job_name()[0]
    logger.info(f"Logging to {logdir}")
    os.mkdir(f"{logdir}/wandb")

    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    run = wandb.init(
        project=cfg.experiment_logger.project,
        entity=cfg.experiment_logger.entity,
        job_type=cfg.experiment_logger.job_type,
        dir=logdir,
        config=OmegaConf.to_container(cfg),
    )

    logger.info("Loading data.")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    data = {"test set": test_dataloader}

    if cfg.external_data is not None:
        logger.info("Loading external data.")
        external_dataset = hydra.utils.instantiate(cfg.external_data.config)
        data.update({cfg.external_data.name: DataLoader(external_dataset, batch_size=cfg.external_data.batch_size)})

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
        },
        compute_groups=False,
    )

    logger.info("Evaluating model.")
    collections, results = {}, {}
    for name, dataloader in data.items():
        m = metrics.clone()
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            x, y = batch
            y_hat = model(x)
            m.update(y_hat, y)
        results[name] = m.compute()
        collections[name] = m

    logger.info("Results")
    table_data = []
    for key in metrics.keys():
        if key in ["confusion_matrix", "roc_auc"]:
            continue
        values = [results[name][key].item() for name in data.keys()]
        table_data.append([key, *values])

    # Log results to wandb, both as a table and as individual metrics
    run.log({"results": wandb.Table(columns=["Metric", *data.keys()], data=table_data)})
    run.log(
        {
            f"{key}_{name}": value
            for name, result in results.items()
            for key, value in result.items()
            if (isinstance(value, torch.Tensor) and value.numel() == 1)
        }
    )

    table = tabulate(
        table_data,
        headers=["Metric", *data.keys()],
        tablefmt="fancy_grid",
        numalign="right",
        stralign="right",
        floatfmt=".4f",
    )
    print(table)

    for name, collection in collections.items():
        for key in ["confusion_matrix", "roc_auc"]:
            fig, _ = collection[key].plot()
            fig.savefig(f"{logdir}/{name}_{key}.png")
            wandb.log({f"{name}_{key}": wandb.Image(fig)})

    logger.info("Confusion matrix and ROC-AUC plots saved.")


if __name__ == "__main__":
    evaluate_model()
