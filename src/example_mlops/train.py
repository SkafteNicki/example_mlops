import os
import shutil

import dotenv
import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf

from example_mlops.data import MnistDataModule
from example_mlops.model import MnistClassifier
from example_mlops.utils import HydraRichLogger

dotenv.load_dotenv()
logger = HydraRichLogger()


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="train")
def train_model(cfg: DictConfig) -> None:
    """Train and evaluate the model."""
    logger.info("Starting training script")
    wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.mkdir(f"{logdir}/checkpoints")
    os.mkdir(f"{logdir}/wandb")

    # Instantiate model and datamodule
    model = MnistClassifier(**cfg.model)
    datamodule = MnistDataModule(**cfg.datamodule)

    # Instantiate logger and callbacks
    experiment_logger = hydra.utils.instantiate(cfg.experiment_logger, save_dir=logdir)
    experiment_logger.log_hyperparams(OmegaConf.to_container(cfg))

    early_stopping_callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=f"{logdir}/checkpoints")
    learning_rate_callback = hydra.utils.instantiate(cfg.callbacks.learning_rate_monitor)
    progress_bar_callback = hydra.utils.instantiate(cfg.callbacks.progress_bar)

    # Instantiate trainer
    trainer = pl.Trainer(
        default_root_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        logger=experiment_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, learning_rate_callback, progress_bar_callback],
        **cfg.trainer,
    )

    logger.info("Starting training")
    trainer.fit(model, datamodule=datamodule)

    logger.info("Starting evaluation")
    results = trainer.test(model, datamodule=datamodule)

    logger.info("Saving model as artifact")
    best_model = checkpoint_callback.best_model_path
    shutil.copy(best_model, f"{logdir}/checkpoints/best.ckpt")
    artifact = wandb.Artifact(
        name="mnist_model",
        type="model",
        metadata={k.lstrip("test_"): round(v, 3) for k, v in results[0].items()},  # remove test_ prefix and round
    )
    artifact.add_file(f"{logdir}/checkpoints/best.ckpt")
    wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    train_model()
