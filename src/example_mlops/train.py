import os
import shutil

import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

import wandb
from example_mlops.utils import HydraRichLogger, get_hydra_dir_and_job_name

dotenv.load_dotenv()
logger = HydraRichLogger(level=os.getenv("LOG_LEVEL", "INFO"))


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="train")
def train_model(cfg: DictConfig) -> None:
    """Train and evaluate the model."""
    logger.info("Starting training script")
    logdir = cfg.logdir or get_hydra_dir_and_job_name()[0]
    logger.info(f"Logging to {logdir}")
    os.mkdir(f"{logdir}/checkpoints")
    os.mkdir(f"{logdir}/wandb")

    # Instantiate model and datamodule
    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Instantiate logger and callbacks
    experiment_logger = hydra.utils.instantiate(cfg.experiment_logger, save_dir=logdir)
    experiment_logger.log_hyperparams(OmegaConf.to_container(cfg))

    early_stopping_callback = hydra.utils.instantiate(cfg.callbacks.early_stopping)
    checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=f"{logdir}/checkpoints")
    learning_rate_callback = hydra.utils.instantiate(cfg.callbacks.learning_rate_monitor)
    progress_bar_callback = hydra.utils.instantiate(cfg.callbacks.progress_bar)

    # Instantiate trainer
    trainer = pl.Trainer(
        default_root_dir=logdir,
        logger=experiment_logger,
        callbacks=[checkpoint_callback, early_stopping_callback, learning_rate_callback, progress_bar_callback],
        **cfg.trainer,
    )

    if cfg.train:
        logger.info("Starting training")
        trainer.fit(model, datamodule=datamodule)

    if cfg.evaluate:
        logger.info("Starting evaluation")
        results = trainer.test(model, datamodule=datamodule)

    if cfg.upload_model:
        logger.info("Saving model as artifact")
        best_model = checkpoint_callback.best_model_path
        shutil.copy(best_model, f"{logdir}/checkpoints/checkpoint.ckpt")
        artifact = wandb.Artifact(
            name="mnist_model",
            type="model",
            metadata={k.lstrip("test_"): round(v, 3) for k, v in results[0].items()},  # remove test_ prefix and round
        )
        artifact.add_file(f"{logdir}/checkpoints/checkpoint.ckpt")
        wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    train_model()
