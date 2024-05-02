import operator
import os

import click
import wandb
from dotenv import load_dotenv

from example_mlops.utils import HydraRichLogger

load_dotenv()
logger = HydraRichLogger()


@click.group()
def cli():
    """Commands for model management."""
    pass


@click.command()
@click.option("--model-name", default="mnist_model", help="Name of the model to be registered.")
@click.option("--metric_name", default="accuracy", help="Name of the metric to choose the best model from.")
@click.option("--higher-is-better", default=True, help="Whether higher metric values are better.")
def stage_best_model_to_registry(model_name, metric_name, higher_is_better):
    """
    Stage the best model to the model registry.

    Args:
        model_name: Name of the model to be registered.
        metric_name: Name of the metric to choose the best model from.
        higher_is_better: Whether higher metric values are better.

    """
    api = wandb.Api(api_key=os.getenv("WANDB_API"))
    artifact_collection = api.artifact_collection(type_name="model", name=model_name)

    best_metric = float("-inf") if higher_is_better else float("inf")
    compare_op = operator.gt if higher_is_better else operator.lt
    best_artifact = None
    for artifact in list(artifact_collection.artifacts()):
        if metric_name in artifact.metadata and compare_op(artifact.metadata[metric_name], best_metric):
            best_metric = artifact.metadata[metric_name]
            best_artifact = artifact

    if best_artifact is None:
        logger.error("No model found in registry.")
        return

    logger.info(f"Best model found in registry: {best_artifact.name} with {metric_name}={best_metric}")
    best_artifact.link(
        target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{model_name}", aliases=["best", "staging"]
    )
    best_artifact.save()
    logger.info("Model staged to registry.")


@click.command()
@click.option("--artifact-path", default="", help="Path to the artifact to stage.")
@click.option("--experiment-path", default="", help="Name of the experiment to stage the artifact from.")
def stage_model(artifact_path: str, experiment_path: str) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        experiment_path: ID of the experiment to stage the artifact from.
            Should be of the format "entity/project/experiment_id".

    """
    if artifact_path == "" and experiment_path == "":
        logger.error("Please provide either artifact_path or experiment_path.")
        return

    api = wandb.Api(api_key=os.getenv("WANDB_API"))
    if artifact_path:
        _, _, artifact_name_version = artifact_path.split("/")
        artifact_name, _ = artifact_name_version.split(":")

        artifact = api.artifact(artifact_path)
        artifact.link(
            target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}", aliases=["best", "staging"]
        )
        artifact.save()
        logger.info("Model staged to registry.")
    else:
        _, _, experiment_id = experiment_path.split("/")
        run = api.run(experiment_id)
        for artifact in list(run.logged_artifacts()):
            if artifact.type == "model":
                artifact_name = artifact.name.split(":")[0]
                artifact.link(
                    target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}",
                    aliases=["best", "staging"],
                )
                artifact.save()
                logger.info("Model staged to registry.")


cli.add_command(stage_best_model_to_registry)
cli.add_command(stage_model)

if __name__ == "__main__":
    cli()
