"""Visualize model predictions."""

import click
import os
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from example_mlops.model import MnistClassifier
from example_mlops.data import MnistDataset


@click.command()
@click.argument("model_checkpoint")
@click.option("--datadir", default="data/", help="Path to processed data directory")
@click.option("--figure_dir", default="reports/figures", help="Path to save figures")
@click.option("--figure_name", default="embeddings.png", help="Name of the figure")
def visualize(model_checkpoint: str, datadir: str, figure_dir: str, figure_name: str) -> None:
    """Visualize model predictions."""
    model = MnistClassifier.load_from_checkpoint(model_checkpoint, map_location="cpu")
    model.eval()
    model.fc = torch.nn.Identity()

    test_dataset = MnistDataset(datadir, train=False)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(f"{figure_dir}/{figure_name}")


if __name__ == "__main__":
    visualize()
