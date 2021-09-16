import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel


def tsne_embedding_plot() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("model_checkpoint", type=str)
    args = parser.parse_args()
    print(args)

    train_set = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.to(device)

    print("Extract embeddings")
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            # Extract features from the backbone
            emb = model.backbone(x.to(device)).reshape(x.shape[0], -1)
            embeddings.append(emb)
            labels.append(y)

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()

    print("Running tsne")
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    for i in np.unique(labels):
        plt.scatter(embeddings_2d[labels == i, 0], embeddings_2d[labels == i, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/2d_tsne_embedding.png")


if __name__ == "__main__":
    tsne_embedding_plot()
