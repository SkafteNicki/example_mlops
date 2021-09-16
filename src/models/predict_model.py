import argparse

import numpy as np
import torch

from src.models.model import MyAwesomeModel


def predict() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("model_checkpoint", type=str)
    parser.add_argument("data_to_predict", type=str)
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(args.model_checkpoint))
    model = model.to(device)

    imgs = np.load(args.data_to_predict)
    imgs = torch.tensor(imgs, dtype=torch.float, device=device)

    log_probs = model(imgs)
    prediction = log_probs.argmax(dim=-1)
    probs = log_probs.softmax(dim=-1)

    print("Predictions")
    for i in range(imgs.shape[0]):
        print(
            f"Image {i+1} predicted to be class {prediction[i].item()} with probability {probs[i, prediction[i]].item()}"
        )


if __name__ == "__main__":
    predict()
