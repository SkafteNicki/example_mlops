import os
from contextlib import asynccontextmanager

import wandb
from fastapi import FastAPI
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms.v2.functional import pil_to_tensor

from example_mlops.data import default_img_transform
from example_mlops.model import MnistClassifier
from example_mlops.utils import HydraRichLogger

logger = HydraRichLogger()

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup and clean up at shutdown."""
    cfg = OmegaConf.load("configs/app.yaml")
    logger.info(f"Config: {cfg}")
    if os.path.exists(cfg.model.checkpoint):
        logger.info(f"Model checkpoint found at {cfg.model.checkpoint}")
        model_checkpoint = cfg.model.checkpoint
    else:  # assume wandb artifact path
        logger.info("Downloading model checkpoint from WandB.")
        api = wandb.Api(api_key=os.getenv("WANDB_API"))
        artifact = api.artifact(cfg.model.checkpoint)
        path = artifact.download("models")
        model_checkpoint = f"{path}/best.ckpt"
    model = MnistClassifier.load_from_checkpoint(model_checkpoint, map_location=cfg.model.map_location)
    model.eval()
    models["mnist"] = model
    logger.info("Model loaded.")

    yield  # Wait for the application to finish

    logger.info("Cleaning up...")
    del model


app = FastAPI(lifespan=lifespan)


class ImageRequest(BaseModel):
    """Request body schema for the image prediction route."""

    image: str  # Path to the image file


@app.get("/")
def read_root():
    """Root endpoint of the API."""
    return {"message": "Welcome to the MNIST model inference API!"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(image_request: ImageRequest):
    """Predict the label of a given image."""
    image_data = pil_to_tensor(Image.open(image_request.image))
    logger.info("Image loaded.")
    input_tensor = default_img_transform(image_data)
    probs, preds = models["mnist"].inference(input_tensor)

    # Return the predicted label
    return {"prediction": int(preds[0]), "probabilities": probs[0].tolist()}
