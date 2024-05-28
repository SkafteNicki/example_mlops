import os

import pytest
import pytorch_lightning as pl
import torch
from example_mlops.app import app
from example_mlops.model import MnistClassifier
from fastapi.testclient import TestClient

from unittests import _TEST_ROOT


@pytest.fixture(scope="module")
def model(tmpdir_factory):
    """Load the model for testing."""
    tmpdir = tmpdir_factory.mktemp("data")
    tempmodel = MnistClassifier()
    temp_checkpoint = os.path.join(tmpdir, "model.ckpt")
    state_dict = {  # dummy checkpoint
        "state_dict": tempmodel.state_dict(),
        "pytorch-lightning_version": pl.__version__,
    }
    torch.save(state_dict, temp_checkpoint)
    os.environ["MODEL_CHECKPOINT"] = temp_checkpoint


def test_read_root(model):
    """Test the root endpoint of the API."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the MNIST model inference API!"}


def test_health(model):
    """Test the health check endpoint of the API."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_modelstats(model):
    """Test the modelstats endpoint of the API."""
    with TestClient(app) as client:
        response = client.get("/modelstats")
        assert response.status_code == 200
        assert "model architecture" in response.json()


@pytest.mark.parametrize("image_path", ["img_1.jpg", "img_2.jpg", "img_3.jpg"])
def test_predict(image_path):
    """Test the prediction endpoint of the API."""
    with TestClient(app) as client:
        image_file = os.path.join(_TEST_ROOT, image_path)
        with open(image_file, "rb") as f:
            image_data = f.read()
        response = client.post("/predict", files={"image": image_data})
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert response.json()["prediction"] in list(range(10))
        assert "probabilities" in response.json()
