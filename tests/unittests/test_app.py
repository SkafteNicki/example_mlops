import os

import pytest
from example_mlops.app import app
from fastapi.testclient import TestClient

from unittests import _TEST_ROOT


def test_read_root():
    """Test the root endpoint of the API."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the MNIST model inference API!"}


def test_health():
    """Test the health check endpoint of the API."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize("image_path", ["img_1.jpg", "img_2.jpg", "img_3.jpg"])
def test_predict(image_path):
    """Test the prediction endpoint of the API."""
    with TestClient(app) as client:
        image_request = {"image": os.path.join(_TEST_ROOT, image_path)}
        response = client.post("/predict", json=image_request)
        assert response.status_code == 200
        assert "prediction" in response.json()
