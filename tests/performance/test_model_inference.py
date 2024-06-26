import os

import pytest
import torch
from dotenv import load_dotenv
from example_mlops.model import load_from_checkpoint

load_dotenv()


@pytest.fixture(scope="module")
def model():
    """Fixture for loading the model."""
    model_checkpoint = os.getenv("MODEL_TO_TEST", None)
    if model_checkpoint is None:
        pytest.skip("No model checkpoint found.")
    model = load_from_checkpoint(model_checkpoint)
    return model


def test_single_image_inference(benchmark, model):
    """Test the inference of a single image."""
    x = torch.randn(1, 3, 28, 28)

    probs, preds = benchmark(model.inference, x=x)
    assert probs.shape == (1, 10)
    assert preds.shape == (1,)
    assert 0 <= preds.item() <= 9
    assert torch.allclose(probs.sum().cpu(), torch.ones(1))

    mean = benchmark.stats.stats.mean
    stddev = benchmark.stats.stats.stddev
    maximum = benchmark.stats.stats.max

    assert mean < 0.2, f"Mean time is {mean}, which is too high"
    assert stddev < 0.2, f"Standard deviation is {stddev}, which is too high"
    assert maximum < 0.2, f"Max time is {maximum}, which is too high"


@pytest.mark.parametrize("batch_size", [16, 32, 64, 128, 256, 512])
def test_batch_inference(benchmark, model, batch_size):
    """Test the inference of a batch of images."""
    x = torch.randn(batch_size, 3, 28, 28)
    probs, preds = benchmark(model.inference, x=x)
    assert probs.shape == (batch_size, 10)
    assert preds.shape == (batch_size,)
    assert (0 <= preds).all() and (preds <= 9).all()
    assert torch.allclose(probs.sum(dim=1).cpu(), torch.ones(batch_size))

    mean = benchmark.stats.stats.mean
    stddev = benchmark.stats.stats.stddev
    maximum = benchmark.stats.stats.max

    scale_factor = max(int(batch_size / 50), 1)  # for larger batch sizes, we allow more time

    assert mean < 0.2 * scale_factor, f"Mean time is {mean}, which is too high"
    assert stddev < 0.2 * scale_factor, f"Standard deviation is {stddev}, which is too high"
    assert maximum < 0.2 * scale_factor, f"Max time is {maximum}, which is too high"
