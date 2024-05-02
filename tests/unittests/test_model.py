import pytest
import torch
from example_mlops.model import MnistClassifier


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu"),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")),
    ],
)
@pytest.mark.parametrize("backbone", ["resnet18", "resnet34", "resnet50"])
def test_expected_output(device, backbone):
    """Test if model returns expected output shape."""
    model = MnistClassifier(backbone=backbone)
    assert model is not None

    x = torch.randn(5, 3, 28, 28)
    output = model(x).to(device)
    assert output.shape == (5, 10)


def test_wrong_model():
    """Test if ValueError is raised for wrong model name."""
    with pytest.raises(ValueError, match="Backbone not_a_model not available."):
        MnistClassifier(backbone="not_a_model")


def test_train_step():
    """Test training step."""
    model = MnistClassifier()
    batch = (torch.randn(5, 3, 28, 28), torch.randint(0, 10, (5,)))
    loss = model.training_step(batch)
    assert loss is not None
