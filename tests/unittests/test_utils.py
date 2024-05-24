import os

import pytest
import torch
from example_mlops.utils import HydraRichLogger, get_dtype_from_string


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_get_dtype_from_string(dtype):
    """Test that the data type is correctly converted from a string."""
    fetched_dtype = get_dtype_from_string(dtype)
    assert fetched_dtype == getattr(torch, dtype)
    assert isinstance(fetched_dtype, torch.dtype)


def test_logger():
    """Test that logger can be initialized and logs messages."""
    logger = HydraRichLogger(level=os.getenv("LOG_LEVEL", "INFO"))
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    logger.critical("This is a critical message")
    logger.exception("This is an exception message")
