"""Test utilities and fixtures."""

import pytest
import logging
from typing import Generator, Any

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_texts() -> list:
    """Fixture providing sample texts."""
    return [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Natural language processing is important.",
    ]


@pytest.fixture
def sample_config() -> dict:
    """Fixture providing sample configuration."""
    return {"model": "gpt2", "batch_size": 16, "learning_rate": 5e-4, "epochs": 3}


@pytest.fixture
def setup_logging() -> Generator:
    """Fixture for setting up logging."""
    logging.basicConfig(level=logging.INFO)
    yield
    logging.shutdown()


class TestBase:
    """Base test class with common setup."""

    @classmethod
    def setup_class(cls):
        """Setup test class."""
        logger.info(f"Setting up {cls.__name__}")

    @classmethod
    def teardown_class(cls):
        """Teardown test class."""
        logger.info(f"Tearing down {cls.__name__}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
