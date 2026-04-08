"""Configuration for document ingestion."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class LoaderType(str, Enum):
    """Supported document loader types."""

    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    DATABASE = "database"
    WEB = "web"
    DIRECTORY = "directory"


@dataclass
class IngestionConfig:
    """Configuration for document ingestion pipeline."""

    loader_type: LoaderType = LoaderType.TEXT
    extract_metadata: bool = True
    preserve_formatting: bool = True
    encoding: str = "utf-8"
    max_file_size_mb: int = 100
    timeout_seconds: int = 30
    retry_attempts: int = 3
    batch_size: int = 10
    num_workers: int = 4
    cleanup_text: bool = True
    remove_duplicates: bool = True

    def __post_init__(self):
        """Validate ingestion configuration."""
        if self.max_file_size_mb <= 0:
            raise ValueError(
                f"max_file_size_mb must be positive, got {self.max_file_size_mb}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        if self.retry_attempts < 0:
            raise ValueError(
                f"retry_attempts must be non-negative, got {self.retry_attempts}"
            )
