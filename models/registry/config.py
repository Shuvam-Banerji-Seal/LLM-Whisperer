"""Configuration and queries for model registry."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegistryBackend(Enum):
    """Supported registry backends."""

    MEMORY = "memory"
    JSON_FILE = "json_file"
    DATABASE = "database"
    REMOTE = "remote"


@dataclass
class RegistryConfig:
    """Configuration for model registry.

    Defines storage backend and registry behavior.
    """

    backend: RegistryBackend
    storage_path: Optional[Path] = None
    database_url: Optional[str] = None
    remote_url: Optional[str] = None

    # Cache settings
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Search settings
    enable_fuzzy_search: bool = True
    search_batch_size: int = 100

    # Custom settings
    auto_save: bool = True
    save_interval_seconds: int = 300

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistryQuery:
    """Query for searching models in registry.

    Allows filtering models by various criteria.
    """

    name: Optional[str] = None
    model_type: Optional[str] = None
    framework: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    organization: Optional[str] = None
    tags: Optional[List[str]] = None
    min_parameters: Optional[int] = None
    max_parameters: Optional[int] = None
    license: Optional[str] = None

    # Metadata filters
    custom_filters: Dict[str, Any] = field(default_factory=dict)

    # Pagination
    limit: int = 100
    offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary.

        Returns:
            Dictionary representation of the query
        """
        return {
            "name": self.name,
            "model_type": self.model_type,
            "framework": self.framework,
            "version": self.version,
            "author": self.author,
            "organization": self.organization,
            "tags": self.tags,
            "min_parameters": self.min_parameters,
            "max_parameters": self.max_parameters,
            "license": self.license,
            "custom_filters": self.custom_filters,
            "limit": self.limit,
            "offset": self.offset,
        }
