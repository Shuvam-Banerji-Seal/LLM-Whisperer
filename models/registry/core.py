"""Core model registry functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RegistryConfig, RegistryQuery, RegistryBackend

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Metadata for a registered model.

    Stores comprehensive information about a model in the registry.
    """

    def __init__(
        self,
        model_id: str,
        name: str,
        version: str,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        author: Optional[str] = None,
        organization: Optional[str] = None,
        license: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        num_parameters: Optional[int] = None,
        source_url: Optional[str] = None,
    ):
        """Initialize model metadata.

        Args:
            model_id: Unique model identifier
            name: Model name
            version: Model version
            model_type: Type of model
            framework: Framework used
            author: Model author
            organization: Organization
            license: Model license
            description: Model description
            tags: Model tags
            num_parameters: Number of parameters
            source_url: Source URL
        """
        self.model_id = model_id
        self.name = name
        self.version = version
        self.model_type = model_type
        self.framework = framework
        self.author = author
        self.organization = organization
        self.license = license
        self.description = description
        self.tags = tags or []
        self.num_parameters = num_parameters
        self.source_url = source_url

        # Metadata timestamps
        self.registered_at = datetime.utcnow().isoformat()
        self.updated_at = self.registered_at
        self.accessed_at = self.registered_at

        # Additional metadata
        self.download_count = 0
        self.rating = 0.0
        self.custom_metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "framework": self.framework,
            "author": self.author,
            "organization": self.organization,
            "license": self.license,
            "description": self.description,
            "tags": self.tags,
            "num_parameters": self.num_parameters,
            "source_url": self.source_url,
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "download_count": self.download_count,
            "rating": self.rating,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ModelMetadata instance
        """
        metadata = cls(
            model_id=data["model_id"],
            name=data["name"],
            version=data["version"],
            model_type=data.get("model_type"),
            framework=data.get("framework"),
            author=data.get("author"),
            organization=data.get("organization"),
            license=data.get("license"),
            description=data.get("description"),
            tags=data.get("tags", []),
            num_parameters=data.get("num_parameters"),
            source_url=data.get("source_url"),
        )

        # Restore timestamps if available
        if "registered_at" in data:
            metadata.registered_at = data["registered_at"]
        if "updated_at" in data:
            metadata.updated_at = data["updated_at"]
        if "accessed_at" in data:
            metadata.accessed_at = data["accessed_at"]

        metadata.download_count = data.get("download_count", 0)
        metadata.rating = data.get("rating", 0.0)
        metadata.custom_metadata = data.get("custom_metadata", {})

        return metadata


class ModelRegistry:
    """Central registry for managing models.

    Provides model discovery, registration, search, and metadata management.
    """

    def __init__(self, config: RegistryConfig):
        """Initialize model registry.

        Args:
            config: Registry configuration
        """
        self.config = config
        self._models: Dict[str, ModelMetadata] = {}
        self._cache: Dict[str, Any] = {}
        self._search_index: Dict[str, List[str]] = {}
        logger.info(f"Initialized ModelRegistry with backend: {config.backend.value}")

    def register_model(
        self,
        model_id: str,
        name: str,
        version: str,
        **kwargs,
    ) -> ModelMetadata:
        """Register a model.

        Args:
            model_id: Unique model identifier
            name: Model name
            version: Model version
            **kwargs: Additional metadata

        Returns:
            ModelMetadata instance

        Raises:
            ValueError: If model already registered
        """
        if model_id in self._models:
            raise ValueError(f"Model {model_id} already registered")

        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            **kwargs,
        )

        self._models[model_id] = metadata
        self._update_search_index(model_id, metadata)

        logger.info(f"Registered model: {model_id}")

        return metadata

    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model.

        Args:
            model_id: Model identifier

        Returns:
            True if unregistered, False if not found
        """
        if model_id in self._models:
            del self._models[model_id]
            self._clear_cache(model_id)
            logger.info(f"Unregistered model: {model_id}")
            return True
        return False

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetadata or None if not found
        """
        metadata = self._models.get(model_id)
        if metadata:
            metadata.accessed_at = datetime.utcnow().isoformat()
        return metadata

    def search(self, query: RegistryQuery) -> List[ModelMetadata]:
        """Search for models.

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        results = []

        for metadata in self._models.values():
            if self._matches_query(metadata, query):
                results.append(metadata)

        # Apply pagination
        return results[query.offset : query.offset + query.limit]

    def _matches_query(self, metadata: ModelMetadata, query: RegistryQuery) -> bool:
        """Check if metadata matches query.

        Args:
            metadata: Model metadata
            query: Search query

        Returns:
            True if matches
        """
        # Name filter
        if query.name and query.name.lower() not in metadata.name.lower():
            return False

        # Type filter
        if query.model_type and metadata.model_type != query.model_type:
            return False

        # Framework filter
        if query.framework and metadata.framework != query.framework:
            return False

        # Author filter
        if query.author and metadata.author != query.author:
            return False

        # Organization filter
        if query.organization and metadata.organization != query.organization:
            return False

        # License filter
        if query.license and metadata.license != query.license:
            return False

        # Tags filter
        if query.tags:
            if not any(tag in metadata.tags for tag in query.tags):
                return False

        # Parameters filter
        if metadata.num_parameters:
            if query.min_parameters and metadata.num_parameters < query.min_parameters:
                return False
            if query.max_parameters and metadata.num_parameters > query.max_parameters:
                return False

        return True

    def _update_search_index(self, model_id: str, metadata: ModelMetadata) -> None:
        """Update search index.

        Args:
            model_id: Model identifier
            metadata: Model metadata
        """
        # Index by various fields
        for tag in metadata.tags:
            if tag not in self._search_index:
                self._search_index[tag] = []
            self._search_index[tag].append(model_id)

    def _clear_cache(self, model_id: str) -> None:
        """Clear cache for a model.

        Args:
            model_id: Model identifier
        """
        if model_id in self._cache:
            del self._cache[model_id]

    def list_models(self) -> List[str]:
        """List all registered model IDs.

        Returns:
            List of model identifiers
        """
        return list(self._models.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary
        """
        total_models = len(self._models)
        total_parameters = sum(m.num_parameters or 0 for m in self._models.values())

        frameworks = set()
        model_types = set()
        for metadata in self._models.values():
            if metadata.framework:
                frameworks.add(metadata.framework)
            if metadata.model_type:
                model_types.add(metadata.model_type)

        return {
            "total_models": total_models,
            "total_parameters": total_parameters,
            "frameworks": list(frameworks),
            "model_types": list(model_types),
            "cache_size": len(self._cache),
        }

    def export_registry(self, path: Path) -> None:
        """Export registry to file.

        Args:
            path: Output file path
        """
        data = {
            "models": {
                model_id: metadata.to_dict()
                for model_id, metadata in self._models.items()
            },
            "exported_at": datetime.utcnow().isoformat(),
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported registry to {path}")
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            raise

    def import_registry(self, path: Path) -> None:
        """Import registry from file.

        Args:
            path: Input file path
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)

            for model_id, metadata_dict in data.get("models", {}).items():
                metadata = ModelMetadata.from_dict(metadata_dict)
                self._models[model_id] = metadata
                self._update_search_index(model_id, metadata)

            logger.info(f"Imported registry from {path}")
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            raise
