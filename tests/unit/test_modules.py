"""Unit tests for core modules."""

import pytest
from typing import List


class TestAgents:
    """Tests for agent module."""

    def test_agent_creation(self):
        """Test agent creation."""
        # Test placeholder
        assert True


class TestInference:
    """Tests for inference module."""

    def test_inference_engine_creation(self):
        """Test inference engine creation."""
        # Test placeholder
        assert True


class TestRAG:
    """Tests for RAG module."""

    def test_document_chunking(self, sample_texts: List[str]):
        """Test document chunking."""
        from rag.src.core import DocumentChunker

        chunker = DocumentChunker(chunk_size=50)
        chunks = chunker.chunk(sample_texts[0])

        assert len(chunks) > 0


class TestInfra:
    """Tests for infrastructure module."""

    def test_infra_config(self):
        """Test infrastructure configuration."""
        from infra.src.core import InfraConfig, DeploymentEnvironment

        config = InfraConfig(environment=DeploymentEnvironment.DEV, region="us-west-2")

        assert config.environment == DeploymentEnvironment.DEV


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
