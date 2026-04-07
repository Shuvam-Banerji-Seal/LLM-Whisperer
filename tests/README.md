# Tests

Comprehensive testing framework for unit, integration, and end-to-end tests.

## Overview

This module provides test infrastructure:
- Unit tests for isolated components
- Integration tests for module interactions
- End-to-end tests for complete workflows
- Test fixtures and utilities
- Coverage measurement and reporting

## Structure

```
tests/
├── README.md (this file)
├── unit/             # Component-level tests
├── integration/      # Cross-module tests
├── e2e/              # Full workflow tests
├── fixtures/         # Shared test data and utilities
├── conftest.py       # Pytest configuration
└── requirements.txt  # Test dependencies
```

## Directory Purposes

### `unit/` - Unit Tests

Test individual functions and classes:

```
unit/
├── test_embeddings.py       # Embedding model tests
├── test_chunking.py         # Text chunking tests
├── test_retrieval.py        # Retrieval logic tests
├── test_training.py         # Training utilities tests
├── test_quantization.py     # Quantization tests
└── test_utils.py            # Utility function tests
```

**Examples**:
- Test embedding model outputs shape
- Test chunking preserves text
- Test retrieval ranking accuracy
- Test data loading functionality

### `integration/` - Integration Tests

Test interactions between modules:

```
integration/
├── test_rag_pipeline.py          # Full RAG workflow
├── test_training_pipeline.py     # Training + evaluation
├── test_inference_serving.py     # Model serving
├── test_fine_tuning_workflow.py  # Fine-tuning end-to-end
└── test_multi_gpu.py             # Distributed training
```

**Examples**:
- Test RAG: load → embed → retrieve → generate
- Test training: prepare data → train → evaluate
- Test inference: load model → batch inference

### `e2e/` - End-to-End Tests

Test complete workflows:

```
e2e/
├── test_basic_rag.py             # Basic RAG workflow
├── test_fine_tuning_and_deploy.py # Train and serve
├── test_rag_with_chat.py         # Multi-turn chat with RAG
└── test_model_comparison.py      # Compare models
```

**Examples**:
- User uploads PDF → search → get answer
- User provides data → train → deploy → inference
- User chats → retrieve → generate responses

### `fixtures/` - Test Data & Utilities

```
fixtures/
├── sample_data.py        # Sample datasets
├── mock_models.py        # Mock model instances
├── test_configs.yaml     # Test configurations
└── sample_pdfs/          # Sample documents
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Suite

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# E2E tests only
pytest tests/e2e/
```

### Run Specific Test

```bash
pytest tests/unit/test_embeddings.py::test_embedding_shape
```

### With Options

```bash
# Verbose output
pytest tests/ -v

# Show print statements
pytest tests/ -s

# Parallel execution
pytest tests/ -n 4

# With coverage
pytest tests/ --cov=src --cov-report=html

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Debug mode
pytest tests/ --pdb
```

## Test Examples

### Unit Test Example

```python
# tests/unit/test_embeddings.py
import pytest
from rag.embeddings import EmbeddingModel


class TestEmbeddingModel:
    @pytest.fixture
    def model(self):
        return EmbeddingModel("all-MiniLM-L6-v2")
    
    def test_embedding_shape(self, model):
        """Test that embeddings have correct shape."""
        text = "This is a test sentence."
        embedding = model.embed(text)
        assert embedding.shape == (384,)
    
    def test_batch_embedding(self, model):
        """Test batch embedding."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = model.embed_batch(texts)
        assert embeddings.shape == (3, 384)
    
    def test_embedding_normalization(self, model):
        """Test that embeddings are normalized."""
        embedding = model.embed("Test")
        norm = (embedding ** 2).sum() ** 0.5
        assert abs(norm - 1.0) < 1e-5
```

### Integration Test Example

```python
# tests/integration/test_rag_pipeline.py
import pytest
from rag.ingestion import DocumentLoader
from rag.chunking import RecursiveChunker
from rag.embeddings import EmbeddingModel
from rag.indexing import VectorIndex
from rag.retrieval import DenseRetriever


def test_rag_pipeline():
    """Test complete RAG pipeline."""
    # Load documents
    loader = DocumentLoader()
    docs = loader.load("tests/fixtures/sample.pdf")
    assert len(docs) > 0
    
    # Chunk
    chunker = RecursiveChunker(chunk_size=512)
    chunks = chunker.chunk(docs)
    assert len(chunks) > 0
    
    # Embed
    embedder = EmbeddingModel()
    index = VectorIndex()
    index.add(chunks, embedder)
    assert index.size() > 0
    
    # Retrieve
    retriever = DenseRetriever(index)
    results = retriever.retrieve("What is this about?", k=5)
    assert len(results) <= 5
    assert all(hasattr(r, 'score') for r in results)
```

### E2E Test Example

```python
# tests/e2e/test_basic_rag.py
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag.pipeline import RAGPipeline


def test_basic_rag_workflow():
    """Test basic RAG workflow end-to-end."""
    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        documents="tests/fixtures/sample.pdf",
        embedding_model="all-MiniLM-L6-v2",
        language_model="mistralai/Mistral-7B-v0.1"
    )
    
    # Query
    query = "What is the main topic?"
    response = pipeline.query(query)
    
    # Validate response
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Error" not in response
```

## Fixtures & Utilities

### Pytest Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path


@pytest.fixture
def sample_docs():
    """Load sample documents for testing."""
    docs = [
        {"text": "Sample document 1"},
        {"text": "Sample document 2"},
        {"text": "Sample document 3"},
    ]
    return docs


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture(scope="session")
def embedding_model():
    """Load embedding model once per session."""
    from rag.embeddings import EmbeddingModel
    return EmbeddingModel("all-MiniLM-L6-v2")
```

### Mock Objects

```python
# tests/fixtures/mock_models.py
from unittest.mock import Mock
import numpy as np


def create_mock_embedding_model():
    """Create mock embedding model."""
    mock = Mock()
    mock.embed.return_value = np.random.randn(384)
    mock.embed_batch.return_value = np.random.randn(10, 384)
    return mock


def create_mock_language_model():
    """Create mock language model."""
    mock = Mock()
    mock.generate.return_value = "Generated response"
    return mock
```

## Coverage Report

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

**Coverage Targets**:
- Overall: >80%
- Critical paths: >90%
- Utils: >70%

## Test Data

### Creating Test Data

```python
# tests/fixtures/sample_data.py
import json
from pathlib import Path


def create_sample_documents():
    """Create sample documents for testing."""
    docs = [
        {"id": "1", "text": "This is a sample document.", "source": "test"},
        {"id": "2", "text": "Another sample document for testing.", "source": "test"},
        {"id": "3", "text": "A third document with different content.", "source": "test"},
    ]
    return docs


def save_test_fixtures():
    """Save test fixtures to files."""
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(exist_ok=True)
    
    # Save sample documents
    docs = create_sample_documents()
    with open(fixtures_dir / "sample_docs.json", "w") as f:
        json.dump(docs, f)
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Best Practices

### Test Organization
1. One test file per module
2. Class-based organization for related tests
3. Clear, descriptive test names
4. Arrange-Act-Assert pattern

### Test Data
1. Use fixtures for reusable data
2. Keep fixtures lightweight
3. Use temporary directories for file operations
4. Mock external dependencies

### Test Coverage
1. Test happy path
2. Test edge cases
3. Test error conditions
4. Aim for >80% coverage

### Performance
1. Use fast-running tests for unit tests
2. Use mock objects to avoid slow operations
3. Mark slow tests with `@pytest.mark.slow`
4. Use parametrization for similar tests

## Adding New Tests

```python
# tests/unit/test_new_feature.py
import pytest
from your_module import YourClass


class TestNewFeature:
    """Tests for new feature."""
    
    @pytest.fixture
    def instance(self):
        return YourClass()
    
    def test_basic_functionality(self, instance):
        """Test basic behavior."""
        result = instance.method()
        assert result is not None
    
    def test_edge_case(self, instance):
        """Test edge case."""
        result = instance.method(edge_value)
        assert result == expected
    
    def test_error_handling(self, instance):
        """Test error handling."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

## References

- [Pytest Documentation](https://docs.pytest.org/)
- See `../pipelines/` for pipeline testing
- See `../fine_tuning/` for training test patterns
- See `../rag/` for RAG testing examples

## Contributing

When adding code:
1. Write tests simultaneously
2. Aim for >80% coverage
3. Test all public APIs
4. Include edge cases
5. Document test purpose

## License

See LICENSE file in repository root.
