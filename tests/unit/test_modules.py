"""Comprehensive unit tests for core modules."""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any
import json
import tempfile
import os


# ==============================================================================
# TestAgents - Tests for agent module
# ==============================================================================
class TestAgents:
    """Tests for agent module."""

    def test_agent_config_creation(self):
        """Test creating agent configuration."""
        from agents.src.core import AgentConfig

        config = AgentConfig(
            name="test-agent",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="You are a helpful assistant",
            tools=["search", "calculator"]
        )

        assert config.name == "test-agent"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.system_prompt == "You are a helpful assistant"
        assert config.tools == ["search", "calculator"]

    def test_agent_creation(self):
        """Test creating a basic agent."""
        from agents.src.core import Agent, AgentConfig

        config = AgentConfig(name="test-agent", model="gpt-4")
        agent = Agent(config)

        assert agent.config.name == "test-agent"
        assert agent.config.model == "gpt-4"
        assert agent.state.value == "idle"
        assert agent.tools == {}
        assert agent.memory == []

    def test_agent_run(self):
        """Test running an agent on a task."""
        from agents.src.core import Agent, AgentConfig

        config = AgentConfig(name="test-agent", model="gpt-4")
        agent = Agent(config)
        result = agent.run("Research quantum computing")

        assert result["task"] == "Research quantum computing"
        assert result["status"] == "completed"
        assert result["result"] == "Task executed successfully"
        assert agent.state.value == "completed"

    def test_agent_run_failure(self):
        """Test agent failure handling."""
        from agents.src.core import Agent, AgentConfig, AgentState

        config = AgentConfig(name="test-agent", model="gpt-4")
        agent = Agent(config)

        # Mock _execute_task to raise an exception
        def failing_task(task):
            raise RuntimeError("Task failed")

        agent._execute_task = failing_task
        result = agent.run("Test task")

        assert "error" in result
        assert agent.state == AgentState.FAILED

    def test_agent_memory_management(self):
        """Test agent memory operations."""
        from agents.src.core import Agent, AgentConfig

        config = AgentConfig(name="test-agent", model="gpt-4")
        agent = Agent(config)

        # Test adding and retrieving memory
        agent.add_memory("key1", "value1")
        agent.add_memory("key2", {"nested": "data"})

        memory = agent.get_memory()
        assert len(memory) == 2
        assert memory[0]["key"] == "key1"
        assert memory[0]["value"] == "value1"
        assert memory[1]["key"] == "key2"
        assert memory[1]["value"] == {"nested": "data"}

    def test_agent_tool_execution(self):
        """Test agent tool execution."""
        from agents.src.core import Agent, AgentConfig, Tool

        # Create a mock tool
        class MockTool(Tool):
            def __init__(self):
                super().__init__("mock_tool", "A mock tool for testing")
                self.executed = False

            def execute(self, **kwargs) -> Any:
                self.executed = True
                return f"Executed with {kwargs}"

        config = AgentConfig(name="test-agent", model="gpt-4")
        agent = Agent(config)
        tool = MockTool()

        agent.add_tool(tool)
        assert "mock_tool" in agent.tools
        assert agent.tools["mock_tool"] == tool
        assert tool.executed is False

    def test_agent_orchestrator_workflow(self):
        """Test agent orchestrator workflow."""
        from agents.src.core import Agent, AgentConfig, AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Create and register agents
        agent1 = Agent(AgentConfig(name="agent1", model="gpt-4"))
        agent2 = Agent(AgentConfig(name="agent2", model="gpt-4"))

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        assert "agent1" in orchestrator.agents
        assert "agent2" in orchestrator.agents

        # Create and execute workflow
        orchestrator.create_workflow("test_workflow", ["agent1", "agent2"])
        result = orchestrator.execute_workflow("test_workflow", "Test task")

        assert result["workflow"] == "test_workflow"
        assert result["task"] == "Test task"
        assert result["agents_executed"] == 2
        assert len(result["results"]) == 2

    def test_agent_orchestrator_invalid_workflow(self):
        """Test orchestrator with invalid workflow."""
        from agents.src.core import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        with pytest.raises(ValueError, match="Unknown workflow"):
            orchestrator.execute_workflow("nonexistent_workflow", "Test task")

    def test_agent_orchestrator_invalid_agent(self):
        """Test orchestrator with invalid agent in workflow."""
        from agents.src.core import Agent, AgentConfig, AgentOrchestrator

        orchestrator = AgentOrchestrator()
        orchestrator.create_workflow("bad_workflow", ["nonexistent_agent"])

        with pytest.raises(ValueError, match="Unknown agent"):
            orchestrator.execute_workflow("bad_workflow", "Test task")


# ==============================================================================
# TestInference - Tests for inference engines
# ==============================================================================
class TestInference:
    """Tests for inference module."""

    def test_inference_config_creation(self):
        """Test creating inference configuration."""
        from inference.src.engines import InferenceConfig

        config = InferenceConfig(
            model_name="gpt2",
            batch_size=32,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            device="cuda",
            quantization=False
        )

        assert config.model_name == "gpt2"
        assert config.batch_size == 32
        assert config.max_length == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.device == "cuda"
        assert config.quantization is False

    @patch("inference.src.engines.AutoTokenizer")
    @patch("inference.src.engines.AutoModelForCausalLM")
    def test_transformers_engine_creation(self, mock_model, mock_tokenizer):
        """Test TransformersEngine creation."""
        from inference.src.engines import InferenceConfig, TransformersEngine

        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        config = InferenceConfig(model_name="gpt2", device="cpu")
        engine = TransformersEngine(config)

        assert engine.config == config
        assert engine.model is not None
        mock_tokenizer.assert_called_once_with("gpt2")
        mock_model.assert_called_once()

    def test_inference_engine_factory_create_transformers(self):
        """Test factory creating TransformersEngine."""
        from inference.src.engines import InferenceEngineFactory, InferenceConfig

        config = InferenceConfig(model_name="gpt2", device="cpu")

        with patch("inference.src.engines.TransformersEngine") as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            result = InferenceEngineFactory.create("transformers", config)
            assert result == mock_engine_instance
            mock_engine.assert_called_once_with(config)

    def test_inference_engine_factory_create_vllm(self):
        """Test factory creating VLLMEngine."""
        from inference.src.engines import InferenceEngineFactory, InferenceConfig

        config = InferenceConfig(model_name="gpt2", device="cpu")

        with patch("inference.src.engines.VLLMEngine") as mock_engine:
            mock_engine_instance = Mock()
            mock_engine.return_value = mock_engine_instance

            result = InferenceEngineFactory.create("vllm", config)
            assert result == mock_engine_instance
            mock_engine.assert_called_once_with(config)

    def test_inference_engine_factory_unknown_engine(self):
        """Test factory with unknown engine type."""
        from inference.src.engines import InferenceEngineFactory, InferenceConfig

        config = InferenceConfig(model_name="gpt2")

        with pytest.raises(ValueError, match="Unknown engine"):
            InferenceEngineFactory.create("unknown_engine", config)

    def test_inference_engine_factory_get_registered_types(self):
        """Test getting registered engine types."""
        from inference.src.engines import InferenceEngineFactory

        types = InferenceEngineFactory.get_registered_types()
        assert "transformers" in types
        assert "vllm" in types

    @patch("inference.src.engines.SentenceTransformer")
    def test_transformers_engine_get_embeddings(self, mock_st):
        """Test getting embeddings from TransformersEngine."""
        from inference.src.engines import InferenceConfig, TransformersEngine

        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = [[0.1] * 384, [0.2] * 384]
        mock_st.return_value = mock_embedding_model

        # Mock tokenizer and model loading
        with patch("inference.src.engines.AutoTokenizer") as mock_tokenizer:
            with patch("inference.src.engines.AutoModelForCausalLM") as mock_model:
                mock_tokenizer_instance = Mock()
                mock_tokenizer.return_value = mock_tokenizer_instance
                mock_model.return_value = Mock()

                config = InferenceConfig(model_name="gpt2", device="cpu")
                engine = TransformersEngine(config)

                embeddings = engine.get_embeddings(["text1", "text2"])

                assert len(embeddings) == 2
                assert len(embeddings[0]) == 384


# ==============================================================================
# TestRAG - Tests for RAG module
# ==============================================================================
class TestRAG:
    """Tests for RAG module."""

    def test_rag_config_creation(self):
        """Test creating RAG configuration."""
        from rag.src.core import RAGConfig

        config = RAGConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
            similarity_threshold=0.5
        )

        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.top_k == 5
        assert config.similarity_threshold == 0.5

    def test_document_chunker_default(self):
        """Test DocumentChunker with default settings."""
        from rag.src.core import DocumentChunker

        chunker = DocumentChunker(chunk_size=100, overlap=20)
        text = "This is a test. " * 50  # Create text longer than chunk_size
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_document_chunker_various_sizes(self):
        """Test DocumentChunker with various chunk sizes."""
        from rag.src.core import DocumentChunker

        text = "Word " * 100

        # Test with small chunks
        chunker_small = DocumentChunker(chunk_size=50, overlap=10)
        chunks_small = chunker_small.chunk(text)
        assert len(chunks_small) > 1

        # Test with large chunks
        chunker_large = DocumentChunker(chunk_size=500, overlap=50)
        chunks_large = chunker_large.chunk(text)
        assert len(chunks_large) >= 1

        # Test with no overlap
        chunker_no_overlap = DocumentChunker(chunk_size=100, overlap=0)
        chunks_no_overlap = chunker_no_overlap.chunk(text)
        assert len(chunks_no_overlap) > 0

    def test_document_chunker_empty_text(self):
        """Test DocumentChunker with empty text."""
        from rag.src.core import DocumentChunker

        chunker = DocumentChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk("")

        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_document_creation(self):
        """Test creating Document objects."""
        from rag.src.core import Document

        doc = Document(
            id="doc_1",
            content="Test content",
            metadata={"source": "test.txt"},
            embedding=[0.1, 0.2, 0.3]
        )

        assert doc.id == "doc_1"
        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test.txt"}
        assert doc.embedding == [0.1, 0.2, 0.3]

    def test_vector_database_add_and_search(self):
        """Test VectorDatabase add and search operations."""
        from rag.src.core import VectorDatabase, Document
        import numpy as np

        db = VectorDatabase()

        # Add documents
        doc1 = Document(
            id="doc1",
            content="Machine learning is a subset of AI",
            metadata={"topic": "ML"},
            embedding=[1.0, 0.0, 0.0]
        )
        doc2 = Document(
            id="doc2",
            content="Deep learning uses neural networks",
            metadata={"topic": "DL"},
            embedding=[0.0, 1.0, 0.0]
        )

        db.add_document(doc1)
        db.add_document(doc2)

        # Search
        results = db.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"
        assert results[0][1] == pytest.approx(1.0, 0.01)

    def test_vector_database_add_documents_batch(self):
        """Test VectorDatabase batch add operation."""
        from rag.src.core import VectorDatabase, Document

        db = VectorDatabase()

        docs = [
            Document(id=f"doc{i}", content=f"Content {i}", metadata={"index": i}, embedding=[float(i), 0.0, 0.0])
            for i in range(5)
        ]

        db.add_documents(docs)
        assert len(db.documents) == 5

    def test_embedding_model_without_sentence_transformers(self):
        """Test EmbeddingModel when sentence_transformers is not installed."""
        from rag.src.core import EmbeddingModel

        with patch("rag.src.core.SentenceTransformer", side_effect=ImportError):
            model = EmbeddingModel("test-model")
            embeddings = model.embed(["text1", "text2"])

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384

    def test_rag_system_initialization(self):
        """Test RAGSystem initialization."""
        from rag.src.core import RAGSystem, RAGConfig

        config = RAGConfig(chunk_size=256, chunk_overlap=25)
        rag = RAGSystem(config)

        assert rag.config == config
        assert rag.chunker.chunk_size == 256
        assert rag.chunker.overlap == 25

    @patch("rag.src.core.EmbeddingModel")
    def test_rag_system_add_documents(self, mock_embedding_model):
        """Test RAGSystem add documents."""
        from rag.src.core import RAGSystem, RAGConfig

        # Mock embedding model
        mock_instance = Mock()
        mock_instance.embed.return_value = [[0.1] * 384]
        mock_embedding_model.return_value = mock_instance

        config = RAGConfig(chunk_size=50, chunk_overlap=10)
        rag = RAGSystem(config)
        rag.embedding_model = mock_instance

        rag.add_documents(["This is a test document"], [{"source": "test.txt"}])

        assert len(rag.vector_db.documents) > 0


# ==============================================================================
# TestModels - Tests for model components
# ==============================================================================
class TestModels:
    """Tests for model components."""

    def test_model_config_creation(self):
        """Test creating ModelConfig."""
        from models.base.config import ModelConfig, ModelType

        config = ModelConfig(
            name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            version="1.0.0",
            description="Test model",
            batch_size=16,
            device="cuda",
            framework="pytorch"
        )

        assert config.name == "test-model"
        assert config.model_type == ModelType.LANGUAGE_MODEL
        assert config.version == "1.0.0"
        assert config.description == "Test model"
        assert config.batch_size == 16
        assert config.device == "cuda"
        assert config.framework == "pytorch"

    def test_model_config_to_dict(self):
        """Test ModelConfig to_dict method."""
        from models.base.config import ModelConfig, ModelType

        config = ModelConfig(
            name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            version="1.0.0"
        )

        config_dict = config.to_dict()
        assert config_dict["name"] == "test-model"
        assert config_dict["model_type"] == "language_model"
        assert config_dict["version"] == "1.0.0"

    def test_model_metadata_creation(self):
        """Test creating ModelMetadata."""
        from models.base.config import ModelMetadata

        metadata = ModelMetadata(
            model_id="test-1.0.0",
            name="test-model",
            version="1.0.0",
            author="Test Author",
            tags=["test", "nlp"]
        )

        assert metadata.model_id == "test-1.0.0"
        assert metadata.name == "test-model"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "nlp"]

    def test_model_status_enum(self):
        """Test ModelStatus enum values."""
        from models.base.core import ModelStatus

        assert ModelStatus.UNINITIALIZED.value == "uninitialized"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.RUNNING.value == "running"
        assert ModelStatus.ERROR.value == "error"
        assert ModelStatus.UNLOADED.value == "unloaded"

    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        from models.base.core import BaseModel
        from models.base.config import ModelConfig, ModelType

        config = ModelConfig(
            name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            version="1.0.0"
        )

        model = BaseModel(config)

        assert model.config == config
        assert model.is_loaded() is False
        assert model.get_status().value == "uninitialized"

    def test_base_model_load_unload(self):
        """Test BaseModel load and unload operations."""
        from models.base.core import BaseModel, ModelStatus
        from models.base.config import ModelConfig, ModelType

        config = ModelConfig(
            name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            version="1.0.0"
        )

        model = BaseModel(config)

        # Test load
        model.load()
        assert model.is_loaded() is True
        assert model.get_status() == ModelStatus.READY

        # Test unload
        model.unload()
        assert model.is_loaded() is False
        assert model.get_status() == ModelStatus.UNLOADED

    def test_base_model_double_load_warning(self):
        """Test BaseModel warning on double load."""
        from models.base.core import BaseModel
        from models.base.config import ModelConfig, ModelType

        config = ModelConfig(name="test-model", model_type=ModelType.LANGUAGE_MODEL)
        model = BaseModel(config)

        model.load()
        # Second load should trigger warning but not fail
        model.load()
        assert model.is_loaded() is True

    def test_base_model_get_info(self):
        """Test BaseModel get_info method."""
        from models.base.core import BaseModel
        from models.base.config import ModelConfig, ModelType

        config = ModelConfig(
            name="test-model",
            model_type=ModelType.LANGUAGE_MODEL,
            version="2.0.0",
            num_parameters=7000000000
        )

        model = BaseModel(config)
        info = model.get_info()

        assert info["name"] == "test-model"
        assert info["version"] == "2.0.0"
        assert info["type"] == "language_model"
        assert info["num_parameters"] == 7000000000

    def test_model_factory_register_and_create(self):
        """Test ModelFactory register and create."""
        from models.base.core import ModelFactory, BaseModel
        from models.base.config import ModelConfig, ModelType

        # Register a custom model builder
        def custom_builder(**kwargs):
            config = ModelConfig(name="custom", model_type=ModelType.CUSTOM_MODEL)
            return BaseModel(config)

        ModelFactory.register_builder("custom_type", custom_builder)

        # Create model
        model = ModelFactory.create("custom_type")
        assert isinstance(model, BaseModel)

    def test_model_factory_unknown_type(self):
        """Test ModelFactory with unknown model type."""
        from models.base.core import ModelFactory

        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create("unknown_type")

    def test_model_factory_get_registered_types(self):
        """Test ModelFactory get_registered_types."""
        from models.base.core import ModelFactory

        types = ModelFactory.get_registered_types()
        assert isinstance(types, list)

    def test_lora_adapter_creation(self):
        """Test LoRAAdapter creation."""
        from models.adapters.core import LoRAAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        config = AdapterConfig(
            name="lora-adapter",
            adapter_type=AdapterType.LORA,
            base_model_name="gpt2"
        )

        adapter = LoRAAdapter(config, rank=8, alpha=16.0)

        assert adapter.config.name == "lora-adapter"
        assert adapter.rank == 8
        assert adapter.alpha == 16.0
        assert adapter.is_active is False

    def test_lora_adapter_enable_disable(self):
        """Test LoRAAdapter enable and disable."""
        from models.adapters.core import LoRAAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        config = AdapterConfig(
            name="lora-adapter",
            adapter_type=AdapterType.LORA,
            base_model_name="gpt2"
        )

        adapter = LoRAAdapter(config)
        assert adapter.is_active is False

        adapter.enable()
        assert adapter.is_active is True

        adapter.disable()
        assert adapter.is_active is False

    def test_lora_adapter_attach_detach(self):
        """Test LoRAAdapter attach and detach."""
        from models.adapters.core import LoRAAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        config = AdapterConfig(
            name="lora-adapter",
            adapter_type=AdapterType.LORA,
            base_model_name="gpt2"
        )

        adapter = LoRAAdapter(config)

        # Create a mock model with config attribute
        mock_model = Mock()
        mock_model.config = Mock()

        adapter.attach(mock_model)
        assert hasattr(adapter, "_model_ref")
        assert adapter._model_ref == mock_model

        adapter.detach()
        assert not hasattr(adapter, "_model_ref")

    def test_lora_adapter_get_trainable_params(self):
        """Test LoRAAdapter get_trainable_params."""
        from models.adapters.core import LoRAAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        config = AdapterConfig(name="lora-adapter", adapter_type=AdapterType.LORA, base_model_name="gpt2")
        adapter = LoRAAdapter(config, rank=8, alpha=16.0)

        # Add a layer
        adapter.add_layer("layer1", 512, 512)

        params = adapter.get_trainable_params()
        assert params["rank"] == 8
        assert params["alpha"] == 16.0
        assert params["num_layers"] == 1

    def test_prefix_tuning_adapter_creation(self):
        """Test PrefixTuningAdapter creation."""
        from models.adapters.core import PrefixTuningAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        config = AdapterConfig(
            name="prefix-adapter",
            adapter_type=AdapterType.PREFIX_TUNING,
            base_model_name="gpt2"
        )

        adapter = PrefixTuningAdapter(config, prefix_length=20)

        assert adapter.prefix_length == 20
        assert adapter.is_active is False

    def test_adapter_registry(self):
        """Test AdapterRegistry operations."""
        from models.adapters.core import AdapterRegistry, LoRAAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        registry = AdapterRegistry()

        # Add adapter
        config = AdapterConfig(name="test-adapter", adapter_type=AdapterType.LORA, base_model_name="gpt2")
        adapter = LoRAAdapter(config)
        registry.add_adapter(adapter)

        assert "test-adapter" in registry.list_adapters()

        # Get adapter
        retrieved = registry.get_adapter("test-adapter")
        assert retrieved == adapter

        # Remove adapter
        assert registry.remove_adapter("test-adapter") is True
        assert registry.remove_adapter("test-adapter") is False

    def test_adapter_registry_duplicate(self):
        """Test AdapterRegistry with duplicate adapter name."""
        from models.adapters.core import AdapterRegistry, LoRAAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        registry = AdapterRegistry()

        config = AdapterConfig(name="duplicate", adapter_type=AdapterType.LORA, base_model_name="gpt2")
        adapter = LoRAAdapter(config)

        registry.add_adapter(adapter)

        with pytest.raises(ValueError, match="already registered"):
            registry.add_adapter(adapter)

    def test_adapter_registry_compose(self):
        """Test AdapterRegistry adapter composition."""
        from models.adapters.core import AdapterRegistry, LoRAAdapter
        from models.adapters.config import AdapterConfig, AdapterType

        registry = AdapterRegistry()

        # Add composable adapters
        config1 = AdapterConfig(name="adapter1", adapter_type=AdapterType.LORA, base_model_name="gpt2", is_composable=True)
        config2 = AdapterConfig(name="adapter2", adapter_type=AdapterType.LORA, base_model_name="gpt2", is_composable=True)

        registry.add_adapter(LoRAAdapter(config1))
        registry.add_adapter(LoRAAdapter(config2))

        # Compose them
        registry.compose_adapters("composite", ["adapter1", "adapter2"])

        composed = registry.get_composed_adapters("composite")
        assert len(composed) == 2

        assert "composite" in registry.list_compositions()


# ==============================================================================
# TestPipelines - Tests for pipeline components
# ==============================================================================
class TestPipelines:
    """Tests for pipeline components."""

    def test_training_config_creation(self):
        """Test TrainingConfig creation."""
        from pipelines.training.src.orchestrator import TrainingConfig

        config = TrainingConfig(
            model_name="gpt2",
            dataset_path="data/train",
            output_dir="./outputs",
            num_epochs=3,
            batch_size=16,
            learning_rate=5e-4,
            training_method="lora"
        )

        assert config.model_name == "gpt2"
        assert config.dataset_path == "data/train"
        assert config.output_dir == "./outputs"
        assert config.num_epochs == 3
        assert config.batch_size == 16
        assert config.learning_rate == 5e-4
        assert config.training_method == "lora"

    @patch("pathlib.Path.mkdir")
    def test_training_orchestrator_setup_directories(self, mock_mkdir):
        """Test TrainingOrchestrator directory setup."""
        from pipelines.training.src.orchestrator import TrainingOrchestrator, TrainingConfig

        config = TrainingConfig(
            model_name="gpt2",
            dataset_path="data/train",
            output_dir="./test_outputs"
        )

        orchestrator = TrainingOrchestrator(config)

        # Check directories would be created
        assert mock_mkdir.called

    def test_training_orchestrator_get_training_stats(self):
        """Test TrainingOrchestrator get_training_stats."""
        from pipelines.training.src.orchestrator import TrainingOrchestrator, TrainingConfig

        config = TrainingConfig(
            model_name="gpt2",
            dataset_path="data/train",
            output_dir="./outputs",
            training_method="lora"
        )

        orchestrator = TrainingOrchestrator(config)

        # Mock model with parameters
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        orchestrator.model = Mock()
        orchestrator.model.parameters.return_value = [mock_param]

        stats = orchestrator.get_training_stats()

        assert stats["model_name"] == "gpt2"
        assert stats["training_method"] == "lora"

    def test_benchmark_config_creation(self):
        """Test BenchmarkConfig creation."""
        from pipelines.evaluation.src.benchmark import BenchmarkConfig

        config = BenchmarkConfig(
            model_path="./checkpoints/model",
            benchmarks=["mmlu", "gsm8k"],
            batch_size=32,
            num_shots=5
        )

        assert config.model_path == "./checkpoints/model"
        assert config.benchmarks == ["mmlu", "gsm8k"]
        assert config.batch_size == 32
        assert config.num_shots == 5

    def test_benchmark_orchestrator_supported_benchmarks(self):
        """Test BenchmarkOrchestrator supported benchmarks."""
        from pipelines.evaluation.src.benchmark import BenchmarkOrchestrator, BenchmarkConfig

        config = BenchmarkConfig(
            model_path="./model",
            benchmarks=["mmlu", "gsm8k", "hellaswag"]
        )

        orchestrator = BenchmarkOrchestrator(config)

        assert "mmlu" in orchestrator.supported_benchmarks
        assert "gsm8k" in orchestrator.supported_benchmarks
        assert "alpacaeval" in orchestrator.supported_benchmarks

    def test_benchmark_orchestrator_get_summary(self):
        """Test BenchmarkOrchestrator get_summary."""
        from pipelines.evaluation.src.benchmark import BenchmarkOrchestrator, BenchmarkConfig

        config = BenchmarkConfig(
            model_path="./model",
            benchmarks=["mmlu", "gsm8k"]
        )

        orchestrator = BenchmarkOrchestrator(config)
        orchestrator.results = {
            "mmlu": {"score": 75.0},
            "gsm8k": {"score": 60.0}
        }

        summary = orchestrator.get_summary()

        assert summary["model"] == "./model"
        assert summary["num_benchmarks"] == 2
        assert summary["completed_benchmarks"] == 2
        assert summary["average_score"] == 67.5

    def test_benchmark_orchestrator_get_summary_with_errors(self):
        """Test BenchmarkOrchestrator get_summary with errors."""
        from pipelines.evaluation.src.benchmark import BenchmarkOrchestrator, BenchmarkConfig

        config = BenchmarkConfig(
            model_path="./model",
            benchmarks=["mmlu", "gsm8k"]
        )

        orchestrator = BenchmarkOrchestrator(config)
        orchestrator.results = {
            "mmlu": {"score": 75.0},
            "gsm8k": {"error": "Failed to load dataset"}
        }

        summary = orchestrator.get_summary()

        assert summary["completed_benchmarks"] == 1
        assert summary["average_score"] == 75.0

    def test_validation_config_creation(self):
        """Test ValidationConfig creation."""
        from pipelines.data.src.validation import ValidationConfig

        config = ValidationConfig(
            required_columns=["text", "label"],
            min_samples=100,
            max_duplicates_percentage=5.0,
            max_missing_percentage=10.0
        )

        assert config.required_columns == ["text", "label"]
        assert config.min_samples == 100
        assert config.max_duplicates_percentage == 5.0

    def test_data_validation_shape_check(self):
        """Test DataValidation shape checking."""
        import pandas as pd
        from pipelines.data.src.validation import DataValidation, ValidationConfig

        config = ValidationConfig(required_columns=["text"])
        validator = DataValidation(config)

        df = pd.DataFrame({"text": ["sample1", "sample2"], "label": [0, 1]})
        is_valid = validator.validate(df)

        assert is_valid is True
        assert validator.validation_results["checks"]["shape"]["num_rows"] == 2
        assert validator.validation_results["checks"]["shape"]["num_columns"] == 2

    def test_data_validation_required_columns(self):
        """Test DataValidation required columns check."""
        import pandas as pd
        from pipelines.data.src.validation import DataValidation, ValidationConfig

        config = ValidationConfig(required_columns=["text", "label"])
        validator = DataValidation(config)

        # Missing "label" column
        df = pd.DataFrame({"text": ["sample1", "sample2"]})
        is_valid = validator.validate(df)

        assert is_valid is False
        assert any("label" in str(e) for e in validator.validation_results["errors"])

    def test_data_validation_minimum_samples(self):
        """Test DataValidation minimum samples check."""
        import pandas as pd
        from pipelines.data.src.validation import DataValidation, ValidationConfig

        config = ValidationConfig(min_samples=10)
        validator = DataValidation(config)

        df = pd.DataFrame({"text": ["sample1"]})
        is_valid = validator.validate(df)

        assert is_valid is False
        assert "Insufficient samples" in str(validator.validation_results["errors"])

    def test_data_validation_duplicates_check(self):
        """Test DataValidation duplicates check."""
        import pandas as pd
        from pipelines.data.src.validation import DataValidation, ValidationConfig

        config = ValidationConfig(max_duplicates_percentage=5.0)
        validator = DataValidation(config)

        # Create dataframe with many duplicates
        df = pd.DataFrame({"text": ["same"] * 100})
        is_valid = validator.validate(df)

        # Check should pass but with warning
        assert "duplicates" in validator.validation_results["checks"]
        assert validator.validation_results["checks"]["duplicates"]["num_duplicates"] == 99

    def test_data_validation_generate_report(self):
        """Test DataValidation generate_report method."""
        import pandas as pd
        from pipelines.data.src.validation import DataValidation, ValidationConfig

        config = ValidationConfig(required_columns=["text"])
        validator = DataValidation(config)

        df = pd.DataFrame({"text": ["sample1", "sample2"]})
        validator.validate(df)

        report = validator.generate_report()

        assert "is_valid" in report
        assert "num_checks" in report
        assert "num_errors" in report
        assert "num_warnings" in report
        assert "details" in report

    def test_data_validation_save_report(self):
        """Test DataValidation save_report method."""
        import pandas as pd
        from pipelines.data.src.validation import DataValidation, ValidationConfig

        config = ValidationConfig(required_columns=["text"])
        validator = DataValidation(config)

        df = pd.DataFrame({"text": ["sample1", "sample2"]})
        validator.validate(df)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            tmp_path = tmp.name

        try:
            validator.save_report(tmp_path)
            assert os.path.exists(tmp_path)

            with open(tmp_path, 'r') as f:
                saved_report = json.load(f)
                assert "is_valid" in saved_report
        finally:
            os.unlink(tmp_path)


# ==============================================================================
# TestRAGIngestion - Tests for RAG document ingestion
# ==============================================================================
class TestRAGIngestion:
    """Tests for RAG document ingestion."""

    def test_ingestion_config_creation(self):
        """Test IngestionConfig creation."""
        from rag.ingestion.config import IngestionConfig, LoaderType

        config = IngestionConfig(
            loader_type=LoaderType.TEXT,
            extract_metadata=True,
            encoding="utf-8",
            max_file_size_mb=100
        )

        assert config.loader_type == LoaderType.TEXT
        assert config.extract_metadata is True
        assert config.encoding == "utf-8"
        assert config.max_file_size_mb == 100

    def test_ingestion_config_validation(self):
        """Test IngestionConfig validation."""
        from rag.ingestion.config import IngestionConfig

        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            IngestionConfig(max_file_size_mb=0)

        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            IngestionConfig(timeout_seconds=0)

        with pytest.raises(ValueError, match="retry_attempts must be non-negative"):
            IngestionConfig(retry_attempts=-1)

    def test_document_loader_text(self):
        """Test DocumentLoader with text file."""
        from rag.ingestion.core import DocumentLoader
        from rag.ingestion.config import IngestionConfig, LoaderType

        config = IngestionConfig(loader_type=LoaderType.TEXT)
        loader = DocumentLoader(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write("This is a test document content.")
            tmp_path = tmp.name

        try:
            docs = loader.load(tmp_path)
            assert len(docs) == 1
            assert docs[0]["content"] == "This is a test document content."
            assert "source" in docs[0]["metadata"]
        finally:
            os.unlink(tmp_path)

    def test_document_loader_nonexistent_file(self):
        """Test DocumentLoader with nonexistent file."""
        from rag.ingestion.core import DocumentLoader
        from rag.ingestion.config import IngestionConfig, LoaderType

        config = IngestionConfig(loader_type=LoaderType.TEXT)
        loader = DocumentLoader(config)

        docs = loader.load("/nonexistent/path/file.txt")
        assert docs == []

    def test_document_loader_directory(self):
        """Test DocumentLoader with directory."""
        from rag.ingestion.core import DocumentLoader
        from rag.ingestion.config import IngestionConfig, LoaderType

        config = IngestionConfig(loader_type=LoaderType.DIRECTORY)
        loader = DocumentLoader(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            with open(os.path.join(tmpdir, "file1.txt"), 'w') as f:
                f.write("Content 1")
            with open(os.path.join(tmpdir, "file2.txt"), 'w') as f:
                f.write("Content 2")

            docs = loader.load(tmpdir)
            assert len(docs) == 2

    def test_metadata_extractor(self):
        """Test MetadataExtractor."""
        from rag.ingestion.core import MetadataExtractor

        extractor = MetadataExtractor()
        content = "Line 1\nLine 2\nLine 3"
        base_metadata = {"source": "test.txt"}

        result = extractor.extract(content, base_metadata)

        assert result["source"] == "test.txt"
        assert result["char_count"] == len(content)
        assert result["word_count"] == 6
        assert result["line_count"] == 3

    def test_document_pipeline(self):
        """Test DocumentPipeline."""
        from rag.ingestion.core import DocumentPipeline
        from rag.ingestion.config import IngestionConfig, LoaderType

        config = IngestionConfig(loader_type=LoaderType.TEXT, extract_metadata=True)
        pipeline = DocumentPipeline(config)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write("Test content for pipeline.")
            tmp_path = tmp.name

        try:
            docs = pipeline.process(tmp_path)
            assert len(docs) == 1
            assert docs[0]["content"] == "Test content for pipeline."
            assert "char_count" in docs[0]["metadata"]
        finally:
            os.unlink(tmp_path)


# ==============================================================================
# TestInfra - Tests for infrastructure module
# ==============================================================================
class TestInfra:
    """Tests for infrastructure module."""

    def test_infra_config_creation(self):
        """Test InfraConfig creation."""
        from infra.src.core import InfraConfig, DeploymentEnvironment

        config = InfraConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            region="us-west-2",
            cpu_cores=8,
            memory_gb=32,
            gpu_enabled=True,
            gpu_type="a100"
        )

        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.region == "us-west-2"
        assert config.cpu_cores == 8
        assert config.memory_gb == 32
        assert config.gpu_enabled is True
        assert config.gpu_type == "a100"

    def test_docker_builder(self):
        """Test DockerBuilder."""
        from infra.src.core import DockerBuilder, InfraConfig, DeploymentEnvironment

        config = InfraConfig(environment=DeploymentEnvironment.DEV, region="us-east-1")
        builder = DockerBuilder(config)

        result = builder.build_image("./Dockerfile", "my-image:latest")

        assert result["status"] == "built"
        assert result["tag"] == "my-image:latest"
        assert result["environment"] == "dev"

    def test_kubernetes_deployer(self):
        """Test KubernetesDeployer."""
        from infra.src.core import KubernetesDeployer, InfraConfig, DeploymentEnvironment

        config = InfraConfig(environment=DeploymentEnvironment.STAGING, region="eu-west-1")
        deployer = KubernetesDeployer(config)

        result = deployer.deploy("./deployment.yaml", namespace="production")

        assert result["status"] == "deployed"
        assert result["namespace"] == "production"
        assert result["environment"] == "staging"
        assert result["replicas"] == 3

    def test_monitoring_system(self):
        """Test MonitoringSystem."""
        from infra.src.core import MonitoringSystem, InfraConfig, DeploymentEnvironment

        config = InfraConfig(environment=DeploymentEnvironment.PRODUCTION, region="us-west-2")
        monitoring = MonitoringSystem(config)

        # Record metrics
        monitoring.record_metric("inference_latency", 150.5, tags={"model": "gpt2"})
        monitoring.record_metric("throughput", 45.2)

        # Get all metrics
        metrics = monitoring.get_metrics()
        assert "inference_latency" in metrics
        assert "throughput" in metrics
        assert metrics["inference_latency"]["value"] == 150.5
        assert metrics["inference_latency"]["tags"]["model"] == "gpt2"

    def test_monitoring_system_filter(self):
        """Test MonitoringSystem metric filtering."""
        from infra.src.core import MonitoringSystem, InfraConfig, DeploymentEnvironment

        config = InfraConfig(environment=DeploymentEnvironment.PRODUCTION, region="us-west-2")
        monitoring = MonitoringSystem(config)

        monitoring.record_metric("inference_latency_ms", 150.5)
        monitoring.record_metric("training_loss", 0.5)
        monitoring.record_metric("inference_throughput", 45.2)

        # Filter metrics
        filtered = monitoring.get_metrics(name_pattern="inference")
        assert "inference_latency_ms" in filtered
        assert "inference_throughput" in filtered
        assert "training_loss" not in filtered


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
