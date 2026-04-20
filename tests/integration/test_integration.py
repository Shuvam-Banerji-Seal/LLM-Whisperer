"""Comprehensive integration tests for LLM-Whisperer codebase.

This module tests component interactions across:
- Pipeline integration (data → training → evaluation → deployment)
- Inference serving (model loading, inference engines, batch processing)
- RAG pipeline (document ingestion, chunking, embedding, retrieval)
- Agent integration (memory, tools, multi-turn conversations)
- Model and inference integration
- Evaluation pipeline
"""

import pytest
import tempfile
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPipelineIntegration:
    """Tests for end-to-end pipeline integration."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_csv_data(self, temp_data_dir):
        """Create sample CSV data for testing."""
        csv_path = Path(temp_dir) / "sample_data.csv"
        df = pd.DataFrame({
            "text": [
                "Machine learning is a subset of AI.",
                "Deep learning uses neural networks.",
                "Natural language processing is important.",
                "Computer vision deals with images.",
            ],
            "label": [1, 1, 0, 0],
        })
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_data_ingestion_to_validation_pipeline(self, sample_csv_data):
        """Test data ingestion → validation pipeline flow."""
        from pipelines.data.src.ingestion import DataIngestion, IngestionConfig
        from pipelines.data.src.validation import DataValidation, ValidationConfig

        # Step 1: Data Ingestion
        ingestion_config = IngestionConfig(
            source_type="csv",
            source_path=sample_csv_data,
            max_samples=100
        )
        ingestion = DataIngestion()
        data = ingestion.load(ingestion_config)

        assert data is not None
        assert len(data) > 0
        assert "text" in data.columns

        # Step 2: Data Validation
        validation_config = ValidationConfig(
            required_columns=["text", "label"],
            min_samples=2,
            max_missing_percentage=50.0
        )
        validator = DataValidation(validation_config)
        is_valid = validator.validate(data)

        assert is_valid is True
        assert validator.validation_results["checks"]["required_columns"] == "OK"

    def test_data_validation_to_training_pipeline(self, temp_data_dir):
        """Test validation → training pipeline integration."""
        from pipelines.data.src.validation import DataValidation, ValidationConfig
        from pipelines.training.src.orchestrator import TrainingOrchestrator, TrainingConfig

        # Create sample data
        df = pd.DataFrame({
            "text": ["Sample text"] * 10,
            "label": [0, 1] * 5,
        })

        # Step 1: Validate data
        validation_config = ValidationConfig(
            required_columns=["text", "label"],
            min_samples=5
        )
        validator = DataValidation(validation_config)
        is_valid = validator.validate(df)

        assert is_valid is True

        # Step 2: Setup training (with mocked model loading)
        training_config = TrainingConfig(
            model_name="gpt2",
            dataset_path=temp_data_dir,
            output_dir=temp_data_dir,
            num_epochs=1,
            batch_size=2
        )

        orchestrator = TrainingOrchestrator(training_config)
        assert orchestrator.config.model_name == "gpt2"
        assert orchestrator.config.num_epochs == 1

        # Verify directory setup
        assert Path(temp_data_dir).exists()

    def test_training_to_evaluation_pipeline(self, temp_data_dir):
        """Test training → evaluation pipeline flow."""
        from pipelines.training.src.orchestrator import TrainingOrchestrator, TrainingConfig
        from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

        # Setup training
        training_config = TrainingConfig(
            model_name="gpt2",
            dataset_path=temp_data_dir,
            output_dir=temp_data_dir,
            num_epochs=1,
        )
        orchestrator = TrainingOrchestrator(training_config)

        # Get training stats
        stats = orchestrator.get_training_stats()
        assert "model_name" in stats
        assert "training_method" in stats

        # Setup evaluation
        metrics_config = MetricsConfig(
            task_benchmarks=True,
            latency_analysis=True
        )
        computer = MetricsComputer(metrics_config)

        # Compute metrics from training results
        results = {"mmlu": {"score": 45.0, "accuracy": 0.45}}
        metrics = computer.compute_metrics(results)

        assert "task_benchmarks" in metrics
        assert metrics["task_benchmarks"]["mmlu_score"] == 45.0

    def test_model_export_to_deployment_pipeline(self, temp_data_dir):
        """Test model export → deployment pipeline flow."""
        from models.exported.core import PyTorchExporter, ExporterFactory
        from models.exported.config import ExportConfig, ExportFormat
        from pipelines.deployment.src.orchestrator import DeploymentOrchestrator, DeploymentConfig

        # Step 1: Export Model
        export_config = ExportConfig(
            export_format=ExportFormat.PYTORCH,
            output_dir=Path(temp_data_dir) / "exports",
            model_name="test-model",
            version="1.0.0"
        )

        exporter = ExporterFactory.create_exporter(export_config)

        # Create mock model
        mock_model = Mock()
        mock_model.state_dict.return_value = {"weight": "value"}

        result = exporter.export(mock_model)
        assert result["success"] is True
        assert result["format"] == "pytorch"

        # Step 2: Deploy Model
        deployment_config = DeploymentConfig(
            model_path=result["path"],
            model_name="test-model",
            model_version="1.0.0",
            output_dir=temp_data_dir
        )

        deployment = DeploymentOrchestrator(deployment_config)
        package_path = deployment.package_model()

        assert Path(package_path).exists()

        # Step 3: Publish Model
        publish_result = deployment.publish_model()
        assert publish_result["status"] == "published"
        assert publish_result["model_name"] == "test-model"

    def test_end_to_end_data_flow(self, temp_data_dir):
        """Test complete data flow between all pipeline components."""
        from pipelines.data.src.ingestion import DataIngestion, IngestionConfig
        from pipelines.data.src.validation import DataValidation, ValidationConfig
        from pipelines.data.src.preprocessing import DataPreprocessor, PreprocessingConfig

        # Create sample data file
        csv_path = Path(temp_data_dir) / "e2e_data.csv"
        df = pd.DataFrame({
            "text": ["Hello world"] * 20,
            "label": [0, 1] * 10,
        })
        df.to_csv(csv_path, index=False)

        # Step 1: Ingest
        ingestion_config = IngestionConfig(
            source_type="csv",
            source_path=str(csv_path)
        )
        ingestion = DataIngestion()
        data = ingestion.load(ingestion_config)

        # Step 2: Validate
        validation_config = ValidationConfig(
            required_columns=["text", "label"],
            min_samples=10
        )
        validator = DataValidation(validation_config)
        is_valid = validator.validate(data)

        assert is_valid is True

        # Step 3: Get statistics
        stats = ingestion.get_statistics()
        assert stats["num_rows"] == 20
        assert stats["num_columns"] == 2


class TestInferenceServing:
    """Tests for inference serving integration."""

    @pytest.fixture
    def inference_config(self):
        """Create sample inference configuration."""
        from inference.src.engines import InferenceConfig
        return InferenceConfig(
            model_name="gpt2",
            batch_size=8,
            max_length=128,
            temperature=0.7,
            device="cpu"
        )

    def test_inference_engine_factory(self, inference_config):
        """Test inference engine factory creates correct engines."""
        from inference.src.engines import InferenceEngineFactory, InferenceConfig

        # Test transformers engine creation
        config = InferenceConfig(model_name="gpt2", device="cpu")
        engine = InferenceEngineFactory.create("transformers", config)

        assert engine is not None
        assert engine.config.model_name == "gpt2"

        # Test vllm engine creation (may raise ImportError if vllm not installed)
        try:
            vllm_engine = InferenceEngineFactory.create("vllm", config)
            assert vllm_engine is not None
        except (ImportError, ValueError):
            # vLLM not installed or unsupported
            pass

        # Test unknown engine raises error
        with pytest.raises(ValueError):
            InferenceEngineFactory.create("unknown", config)

    def test_model_loading_and_inference_mock(self, inference_config):
        """Test model loading and inference with mocking."""
        from inference.src.engines import TransformersEngine

        with patch("inference.src.engines.AutoModelForCausalLM") as mock_model:
            with patch("inference.src.engines.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.pad_token = None
                mock_tokenizer_instance.eos_token = "<|endoftext|>"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

                mock_model_instance = Mock()
                mock_model.from_pretrained.return_value = mock_model_instance

                # Create engine
                engine = TransformersEngine(inference_config)

                # Verify model was loaded
                assert engine.model is not None
                assert engine.tokenizer is not None

    def test_embedding_computation_pipeline(self, inference_config):
        """Test embedding computation pipeline."""
        from inference.src.engines import InferenceConfig

        config = InferenceConfig(model_name="gpt2", device="cpu")

        # Create a mock for embedding computation
        with patch("inference.src.engines.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
            mock_st.return_value = mock_model

            from inference.src.engines import TransformersEngine
            engine = TransformersEngine(config)

            # Mock the _embedding_model attribute
            engine._embedding_model = mock_model

            # Test embedding
            texts = ["Hello world", "Test sentence"]
            embeddings = engine.get_embeddings(texts)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384

    def test_batch_inference(self, inference_config):
        """Test batch inference processing."""
        from inference.src.engines import InferenceConfig, TransformersEngine

        config = InferenceConfig(
            model_name="gpt2",
            batch_size=4,
            device="cpu"
        )

        with patch("inference.src.engines.AutoModelForCausalLM") as mock_model:
            with patch("inference.src.engines.AutoTokenizer") as mock_tokenizer:
                # Setup mocks
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.pad_token = "<pad>"
                mock_tokenizer_instance.eos_token = "<|endoftext|>"
                mock_tokenizer_instance.encode.return_value = {"input_ids": [[1, 2, 3]]}
                mock_tokenizer_instance.decode.return_value = "Generated text"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

                mock_model_instance = Mock()
                mock_outputs = Mock()
                mock_outputs.cpu.return_value = Mock()
                mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
                mock_model.from_pretrained.return_value = mock_model_instance

                engine = TransformersEngine(config)

                # Test batch generation
                prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
                # Note: In real scenario, would need proper mocking of tokenizer.encode

                assert engine.config.batch_size == 4


class TestRAGPipeline:
    """Tests for RAG (Retrieval Augmented Generation) pipeline."""

    @pytest.fixture
    def rag_config(self):
        """Create RAG configuration."""
        from rag.src.core import RAGConfig
        return RAGConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=256,
            chunk_overlap=50,
            top_k=3
        )

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for RAG testing."""
        return [
            "Machine learning is a subset of artificial intelligence. "
            "It involves training algorithms on data to make predictions. "
            "Deep learning is a type of machine learning using neural networks.",
            "Natural language processing (NLP) is a field of AI. "
            "NLP focuses on enabling computers to understand human language. "
            "Applications include chatbots, translation, and sentiment analysis.",
            "Computer vision allows machines to interpret visual information. "
            "It uses convolutional neural networks for image recognition. "
            "Applications include self-driving cars and medical imaging.",
        ]

    def test_document_chunking(self, rag_config, sample_documents):
        """Test document chunking functionality."""
        from rag.src.core import DocumentChunker

        chunker = DocumentChunker(
            chunk_size=rag_config.chunk_size,
            overlap=rag_config.chunk_overlap
        )

        chunks = chunker.chunk(sample_documents[0])

        assert len(chunks) > 0
        # Each chunk should be within size limit
        for chunk in chunks:
            assert len(chunk) <= rag_config.chunk_size + rag_config.chunk_overlap

    def test_embedding_generation(self, rag_config):
        """Test embedding generation."""
        from rag.src.core import EmbeddingModel

        with patch("rag.src.core.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1] * 384])
            mock_st.return_value = mock_model

            embedding_model = EmbeddingModel(rag_config.embedding_model)
            texts = ["Test sentence for embedding"]
            embeddings = embedding_model.embed(texts)

            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384

    def test_vector_database_operations(self, rag_config):
        """Test vector database add and search operations."""
        from rag.src.core import VectorDatabase, Document

        vector_db = VectorDatabase()

        # Add documents
        docs = [
            Document(id="doc1", content="ML content", metadata={}, embedding=[0.1] * 384),
            Document(id="doc2", content="NLP content", metadata={}, embedding=[0.2] * 384),
        ]

        for doc in docs:
            vector_db.add_document(doc)

        assert len(vector_db.documents) == 2

        # Search
        query_embedding = [0.15] * 384
        results = vector_db.search(query_embedding, top_k=2)

        assert len(results) <= 2

    def test_full_rag_pipeline(self, rag_config, sample_documents):
        """Test full RAG pipeline: ingestion → chunking → embedding → retrieval."""
        from rag.src.core import RAGSystem

        # Initialize RAG system
        with patch("rag.src.core.SentenceTransformer") as mock_st:
            mock_model = Mock()
            # Generate dummy embeddings
            mock_model.encode.return_value = np.array([[0.1] * 384 for _ in range(10)])
            mock_st.return_value = mock_model

            rag = RAGSystem(rag_config)

            # Step 1: Add documents (triggers chunking + embedding)
            rag.add_documents(sample_documents)

            # Verify documents were added
            assert len(rag.vector_db.documents) > 0

            # Step 2: Retrieve relevant documents
            query = "What is machine learning?"
            results = rag.retrieve(query)

            # Should return relevant documents
            assert isinstance(results, list)

    def test_rag_query_pipeline(self, rag_config):
        """Test RAG query processing pipeline."""
        from rag.src.core import RAGSystem, Document

        with patch("rag.src.core.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1] * 384])
            mock_st.return_value = mock_model

            rag = RAGSystem(rag_config)

            # Manually add documents with embeddings
            docs = [
                Document(id="1", content="ML is AI", metadata={}, embedding=[0.9] * 384),
                Document(id="2", content="Deep learning is ML", metadata={}, embedding=[0.8] * 384),
                Document(id="3", content="Python is a language", metadata={}, embedding=[0.1] * 384),
            ]

            for doc in docs:
                rag.vector_db.add_document(doc)

            # Query for ML-related content
            query = "Tell me about machine learning"
            results = rag.retrieve(query)

            assert len(results) > 0
            # First result should be most relevant (ML is AI)
            assert results[0].id in ["1", "2"]


class TestAgentIntegration:
    """Tests for agent integration with memory and tools."""

    @pytest.fixture
    def agent_config(self):
        """Create agent configuration."""
        from agents.src.core import AgentConfig
        return AgentConfig(
            name="test-agent",
            model="gpt2",
            temperature=0.7,
            max_tokens=256
        )

    def test_agent_with_memory(self, agent_config):
        """Test agent with memory integration."""
        from agents.src.core import Agent
        from agents.src.langchain_memory_systems import ConversationBufferMemory, MessageRole

        # Create agent
        agent = Agent(agent_config)

        # Create memory
        memory = ConversationBufferMemory(max_messages=10)
        memory.add_message(MessageRole.USER, "Hello")
        memory.add_message(MessageRole.ASSISTANT, "Hi there!")

        # Add memory to agent
        agent.add_memory("conversation", memory.get_messages())

        # Verify memory was added
        assert len(agent.get_memory()) > 0

        # Run agent
        result = agent.run("What did we discuss?")
        assert result is not None
        assert "task" in result

    def test_agent_with_tools(self, agent_config):
        """Test agent with tools integration."""
        from agents.src.core import Agent, Tool
        from agents.src.langchain_tools_integration import FunctionTool, ToolCategory

        # Create agent
        agent = Agent(agent_config)

        # Create a test tool
        def calculate_sum(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool(
            calculate_sum,
            category=ToolCategory.COMPUTATION,
            tags=["math"]
        )

        # Add tool to agent
        agent.add_tool(tool)

        # Verify tool was added
        assert "calculate_sum" in agent.tools
        assert agent.tools["calculate_sum"].get_schema().name == "calculate_sum"

    def test_multi_turn_conversation(self, agent_config):
        """Test multi-turn conversation with memory."""
        from agents.src.core import Agent
        from agents.src.langchain_memory_systems import (
            ConversationBufferMemory, MessageRole, ConversationSummaryMemory
        )

        # Create agent with buffer memory
        agent = Agent(agent_config)
        memory = ConversationBufferMemory(max_messages=20)

        # Simulate multi-turn conversation
        conversation_turns = [
            (MessageRole.USER, "What is Python?"),
            (MessageRole.ASSISTANT, "Python is a programming language."),
            (MessageRole.USER, "What can I build with it?"),
            (MessageRole.ASSISTANT, "You can build web apps, ML models, and more."),
            (MessageRole.USER, "Tell me more about ML."),
        ]

        for role, content in conversation_turns:
            memory.add_message(role, content)

        # Add to agent memory
        agent.add_memory("conversation_history", memory.get_messages())

        # Verify conversation history
        stored_memory = agent.get_memory()
        assert len(stored_memory) > 0

        # Test memory stats
        stats = memory.get_stats()
        assert stats["total_messages"] == 5
        assert stats["user_messages"] == 3
        assert stats["assistant_messages"] == 2

    def test_agent_orchestrator_workflow(self):
        """Test agent orchestrator with multiple agents."""
        from agents.src.core import AgentOrchestrator, Agent, AgentConfig

        # Create orchestrator
        orchestrator = AgentOrchestrator()

        # Create agents
        agent1_config = AgentConfig(name="researcher", model="gpt2")
        agent2_config = AgentConfig(name="writer", model="gpt2")

        agent1 = Agent(agent1_config)
        agent2 = Agent(agent2_config)

        # Register agents
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Create workflow
        orchestrator.create_workflow("research_and_write", ["researcher", "writer"])

        # Execute workflow
        result = orchestrator.execute_workflow("research_and_write", "Test task")

        assert result["workflow"] == "research_and_write"
        assert result["agents_executed"] == 2
        assert len(result["results"]) == 2

    def test_agent_with_entity_memory(self, agent_config):
        """Test agent with entity tracking memory."""
        from agents.src.core import Agent
        from agents.src.langchain_memory_systems import EntityMemory

        agent = Agent(agent_config)
        entity_memory = EntityMemory()

        # Add entities
        entity_memory.add_entity(
            "Alice",
            {"age": "30", "role": "engineer"},
            description="Senior software engineer"
        )
        entity_memory.add_entity(
            "Bob",
            {"age": "25", "role": "designer"},
            description="UX designer"
        )

        # Add relationship
        entity_memory.add_relationship("Alice", "manages", "Bob")

        # Add to agent
        agent.add_memory("entities", entity_memory.get_entities())
        agent.add_memory("relationships", entity_memory.get_relationships())

        # Verify
        entities = entity_memory.get_entities()
        assert "Alice" in entities
        assert "Bob" in entities

        relationships = entity_memory.get_relationships()
        assert len(relationships) == 1


class TestModelInferenceIntegration:
    """Tests for model and inference integration."""

    def test_exported_model_loading(self):
        """Test loading of exported models."""
        from models.exported.core import ModelLoader, ExporterFactory
        from models.exported.config import LoadConfig, ExportConfig, ExportFormat
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Export a mock model first
            export_config = ExportConfig(
                export_format=ExportFormat.PYTORCH,
                output_dir=Path(temp_dir),
                model_name="test-model",
                version="1.0.0"
            )

            exporter = ExporterFactory.create_exporter(export_config)

            mock_model = Mock()
            mock_model.state_dict.return_value = {"layer.weight": [1, 2, 3]}
            export_result = exporter.export(mock_model)

            # Now load it
            load_config = LoadConfig(
                model_path=Path(temp_dir),
                format=ExportFormat.PYTORCH,
                model_name="test-model",
                version="1.0.0"
            )

            loader = ModelLoader(load_config)
            # Note: _load_pytorch would need actual files, test is conceptual
            assert loader.config.model_path.exists()

    def test_inference_with_exported_model(self):
        """Test inference using exported model."""
        from inference.src.engines import InferenceConfig, TransformersEngine

        config = InferenceConfig(
            model_name="gpt2",
            batch_size=1,
            max_length=50,
            device="cpu"
        )

        with patch("inference.src.engines.AutoModelForCausalLM") as mock_model:
            with patch("inference.src.engines.AutoTokenizer") as mock_tokenizer:
                mock_tokenizer_instance = Mock()
                mock_tokenizer_instance.pad_token = "<pad>"
                mock_tokenizer_instance.eos_token = "<|endoftext|>"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

                mock_model_instance = Mock()
                mock_model.from_pretrained.return_value = mock_model_instance

                engine = TransformersEngine(config)

                assert engine.config.model_name == "gpt2"
                assert engine.config.device == "cpu"


class TestEvaluationPipeline:
    """Tests for evaluation pipeline integration."""

    def test_benchmark_orchestrator_integration(self):
        """Test benchmark orchestrator with metrics computation."""
        from pipelines.evaluation.src.benchmark import BenchmarkConfig
        from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

        # Setup benchmark config
        config = BenchmarkConfig(
            model_path="gpt2",
            benchmarks=["mmlu"],
            batch_size=8,
            max_samples=10
        )

        # Setup metrics config
        metrics_config = MetricsConfig(
            task_benchmarks=True,
            latency_analysis=True
        )
        computer = MetricsComputer(metrics_config)

        # Mock results
        mock_results = {
            "mmlu": {"score": 45.0, "accuracy": 0.45, "num_samples": 10}
        }

        metrics = computer.compute_metrics(mock_results)

        assert "task_benchmarks" in metrics
        assert "latency" in metrics
        assert metrics["task_benchmarks"]["mmlu_score"] == 45.0

    def test_metrics_computation_integration(self):
        """Test metrics computation with various result types."""
        from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

        config = MetricsConfig(
            task_benchmarks=True,
            latency_analysis=True,
            safety_checks=True
        )
        computer = MetricsComputer(config)

        # Test with multiple benchmarks
        results = {
            "mmlu": {"score": 45.0, "accuracy": 0.45},
            "gsm8k": {"score": 30.0, "accuracy": 0.30},
            "hellaswag": {"score": 60.0, "accuracy": 0.60},
        }

        metrics = computer.compute_metrics(results)

        assert "task_benchmarks" in metrics
        assert metrics["task_benchmarks"]["average_score"] == 45.0  # (45+30+60)/3

    def test_regression_detection_integration(self):
        """Test regression detection with baseline comparison."""
        from pipelines.evaluation.src.metrics import RegressionDetector

        baseline = {
            "accuracy": 0.80,
            "f1_score": 0.75,
            "p50_latency_ms": 100.0,
        }

        detector = RegressionDetector(baseline)

        # Test with improved metrics
        current = {
            "accuracy": 0.85,  # Improved
            "f1_score": 0.70,  # Regression
            "p50_latency_ms": 120.0,  # Regression (higher latency)
        }

        report = detector.detect_regressions(current, threshold=0.05)

        assert "has_regressions" in report
        assert "num_regressions" in report
        assert "num_improvements" in report

    def test_evaluation_metrics_summary(self):
        """Test evaluation metrics summary generation."""
        from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

        config = MetricsConfig(task_benchmarks=True, latency_analysis=True)
        computer = MetricsComputer(config)

        # Compute metrics
        results = {"mmlu": {"score": 50.0}, "hellaswag": {"score": 70.0}}
        computer.compute_metrics(results)

        # Get summary
        summary = computer.get_metrics_summary()

        assert "total_metrics" in summary
        assert "metric_categories" in summary
        assert "details" in summary


class TestCrossComponentIntegration:
    """Tests for cross-component integration scenarios."""

    def test_training_to_inference_pipeline(self):
        """Test complete training to inference pipeline."""
        from pipelines.training.src.orchestrator import TrainingOrchestrator, TrainingConfig
        from inference.src.engines import InferenceConfig

        # Setup training
        training_config = TrainingConfig(
            model_name="gpt2",
            dataset_path="./data",
            output_dir="./output",
            num_epochs=1
        )
        training = TrainingOrchestrator(training_config)

        # Get training stats
        stats = training.get_training_stats()

        # Setup inference with same model
        inference_config = InferenceConfig(
            model_name=training_config.model_name,
            batch_size=8,
            device="cpu"
        )

        # Verify model names match
        assert stats["model_name"] == inference_config.model_name

    def test_rag_to_agent_integration(self):
        """Test RAG system integration with agents."""
        from rag.src.core import RAGSystem, RAGConfig
        from agents.src.core import Agent, AgentConfig

        with patch("rag.src.core.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1] * 384])
            mock_st.return_value = mock_model

            # Setup RAG
            rag_config = RAGConfig(chunk_size=256, top_k=3)
            rag = RAGSystem(rag_config)

            # Add documents
            rag.add_documents(["Test document content"])

            # Setup Agent
            agent_config = AgentConfig(name="rag-agent", model="gpt2")
            agent = Agent(agent_config)

            # Integrate RAG with Agent
            agent.add_memory("retrieved_docs", rag.retrieve("test query"))

            # Verify integration
            assert len(agent.get_memory()) > 0

    def test_data_to_rag_pipeline(self):
        """Test data pipeline integration with RAG."""
        from pipelines.data.src.ingestion import DataIngestion, IngestionConfig
        from rag.src.core import RAGSystem, RAGConfig
        import tempfile
        import pandas as pd

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data
            csv_path = Path(temp_dir) / "rag_data.csv"
            df = pd.DataFrame({
                "text": ["Document 1 content", "Document 2 content", "Document 3 content"],
                "category": ["A", "B", "A"]
            })
            df.to_csv(csv_path, index=False)

            # Ingest data
            ingestion_config = IngestionConfig(
                source_type="csv",
                source_path=str(csv_path)
            )
            ingestion = DataIngestion()
            data = ingestion.load(ingestion_config)

            # Feed into RAG
            with patch("rag.src.core.SentenceTransformer") as mock_st:
                mock_model = Mock()
                mock_model.encode.return_value = np.array([[0.1] * 384 for _ in range(len(data))])
                mock_st.return_value = mock_model

                rag_config = RAGConfig()
                rag = RAGSystem(rag_config)

                rag.add_documents(data["text"].tolist())

                assert len(rag.vector_db.documents) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
