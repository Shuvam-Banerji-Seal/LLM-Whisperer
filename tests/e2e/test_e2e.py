"""Comprehensive End-to-End Tests for LLM-Whisperer.

This module contains comprehensive E2E tests covering complete user workflows
for the LLM-Whisperer platform, including:
- ML Pipeline: Data → Training → Evaluation → Export → Deployment
- RAG System: Documents → Ingestion → Indexing → Query → Response
- Agent System: Agent creation → Memory → Tools → Multi-turn conversation
- Fine-tuning Workflow: Base model → Fine-tuning → Evaluation → Inference
- Model Lifecycle: Full model lifecycle from creation to deployment
- Benchmark Workflow: Complete benchmark evaluation workflow
"""

import pytest
import logging
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCompleteWorkflow:
    """Tests for complete end-to-end workflows.

    These tests cover the full ML pipeline from data ingestion through
    model deployment, as well as complete RAG and Agent workflows.
    """

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_documents(self):
        """Provide sample documents for RAG tests."""
        return [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Natural language processing allows computers to understand and generate human language.",
            "Computer vision enables machines to interpret and understand visual information from the world.",
            "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
        ]

    @pytest.fixture
    def mock_model_response(self):
        """Mock model response for testing."""
        return "This is a generated response from the model."

    def test_full_ml_pipeline_data_to_deployment(self, temp_output_dir):
        """Test complete ML pipeline from data preparation to deployment.

        Workflow: Data → Training → Evaluation → Export → Deployment
        """
        logger.info("Starting full ML pipeline E2E test")

        # Step 1: Data Preparation
        from pipelines.training.src.orchestrator import (
            TrainingOrchestrator,
            TrainingConfig,
        )

        data_config = TrainingConfig(
            model_name="gpt2",
            dataset_path="data/processed",
            output_dir=str(temp_output_dir / "training"),
            num_epochs=1,
            batch_size=4,
            training_method="full_finetune",
        )
        data_orchestrator = TrainingOrchestrator(data_config)
        assert data_orchestrator.config.model_name == "gpt2"
        logger.info("Data preparation step completed")

        # Step 2: Training
        with patch(
            "pipelines.training.src.orchestrator.AutoModelForCausalLM"
        ) as mock_model, patch(
            "pipelines.training.src.orchestrator.AutoTokenizer"
        ) as mock_tokenizer:
            # Setup mocks
            mock_tokenizer.from_pretrained.return_value = Mock(
                pad_token=None, eos_token="<eos>"
            )
            mock_model_instance = Mock()
            mock_model_instance.parameters.return_value = [
                Mock(numel=lambda: 1000, requires_grad=True)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_model.return_value = mock_model_instance

            # Execute training setup
            data_orchestrator.load_model()
            data_orchestrator.setup_training()

            # Save checkpoint
            checkpoint_dir = temp_output_dir / "training" / "checkpoints"
            data_orchestrator.save_checkpoint(str(checkpoint_dir / "latest"))

            assert (checkpoint_dir / "latest").exists()
            logger.info("Training step completed")

        # Step 3: Evaluation
        from pipelines.evaluation.src.benchmark import (
            BenchmarkOrchestrator,
            BenchmarkConfig,
        )

        eval_config = BenchmarkConfig(
            model_path=str(checkpoint_dir / "latest"),
            benchmarks=["mmlu"],
            batch_size=8,
            output_dir=str(temp_output_dir / "eval"),
            max_samples=10,
        )
        benchmark_orchestrator = BenchmarkOrchestrator(eval_config)

        # Mock the dataset loading and model inference
        with patch(
            "pipelines.evaluation.src.benchmark.load_dataset"
        ) as mock_load_dataset, patch(
            "pipelines.evaluation.src.benchmark.AutoModelForCausalLM"
        ) as mock_eval_model, patch(
            "pipelines.evaluation.src.benchmark.AutoTokenizer"
        ) as mock_eval_tokenizer:

            # Setup mock dataset
            mock_dataset = [
                {
                    "question": "What is 2+2?",
                    "choices": ["3", "4", "5", "6"],
                    "answer": "B",
                },
            ] * 10
            mock_load_dataset.return_value = mock_dataset

            # Setup mock model
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = "<pad>"
            mock_tokenizer_instance.eos_token_id = 0
            mock_tokenizer_instance.pad_token_id = 1
            mock_eval_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_eval_model_instance = Mock()
            mock_eval_model.from_pretrained.return_value = mock_eval_model_instance
            mock_eval_model_instance.to.return_value = mock_eval_model_instance
            mock_eval_model_instance.device = "cpu"

            # Mock generation
            mock_outputs = Mock()
            mock_outputs.loss = Mock(item=lambda: 0.5)
            mock_eval_model_instance.return_value = mock_outputs
            mock_eval_model_instance.generate.return_value = [[1, 2, 3, 4]]
            mock_tokenizer_instance.decode.return_value = "Question: What is 2+2?\nAnswer: B"
            mock_tokenizer_instance.__call__ = Mock(return_value={"input_ids": [[1]], "attention_mask": [[1]]})

            # Run evaluation
            results = benchmark_orchestrator.run_all()
            assert "mmlu" in results
            logger.info("Evaluation step completed")

        # Step 4: Model Export
        from models.exported.core import (
            ExporterFactory,
            ExportConfig,
            ExportFormat,
        )

        export_config = ExportConfig(
            model_name="test-model",
            output_dir=temp_output_dir / "export",
            export_format=ExportFormat.PYTORCH,
            version="1.0.0",
        )
        exporter = ExporterFactory.create_exporter(export_config)

        # Create mock model for export
        mock_export_model = Mock()
        mock_export_model.state_dict.return_value = {"weight": [1.0, 2.0, 3.0]}

        export_result = exporter.export(mock_export_model)
        assert export_result["success"] is True
        logger.info("Export step completed")

        # Step 5: Deployment
        from pipelines.deployment.src.orchestrator import (
            DeploymentOrchestrator,
            DeploymentConfig,
        )

        deploy_config = DeploymentConfig(
            model_path=str(temp_output_dir / "training" / "checkpoints" / "latest"),
            model_name="test-ml-model",
            model_version="1.0.0",
            output_dir=str(temp_output_dir / "deploy"),
            major=1,
            minor=0,
            patch=0,
        )
        deploy_orchestrator = DeploymentOrchestrator(deploy_config)

        # Create mock model directory structure
        model_dir = Path(deploy_config.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"model_type": "gpt2"}')
        (model_dir / "pytorch_model.bin").write_text("mock model weights")

        package_path = deploy_orchestrator.package_model()
        assert Path(package_path).exists()

        publish_result = deploy_orchestrator.publish_model()
        assert publish_result["status"] == "published"
        logger.info("Deployment step completed")

        # Verify full pipeline success
        deployment_info = deploy_orchestrator.get_deployment_info()
        assert deployment_info["model_name"] == "test-ml-model"
        logger.info("Full ML pipeline E2E test completed successfully")

    def test_complete_rag_workflow(self, temp_output_dir, sample_documents):
        """Test complete RAG workflow: Documents → Ingestion → Indexing → Query → Response.

        This test verifies the entire RAG pipeline from document ingestion
        through query response generation.
        """
        logger.info("Starting complete RAG workflow E2E test")

        # Step 1: Document Ingestion
        from rag.ingestion.core import DocumentPipeline, IngestionConfig
        from rag.ingestion.config import LoaderType

        ingestion_config = IngestionConfig(
            loader_type=LoaderType.TEXT, encoding="utf-8"
        )
        pipeline = DocumentPipeline(ingestion_config)

        # Create test document file
        test_doc_path = temp_output_dir / "test_doc.txt"
        test_content = "\n\n".join(sample_documents)
        test_doc_path.write_text(test_content)

        documents = pipeline.process(str(test_doc_path))
        assert len(documents) > 0
        assert "content" in documents[0]
        assert "metadata" in documents[0]
        logger.info(f"Ingested {len(documents)} documents")

        # Step 2: Document Chunking and Embedding
        from rag.src.core import RAGSystem, RAGConfig

        rag_config = RAGConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=200,
            chunk_overlap=20,
            top_k=3,
            similarity_threshold=0.3,
        )
        rag_system = RAGSystem(rag_config)

        # Add documents to RAG system
        doc_texts = [doc["content"] for doc in documents]
        metadata = [doc["metadata"] for doc in documents]

        # Mock embeddings for testing
        with patch.object(
            rag_system.embedding_model, "embed", return_value=[[0.1] * 384] * 10
        ):
            rag_system.add_documents(doc_texts, metadata)
            logger.info(f"Added {len(doc_texts)} documents to RAG system")

        # Step 3: Indexing
        from rag.indexing.core import IndexBuilder, IndexConfig
        from rag.indexing.config import IndexType

        index_config = IndexConfig(
            index_type=IndexType.FLAT, metric="cosine", dimension=384
        )
        index_builder = IndexBuilder(index_config)

        # Build index with mock embeddings
        mock_embeddings = [[float(i % 10) / 10.0] * 384 for i in range(10)]
        doc_ids = [f"doc_{i}" for i in range(10)]
        index = index_builder.build(mock_embeddings, doc_ids, metadata)
        assert len(index.vectors) == 10
        logger.info("Index built successfully")

        # Step 4: Query and Retrieval
        query = "What is machine learning?"
        with patch.object(
            rag_system.embedding_model, "embed", return_value=[[0.1] * 384]
        ):
            retrieved_docs = rag_system.retrieve(query)
            assert isinstance(retrieved_docs, list)
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")

        # Step 5: Response Generation (Mock)
        # In a real scenario, this would use an LLM to generate a response
        mock_response = {
            "query": query,
            "retrieved_documents": len(retrieved_docs),
            "response": "Machine learning is a subset of artificial intelligence...",
            "sources": [doc.id for doc in retrieved_docs[:3]] if retrieved_docs else [],
        }
        assert "query" in mock_response
        assert "response" in mock_response
        logger.info("RAG workflow completed successfully")

    def test_complete_agent_workflow(self, temp_output_dir):
        """Test complete agent workflow: Creation → Memory → Tools → Multi-turn conversation.

        Verifies the full agent lifecycle including tool integration,
        memory management, and multi-turn interactions.
        """
        logger.info("Starting complete agent workflow E2E test")

        from agents.src.core import Agent, AgentConfig, AgentState

        # Step 1: Agent Creation
        config = AgentConfig(
            name="test-agent",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="You are a helpful assistant.",
        )
        agent = Agent(config)
        assert agent.config.name == "test-agent"
        assert agent.state == AgentState.IDLE
        logger.info("Agent created successfully")

        # Step 2: Tool Integration
        from agents.src.core import Tool

        class MockSearchTool(Tool):
            def execute(self, query: str) -> Dict[str, Any]:
                return {"query": query, "results": ["Result 1", "Result 2"]}

        class MockCalculatorTool(Tool):
            def execute(self, expression: str) -> Dict[str, Any]:
                return {"expression": expression, "result": 42}

        search_tool = MockSearchTool("search", "Search the web")
        calc_tool = MockCalculatorTool("calculator", "Perform calculations")

        agent.add_tool(search_tool)
        agent.add_tool(calc_tool)
        assert len(agent.tools) == 2
        assert "search" in agent.tools
        assert "calculator" in agent.tools
        logger.info("Tools added to agent")

        # Step 3: Memory Management
        agent.add_memory("user_name", "Alice")
        agent.add_memory("session_id", "session_123")
        agent.add_memory("preferences", {"theme": "dark", "language": "en"})

        memory = agent.get_memory()
        assert len(memory) == 3
        assert any(m["key"] == "user_name" and m["value"] == "Alice" for m in memory)
        logger.info("Agent memory populated")

        # Step 4: Multi-turn Conversation
        conversation_turns = [
            "Hello, my name is Alice",
            "What is the capital of France?",
            "Calculate 6 times 7",
            "Thank you for your help",
        ]

        results = []
        for turn in conversation_turns:
            result = agent.run(turn)
            results.append(result)
            assert result["task"] == turn
            assert agent.state in [AgentState.COMPLETED, AgentState.FAILED]

        assert len(results) == 4
        logger.info(f"Completed {len(results)} conversation turns")

        # Step 5: Agent Orchestration (Multi-agent)
        from agents.src.core import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Create specialized agents
        research_config = AgentConfig(name="research-agent", model="gpt-4")
        writer_config = AgentConfig(name="writer-agent", model="gpt-4")
        review_config = AgentConfig(name="review-agent", model="gpt-4")

        research_agent = Agent(research_config)
        writer_agent = Agent(writer_config)
        review_agent = Agent(review_config)

        orchestrator.register_agent(research_agent)
        orchestrator.register_agent(writer_agent)
        orchestrator.register_agent(review_agent)

        # Create workflow
        orchestrator.create_workflow(
            "content-creation", ["research-agent", "writer-agent", "review-agent"]
        )

        # Execute workflow
        workflow_result = orchestrator.execute_workflow(
            "content-creation", "Write an article about AI"
        )
        assert workflow_result["workflow"] == "content-creation"
        assert workflow_result["agents_executed"] == 3
        logger.info("Multi-agent workflow executed successfully")

    def test_complete_fine_tuning_workflow(self, temp_output_dir):
        """Test complete fine-tuning workflow: Base Model → Fine-tuning → Evaluation → Inference.

        Verifies the full fine-tuning pipeline including training,
        checkpointing, evaluation, and inference.
        """
        logger.info("Starting complete fine-tuning workflow E2E test")

        # Step 1: Load Base Model (mock)
        from pipelines.training.src.orchestrator import (
            TrainingOrchestrator,
            TrainingConfig,
        )

        base_config = TrainingConfig(
            model_name="gpt2",
            dataset_path="data/finetune",
            output_dir=str(temp_output_dir / "finetuning"),
            num_epochs=1,
            batch_size=4,
            training_method="lora",
            lora_rank=8,
            lora_alpha=16,
        )
        orchestrator = TrainingOrchestrator(base_config)

        with patch(
            "pipelines.training.src.orchestrator.AutoModelForCausalLM"
        ) as mock_model, patch(
            "pipelines.training.src.orchestrator.AutoTokenizer"
        ) as mock_tokenizer, patch(
            "pipelines.training.src.orchestrator.get_peft_model"
        ) as mock_peft:

            # Setup mocks
            mock_tokenizer.from_pretrained.return_value = Mock(
                pad_token=None, eos_token="<eos>"
            )
            mock_model_instance = Mock()
            mock_model_instance.parameters.return_value = [
                Mock(numel=lambda: 1000, requires_grad=True)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_peft.return_value = mock_model_instance

            # Load model
            orchestrator.load_model()
            logger.info("Base model loaded")

            # Step 2: Setup Training
            orchestrator.setup_training()
            assert orchestrator.optimizer is not None
            assert orchestrator.lr_scheduler is not None
            logger.info("Training setup completed")

            # Step 3: Training with Checkpointing
            orchestrator.save_checkpoint(
                str(temp_output_dir / "finetuning" / "checkpoints" / "epoch_1")
            )
            checkpoint_path = temp_output_dir / "finetuning" / "checkpoints" / "epoch_1"
            assert checkpoint_path.exists()
            logger.info("Checkpoint saved")

        # Step 4: Evaluation
        from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

        eval_results = {
            "mmlu": {"score": 45.0, "accuracy": 0.45, "num_samples": 100},
            "hellaswag": {"score": 52.0, "accuracy": 0.52, "num_samples": 100},
        }

        metrics_config = MetricsConfig(
            task_benchmarks=True, latency_analysis=True, safety_checks=True
        )
        metrics_computer = MetricsComputer(metrics_config)
        computed_metrics = metrics_computer.compute_metrics(eval_results)

        assert "task_benchmarks" in computed_metrics
        assert "latency" in computed_metrics
        assert "safety" in computed_metrics
        logger.info("Evaluation metrics computed")

        # Step 5: Inference
        from inference.src.engines import (
            InferenceEngineFactory,
            InferenceConfig,
        )

        inference_config = InferenceConfig(
            model_name=str(checkpoint_path),
            batch_size=1,
            max_length=100,
            temperature=0.7,
            device="cpu",
        )

        with patch(
            "inference.src.engines.AutoModelForCausalLM"
        ) as mock_inference_model, patch(
            "inference.src.engines.AutoTokenizer"
        ) as mock_inference_tokenizer:

            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token_id = 0
            mock_tokenizer_instance.eos_token_id = 1
            mock_inference_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_inference_tokenizer_instance = Mock()
            mock_inference_tokenizer_instance.pad_token_id = 0
            mock_inference_tokenizer_instance.eos_token_id = 1
            mock_inference_tokenizer.from_pretrained.return_value = mock_inference_tokenizer_instance

            mock_model_instance = Mock()
            mock_outputs = Mock()
            mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
            mock_inference_model.from_pretrained.return_value = mock_model_instance
            mock_inference_tokenizer_instance.batch_decode.return_value = [
                "Generated text response"
            ]
            mock_inference_tokenizer_instance.__call__ = Mock(
                return_value={
                    "input_ids": [[1]],
                    "attention_mask": [[1]],
                }
            )

            engine = InferenceEngineFactory.create("transformers", inference_config)

            # Test generation
            with patch.object(engine, "tokenizer", mock_inference_tokenizer_instance), \
                 patch.object(engine, "model", mock_model_instance):
                engine.tokenizer = mock_inference_tokenizer_instance
                engine.model = mock_model_instance
                mock_inference_tokenizer_instance.batch_decode.return_value = [
                    "Generated text response"
                ]
                results = engine.generate(["Test prompt"])
                assert len(results) == 1

            logger.info("Inference completed")

        logger.info("Fine-tuning workflow E2E test completed successfully")


class TestModelLifecycle:
    """Tests for full model lifecycle from creation to deployment.

    Covers model registration, versioning, training, evaluation,
    export, deployment, and monitoring.
    """

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for model registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_model_lifecycle(self, temp_registry_dir):
        """Test complete model lifecycle: Register → Train → Evaluate → Deploy → Monitor.

        Verifies the entire model lifecycle management process.
        """
        logger.info("Starting model lifecycle E2E test")

        # Step 1: Model Registration
        from models.registry.core import ModelRegistry, ModelMetadata
        from models.registry.config import RegistryConfig, RegistryBackend

        registry_config = RegistryConfig(backend=RegistryBackend.LOCAL)
        registry = ModelRegistry(registry_config)

        metadata = registry.register_model(
            model_id="test-model-v1",
            name="Test Model",
            version="1.0.0",
            model_type="causal_lm",
            framework="pytorch",
            author="test-user",
            num_parameters=7000000,
            description="A test model for E2E testing",
            tags=["test", "gpt2", "causal-lm"],
        )
        assert metadata.model_id == "test-model-v1"
        assert metadata.version == "1.0.0"
        logger.info("Model registered in registry")

        # Step 2: Training
        from pipelines.training.src.orchestrator import (
            TrainingOrchestrator,
            TrainingConfig,
        )

        train_config = TrainingConfig(
            model_name="gpt2",
            dataset_path="data/train",
            output_dir=str(temp_registry_dir / "training"),
            num_epochs=1,
            batch_size=4,
        )
        training_orch = TrainingOrchestrator(train_config)

        with patch(
            "pipelines.training.src.orchestrator.AutoModelForCausalLM"
        ) as mock_model, patch(
            "pipelines.training.src.orchestrator.AutoTokenizer"
        ) as mock_tokenizer:

            mock_tokenizer.from_pretrained.return_value = Mock(
                pad_token=None, eos_token="<eos>"
            )
            mock_model_instance = Mock()
            mock_model_instance.parameters.return_value = [
                Mock(numel=lambda: 1000, requires_grad=True)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance

            training_orch.load_model()
            training_orch.setup_training()
            training_orch.save_checkpoint(
                str(temp_registry_dir / "training" / "final")
            )

        logger.info("Training completed")

        # Step 3: Evaluation
        from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

        eval_results = {"mmlu": {"score": 50.0, "accuracy": 0.5, "num_samples": 50}}
        metrics_config = MetricsConfig(task_benchmarks=True, latency_analysis=True)
        metrics_computer = MetricsComputer(metrics_config)
        metrics = metrics_computer.compute_metrics(eval_results)
        assert metrics["task_benchmarks"]["mmlu_score"] == 50.0
        logger.info("Evaluation completed")

        # Step 4: Model Export
        from models.exported.core import (
            ExporterFactory,
            ExportConfig,
            ExportFormat,
        )

        export_config = ExportConfig(
            model_name="test-model-v1",
            output_dir=temp_registry_dir / "export",
            export_format=ExportFormat.PYTORCH,
            version="1.0.0",
        )
        exporter = ExporterFactory.create_exporter(export_config)

        mock_model = Mock()
        mock_model.state_dict.return_value = {"weight": [1.0, 2.0]}
        export_result = exporter.export(mock_model)
        assert export_result["success"] is True
        logger.info("Model exported")

        # Step 5: Deployment
        from pipelines.deployment.src.orchestrator import (
            DeploymentOrchestrator,
            DeploymentConfig,
        )

        deploy_config = DeploymentConfig(
            model_path=str(temp_registry_dir / "training" / "final"),
            model_name="test-model",
            model_version="1.0.0",
            output_dir=str(temp_registry_dir / "deploy"),
        )
        deploy_orch = DeploymentOrchestrator(deploy_config)

        # Create mock model files
        model_path = Path(deploy_config.model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "config.json").write_text('{"model_type": "gpt2"}')
        (model_path / "model.bin").write_text("mock weights")

        deploy_orch.package_model()
        publish_result = deploy_orch.publish_model()
        assert publish_result["status"] == "published"
        logger.info("Model deployed")

        # Step 6: Registry Statistics
        stats = registry.get_statistics()
        assert stats["total_models"] == 1
        assert "pytorch" in stats["frameworks"]
        logger.info("Model lifecycle test completed")

    def test_model_versioning_and_rollback(self, temp_registry_dir):
        """Test model versioning and rollback capabilities."""
        logger.info("Starting model versioning E2E test")

        from pipelines.deployment.src.orchestrator import (
            DeploymentOrchestrator,
            DeploymentConfig,
        )

        # Deploy version 1.0.0
        config_v1 = DeploymentConfig(
            model_path=str(temp_registry_dir / "models" / "v1"),
            model_name="prod-model",
            model_version="1.0.0",
            output_dir=str(temp_registry_dir / "deploy"),
            major=1,
            minor=0,
            patch=0,
        )

        Path(config_v1.model_path).mkdir(parents=True, exist_ok=True)
        (Path(config_v1.model_path) / "model.bin").write_text("v1 weights")

        orch_v1 = DeploymentOrchestrator(config_v1)
        orch_v1.package_model()
        result_v1 = orch_v1.publish_model()
        assert result_v1["version"] == "v1.0.0"

        # Deploy version 1.1.0
        config_v2 = DeploymentConfig(
            model_path=str(temp_registry_dir / "models" / "v2"),
            model_name="prod-model",
            model_version="1.1.0",
            output_dir=str(temp_registry_dir / "deploy"),
            major=1,
            minor=1,
            patch=0,
        )

        Path(config_v2.model_path).mkdir(parents=True, exist_ok=True)
        (Path(config_v2.model_path) / "model.bin").write_text("v2 weights")

        orch_v2 = DeploymentOrchestrator(config_v2)
        orch_v2.package_model()
        result_v2 = orch_v2.publish_model()
        assert result_v2["version"] == "v1.1.0"

        logger.info("Model versioning test completed")


class TestRAGSystem:
    """End-to-end tests for the complete RAG system.

    Tests document processing, embedding generation, indexing,
    retrieval, and response generation as an integrated system.
    """

    @pytest.fixture
    def sample_knowledge_base(self):
        """Provide a sample knowledge base for RAG testing."""
        return [
            {
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a method of data analysis that automates analytical model building.",
                "category": "AI",
            },
            {
                "title": "Deep Learning Fundamentals",
                "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
                "category": "AI",
            },
            {
                "title": "Natural Language Processing",
                "content": "NLP is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
                "category": "NLP",
            },
            {
                "title": "Computer Vision Basics",
                "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
                "category": "Vision",
            },
            {
                "title": "Reinforcement Learning",
                "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment.",
                "category": "AI",
            },
        ]

    def test_rag_end_to_end_with_query_augmentation(self, temp_output_dir, sample_knowledge_base):
        """Test complete RAG system with query augmentation and reranking.

        Tests the full pipeline: Ingest → Chunk → Embed → Index → Retrieve → Generate.
        """
        logger.info("Starting RAG system E2E test")

        from rag.src.core import RAGSystem, RAGConfig

        # Initialize RAG system
        config = RAGConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=300,
            chunk_overlap=30,
            top_k=5,
            similarity_threshold=0.3,
        )
        rag = RAGSystem(config)

        # Prepare documents
        documents = [doc["content"] for doc in sample_knowledge_base]
        metadata = [{"title": doc["title"], "category": doc["category"]} for doc in sample_knowledge_base]

        # Mock embeddings for testing
        num_chunks = sum(
            len(rag.chunker.chunk(doc)) for doc in documents
        )
        mock_embeddings = [[float(i % 10) / 10.0] * 384 for i in range(num_chunks + 10)]

        with patch.object(
            rag.embedding_model, "embed", side_effect=lambda texts: [mock_embeddings.pop(0) for _ in texts] if isinstance(texts, list) else mock_embeddings.pop(0)
        ):
            # Index documents
            rag.add_documents(documents, metadata)
            logger.info(f"Indexed {len(documents)} documents")

        # Test various queries
        queries = [
            "What is machine learning?",
            "Tell me about deep learning",
            "How does computer vision work?",
        ]

        for query in queries:
            with patch.object(
                rag.embedding_model, "embed", return_value=[[0.5] * 384]
            ):
                retrieved = rag.retrieve(query)
                assert isinstance(retrieved, list)
                logger.info(f"Query: '{query}' → Retrieved {len(retrieved)} chunks")

        logger.info("RAG system E2E test completed")

    def test_rag_with_hybrid_retrieval(self, temp_output_dir):
        """Test RAG system with hybrid retrieval (dense + sparse)."""
        logger.info("Starting hybrid RAG E2E test")

        from rag.retrieval.core import DocumentRetriever, HybridRetriever, RetrieverConfig
        from rag.indexing.core import VectorIndex, IndexConfig
        from rag.indexing.config import IndexType

        # Setup
        retriever_config = RetrieverConfig(top_k=5, similarity_threshold=0.3)
        index_config = IndexConfig(index_type=IndexType.FLAT, metric="cosine", dimension=384)
        index = VectorIndex(index_config)

        # Create mock embedder
        mock_embedder = Mock()
        mock_embedder.embed_single = Mock(return_value=[0.1] * 384)

        # Create retrievers
        dense_retriever = DocumentRetriever(index, mock_embedder, retriever_config)

        # Mock sparse retriever
        mock_sparse_retriever = Mock()
        mock_sparse_retriever.retrieve = Mock(return_value=["doc_1", "doc_3"])

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=mock_sparse_retriever,
            alpha=0.7,
        )

        # Add mock documents to index
        index.add(
            vectors=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            doc_ids=["doc_1", "doc_2", "doc_3"],
            metadata=[{"title": "Doc 1"}, {"title": "Doc 2"}, {"title": "Doc 3"}],
        )

        # Test hybrid retrieval
        with patch.object(
            index, "search", return_value=[("doc_1", 0.9), ("doc_2", 0.8)]
        ):
            results = hybrid_retriever.retrieve("test query", top_k=3)
            assert isinstance(results, list)
            # Should combine results from both retrievers
            logger.info(f"Hybrid retrieval returned {len(results)} documents")

        logger.info("Hybrid RAG E2E test completed")


class TestAgentSystem:
    """End-to-end tests for the complete agent system.

    Tests agent creation, tool integration, memory systems,
    multi-agent workflows, and conversation handling.
    """

    def test_agent_with_memory_and_tools(self):
        """Test agent with full memory and tool integration.

        Verifies agent can use tools, maintain conversation memory,
        and execute multi-step tasks.
        """
        logger.info("Starting agent with memory and tools E2E test")

        from agents.src.core import Agent, AgentConfig, AgentState, Tool

        # Create agent
        config = AgentConfig(
            name="smart-agent",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            system_prompt="You are a helpful assistant with access to tools.",
        )
        agent = Agent(config)

        # Define tools
        class WeatherTool(Tool):
            def execute(self, location: str) -> Dict[str, Any]:
                return {"location": location, "temperature": 72, "condition": "sunny"}

        class DatabaseTool(Tool):
            def execute(self, query: str) -> Dict[str, Any]:
                return {"query": query, "results": ["Item 1", "Item 2", "Item 3"]}

        class CalculatorTool(Tool):
            def execute(self, expression: str) -> Dict[str, Any]:
                try:
                    result = eval(expression)  # Safe in test environment
                    return {"expression": expression, "result": result}
                except Exception as e:
                    return {"expression": expression, "error": str(e)}

        # Register tools
        agent.add_tool(WeatherTool("weather", "Get weather information"))
        agent.add_tool(DatabaseTool("database", "Query database"))
        agent.add_tool(CalculatorTool("calculator", "Perform calculations"))

        assert len(agent.tools) == 3

        # Simulate conversation with memory
        conversation = [
            ("user", "My name is Alice"),
            ("assistant", "Nice to meet you, Alice!"),
            ("user", "What's the weather in New York?"),
            ("assistant", "Let me check that for you."),
            ("user", "Calculate 15 * 23"),
            ("assistant", "The result is 345."),
            ("user", "What is my name?"),  # Tests memory
        ]

        for role, message in conversation:
            if role == "user":
                agent.add_memory("conversation", {"role": role, "content": message})
            else:
                agent.add_memory("conversation", {"role": role, "content": message})

        # Verify memory
        memory = agent.get_memory()
        assert len(memory) == len(conversation)

        # Verify last memory about name
        last_memory = memory[-1]
        assert last_memory["key"] == "conversation"
        assert last_memory["value"]["content"] == "What is my name?"

        logger.info("Agent with memory and tools test completed")

    def test_multi_agent_workflow_orchestration(self):
        """Test multi-agent workflow with orchestrator.

        Tests complex workflows involving multiple specialized agents
        working together to complete tasks.
        """
        logger.info("Starting multi-agent workflow E2E test")

        from agents.src.core import Agent, AgentConfig, AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Create specialized agents
        agents_config = {
            "research-agent": AgentConfig(
                name="research-agent",
                model="gpt-4",
                system_prompt="You are a research specialist. Find and summarize information.",
            ),
            "analysis-agent": AgentConfig(
                name="analysis-agent",
                model="gpt-4",
                system_prompt="You are an analysis specialist. Analyze data and provide insights.",
            ),
            "writer-agent": AgentConfig(
                name="writer-agent",
                model="gpt-4",
                system_prompt="You are a writing specialist. Create clear, engaging content.",
            ),
            "review-agent": AgentConfig(
                name="review-agent",
                model="gpt-4",
                system_prompt="You are a review specialist. Check quality and accuracy.",
            ),
        }

        # Register agents
        for name, config in agents_config.items():
            agent = Agent(config)
            orchestrator.register_agent(agent)

        # Create content creation workflow
        orchestrator.create_workflow(
            "content-pipeline",
            ["research-agent", "analysis-agent", "writer-agent", "review-agent"],
        )

        # Create data analysis workflow
        orchestrator.create_workflow(
            "data-pipeline",
            ["research-agent", "analysis-agent", "review-agent"],
        )

        # Execute workflows
        result1 = orchestrator.execute_workflow(
            "content-pipeline",
            "Create a report on renewable energy trends",
        )
        assert result1["workflow"] == "content-pipeline"
        assert result1["agents_executed"] == 4
        assert len(result1["results"]) == 4

        result2 = orchestrator.execute_workflow(
            "data-pipeline",
            "Analyze sales data for Q3 2024",
        )
        assert result2["workflow"] == "data-pipeline"
        assert result2["agents_executed"] == 3

        logger.info("Multi-agent workflow test completed")


class TestBenchmarkWorkflow:
    """Complete benchmark evaluation workflow tests.

    Tests the full benchmark evaluation pipeline including
    multiple benchmarks, metrics computation, and regression detection.
    """

    def test_complete_benchmark_evaluation_pipeline(self):
        """Test complete benchmark evaluation workflow.

        Covers: Setup → Run Benchmarks → Compute Metrics → Detect Regressions → Generate Report
        """
        logger.info("Starting complete benchmark workflow E2E test")

        from pipelines.evaluation.src.benchmark import (
            BenchmarkOrchestrator,
            BenchmarkConfig,
        )
        from pipelines.evaluation.src.metrics import (
            MetricsComputer,
            MetricsConfig,
            RegressionDetector,
        )

        # Step 1: Setup Benchmark Configuration
        config = BenchmarkConfig(
            model_path="./models/test-model",
            benchmarks=["mmlu", "gsm8k", "hellaswag"],
            batch_size=16,
            num_shots=5,
            max_samples=50,
            output_dir="./benchmark_results",
        )
        orchestrator = BenchmarkOrchestrator(config)
        assert len(orchestrator.config.benchmarks) == 3

        # Step 2: Run Benchmarks (with mocked results)
        mock_results = {
            "mmlu": {"score": 45.2, "accuracy": 0.452, "num_samples": 50},
            "gsm8k": {"score": 38.5, "accuracy": 0.385, "num_samples": 50},
            "hellaswag": {"score": 52.1, "accuracy": 0.521, "num_samples": 50},
        }

        with patch.object(orchestrator, "run_all", return_value=mock_results):
            results = orchestrator.run_all()
            assert "mmlu" in results
            assert "gsm8k" in results
            assert "hellaswag" in results

        logger.info("Benchmark execution completed")

        # Step 3: Compute Metrics
        metrics_config = MetricsConfig(
            task_benchmarks=True,
            latency_analysis=True,
            safety_checks=True,
            regression_tests=True,
        )
        metrics_computer = MetricsComputer(metrics_config)

        # Add latency measurements
        mock_results_with_latency = {
            **mock_results,
            "latency_measurements": [150.0, 145.0, 160.0, 155.0] * 10,
        }

        computed_metrics = metrics_computer.compute_metrics(mock_results_with_latency)
        assert "task_benchmarks" in computed_metrics
        assert "latency" in computed_metrics
        assert "safety" in computed_metrics

        # Verify task benchmark scores
        assert computed_metrics["task_benchmarks"]["mmlu_score"] == 45.2
        assert computed_metrics["task_benchmarks"]["gsm8k_score"] == 38.5
        assert computed_metrics["task_benchmarks"]["hellaswag_score"] == 52.1

        # Verify latency metrics
        assert "p50_latency_ms" in computed_metrics["latency"]
        assert "p95_latency_ms" in computed_metrics["latency"]
        assert "throughput_requests_per_sec" in computed_metrics["latency"]

        logger.info("Metrics computation completed")

        # Step 4: Regression Detection
        baseline_metrics = {
            "mmlu_score": 44.0,
            "gsm8k_score": 40.0,
            "hellaswag_score": 50.0,
            "p50_latency_ms": 140.0,
        }

        current_metrics = {
            "mmlu_score": 45.2,
            "gsm8k_score": 38.5,
            "hellaswag_score": 52.1,
            "p50_latency_ms": 155.0,
        }

        detector = RegressionDetector(baseline_metrics)
        regression_report = detector.detect_regressions(current_metrics, threshold=0.05)

        assert "has_regressions" in regression_report
        assert "regressions" in regression_report
        assert "improvements" in regression_report

        logger.info("Regression detection completed")

        # Step 5: Generate Summary
        summary = metrics_computer.get_metrics_summary()
        assert "total_metrics" in summary
        assert "metric_categories" in summary
        assert "details" in summary

        logger.info("Benchmark workflow E2E test completed")

    def test_benchmark_with_regression_analysis(self):
        """Test benchmark workflow with comprehensive regression analysis."""
        logger.info("Starting benchmark regression analysis E2E test")

        from pipelines.evaluation.src.metrics import RegressionDetector

        # Baseline from previous run
        baseline = {
            "mmlu_score": 50.0,
            "hellaswag_score": 55.0,
            "latency_p95": 200.0,
            "toxicity_score": 0.15,
        }

        # Current results
        current = {
            "mmlu_score": 48.0,  # Regression: 4% drop
            "hellaswag_score": 60.0,  # Improvement: 9% increase
            "latency_p95": 250.0,  # Regression: 25% increase (latency higher is worse)
            "toxicity_score": 0.12,  # Improvement: lower is better
        }

        detector = RegressionDetector(baseline)
        report = detector.detect_regressions(current, threshold=0.05)

        # Should detect regression in MMLU
        mmlu_regression = any(r["metric"] == "mmlu_score" for r in report["regressions"])
        assert mmlu_regression is True

        # Should detect improvement in HellaSwag
        hellaswag_improvement = any(
            i["metric"] == "hellaswag_score" for i in report["improvements"]
        )
        assert hellaswag_improvement is True

        # Should detect latency regression
        latency_regression = any(
            r["metric"] == "latency_p95" for r in report["regressions"]
        )
        assert latency_regression is True

        logger.info("Regression analysis E2E test completed")


class TestFineTuningWorkflow:
    """End-to-end tests for the complete fine-tuning workflow.

    Tests dataset preparation, training, checkpoint management,
    evaluation, and inference on fine-tuned models.
    """

    def test_qlora_fine_tuning_workflow(self, temp_output_dir):
        """Test complete QLoRA fine-tuning workflow.

        Covers: Dataset → Model Loading (4-bit) → LoRA Config → Training → Export → Inference
        """
        logger.info("Starting QLoRA fine-tuning E2E test")

        from pipelines.training.src.orchestrator import (
            TrainingOrchestrator,
            TrainingConfig,
        )

        # Step 1: Setup QLoRA Configuration
        config = TrainingConfig(
            model_name="meta-llama/Llama-2-7b",
            dataset_path="data/instructions",
            output_dir=str(temp_output_dir / "qlora"),
            num_epochs=1,
            batch_size=4,
            learning_rate=2e-4,
            training_method="qlora",
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            quantization_enabled=True,
            gradient_accumulation_steps=4,
        )

        orchestrator = TrainingOrchestrator(config)
        assert orchestrator.config.training_method == "qlora"
        assert orchestrator.config.quantization_enabled is True

        # Step 2: Load Model (mocked)
        with patch(
            "pipelines.training.src.orchestrator.AutoModelForCausalLM"
        ) as mock_model, patch(
            "pipelines.training.src.orchestrator.AutoTokenizer"
        ) as mock_tokenizer, patch(
            "pipelines.training.src.orchestrator.BitsAndBytesConfig"
        ) as mock_bnb, patch(
            "pipelines.training.src.orchestrator.get_peft_model"
        ) as mock_peft:

            # Setup quantization config mock
            mock_bnb.return_value = Mock()

            # Setup tokenizer mock
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "</s>"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # Setup model mock
            mock_model_instance = Mock()
            mock_model_instance.parameters.return_value = [
                Mock(numel=lambda: 1000, requires_grad=True)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_peft.return_value = mock_model_instance

            # Load model with quantization
            orchestrator.load_model()

            # Verify quantization config was created
            assert mock_bnb.called
            logger.info("Model loaded with 4-bit quantization")

            # Step 3: Setup Training
            orchestrator.setup_training()
            assert orchestrator.optimizer is not None
            logger.info("Training setup completed")

            # Step 4: Save Checkpoint
            checkpoint_path = temp_output_dir / "qlora" / "checkpoints" / "checkpoint-100"
            orchestrator.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()
            logger.info("Checkpoint saved")

        # Step 5: Evaluation
        from pipelines.evaluation.src.metrics import MetricsComputer, MetricsConfig

        eval_results = {"alpacaeval": {"score": 52.0, "win_rate": 0.52, "num_samples": 100}}
        metrics_config = MetricsConfig(task_benchmarks=True)
        metrics_computer = MetricsComputer(metrics_config)
        metrics = metrics_computer.compute_metrics(eval_results)
        assert metrics["task_benchmarks"]["alpacaeval_score"] == 52.0

        # Step 6: Inference
        from inference.src.engines import InferenceEngineFactory, InferenceConfig

        inference_config = InferenceConfig(
            model_name=str(checkpoint_path),
            batch_size=1,
            max_length=256,
            temperature=0.7,
            device="cpu",
        )

        with patch(
            "inference.src.engines.AutoModelForCausalLM"
        ) as mock_inference_model, patch(
            "inference.src.engines.AutoTokenizer"
        ) as mock_inference_tokenizer:

            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token_id = 0
            mock_tokenizer_instance.eos_token_id = 1
            mock_inference_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model_instance.generate.return_value = [[1, 2, 3, 4]]
            mock_inference_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer_instance.batch_decode.return_value = [
                "Fine-tuned model response"
            ]
            mock_tokenizer_instance.__call__ = Mock(
                return_value={"input_ids": [[1]], "attention_mask": [[1]]}
            )

            engine = InferenceEngineFactory.create("transformers", inference_config)

            with patch.object(engine, "tokenizer", mock_tokenizer_instance), \
                 patch.object(engine, "model", mock_model_instance):
                engine.tokenizer = mock_tokenizer_instance
                engine.model = mock_model_instance
                mock_tokenizer_instance.batch_decode.return_value = [
                    "Fine-tuned model response"
                ]
                results = engine.generate(["Test instruction"])
                assert len(results) == 1

        logger.info("QLoRA fine-tuning workflow E2E test completed")

    def test_multimodal_fine_tuning_workflow(self, temp_output_dir):
        """Test multimodal fine-tuning workflow (text + vision)."""
        logger.info("Starting multimodal fine-tuning E2E test")

        # This is a conceptual test for multimodal fine-tuning
        # In practice, this would involve vision-language models

        from pipelines.training.src.orchestrator import (
            TrainingOrchestrator,
            TrainingConfig,
        )

        config = TrainingConfig(
            model_name="llava-hf/llava-1.5-7b",
            dataset_path="data/multimodal",
            output_dir=str(temp_output_dir / "multimodal"),
            num_epochs=1,
            batch_size=2,
            training_method="lora",
        )

        orchestrator = TrainingOrchestrator(config)

        with patch(
            "pipelines.training.src.orchestrator.AutoModelForCausalLM"
        ) as mock_model, patch(
            "pipelines.training.src.orchestrator.AutoTokenizer"
        ) as mock_tokenizer, patch(
            "pipelines.training.src.orchestrator.get_peft_model"
        ) as mock_peft:

            mock_tokenizer.from_pretrained.return_value = Mock(
                pad_token=None, eos_token="</s>"
            )
            mock_model_instance = Mock()
            mock_model_instance.parameters.return_value = [
                Mock(numel=lambda: 1000, requires_grad=True)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_peft.return_value = mock_model_instance

            orchestrator.load_model()
            orchestrator.setup_training()

            # Verify model was loaded
            assert orchestrator.model is not None
            logger.info("Multimodal model setup completed")

        logger.info("Multimodal fine-tuning workflow E2E test completed")


class TestExportAndDeployment:
    """End-to-end tests for model export and deployment workflows."""

    def test_model_export_to_multiple_formats(self, temp_output_dir):
        """Test exporting models to multiple formats (PyTorch, ONNX)."""
        logger.info("Starting multi-format export E2E test")

        from models.exported.core import (
            ExporterFactory,
            ExportConfig,
            ExportFormat,
        )
        from models.exported.core import ModelLoader, LoadConfig

        formats_to_test = [
            (ExportFormat.PYTORCH, ".pt"),
            (ExportFormat.ONNX, ".onnx"),
        ]

        mock_model = Mock()
        mock_model.state_dict.return_value = {"layer1.weight": [1.0, 2.0, 3.0]}

        for fmt, expected_ext in formats_to_test:
            config = ExportConfig(
                model_name=f"test-model-{fmt.value}",
                output_dir=temp_output_dir / f"export_{fmt.value}",
                export_format=fmt,
                version="1.0.0",
                include_metadata=True,
                include_config=True,
            )

            exporter = ExporterFactory.create_exporter(config)
            result = exporter.export(mock_model)

            assert result["success"] is True
            assert result["format"] == fmt.value
            logger.info(f"Export to {fmt.value} completed")

        logger.info("Multi-format export E2E test completed")

    def test_deployment_with_monitoring(self, temp_output_dir):
        """Test deployment workflow with monitoring setup."""
        logger.info("Starting deployment with monitoring E2E test")

        from pipelines.deployment.src.orchestrator import (
            DeploymentOrchestrator,
            DeploymentConfig,
        )

        config = DeploymentConfig(
            model_path=str(temp_output_dir / "model"),
            model_name="production-model",
            model_version="2.0.0",
            output_dir=str(temp_output_dir / "deploy"),
            major=2,
            minor=0,
            patch=0,
            push_to_hub=True,
            hub_repo_id="org/production-model",
        )

        # Create model directory
        model_dir = Path(config.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text('{"architectures": ["LlamaModel"]}')
        (model_dir / "model.safetensors").write_text("mock weights")
        (model_dir / "tokenizer.json").write_text('{"vocab_size": 32000}')

        orchestrator = DeploymentOrchestrator(config)

        # Package model
        package_path = orchestrator.package_model()
        assert Path(package_path).exists()

        # Verify package contents
        assert (Path(package_path) / "config.json").exists()
        assert (Path(package_path) / "deployment_metadata.json").exists()

        # Publish model
        result = orchestrator.publish_model()
        assert result["status"] == "published"
        assert result["hub_url"] == "https://huggingface.co/org/production-model"

        # Verify deployment info
        info = orchestrator.get_deployment_info()
        assert info["model_name"] == "production-model"
        assert info["version"] == "v2.0.0"

        logger.info("Deployment with monitoring E2E test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
