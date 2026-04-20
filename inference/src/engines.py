"""Inference engines for model serving."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""

    model_name: str
    batch_size: int = 32
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda"
    quantization: bool = False


class InferenceEngine(ABC):
    """Base inference engine."""

    def __init__(self, config: InferenceConfig):
        """Initialize inference engine.

        Args:
            config: Inference configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate text from prompts.

        Args:
            prompts: List of input prompts

        Returns:
            List of generated texts
        """
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        pass


class TransformersEngine(InferenceEngine):
    """Inference engine using transformers library."""

    def __init__(self, config: InferenceConfig):
        """Initialize transformers engine.

        Args:
            config: Inference configuration
        """
        super().__init__(config)
        self._load_model()

    def _load_model(self):
        """Load model from transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required")

        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map=self.config.device if self.config.device == "cuda" else None,
        )

        logger.info("Model loaded successfully")

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate text from prompts.

        Args:
            prompts: List of input prompts

        Returns:
            List of generated texts
        """
        logger.info(f"Generating for {len(prompts)} prompts")

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)

        outputs = self.model.generate(
            **inputs,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            num_return_sequences=1,
        )

        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return results

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        logger.info(f"Computing embeddings for {len(texts)} texts")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers library required for embeddings")

        # Load embedding model if not already loaded
        if not hasattr(self, '_embedding_model') or self._embedding_model is None:
            # Use a default embedding model or derive from model_name
            embedding_model_name = "all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {embedding_model_name}")
            self._embedding_model = SentenceTransformer(embedding_model_name)

        # Compute embeddings
        embeddings = self._embedding_model.encode(texts, convert_to_numpy=True)

        # Convert to list of lists
        return [embedding.tolist() for embedding in embeddings]


class VLLMEngine(InferenceEngine):
    """Inference engine using vLLM."""

    def __init__(self, config: InferenceConfig):
        """Initialize vLLM engine.

        Args:
            config: Inference configuration
        """
        super().__init__(config)
        self._load_model()

    def _load_model(self):
        """Load model using vLLM."""
        logger.info(f"Loading model with vLLM: {self.config.model_name}")

        try:
            from vllm import LLM

            self.model = LLM(
                model=self.config.model_name,
                quantization="awq" if self.config.quantization else None,
            )
        except ImportError:
            logger.warning("vLLM not installed, skipping initialization")

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate text using vLLM.

        Args:
            prompts: List of input prompts

        Returns:
            List of generated texts
        """
        logger.info(f"Generating with vLLM for {len(prompts)} prompts")

        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("vLLM library required")

        if self.model is None:
            raise RuntimeError("vLLM model not loaded")

        # Configure sampling parameters from config
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_length,
        )

        # Generate outputs
        outputs = self.model.generate(prompts, sampling_params)

        # Extract generated text from outputs
        results = [output.outputs[0].text for output in outputs]

        return results

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings (not supported in vLLM for generation)."""
        logger.warning("Embeddings not directly supported in vLLM")
        return [[0.0] * 768 for _ in texts]


class InferenceEngineFactory:
    """Factory for creating inference engines."""

    engines = {
        "transformers": TransformersEngine,
        "vllm": VLLMEngine,
    }

    @classmethod
    def create(cls, engine_type: str, config: InferenceConfig) -> InferenceEngine:
        """Create inference engine.

        Args:
            engine_type: Type of engine
            config: Engine configuration

        Returns:
            Inference engine instance
        """
        if engine_type not in cls.engines:
            raise ValueError(f"Unknown engine: {engine_type}")

        return cls.engines[engine_type](config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = InferenceConfig(model_name="gpt2", batch_size=8, device="cpu")

    engine = InferenceEngineFactory.create("transformers", config)
    results = engine.generate(["Hello, how are you?"])
    print(results)
