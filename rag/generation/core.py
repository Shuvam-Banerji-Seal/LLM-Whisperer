"""Core generation implementations."""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .config import GenerationConfig, GenerationMode

logger = logging.getLogger(__name__)


class TextAugmenter:
    """Augments text for generation tasks."""

    @staticmethod
    def augment(text: str, context: List[str]) -> str:
        """Augment text with context.

        Args:
            text: Original text
            context: Context documents

        Returns:
            Augmented text
        """
        context_str = "\n\n".join(context)
        augmented = f"Context:\n{context_str}\n\nQuestion: {text}"
        return augmented


class PromptAssembler:
    """Assembles prompts for generation."""

    def __init__(self, config: GenerationConfig):
        """Initialize prompt assembler.

        Args:
            config: Generation configuration
        """
        self.config = config

    def assemble(
        self,
        query: str,
        context_docs: List[str],
        citations: bool = True,
    ) -> str:
        """Assemble prompt from query and context.

        Args:
            query: Original query
            context_docs: Retrieved context documents
            citations: Whether to include citations

        Returns:
            Assembled prompt
        """
        context_str = "\n\n".join(
            [f"[{i + 1}] {doc}" for i, doc in enumerate(context_docs)]
        )

        if self.config.mode == GenerationMode.GROUNDED:
            prompt = (
                f"Based on the following context, answer the question.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
        else:
            prompt = f"Question: {query}\n\nAnswer:"

        return prompt


class GenerationOrchestrator:
    """Orchestrates end-to-end generation."""

    def __init__(
        self,
        retriever,
        reranker,
        prompt_assembler: PromptAssembler,
        config: GenerationConfig,
    ):
        """Initialize generation orchestrator.

        Args:
            retriever: Document retriever
            reranker: Document reranker
            prompt_assembler: Prompt assembler
            config: Generation configuration
        """
        self.retriever = retriever
        self.reranker = reranker
        self.assembler = prompt_assembler
        self.config = config

    def generate(
        self,
        query: str,
        llm_generate_fn: callable,
    ) -> Dict[str, Any]:
        """Generate response for query.

        Args:
            query: Input query
            llm_generate_fn: Function to call LLM

        Returns:
            Dictionary with response and metadata
        """
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, self.config.context_window_size)

        # Optionally rerank
        if self.reranker:
            reranked = self.reranker.rerank(query, retrieved_docs)
            retrieved_docs = [doc for doc, _ in reranked]

        # Assemble prompt
        prompt = self.assembler.assemble(query, retrieved_docs)

        # Generate response
        response = llm_generate_fn(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return {
            "response": response,
            "context_docs": retrieved_docs,
            "prompt": prompt,
        }
