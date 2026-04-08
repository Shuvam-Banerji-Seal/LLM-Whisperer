"""Core document chunking implementations."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import ChunkingConfig, ChunkingMethod

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of a document."""

    id: str
    content: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    source_doc_id: Optional[str] = None


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Chunk text according to strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata for chunks

        Returns:
            List of Chunk objects
        """
        pass


class RecursiveChunker(ChunkingStrategy):
    """Chunks text recursively using semantic separators.

    Attempts to chunk by progressively smaller separators to maintain
    semantic coherence. Good general-purpose chunker.
    """

    def __init__(self, config: ChunkingConfig):
        """Initialize recursive chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.separators = config.separators

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Chunk text recursively by semantic boundaries.

        Args:
            text: Text to chunk
            metadata: Optional metadata for chunks

        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}

        chunks = self._recursive_split(text)
        chunk_objects = []

        current_pos = 0
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) < self.config.min_chunk_size:
                continue

            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = i
            chunk_meta["chunk_text_length"] = len(chunk_text)

            chunk_obj = Chunk(
                id=f"{metadata.get('doc_id', 'unknown')}_chunk_{i}",
                content=chunk_text,
                start_idx=current_pos,
                end_idx=current_pos + len(chunk_text),
                metadata=chunk_meta,
                source_doc_id=metadata.get("doc_id"),
            )
            chunk_objects.append(chunk_obj)
            current_pos += len(chunk_text)

        logger.info(
            f"Recursive chunking created {len(chunk_objects)} chunks "
            f"from text of length {len(text)}"
        )
        return chunk_objects

    def _recursive_split(
        self, text: str, separators: Optional[List[str]] = None
    ) -> List[str]:
        """Recursively split text by separators.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        if separators is None:
            separators = self.separators

        good_splits = []
        separator = separators[-1]

        for _s in separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = [s for s in splits if len(s) >= self.config.min_chunk_size]

        if not good_splits:
            return splits

        # Merge small chunks
        merged_chunks = self._merge_splits(good_splits, separator)
        return merged_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into larger chunks.

        Args:
            splits: List of text splits
            separator: Separator to use when merging

        Returns:
            List of merged chunks
        """
        separator_len = len(separator)
        good_splits = []
        current_chunk = []
        current_length = 0

        for s in splits:
            if current_length + len(s) + separator_len <= self.config.chunk_size:
                current_chunk.append(s)
                current_length += len(s) + separator_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk)
                    good_splits.append(merged)
                current_chunk = [s]
                current_length = len(s) + separator_len

        if current_chunk:
            merged = separator.join(current_chunk)
            good_splits.append(merged)

        return good_splits


class SlidingWindowChunker(ChunkingStrategy):
    """Chunks text using a sliding window approach.

    Creates overlapping chunks by sliding a fixed-size window through the text.
    Useful for preserving context at chunk boundaries.
    """

    def __init__(self, config: ChunkingConfig):
        """Initialize sliding window chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Chunk text using sliding window.

        Args:
            text: Text to chunk
            metadata: Optional metadata for chunks

        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}

        chunks = []
        start = 0

        chunk_index = 0
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            chunk_text = text[start:end]

            if len(chunk_text) >= self.config.min_chunk_size:
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["chunk_text_length"] = len(chunk_text)

                chunk_obj = Chunk(
                    id=f"{metadata.get('doc_id', 'unknown')}_chunk_{chunk_index}",
                    content=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    metadata=chunk_meta,
                    source_doc_id=metadata.get("doc_id"),
                )
                chunks.append(chunk_obj)
                chunk_index += 1

            start = end - self.config.chunk_overlap

        logger.info(
            f"Sliding window chunking created {len(chunks)} chunks "
            f"from text of length {len(text)}"
        )
        return chunks


class DocumentChunker:
    """Factory class for creating chunkers based on configuration."""

    def __init__(self, config: ChunkingConfig):
        """Initialize document chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.strategy = self._create_strategy()

    def _create_strategy(self) -> ChunkingStrategy:
        """Create chunking strategy based on config.

        Returns:
            ChunkingStrategy instance

        Raises:
            ValueError: If chunking method is not supported
        """
        if self.config.method == ChunkingMethod.RECURSIVE:
            return RecursiveChunker(self.config)
        elif self.config.method == ChunkingMethod.SLIDING_WINDOW:
            return SlidingWindowChunker(self.config)
        else:
            # Default to recursive
            return RecursiveChunker(self.config)

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Chunk text using configured strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata for chunks

        Returns:
            List of Chunk objects
        """
        return self.strategy.chunk(text, metadata)

    def chunk_multiple(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Chunk]:
        """Chunk multiple texts.

        Args:
            texts: List of texts to chunk
            metadata: Optional list of metadata dicts

        Returns:
            List of Chunk objects
        """
        all_chunks = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata else {}
            if "doc_id" not in meta:
                meta["doc_id"] = f"doc_{i}"
            chunks = self.chunk(text, meta)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(texts)} documents into {len(all_chunks)} chunks")
        return all_chunks


class ChunkMerger:
    """Merges small or similar chunks."""

    def __init__(self, min_size: int = 100, similarity_threshold: float = 0.8):
        """Initialize chunk merger.

        Args:
            min_size: Minimum chunk size before merging
            similarity_threshold: Threshold for merging similar chunks
        """
        self.min_size = min_size
        self.similarity_threshold = similarity_threshold

    def merge_by_size(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are smaller than min_size with adjacent chunks.

        Args:
            chunks: List of chunks to merge

        Returns:
            List of merged chunks
        """
        if not chunks:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If chunk is large enough, keep it
            if len(current.content) >= self.min_size:
                merged.append(current)
                i += 1
                continue

            # Try to merge with next chunk
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                merged_content = current.content + " " + next_chunk.content
                merged_chunk = Chunk(
                    id=f"{current.id}_merged",
                    content=merged_content,
                    start_idx=current.start_idx,
                    end_idx=next_chunk.end_idx,
                    metadata={**current.metadata, "merged": True},
                    source_doc_id=current.source_doc_id,
                )
                merged.append(merged_chunk)
                i += 2
            else:
                # Last chunk, just keep it
                merged.append(current)
                i += 1

        logger.info(f"Merged chunks: {len(chunks)} -> {len(merged)}")
        return merged

    def merge_by_separator(
        self, chunks: List[Chunk], separator: str = " "
    ) -> List[Chunk]:
        """Merge all chunks with a separator.

        Args:
            chunks: List of chunks to merge
            separator: Separator to use between chunks

        Returns:
            List with single merged chunk (or empty if no chunks)
        """
        if not chunks:
            return []

        merged_content = separator.join(chunk.content for chunk in chunks)
        merged_chunk = Chunk(
            id="merged_all",
            content=merged_content,
            start_idx=chunks[0].start_idx,
            end_idx=chunks[-1].end_idx,
            metadata={"merged": True, "original_count": len(chunks)},
            source_doc_id=chunks[0].source_doc_id,
        )
        return [merged_chunk]
