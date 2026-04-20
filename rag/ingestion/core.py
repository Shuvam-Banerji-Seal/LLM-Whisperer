"""Core document ingestion implementations."""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .config import IngestionConfig, LoaderType

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Main document loader supporting multiple formats."""

    def __init__(self, config: IngestionConfig):
        """Initialize document loader.

        Args:
            config: Ingestion configuration
        """
        self.config = config

    def load(self, source: str) -> List[Dict[str, Any]]:
        """Load documents from source.

        Args:
            source: Path or URL to load from

        Returns:
            List of document dicts with 'content' and 'metadata'
        """
        if self.config.loader_type == LoaderType.TEXT:
            return self._load_text(source)
        elif self.config.loader_type == LoaderType.PDF:
            return self._load_pdf(source)
        elif self.config.loader_type == LoaderType.DIRECTORY:
            return self._load_directory(source)
        else:
            raise ValueError(f"Unsupported loader type: {self.config.loader_type}")

    def _load_text(self, source: str) -> List[Dict[str, Any]]:
        """Load text file."""
        try:
            with open(source, "r", encoding=self.config.encoding) as f:
                content = f.read()
            return [{"content": content, "metadata": {"source": source}}]
        except Exception as e:
            logger.error(f"Failed to load text file {source}: {e}")
            return []

    def _load_pdf(self, source: str) -> List[Dict[str, Any]]:
        """Load PDF file.

        Args:
            source: Path to PDF file.

        Returns:
            List of document dicts with 'content' and 'metadata'.
        """
        logger.info(f"Loading PDF from {source}")

        try:
            import pdfplumber

            documents = []
            with pdfplumber.open(source) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append({
                            "content": text,
                            "metadata": {
                                "source": source,
                                "page_number": page_num,
                                "total_pages": len(pdf.pages)
                            }
                        })
            return documents
        except ImportError:
            logger.error("pdfplumber not installed. Install with: pip install pdfplumber")
            return []
        except Exception as e:
            logger.error(f"Failed to load PDF {source}: {e}")
            return []

    def _load_directory(self, source: str) -> List[Dict[str, Any]]:
        """Load all files from directory."""
        import os

        documents = []
        for filename in os.listdir(source):
            filepath = os.path.join(source, filename)
            if os.path.isfile(filepath):
                docs = self._load_text(filepath)
                documents.extend(docs)
        return documents


class MetadataExtractor:
    """Extracts metadata from documents."""

    @staticmethod
    def extract(content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract metadata from content.

        Args:
            content: Document content
            metadata: Base metadata dict

        Returns:
            Enriched metadata dict
        """
        if metadata is None:
            metadata = {}

        metadata["char_count"] = len(content)
        metadata["word_count"] = len(content.split())
        metadata["line_count"] = len(content.split("\n"))

        return metadata


class DocumentPipeline:
    """Complete ingestion pipeline."""

    def __init__(self, config: IngestionConfig):
        """Initialize pipeline.

        Args:
            config: Ingestion configuration
        """
        self.config = config
        self.loader = DocumentLoader(config)
        self.extractor = MetadataExtractor()

    def process(self, source: str) -> List[Dict[str, Any]]:
        """Process documents through pipeline.

        Args:
            source: Source to ingest

        Returns:
            List of processed documents
        """
        # Load documents
        documents = self.loader.load(source)

        # Extract metadata
        for doc in documents:
            doc["metadata"] = self.extractor.extract(
                doc["content"], doc.get("metadata")
            )

        logger.info(f"Pipeline processed {len(documents)} documents")
        return documents
