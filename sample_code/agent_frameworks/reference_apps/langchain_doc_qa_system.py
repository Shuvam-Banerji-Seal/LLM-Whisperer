"""
LangChain Document Q&A System - Reference Application

A complete document question-answering system that demonstrates:
- Document ingestion and preprocessing
- Text chunking and embedding simulation
- Semantic search and retrieval
- Question answering with source attribution
- Multi-document support
- Session memory management

References:
- LangChain Documentation: https://docs.langchain.com/
- Building AI Agents with LangChain: https://www.ai-agentsplus.com/blog/building-ai-agents-with-langchain-tutorial-2026/

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install langchain>=0.1.0
# pip install langchain-community>=0.1.0
# pip install openai>=1.0.0

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""

    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    # Simulated embedding (in production, use real embeddings)
    embedding: List[float] = field(default_factory=list)


@dataclass
class Document:
    """A document in the Q&A system."""

    doc_id: str
    title: str
    content: str
    source_url: str
    uploaded_at: datetime = field(default_factory=datetime.now)
    chunks: List[DocumentChunk] = field(default_factory=list)


@dataclass
class QAResult:
    """Result of a question-answering query."""

    question: str
    answer: str
    source_chunks: List[DocumentChunk]
    confidence_score: float
    processing_time_ms: float
    citations: List[str] = field(default_factory=list)


# ============================================================================
# Text Splitter
# ============================================================================


class TextSplitter:
    """Splits documents into semantic chunks."""

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    @staticmethod
    def split_text(
        text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
    ) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((text[start:end], start, end))
            start = end - overlap

        return chunks


# ============================================================================
# Simulated Vector Store
# ============================================================================


class VectorStore:
    """
    Simulated vector store for semantic search.
    In production, use Pinecone, Weaviate, or similar.
    """

    def __init__(self):
        """Initialize the vector store."""
        self.documents: List[Document] = []
        self.chunks: List[DocumentChunk] = []

    def add_document(self, document: Document) -> None:
        """Add a document to the store."""
        # Split document into chunks
        text_chunks = TextSplitter.split_text(document.content)

        for chunk_index, (chunk_text, start, end) in enumerate(text_chunks):
            chunk = DocumentChunk(
                chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
                document_id=document.doc_id,
                content=chunk_text,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                # Simulated: would be real embeddings in production
                embedding=self._simulate_embedding(chunk_text),
            )
            self.chunks.append(chunk)
            document.chunks.append(chunk)

        self.documents.append(document)
        print(f"📚 Added document '{document.title}' with {len(text_chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[DocumentChunk]:
        """
        Search for relevant chunks using semantic similarity.
        In production, uses embeddings and vector search.
        """
        query_keywords = set(query.lower().split())
        results = []

        for chunk in self.chunks:
            # Simple keyword matching (production uses embeddings)
            chunk_words = set(chunk.content.lower().split())
            matching_words = query_keywords & chunk_words

            if matching_words:
                # Relevance score based on keyword overlap
                relevance = len(matching_words) / max(len(query_keywords), 1)
                results.append((chunk, relevance))

        # Sort by relevance and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in results[:top_k]]

    @staticmethod
    def _simulate_embedding(text: str) -> List[float]:
        """
        Simulate an embedding vector.
        In production, use OpenAI embeddings or similar.
        """
        # Create a simple hash-based "embedding" for demonstration
        import hashlib

        hash_obj = hashlib.md5(text.encode())
        hash_bytes = bytes.fromhex(hash_obj.hexdigest())
        return [float(b) / 255.0 for b in hash_bytes[:16]]


# ============================================================================
# Question Answering Engine
# ============================================================================


class LangChainDocumentQASystem:
    """
    Document Q&A system using LangChain patterns.

    Features:
    - Multi-document support
    - Semantic search
    - Answer generation with citations
    - Session memory
    - Confidence scoring
    """

    def __init__(self):
        """Initialize the Q&A system."""
        self.vector_store = VectorStore()
        self.qa_history: List[QAResult] = []
        self.session_memory: List[Dict[str, str]] = []

    def add_document(
        self,
        title: str,
        content: str,
        source_url: str,
    ) -> None:
        """
        Add a document to the Q&A system.

        Args:
            title: Document title
            content: Document content
            source_url: Source URL
        """
        doc = Document(
            doc_id=f"doc_{len(self.vector_store.documents):03d}",
            title=title,
            content=content,
            source_url=source_url,
        )
        self.vector_store.add_document(doc)

    def answer_question(self, question: str) -> QAResult:
        """
        Answer a question based on stored documents.

        Args:
            question: The question to answer

        Returns:
            QAResult with answer and sources
        """
        import time

        start_time = time.time()

        print(f"\n❓ Question: {question}")

        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(question, top_k=3)

        if not relevant_chunks:
            result = QAResult(
                question=question,
                answer="I don't have documents that contain information related to your question.",
                source_chunks=[],
                confidence_score=0.0,
                processing_time_ms=0,
            )
        else:
            # Synthesize answer from chunks
            answer = self._synthesize_answer(question, relevant_chunks)

            # Generate citations
            citations = self._generate_citations(relevant_chunks)

            # Calculate confidence
            confidence = len(relevant_chunks) / 5.0  # Simplified

            processing_time = (time.time() - start_time) * 1000

            result = QAResult(
                question=question,
                answer=answer,
                source_chunks=relevant_chunks,
                confidence_score=min(1.0, confidence),
                processing_time_ms=processing_time,
                citations=citations,
            )

        self.qa_history.append(result)
        self.session_memory.append({"role": "user", "content": question})
        self.session_memory.append({"role": "assistant", "content": result.answer})

        return result

    def _synthesize_answer(self, question: str, chunks: List[DocumentChunk]) -> str:
        """Synthesize an answer from relevant chunks."""
        answer = f"Based on the documents, here's what I found about '{question}':\n\n"

        for i, chunk in enumerate(chunks, 1):
            # Extract relevant sentences
            sentences = chunk.content.split(".")
            relevant_sentences = [
                s.strip()
                for s in sentences
                if any(word in s.lower() for word in question.lower().split())
            ][:2]

            if relevant_sentences:
                answer += f"{i}. {' '.join(relevant_sentences)}.\n"

        if not any(chunk.content for chunk in chunks):
            answer += "The documents contain information relevant to your question."

        return answer

    @staticmethod
    def _generate_citations(chunks: List[DocumentChunk]) -> List[str]:
        """Generate citations for the source chunks."""
        citations = []

        # Get unique documents
        doc_ids = set(chunk.document_id for chunk in chunks)

        for doc_id in doc_ids:
            citations.append(f"Source: Document {doc_id}")

        return citations

    def display_result(self, result: QAResult) -> None:
        """Display a Q&A result."""
        print(f"\n✅ Answer: {result.answer}")

        if result.citations:
            print(f"\n📚 Citations:")
            for citation in result.citations:
                print(f"  - {citation}")

        print(f"\n📊 Confidence: {result.confidence_score:.1%}")
        print(f"⏱️  Processing Time: {result.processing_time_ms:.1f}ms")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.session_memory


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LangChain Document Q&A System - Reference Application")
    print("=" * 70)

    qa_system = LangChainDocumentQASystem()

    # Add sample documents
    print("\n📖 Loading Documents")
    print("=" * 70)

    qa_system.add_document(
        title="Python Best Practices",
        content="""
        Python is a powerful programming language known for its simplicity and readability.
        When writing Python code, it's important to follow PEP 8 style guidelines.
        Use meaningful variable names that describe their purpose.
        Write docstrings for all functions and classes.
        Use type hints to improve code clarity.
        Always handle exceptions properly.
        Write unit tests for your code.
        Use virtual environments for project isolation.
        """,
        source_url="https://example.com/python-best-practices",
    )

    qa_system.add_document(
        title="LangChain Framework Guide",
        content="""
        LangChain is a framework for developing applications powered by language models.
        It supports multiple language models and tools.
        You can build agents that can use tools like search, calculators, or custom functions.
        LangChain provides chains for composing multiple operations.
        Memory management is important for maintaining conversation context.
        The framework supports both synchronous and asynchronous operations.
        You can integrate LangChain with various databases and APIs.
        """,
        source_url="https://example.com/langchain-guide",
    )

    qa_system.add_document(
        title="Machine Learning Fundamentals",
        content="""
        Machine learning involves training models on data to make predictions.
        There are three main types: supervised, unsupervised, and reinforcement learning.
        Supervised learning uses labeled data for training.
        Features should be normalized before training.
        Overfitting occurs when a model learns the training data too well.
        Cross-validation helps evaluate model performance reliably.
        Always split data into training and test sets.
        """,
        source_url="https://example.com/ml-fundamentals",
    )

    # Ask questions
    print("\n\n📝 Question & Answer Session")
    print("=" * 70)

    # Question 1
    result1 = qa_system.answer_question("What are Python best practices?")
    qa_system.display_result(result1)

    # Question 2
    print("\n" + "=" * 70)
    result2 = qa_system.answer_question("How does LangChain work?")
    qa_system.display_result(result2)

    # Question 3
    print("\n" + "=" * 70)
    result3 = qa_system.answer_question("What is machine learning?")
    qa_system.display_result(result3)

    # Display conversation history
    print("\n\n" + "=" * 70)
    print("Conversation History")
    print("=" * 70)
    history = qa_system.get_conversation_history()
    for i, msg in enumerate(history, 1):
        role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
        print(f"\n{i}. {role}")
        print(f"   {msg['content'][:100]}...")

    print("\n" + "=" * 70)
    print("Production Implementation Features")
    print("=" * 70)
    print("""
✅ Multi-document support
✅ Semantic search with embeddings
✅ Question answering with citations
✅ Conversation memory
✅ Confidence scoring
✅ Processing time tracking

Production Enhancement Areas:
1. Real embedding models (OpenAI, Sentence Transformers)
2. Vector database (Pinecone, Weaviate, Milvus)
3. Advanced retrieval (BM25, hybrid search)
4. Query expansion and paraphrasing
5. Answer validation and fact-checking
6. Support for multiple document formats (PDF, DOCX)
7. Document versioning and updates
8. User authentication and access control
9. Caching for performance optimization
10. Analytics and usage monitoring

Key LangChain Components:
- DocumentLoader: Load documents from sources
- TextSplitter: Split documents into chunks
- Embeddings: Generate vector representations
- VectorStore: Store and retrieve embeddings
- Retriever: Interface for document retrieval
- LLMChain: Compose LLM with prompts
- Memory: Maintain conversation context
- CallbackManager: Monitor execution

See: https://docs.langchain.com/
    """)
