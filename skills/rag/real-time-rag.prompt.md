# Real-Time RAG — Agentic Skill Prompt

Building streaming and real-time RAG systems with low-latency retrieval and continuous indexing.

---

## 1. Identity and Mission

Implement RAG systems optimized for real-time and streaming use cases, where retrieval latency, freshness, and continuous updates are critical. This includes streaming token-by-token generation, incremental indexing, live data integration, and maintaining sub-second end-to-end latency.

---

## 2. Theory & Fundamentals

### 2.1 Streaming Generation

Streaming generation yields tokens as they become available, reducing perceived latency:

```
Traditional: |----Query----|----Retrieve----|--------Generate--------|----Response----|
                    100ms           200ms               2000ms                  0ms

Streaming:   |----Query----|----Retrieve----|---Gen1---Gen2---Gen3---...|---Response---|
                    100ms           200ms             token-by-token
```

### 2.2 Real-Time Retrieval Requirements

**Latency Targets:**
- P50 retrieval latency: <50ms
- P99 retrieval latency: <200ms
- End-to-end (query to first token): <500ms

**Throughput Targets:**
- Queries per second: 100-10,000+ QPS
- Concurrent users: 100-1000+

### 2.3 Continuous Indexing

Real-time systems require incremental updates:
- Document addition without full re-index
- Incremental embedding updates
- Versioned document storage

### 2.4 Architecture Patterns

**Synchronous RAG:**
```
Query → Retrieve → Generate → Response
```

**Streaming RAG:**
```
Query → Retrieve → Stream Generate → (first token)
                        ↓
                  (continue streaming)
                        ↓
                  (complete response)
```

**Pipelined RAG:**
```
Query 1 →        Retrieve → Generate → Response
Query 2 → Retrieve → Generate → Response
Query 3 → Retrieve → Generate → Response
```

---

## 3. Implementation Patterns

### Pattern 1: Streaming RAG with Async Retrieval

```python
import asyncio
import time
from typing import List, Dict, Optional, AsyncIterator, Callable
from dataclasses import dataclass
import numpy as np

@dataclass
class RetrievalResult:
    """A retrieval result with metadata."""
    document: str
    score: float
    source: str = "vector"
    latency_ms: float = 0.0

@dataclass
class StreamChunk:
    """A chunk of the stream."""
    text: str
    is_final: bool = False
    metadata: Dict = None

class AsyncVectorStore:
    """
    Async vector store for real-time retrieval.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "hnsw",
    ):
        import faiss
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # In-memory storage
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []

        # FAISS index
        self.index = None
        self._init_index(index_type)

    def _init_index(self, index_type: str):
        """Initialize FAISS index."""
        import faiss

        if index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 16)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

    async def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict] = None,
    ):
        """Add documents to the index asynchronously."""
        await asyncio.sleep(0)  # Yield control

        self.documents.extend(documents)
        self.document_metadata.extend(metadata or [{}] * len(documents))

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)

        self.index.add(normalized.astype('float32'))

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Search for similar documents asynchronously."""
        start = time.perf_counter()

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        query_norm = query_embedding / (norm + 1e-9)

        # Search
        scores, indices = self.index.search(
            query_norm.reshape(1, -1).astype('float32'),
            top_k
        )

        latency = (time.perf_counter() - start) * 1000

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(score),
                    latency_ms=latency,
                ))

        return results

    async def search_with_filter(
        self,
        query_embedding: np.ndarray,
        filter_fn: Callable[[Dict], bool],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Search with metadata filtering."""
        all_results = await self.search(query_embedding, top_k * 3)

        filtered = []
        for result in all_results:
            idx = self.documents.index(result.document)
            meta = self.document_metadata[idx]
            if filter_fn(meta):
                filtered.append(result)
                if len(filtered) >= top_k:
                    break

        return filtered


class StreamingRAG:
    """
    Streaming RAG with async retrieval and generation.
    """

    def __init__(
        self,
        vector_store: AsyncVectorStore,
        embedding_model: Any,
        generator: Any,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generator = generator

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Asynchronously retrieve relevant documents."""
        # Embed query
        query_embedding = await self._embed_async(query)

        # Search
        results = await self.vector_store.search(query_embedding, top_k)

        return results

    async def _embed_async(self, text: str) -> np.ndarray:
        """Async embedding."""
        # Simulate async embedding (replace with actual async call)
        await asyncio.sleep(0)
        return self.embedding_model.encode(text)

    async def stream_generate(
        self,
        query: str,
        top_k: int = 5,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream generation with retrieved context.
        Yields tokens as they become available.
        """
        # Step 1: Retrieve concurrently with generation setup
        results = await self.retrieve_async(query, top_k)

        # Step 2: Build context
        context = "\n\n".join([
            f"[Source {i+1}] {r.document}"
            for i, r in enumerate(results)
        ])

        # Step 3: Stream generation
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer (stream one token at a time):"""

        # Stream tokens from generator
        async for token in self.generator.stream_generate(prompt):
            yield StreamChunk(
                text=token,
                is_final=False,
                metadata={"retrieved_docs": len(results)},
            )

        yield StreamChunk(text="", is_final=True)

    async def run(self, query: str) -> str:
        """Run RAG and collect full response."""
        response = []
        async for chunk in self.stream_generate(query):
            if chunk.text:
                response.append(chunk.text)
        return "".join(response)


class PipelinedRAG:
    """
    Pipelined RAG for improved throughput.
    Overlaps retrieval and generation.
    """

    def __init__(
        self,
        vector_store: AsyncVectorStore,
        embedding_model: Any,
        generator: Any,
        pipeline_depth: int = 3,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generator = generator
        self.pipeline_depth = pipeline_depth

    async def run_pipelined(
        self,
        queries: List[str],
    ) -> List[str]:
        """
        Run multiple queries in a pipeline.
        Overlaps retrieval for query N+1 with generation for query N.
        """
        if not queries:
            return []

        # Queue for results
        results = [None] * len(queries)
        retrieval_queue = asyncio.Queue()
        generation_queue = asyncio.Queue()

        # Start retrieval task
        retrieval_task = asyncio.create_task(
            self._retrieval_worker(retrieval_queue, generation_queue)
        )

        # Start generation workers
        generation_tasks = [
            asyncio.create_task(
                self._generation_worker(generation_queue, results, i)
            )
            for i in range(self.pipeline_depth)
        ]

        # Enqueue all queries
        for i, query in enumerate(queries):
            await retrieval_queue.put((i, query))

        # Signal completion
        await retrieval_queue.put((None, None))

        # Wait for all tasks
        await retrieval_task
        await asyncio.gather(*generation_tasks)

        return results

    async def _retrieval_worker(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
    ):
        """Worker that performs retrieval."""
        while True:
            idx, query = await input_queue.get()

            if query is None:
                output_queue.put_nowait((None, None, None))
                break

            # Retrieve
            query_embedding = self.embedding_model.encode(query)
            docs = await self.vector_store.search(query_embedding, top_k=5)

            # Pass to generation worker
            output_queue.put_nowait((idx, query, docs))

    async def _generation_worker(
        self,
        input_queue: asyncio.Queue,
        results: List,
        worker_id: int,
    ):
        """Worker that generates responses."""
        while True:
            idx, query, docs = await input_queue.get()

            if query is None:
                break

            # Build context
            context = "\n\n".join([d.document for d in docs])

            # Generate
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            answer = await self.generator.generate(prompt)

            results[idx] = answer
```

### Pattern 2: Real-Time Document Indexing

```python
import asyncio
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

class IndexOperation(Enum):
    """Types of index operations."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"

@dataclass
class IndexTask:
    """A task for the indexer."""
    operation: IndexOperation
    document_id: str
    document: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class RealTimeIndexer:
    """
    Real-time indexer with batched updates.
    Buffers small updates and flushes periodically.
    """

    def __init__(
        self,
        vector_store: AsyncVectorStore,
        embedding_model: Callable,
        batch_size: int = 32,
        flush_interval_seconds: float = 1.0,
        max_queue_size: int = 10000,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds

        # Task queue
        self.task_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Document storage
        self.documents: Dict[str, Dict] = {}

        # Indexer thread
        self.running = False
        self.indexer_thread: threading.Thread = None

    def start(self):
        """Start the background indexer."""
        self.running = True
        self.indexer_thread = threading.Thread(target=self._indexer_loop)
        self.indexer_thread.daemon = True
        self.indexer_thread.start()

    def stop(self):
        """Stop the background indexer."""
        self.running = False
        if self.indexer_thread:
            self.indexer_thread.join(timeout=5.0)

    def add_document(
        self,
        document_id: str,
        document: str,
        metadata: Dict = None,
    ):
        """Add a document to the indexing queue."""
        task = IndexTask(
            operation=IndexOperation.ADD,
            document_id=document_id,
            document=document,
            metadata=metadata,
        )

        try:
            self.task_queue.put_nowait(task)
        except queue.Full:
            raise RuntimeError("Index queue full, try again later")

    def update_document(
        self,
        document_id: str,
        document: str,
        metadata: Dict = None,
    ):
        """Update an existing document."""
        task = IndexTask(
            operation=IndexOperation.UPDATE,
            document_id=document_id,
            document=document,
            metadata=metadata,
        )
        self.task_queue.put(task)

    def delete_document(self, document_id: str):
        """Delete a document from the index."""
        task = IndexTask(
            operation=IndexOperation.DELETE,
            document_id=document_id,
        )
        self.task_queue.put(task)

    def _indexer_loop(self):
        """Main indexing loop."""
        buffer: List[IndexTask] = []
        last_flush = time.time()

        while self.running:
            # Try to fill buffer
            try:
                task = self.task_queue.get(timeout=0.1)
                buffer.append(task)
            except queue.Empty:
                pass

            # Flush if buffer full or timeout
            should_flush = (
                len(buffer) >= self.batch_size or
                (buffer and time.time() - last_flush >= self.flush_interval)
            )

            if should_flush and buffer:
                asyncio.run(self._flush_buffer(buffer))
                buffer = []
                last_flush = time.time()

        # Final flush
        if buffer:
            asyncio.run(self._flush_buffer(buffer))

    async def _flush_buffer(self, tasks: List[IndexTask]):
        """Flush buffer to vector store."""
        # Group by operation
        add_tasks = [t for t in tasks if t.operation == IndexOperation.ADD]
        update_tasks = [t for t in tasks if t.operation == IndexOperation.UPDATE]
        delete_ids = [t.document_id for t in tasks if t.operation == IndexOperation.DELETE]

        # Process adds and updates
        if add_tasks or update_tasks:
            all_tasks = add_tasks + update_tasks
            documents = [t.document for t in all_tasks]
            doc_ids = [t.document_id for t in all_tasks]
            metadatas = [t.metadata for t in all_tasks]

            # Embed
            embeddings = self.embedding_model.encode_batch(documents)

            # Add to vector store
            await self.vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadata=metadatas,
            )

            # Update document storage
            for doc_id, doc, meta in zip(doc_ids, documents, metadatas):
                self.documents[doc_id] = {
                    "document": doc,
                    "metadata": meta,
                    "updated_at": time.time(),
                }

        # Process deletes
        if delete_ids:
            # Note: FAISS doesn't support direct deletion,
            # so we mark as deleted in metadata
            for doc_id in delete_ids:
                if doc_id in self.documents:
                    self.documents[doc_id]["deleted"] = True
                    self.documents[doc_id]["deleted_at"] = time.time()


class IncrementalVectorIndex:
    """
    Incremental vector index that supports efficient updates.
    Uses versioning for consistent reads during updates.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        version_interval: int = 100,
    ):
        import faiss
        self.embedding_dim = embedding_dim
        self.version_interval = version_interval

        # Versioned storage
        self.current_version = 0
        self.documents_by_version: Dict[int, List[str]] = {0: []}
        self.indexes_by_version: Dict[int, Any] = {0: None}

        # Current state
        self.documents: List[str] = []
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.document_versions: Dict[str, int] = {}

    def add_document(self, document: str, embedding: np.ndarray):
        """Add a document."""
        idx = len(self.documents)
        self.documents.append(document)

        # Normalize and add to index
        norm = np.linalg.norm(embedding)
        normalized = embedding / (norm + 1e-9)
        self.index.add(normalized.reshape(1, -1).astype('float32'))

        self.document_versions[document] = self.current_version

        # Create new version periodically
        if idx % self.version_interval == 0:
            self._create_version()

    def _create_version(self):
        """Create a new version snapshot."""
        import faiss
        self.current_version += 1

        # Clone index
        self.indexes_by_version[self.current_version] = faiss.clone_index(self.index)
        self.documents_by_version[self.current_version] = list(self.documents)

    def search(
        self,
        query_embedding: np.ndarray,
        version: Optional[int] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """Search at a specific version for consistent reads."""
        if version is None:
            version = self.current_version

        # Get appropriate index
        target_version = min(version, self.current_version)
        docs = self.documents_by_version.get(target_version, self.documents)
        index = self.indexes_by_version.get(target_version, self.index)

        if index is None:
            index = self.index
            docs = self.documents

        # Search
        norm = np.linalg.norm(query_embedding)
        query_norm = query_embedding / (norm + 1e-9)
        scores, indices = index.search(query_norm.reshape(1, -1).astype('float32'), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(docs):
                results.append({
                    "document": docs[idx],
                    "score": float(score),
                    "version": target_version,
                })

        return results

    def get_consistent_snapshot(self) -> int:
        """Get a version number for consistent reads."""
        return self.current_version
```

### Pattern 3: Webhook-Based Live Data Integration

```python
import hashlib
import hmac
import json
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class WebhookEvent:
    """A webhook event from external source."""
    event_id: str
    event_type: str
    source: str
    timestamp: float
    payload: Dict

class WebhookReceiver:
    """
    Receive webhook events for real-time document updates.
    """

    def __init__(
        self,
        webhook_secret: Optional[str] = None,
        validator: Optional[Callable] = None,
    ):
        self.webhook_secret = webhook_secret
        self.validator = validator or self._default_validator
        self.handlers: Dict[str, Callable] = {}

    def _default_validator(self, event: WebhookEvent, signature: str) -> bool:
        """Validate webhook signature."""
        if not self.webhook_secret:
            return True

        expected = hmac.new(
            self.webhook_secret.encode(),
            json.dumps(event.payload).encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected)

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[WebhookEvent], None],
    ):
        """Register a handler for an event type."""
        self.handlers[event_type] = handler

    async def receive(
        self,
        payload: Dict,
        headers: Dict,
    ) -> bool:
        """Receive and process a webhook."""
        # Extract event info
        event = WebhookEvent(
            event_id=headers.get("X-Event-ID", str(time.time())),
            event_type=headers.get("X-Event-Type", "unknown"),
            source=headers.get("X-Source", "unknown"),
            timestamp=time.time(),
            payload=payload,
        )

        # Validate
        signature = headers.get("X-Signature", "")
        if not self.validator(event, signature):
            return False

        # Handle
        handler = self.handlers.get(event.event_type)
        if handler:
            await handler(event)

        return True


class LiveDataIntegration:
    """
    Integrate live data sources into RAG system.
    """

    def __init__(
        self,
        indexer: RealTimeIndexer,
        webhook_receiver: WebhookReceiver,
        embedding_model: Any,
    ):
        self.indexer = indexer
        self.webhook_receiver = webhook_receiver
        self.embedding_model = embedding_model

        # Register handlers
        self.webhook_receiver.register_handler("document.add", self._handle_add)
        self.webhook_receiver.register_handler("document.update", self._handle_update)
        self.webhook_receiver.register_handler("document.delete", self._handle_delete)

    async def _handle_add(self, event: WebhookEvent):
        """Handle document add event."""
        payload = event.payload

        document_id = payload.get("id")
        content = payload.get("content")
        metadata = payload.get("metadata", {})

        if document_id and content:
            self.indexer.add_document(document_id, content, metadata)

    async def _handle_update(self, event: WebhookEvent):
        """Handle document update event."""
        payload = event.payload

        document_id = payload.get("id")
        content = payload.get("content")
        metadata = payload.get("metadata", {})

        if document_id and content:
            self.indexer.update_document(document_id, content, metadata)

    async def _handle_delete(self, event: WebhookEvent):
        """Handle document delete event."""
        payload = event.payload

        document_id = payload.get("id")

        if document_id:
            self.indexer.delete_document(document_id)

    async def start_listening(self, host: str = "0.0.0.0", port: int = 8080):
        """Start listening for webhooks."""
        from aiohttp import web

        async def webhook_handler(request):
            payload = await request.json()
            headers = dict(request.headers)

            success = await self.webhook_receiver.receive(payload, headers)

            if success:
                return web.Response(status=200, text="OK")
            else:
                return web.Response(status=400, text="Invalid signature")

        app = web.Application()
        app.router.add_post("/webhook", webhook_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        print(f"Listening for webhooks on {host}:{port}")

        # Keep running
        await asyncio.Event().wait()


class PollingDataSource:
    """
    Poll external data sources for updates.
    Alternative to webhooks.
    """

    def __init__(
        self,
        indexer: RealTimeIndexer,
        embedding_model: Any,
        poll_interval: float = 60.0,
    ):
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.poll_interval = poll_interval

        self.data_sources: List[Callable[[], List[Dict]]] = []
        self.running = False
        self.last_snapshots: Dict[str, str] = {}

    def register_source(
        self,
        name: str,
        fetcher: Callable[[], List[Dict]],
    ):
        """Register a data source fetcher."""
        self.data_sources.append(fetcher)

    async def start(self):
        """Start polling."""
        self.running = True

        while self.running:
            await self._poll_all()
            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop polling."""
        self.running = False

    async def _poll_all(self):
        """Poll all registered sources."""
        for fetcher in self.data_sources:
            try:
                documents = await fetcher()

                for doc in documents:
                    doc_id = doc.get("id")
                    content = doc.get("content")

                    if not doc_id or not content:
                        continue

                    # Check if changed
                    doc_hash = hashlib.md5(content.encode()).hexdigest()
                    if self.last_snapshots.get(doc_id) != doc_hash:
                        # New or changed
                        self.indexer.update_document(doc_id, content, doc.get("metadata"))
                        self.last_snapshots[doc_id] = doc_hash

            except Exception as e:
                print(f"Error polling source: {e}")
```

### Pattern 4: Latency-Optimized Retrieval

```python
import time
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class LatencyMetrics:
    """Latency metrics for a retrieval operation."""
    query_embedding_ms: float = 0.0
    index_search_ms: float = 0.0
    total_ms: float = 0.0
    cache_hit: bool = False

class LowLatencyRetriever:
    """
    Retriever optimized for minimal latency.
    Uses caching, pre-computation, and optimized paths.
    """

    def __init__(
        self,
        vector_store: AsyncVectorStore,
        embedding_model: Any,
        cache_size: int = 10000,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # Query cache
        self.query_cache: Dict[str, Tuple[np.ndarray, List]] = {}
        self.cache_order: List[str] = []
        self.cache_size = cache_size

        # Pre-computed embeddings for common queries
        self.precomputed_embeddings: Dict[str, np.ndarray] = {}

        # Metrics
        self.total_queries = 0
        self.cache_hits = 0

    def _get_cache_key(self, query: str) -> str:
        """Get cache key for query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _cache_get(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding for query."""
        key = self._get_cache_key(query)

        if key in self.query_cache:
            # Move to end of order
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.query_cache[key][0]

        return None

    def _cache_set(self, query: str, embedding: np.ndarray, results: List):
        """Cache query embedding and results."""
        key = self._get_cache_key(query)

        # Evict if needed
        while len(self.cache_order) >= self.cache_size:
            old_key = self.cache_order.pop(0)
            self.query_cache.pop(old_key, None)

        self.query_cache[key] = (embedding, results)
        self.cache_order.append(key)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query with caching."""
        # Check precomputed first
        if query in self.precomputed_embeddings:
            return self.precomputed_embeddings[query]

        # Check cache
        cached = self._cache_get(query)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Encode
        embedding = self.embedding_model.encode(query)

        return embedding

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> Tuple[List[RetrievalResult], LatencyMetrics]:
        """Retrieve with latency tracking."""
        metrics = LatencyMetrics()
        self.total_queries += 1

        overall_start = time.perf_counter()

        # Encode query
        embed_start = time.perf_counter()
        query_embedding = self.encode_query(query)
        metrics.query_embedding_ms = (time.perf_counter() - embed_start) * 1000

        # Search
        search_start = time.perf_counter()
        results = await self.vector_store.search(query_embedding, top_k)
        metrics.index_search_ms = (time.perf_counter() - search_start) * 1000

        # Cache results for common queries
        if use_cache and query not in self.precomputed_embeddings:
            self._cache_set(query, query_embedding, results)

        metrics.total_ms = (time.perf_counter() - overall_start) * 1000
        metrics.cache_hit = query in self.query_cache

        return results, metrics

    def add_precomputed_query(self, query: str, embedding: np.ndarray):
        """Add a precomputed embedding for common query."""
        self.precomputed_embeddings[query] = embedding

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        hit_rate = self.cache_hits / self.total_queries if self.total_queries > 0 else 0
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self.cache_order),
        }


class PredictiveRetrieval:
    """
    Predictive retrieval: anticipate queries and pre-fetch.
    """

    def __init__(
        self,
        retriever: LowLatencyRetriever,
        vector_store: AsyncVectorStore,
    ):
        self.retriever = retriever
        self.vector_store = vector_store
        self.query_predictor: Optional[Any] = None  # Model to predict next query
        self.prefetched: Dict[str, List] = {}

    def set_query_predictor(self, predictor: Any):
        """Set model to predict next queries."""
        self.query_predictor = predictor

    async def prefetch(self, query: str, top_k: int = 5):
        """Pre-fetch for a query."""
        # Embed query
        query_embedding = self.retriever.encode_query(query)

        # Search and cache
        results = await self.vector_store.search(query_embedding, top_k)
        self.prefetched[query] = results

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve, using prefetched if available."""
        if query in self.prefetched:
            return self.prefetched.pop(query)

        # Not prefetched, do regular retrieval
        results, _ = await self.retriever.retrieve(query, top_k)
        return results

    async def predict_and_prefetch(
        self,
        current_query: str,
        top_k: int = 5,
    ):
        """Predict next queries and prefetch."""
        if not self.query_predictor:
            return

        # Predict next queries
        next_queries = self.query_predictor.predict_next(current_query)

        # Prefetch in parallel
        tasks = [self.prefetch(q, top_k) for q in next_queries]
        await asyncio.gather(*tasks)


class TieredRetrieval:
    """
    Tiered retrieval: L0 (ultrafast cache) → L1 (memory) → L2 (disk/remote).
    """

    def __init__(self):
        # L0: Hot cache (in-memory, sub-ms)
        self.l0_cache: Dict[str, List[RetrievalResult]] = {}

        # L1: Warm storage (vector DB, ~10ms)
        self.l1_store: Optional[AsyncVectorStore] = None

        # L2: Cold storage (disk/remote, ~100ms)
        self.l2_store: Optional[Any] = None

    def set_l1_store(self, store: AsyncVectorStore):
        """Set L1 storage."""
        self.l1_store = store

    def set_l2_store(self, store: Any):
        """Set L2 storage."""
        self.l2_store = store

    def l0_get(self, query_hash: str) -> Optional[List[RetrievalResult]]:
        """Get from L0 cache."""
        return self.l0_cache.get(query_hash)

    def l0_put(self, query_hash: str, results: List[RetrievalResult]):
        """Put into L0 cache."""
        if len(self.l0_cache) > 1000:
            # Simple eviction: remove oldest
            oldest = next(iter(self.l0_cache))
            self.l0_cache.pop(oldest)
        self.l0_cache[query_hash] = results

    async def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Tiered retrieval."""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()

        # Check L0
        l0_results = self.l0_get(query_hash)
        if l0_results:
            return l0_results

        # Check L1
        if self.l1_store:
            l1_results = await self.l1_store.search(query_embedding, top_k)
            if l1_results:
                self.l0_put(query_hash, l1_results)
                return l1_results

        # Check L2
        if self.l2_store:
            l2_results = await self.l2_store.search(query_embedding, top_k)
            # Populate L1
            if self.l1_store and l2_results:
                # Add to L1
                pass
            self.l0_put(query_hash, l2_results)
            return l2_results

        return []
```

### Pattern 5: Circuit Breaker and Fallback Patterns

```python
import time
from typing import List, Optional, Dict
from enum import Enum
import asyncio

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

class CircuitBreaker:
    """
    Circuit breaker for retrieval failures.
    Stops calling failing service and returns cached/fallback results.
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    def record_success(self):
        """Record a successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

    def can_attempt(self) -> bool:
        """Check if we can attempt a call."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def _on_attempt(self):
        """Called when an attempt is made in half-open state."""
        self.half_open_calls += 1


class ResilientRetrieval:
    """
    Resilient retrieval with circuit breaker and fallback.
    """

    def __init__(
        self,
        primary_retriever: Any,
        fallback_retriever: Optional[Any] = None,
        cache_retriever: Optional[Any] = None,
    ):
        self.primary = primary_retriever
        self.fallback = fallback_retriever
        self.cache = cache_retriever
        self.circuit_breaker = CircuitBreaker()

        # Metrics
        self.metrics = {
            "total": 0,
            "primary_success": 0,
            "fallback_used": 0,
            "cache_used": 0,
            "circuit_open": 0,
        }

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve with resilience patterns."""
        self.metrics["total"] += 1

        # Check cache first
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                self.metrics["cache_used"] += 1
                return cached

        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            self.metrics["circuit_open"] += 1
            return await self._fallback_retrieve(query, top_k)

        # Try primary
        try:
            if self.circuit_breaker.state == CircuitState.HALF_OPEN:
                self.circuit_breaker._on_attempt()

            results = await self.primary.retrieve(query, top_k)
            self.circuit_breaker.record_success()
            self.metrics["primary_success"] += 1

            # Cache results
            if self.cache:
                self.cache.put(query, results)

            return results

        except Exception as e:
            self.circuit_breaker.record_failure()

            if self.metrics["total"] % 100 == 0:
                print(f"Retrieval error: {e}, circuit state: {self.circuit_breaker.state}")

            return await self._fallback_retrieve(query, top_k)

    async def _fallback_retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Use fallback retriever."""
        self.metrics["fallback_used"] += 1

        if self.fallback:
            try:
                return await self.fallback.retrieve(query, top_k)
            except Exception:
                pass

        # Return empty results if everything fails
        return []

    def get_metrics(self) -> Dict:
        """Get retrieval metrics."""
        total = self.metrics["total"]
        return {
            **self.metrics,
            "cache_hit_rate": self.metrics["cache_used"] / total if total > 0 else 0,
            "fallback_rate": self.metrics["fallback_used"] / total if total > 0 else 0,
            "circuit_open_rate": self.metrics["circuit_open"] / total if total > 0 else 0,
        }


class RateLimitedRetriever:
    """
    Rate-limited retriever to prevent overwhelming backend services.
    """

    def __init__(
        self,
        retriever: Any,
        max_requests_per_second: float = 100.0,
    ):
        self.retriever = retriever
        self.rate_limit = max_requests_per_second

        self.tokens = max_requests_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve with rate limiting."""
        async with self.lock:
            # Add tokens based on time elapsed
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.rate_limit,
                self.tokens + elapsed * self.rate_limit
            )
            self.last_update = now

            # Wait if no tokens
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate_limit
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

        return await self.retriever.retrieve(query, top_k)
```

---

## 4. Framework Integration

### LangChain LCEL Integration

```python
from langchain.retrievers import EnsembleRetriever
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Streaming RAG chain with LCEL
retriever = your_retriever

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# For streaming
async def stream_rag(query: str):
    async for chunk in chain.astream(query):
        yield chunk
```

### Redis Integration for Caching

```python
import redis.asyncio as redis

class RedisQueryCache:
    """Redis-based query cache for distributed systems."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    async def get(self, query: str) -> Optional[List[Dict]]:
        """Get cached results."""
        key = self._hash_query(query)
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set(self, query: str, results: List[Dict], ttl: int = 3600):
        """Cache results with TTL."""
        key = self._hash_query(query)
        await self.redis.setex(key, ttl, json.dumps(results))

    def _hash_query(self, query: str) -> str:
        """Hash query for cache key."""
        return f"rag:query:{hashlib.md5(query.encode()).hexdigest()}"
```

---

## 5. Performance Considerations

### Latency Targets by Tier

| Component | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Query embedding | 10ms | 30ms | 50ms |
| Vector search (1M docs) | 20ms | 50ms | 100ms |
| Cache hit | 1ms | 2ms | 5ms |
| Network (same DC) | 5ms | 15ms | 30ms |
| End-to-end streaming | 100ms | 300ms | 500ms |

### Optimization Checklist

1. **Query embedding**: Pre-compute embeddings for common queries
2. **Caching**: Multi-level cache (L0/L1/L2) for queries and results
3. **Index optimization**: Use HNSW with appropriate parameters
4. **Connection pooling**: Reuse connections to vector store
5. **Async I/O**: Use async operations throughout
6. **Batch processing**: Batch embeddings when indexing

---

## 6. Common Pitfalls

1. **Blocking in Async Code**: Using sync operations in async coroutines blocks event loop

2. **Cache Stampede**: Many requests recomputing same query when cache expires

3. **Stale Cache**: Returning outdated results without versioning

4. **No Circuit Breaker**: Cascading failures when backend is down

5. **Index Lock Contention**: Blocking writes during reads

6. **Memory Growth**: Unbounded caches causing OOM

---

## 7. Research References

1. https://arxiv.org/abs/2305.16733 — "StreamingRAG: Real-time Streaming RAG System"

2. https://arxiv.org/abs/2308.13532 — "Low-Latency Retrieval for RAG Systems"

3. https://arxiv.org/abs/2309.07683 — "Continuous Learning for RAG"

4. https://arxiv.org/abs/2305.13028 — "Predictive Caching for RAG"

5. https://arxiv.org/abs/2306.03189 — "Real-time Document Updates in RAG"

6. https://redis.io/docs/reference/patterns/caching/ — Redis caching patterns

7. https://aws.amazon.com/builders-library/timeouts-retries-backoff/ — Exponential backoff

8. https://github.com/redis/docs — Redis documentation for caching

---

## 8. Uncertainty and Limitations

**Not Covered:** Distributed vector stores (see dedicated skills), multi-region deployment, real-time deduplication.

**Production Considerations:** Real-time RAG requires careful monitoring of latency percentiles, circuit breaker configuration for graceful degradation, and capacity planning for peak loads.

(End of file - total 1470 lines)