# Agent Memory Strategies: Comprehensive Implementation Guide

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026  
**Version:** 1.0

## Table of Contents

1. [Introduction](#introduction)
2. [Memory Architecture Overview](#memory-architecture-overview)
3. [Short-Term Memory Strategies](#short-term-memory-strategies)
4. [Long-Term Memory Strategies](#long-term-memory-strategies)
5. [Memory Retrieval Patterns](#memory-retrieval-patterns)
6. [Context Window Management](#context-window-management)
7. [Conversation History Management](#conversation-history-management)
8. [Knowledge Base Integration](#knowledge-base-integration)
9. [Performance Considerations](#performance-considerations)
10. [Trade-offs Between Approaches](#trade-offs-between-approaches)

---

## Introduction

Memory is the critical infrastructure that transforms stateless LLM calls into coherent, evolving agent systems. Without proper memory architecture, agents forget conversations, repeat mistakes, and fail to leverage past experiences.

By 2026, the field has converged on a three-layer memory model:

1. **Working Memory** (Context Window): For immediate reasoning
2. **Session Memory** (Short-term): For conversation continuity
3. **Episodic/Semantic Memory** (Long-term): For learning and retrieval

This guide covers implementation patterns for each layer, with code examples for AGNO, LangChain/LangGraph, and CrewAI.

---

## Memory Architecture Overview

### The Three-Layer Model

```
┌─────────────────────────────────────────────────┐
│     Working Memory (Context Window)             │
│     - Messages: last 10-50                      │
│     - Pinned context: requirements              │
│     - Active state: current task                │
│     - Lifespan: single conversation turn       │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│     Session Memory (Short-term)                 │
│     - Conversation history: last N turns        │
│     - Task context: current task state          │
│     - Execution results: tool outputs           │
│     - Lifespan: single session (hours)         │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│     Long-term Memory (Episodic/Semantic)        │
│     - Knowledge base: documents, specs          │
│     - Learned patterns: successful approaches   │
│     - User profiles: preferences, history       │
│     - Lifespan: permanent (until updated)      │
└─────────────────────────────────────────────────┘
```

---

## Short-Term Memory Strategies

Short-term memory enables coherent single-session interactions.

### Strategy 1: Sliding Window Over Conversation History

Keep only the most recent N messages in context.

```python
from typing import TypedDict, Annotated
import operator
from datetime import datetime, timedelta

class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    message_count: int
    session_start: str

class SlidingWindowMemory:
    """Maintain a sliding window of recent messages."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def update_context(self, state: ConversationState) -> ConversationState:
        """Keep only recent messages, discard old ones."""
        messages = state["messages"]
        
        # Keep system messages and recent user/assistant messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        conversation_messages = [m for m in messages if m.get("role") != "system"]
        
        # Slide the window: keep only the most recent N conversation messages
        recent_messages = conversation_messages[-self.window_size:]
        
        # Combine: system messages + recent conversation
        trimmed_messages = system_messages + recent_messages
        
        return {
            **state,
            "messages": trimmed_messages,
            "message_count": len(trimmed_messages),
            "messages_discarded": max(0, len(conversation_messages) - self.window_size)
        }

# Example usage
memory = SlidingWindowMemory(window_size=10)

state = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        *[{"role": "user", "content": f"Question {i}"} for i in range(50)],
        *[{"role": "assistant", "content": f"Answer {i}"} for i in range(50)],
    ],
    "message_count": 0,
    "session_start": datetime.now().isoformat()
}

trimmed = memory.update_context(state)
print(f"Kept {len(trimmed['messages'])} messages, discarded {trimmed['messages_discarded']}")
# Output: Kept 11 messages (1 system + 10 recent), discarded 90
```

**Advantages:**
- Simple to implement
- Predictable memory usage
- Works with any LLM

**Disadvantages:**
- Loses historical context
- No way to recover forgotten information
- Early conversation context disappears

### Strategy 2: Summarization Memory

Periodically summarize conversation history to compress it.

```python
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMemory, ConversationSummaryBufferMemory

class AgentSummarizationMemory:
    """Summarize conversation when it exceeds size threshold."""
    
    def __init__(self, llm: ChatOpenAI, max_messages: int = 20, summary_k: int = 5):
        self.llm = llm
        self.max_messages = max_messages
        self.summary_k = summary_k
        self.summary_history = []
    
    def should_summarize(self, messages: list) -> bool:
        """Check if summarization is needed."""
        return len(messages) > self.max_messages
    
    async def summarize_messages(self, messages: list) -> str:
        """Summarize conversation messages."""
        conversation_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in messages[-self.summary_k:]
        ])
        
        summarization_prompt = f"""Summarize this conversation concisely, 
        preserving key decisions, facts, and user preferences:
        
{conversation_text}

Summary (2-3 sentences):"""
        
        response = await self.llm.ainvoke([
            {"role": "user", "content": summarization_prompt}
        ])
        return response.content
    
    async def compress_memory(self, state: ConversationState) -> ConversationState:
        """Compress conversation by summarizing old messages."""
        if not self.should_summarize(state["messages"]):
            return state
        
        messages = state["messages"]
        system_messages = [m for m in messages if m.get("role") == "system"]
        conversation = [m for m in messages if m.get("role") != "system"]
        
        # Summarize oldest half
        old_messages = conversation[:len(conversation)//2]
        recent_messages = conversation[len(conversation)//2:]
        
        summary = await self.summarize_messages(old_messages)
        self.summary_history.append(summary)
        
        # Create summary message
        summary_message = {
            "role": "system",
            "content": f"Previous conversation summary: {summary}"
        }
        
        # Combine: system messages + summary + recent messages
        compressed_messages = system_messages + [summary_message] + recent_messages
        
        return {
            **state,
            "messages": compressed_messages,
            "summary_count": len(self.summary_history)
        }

# Using with LangChain's built-in summarization
from langchain.memory import ConversationSummaryBufferMemory

summary_memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-4o"),
    max_token_limit=1000,  # Summarize when exceeding token limit
    human_prefix="User",
    ai_prefix="Assistant"
)

# Add messages
summary_memory.save_context(
    {"input": "What's the capital of France?"},
    {"output": "The capital of France is Paris."}
)

# Memory automatically summarizes when token limit exceeded
```

**Advantages:**
- Preserves important context
- Reduces token usage
- More intelligent than simple window truncation

**Disadvantages:**
- Summarization introduces information loss
- Adds latency (LLM call for summarization)
- May lose nuanced details

### Strategy 3: Hierarchical Message Grouping

Group messages by topic/task, store summaries of groups.

```python
from pydantic import BaseModel
from enum import Enum

class MessageType(str, Enum):
    PROBLEM_ANALYSIS = "problem_analysis"
    DATA_GATHERING = "data_gathering"
    SOLUTION_DESIGN = "solution_design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"

class MessageGroup(BaseModel):
    """Group of related messages."""
    type: MessageType
    messages: list[dict]
    summary: str
    timestamp: str
    resolution: str  # How this group was resolved

class HierarchicalMemory:
    """Store conversation in grouped, summarized form."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.message_groups: list[MessageGroup] = []
    
    async def classify_message(self, message: str) -> MessageType:
        """Classify message into a category."""
        classification_prompt = f"""Classify this message into one category:
        
Message: {message}

Categories:
- problem_analysis: Understanding the problem
- data_gathering: Collecting information
- solution_design: Planning the solution
- implementation: Writing/building code
- testing: Testing and validation

Return only the category name."""
        
        response = await self.llm.ainvoke([
            {"role": "user", "content": classification_prompt}
        ])
        return MessageType(response.content.strip().lower())
    
    async def group_and_summarize(self, messages: list[dict]) -> list[MessageGroup]:
        """Group messages by type and summarize each group."""
        groups = {}
        
        # Classify and group messages
        for msg in messages:
            msg_type = await self.classify_message(msg["content"])
            if msg_type not in groups:
                groups[msg_type] = []
            groups[msg_type].append(msg)
        
        # Summarize each group
        message_groups = []
        for msg_type, group_messages in groups.items():
            conversation = "\n".join([
                f"{m['role']}: {m['content']}"
                for m in group_messages
            ])
            
            summary_prompt = f"""Summarize this {msg_type} conversation:
            
{conversation}

Summary (1-2 sentences):"""
            
            response = await self.llm.ainvoke([
                {"role": "user", "content": summary_prompt}
            ])
            
            message_groups.append(MessageGroup(
                type=msg_type,
                messages=group_messages,
                summary=response.content,
                timestamp=datetime.now().isoformat(),
                resolution="pending"
            ))
        
        return message_groups

# Usage
memory = HierarchicalMemory(llm=ChatOpenAI(model="gpt-4o"))

messages = [
    {"role": "user", "content": "I need to build a REST API"},
    {"role": "assistant", "content": "What should it do?"},
    {"role": "user", "content": "It should manage user accounts"},
    {"role": "assistant", "content": "Here's a design..."},
    {"role": "user", "content": "Let's implement it"},
    {"role": "assistant", "content": "Here's the code..."},
]

grouped = asyncio.run(memory.group_and_summarize(messages))
```

**Advantages:**
- Preserves task-specific context
- Enables retrieval by task type
- Structured organization

**Disadvantages:**
- Requires classification (adds complexity)
- Manual resolution tracking needed
- More storage overhead

---

## Long-Term Memory Strategies

Long-term memory enables learning and knowledge accumulation across sessions.

### Strategy 1: Episodic Memory (Experience Recording)

Record complete experiences with outcomes for later retrieval.

```python
from datetime import datetime
from typing import Optional
import json

class Episode(BaseModel):
    """A complete interaction episode."""
    id: str
    timestamp: str
    task_type: str
    user_query: str
    agent_reasoning: str
    actions_taken: list[dict]
    outcome: str  # success, partial, failure
    result: str
    duration_seconds: float
    tokens_used: int
    user_satisfaction: Optional[int] = None  # 1-5 rating

class EpisodicMemory:
    """Store complete episodes for retrieval and learning."""
    
    def __init__(self, storage_path: str = "episodes.json"):
        self.storage_path = storage_path
        self.episodes = []
        self.load()
    
    def record_episode(self, episode: Episode):
        """Record a completed episode."""
        self.episodes.append(episode)
        self.save()
    
    def retrieve_similar(self, query: str, task_type: str, top_k: int = 3) -> list[Episode]:
        """Retrieve similar past episodes."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Filter by task type
        same_type = [e for e in self.episodes if e.task_type == task_type]
        
        if not same_type:
            return []
        
        # Find most similar by text similarity
        vectorizer = TfidfVectorizer()
        queries = [query] + [e.user_query for e in same_type]
        vectors = vectorizer.fit_transform(queries)
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        
        # Return top K by similarity
        indices = similarities.argsort()[-top_k:][::-1]
        return [same_type[i] for i in indices if similarities[i] > 0.3]
    
    def get_success_rate_by_type(self, task_type: str) -> float:
        """Get success rate for a task type."""
        matching = [e for e in self.episodes if e.task_type == task_type]
        if not matching:
            return 0.0
        successful = sum(1 for e in matching if e.outcome == "success")
        return successful / len(matching)
    
    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump([e.model_dump() for e in self.episodes], f)
    
    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.episodes = [Episode(**item) for item in data]
        except FileNotFoundError:
            self.episodes = []

# Usage in agent
episodic_memory = EpisodicMemory()

# After task completion
episode = Episode(
    id=f"ep_{datetime.now().timestamp()}",
    timestamp=datetime.now().isoformat(),
    task_type="data_analysis",
    user_query="Analyze sales trends from Q1-Q4",
    agent_reasoning="Need to load data, compute trends, create visualizations",
    actions_taken=[
        {"tool": "load_csv", "file": "sales.csv"},
        {"tool": "analyze", "method": "time_series"},
        {"tool": "visualize", "type": "line_chart"}
    ],
    outcome="success",
    result="Identified 15% growth trend",
    duration_seconds=42.3,
    tokens_used=2847,
    user_satisfaction=5
)
episodic_memory.record_episode(episode)

# Before new task, retrieve similar episodes
similar = episodic_memory.retrieve_similar(
    "Show me last quarter sales performance",
    task_type="data_analysis"
)
print(f"Found {len(similar)} similar episodes to learn from")
```

**Advantages:**
- Complete context preserved
- Enables learning from success/failure
- Supports reasoning by analogy
- Trackable outcomes

**Disadvantages:**
- Large storage requirements
- Retrieval must be efficient
- Privacy concerns with storing complete interactions

### Strategy 2: Semantic Memory (Knowledge Base)

Store distilled knowledge as semantic triples or embeddings.

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.docstore.document import Document

class SemanticMemory:
    """Store semantic knowledge in vector database."""
    
    def __init__(self, persist_directory: str = "semantic_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name="semantic_knowledge"
        )
    
    def add_fact(self, fact: str, source: str, category: str):
        """Add a semantic fact."""
        doc = Document(
            page_content=fact,
            metadata={
                "source": source,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.9
            }
        )
        self.vectorstore.add_documents([doc])
    
    def add_rule(self, rule: str, condition: str, action: str):
        """Add a conditional rule."""
        fact = f"IF {condition} THEN {action}"
        doc = Document(
            page_content=fact,
            metadata={
                "type": "rule",
                "condition": condition,
                "action": action,
                "timestamp": datetime.now().isoformat()
            }
        )
        self.vectorstore.add_documents([doc])
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve relevant knowledge."""
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [f"{result.page_content} (source: {result.metadata['source']})" 
                for result in results]
    
    def retrieve_by_category(self, category: str, top_k: int = 10) -> list[str]:
        """Retrieve knowledge by category."""
        # Use metadata filter
        results = self.vectorstore.get()
        matching = [
            doc for doc in results if doc.metadata.get("category") == category
        ]
        return matching[:top_k]

# Example usage
semantic_memory = SemanticMemory()

# Add facts from learned experiences
semantic_memory.add_fact(
    fact="For data analysis tasks, always load data first before analysis",
    source="successful_data_analysis_episodes",
    category="data_analysis"
)

semantic_memory.add_rule(
    rule="When user requests performance report",
    condition="task_type == 'performance_analysis'",
    action="Load metrics, compute trends, create visualizations"
)

semantic_memory.add_fact(
    fact="User prefers functional programming style over OOP",
    source="user_profile_learning",
    category="user_preferences"
)

# Retrieve relevant knowledge for new task
relevant = semantic_memory.retrieve_relevant(
    "I need to analyze performance data"
)
print("Relevant knowledge:", relevant)
```

**Advantages:**
- Efficient retrieval
- Structured knowledge
- Enables reasoning with retrieved facts
- Good for rules and preferences

**Disadvantages:**
- Information loss during extraction
- Requires careful fact curation
- Semantic extraction is non-trivial

### Strategy 3: Procedural Memory (Action Sequences)

Remember successful action sequences for specific task types.

```python
from typing import Sequence

class ActionSequence(BaseModel):
    """A sequence of actions that achieved a goal."""
    id: str
    task_type: str
    description: str
    steps: list[dict]  # [{"tool": "...", "params": {...}}, ...]
    success_rate: float
    average_tokens: int
    average_duration: float
    last_used: str

class ProceduralMemory:
    """Store successful action sequences."""
    
    def __init__(self, storage_path: str = "procedures.json"):
        self.storage_path = storage_path
        self.procedures = []
        self.load()
    
    def register_procedure(self, sequence: ActionSequence):
        """Register a successful action sequence."""
        self.procedures.append(sequence)
        self.save()
    
    def get_procedure(self, task_type: str) -> Optional[ActionSequence]:
        """Get the best procedure for a task type."""
        matching = [p for p in self.procedures if p.task_type == task_type]
        if not matching:
            return None
        # Return highest success rate
        return max(matching, key=lambda p: p.success_rate)
    
    def suggest_next_action(self, task_type: str, current_step: int) -> Optional[dict]:
        """Suggest next action based on known procedures."""
        procedure = self.get_procedure(task_type)
        if not procedure or current_step >= len(procedure.steps):
            return None
        return procedure.steps[current_step]
    
    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump([p.model_dump() for p in self.procedures], f)
    
    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.procedures = [ActionSequence(**item) for item in data]
        except FileNotFoundError:
            self.procedures = []

# Usage in agent
procedural_memory = ProceduralMemory()

# Register a procedure after successful execution
code_review_procedure = ActionSequence(
    id="proc_code_review_001",
    task_type="code_review",
    description="Standard code review procedure",
    steps=[
        {"tool": "read_file", "params": {"filepath": "${target_file}"}},
        {"tool": "analyze_code", "params": {"focus": ["style", "security", "performance"]}},
        {"tool": "check_tests", "params": {"filepath": "${test_file}"}},
        {"tool": "generate_report", "params": {"format": "markdown"}}
    ],
    success_rate=0.95,
    average_tokens=3400,
    average_duration=45.2,
    last_used=datetime.now().isoformat()
)
procedural_memory.register_procedure(code_review_procedure)

# In agent, use procedure as guide
procedure = procedural_memory.get_procedure("code_review")
if procedure:
    next_step = procedural_memory.suggest_next_action("code_review", current_step=0)
    print(f"Suggested action: {next_step}")
```

**Advantages:**
- Encodes successful strategies
- Reduces reasoning overhead
- Improves consistency
- Measurable success metrics

**Disadvantages:**
- Static procedures may not adapt
- Requires expert procedures initially
- May not work for novel tasks

---

## Memory Retrieval Patterns

Efficient retrieval is as important as storage.

### Pattern 1: Hybrid Search (Semantic + Keyword)

Combine semantic and keyword search for best recall.

```python
from langchain.vectorstores import Chroma, Weaviate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

class HybridRetriever:
    """Combine semantic and keyword search."""
    
    def __init__(self, docs: list, embeddings):
        # Semantic search with embeddings
        self.semantic_retriever = Chroma.from_documents(
            docs,
            embeddings,
            collection_name="semantic"
        ).as_retriever(search_kwargs={"k": 5})
        
        # Keyword search (BM25)
        self.keyword_retriever = BM25Retriever.from_documents(docs)
        self.keyword_retriever.k = 5
        
        # Ensemble: combine both
        self.ensemble = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.keyword_retriever],
            weights=[0.6, 0.4]  # More weight to semantic
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve using hybrid approach."""
        results = self.ensemble.get_relevant_documents(query)
        return [result.page_content for result in results[:top_k]]

# Usage
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

loader = TextLoader("project_docs.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000)
split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
retriever = HybridRetriever(split_docs, embeddings)

results = retriever.retrieve("How do I set up the database?")
```

**Advantages:**
- Better recall than single approach
- Handles both semantic and exact matches
- More robust

**Disadvantages:**
- More complex
- Slower than single retriever
- Weight tuning needed

### Pattern 2: Re-ranking with LLM

Retrieve candidates, then re-rank with LLM for relevance.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseReranker

class RerankingRetriever:
    """Re-rank retrieval results using LLM."""
    
    def __init__(self, base_retriever, llm: ChatOpenAI):
        # Re-rank top-k results
        compressor = LLMListwiseReranker.from_llm(
            llm=llm,
            top_n=3
        )
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def retrieve(self, query: str) -> list[str]:
        """Retrieve and re-rank."""
        docs = self.retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]

# Usage
base_retriever = Chroma.from_documents(split_docs, embeddings).as_retriever()
reranker = RerankingRetriever(base_retriever, ChatOpenAI(model="gpt-4o"))

results = reranker.retrieve("How do I set up the database?")
```

**Advantages:**
- More relevant results
- Semantic understanding in ranking
- Better quality outputs

**Disadvantages:**
- Adds latency (LLM call)
- Increases cost
- Slower than simple retrieval

### Pattern 3: Metadata-Based Filtering

Filter memory by metadata before retrieval.

```python
class FilteredRetriever:
    """Filter memory by metadata before retrieval."""
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
    
    def retrieve_by_source(self, query: str, source: str) -> list[str]:
        """Retrieve only from specific source."""
        results = self.vectorstore.similarity_search(
            query,
            k=5,
            filter={"source": source}
        )
        return [result.page_content for result in results]
    
    def retrieve_by_date_range(self, query: str, start_date: str, end_date: str) -> list[str]:
        """Retrieve within date range."""
        results = self.vectorstore.similarity_search(
            query,
            k=5,
            filter={
                "$and": [
                    {"timestamp": {"$gte": start_date}},
                    {"timestamp": {"$lte": end_date}}
                ]
            }
        )
        return [result.page_content for result in results]
    
    def retrieve_by_confidence(self, query: str, min_confidence: float = 0.8) -> list[str]:
        """Retrieve only high-confidence facts."""
        results = self.vectorstore.similarity_search(
            query,
            k=5,
            filter={"confidence": {"$gte": min_confidence}}
        )
        return [result.page_content for result in results]
```

**Advantages:**
- Precise control
- Faster filtering
- Better relevance

**Disadvantages:**
- Requires good metadata
- Manual filter creation
- May miss relevant results outside filters

---

## Context Window Management

Context windows are precious and finite. Manage them carefully.

### Strategy 1: Dynamic Context Assembly

Build minimal context for each request.

```python
from typing import TypedDict

class MinimalContext(TypedDict):
    system_prompt: str
    pinned_requirements: str
    recent_history: list[dict]
    relevant_knowledge: list[str]

class DynamicContextAssembler:
    """Assemble minimal context for each request."""
    
    def __init__(self, llm: ChatOpenAI, memory_systems: dict):
        self.llm = llm
        self.episodic = memory_systems.get("episodic")
        self.semantic = memory_systems.get("semantic")
        self.procedural = memory_systems.get("procedural")
    
    def assemble_context(
        self,
        task: str,
        task_type: str,
        current_conversation: list[dict],
        max_tokens: int = 6000
    ) -> MinimalContext:
        """Assemble minimal context."""
        
        context_tokens = 0
        components = {}
        
        # 1. System prompt (fixed)
        system = f"You are an AI agent specialized in {task_type}."
        components["system"] = system
        context_tokens += len(system.split()) // 1.3  # Rough token estimate
        
        # 2. Pinned requirements (fixed)
        requirements = "Always follow these standards: type hints, docstrings, error handling"
        components["requirements"] = requirements
        context_tokens += len(requirements.split()) // 1.3
        
        # 3. Recent conversation history (sliding window)
        conversation = current_conversation[-5:]  # Last 5 turns
        components["conversation"] = conversation
        context_tokens += sum(len(msg.get("content", "").split()) for msg in conversation) // 1.3
        
        # 4. Relevant knowledge (if space available)
        if context_tokens < max_tokens * 0.7:  # Reserve space
            if self.semantic:
                knowledge = self.semantic.retrieve_relevant(task, top_k=3)
                components["knowledge"] = knowledge
                context_tokens += sum(len(k.split()) for k in knowledge) // 1.3
        
        # 5. Procedure (if space available)
        if context_tokens < max_tokens * 0.8:
            if self.procedural:
                procedure = self.procedural.get_procedure(task_type)
                if procedure:
                    components["procedure"] = procedure
                    context_tokens += len(str(procedure).split()) // 1.3
        
        return MinimalContext(
            system_prompt=components.get("system", ""),
            pinned_requirements=components.get("requirements", ""),
            recent_history=components.get("conversation", []),
            relevant_knowledge=components.get("knowledge", [])
        )
    
    def render_to_prompt(self, context: MinimalContext) -> str:
        """Render assembled context into prompt."""
        prompt = context["system_prompt"]
        prompt += f"\n\nRequirements:\n{context['pinned_requirements']}"
        
        if context["relevant_knowledge"]:
            prompt += "\n\nRelevant Knowledge:\n"
            prompt += "\n".join(context["relevant_knowledge"])
        
        return prompt

# Usage
memory_systems = {
    "episodic": episodic_memory,
    "semantic": semantic_memory,
    "procedural": procedural_memory
}
assembler = DynamicContextAssembler(llm, memory_systems)

context = assembler.assemble_context(
    task="Analyze sales trends",
    task_type="data_analysis",
    current_conversation=state["messages"],
    max_tokens=8000
)

prompt = assembler.render_to_prompt(context)
```

**Advantages:**
- Minimal token usage
- Dynamic assembly based on need
- Handles token limits gracefully

**Disadvantages:**
- Requires multiple memory systems
- Complex assembly logic
- May miss relevant context

### Strategy 2: Hierarchical Prompting

Use separate prompts for different reasoning levels.

```python
class HierarchicalPrompting:
    """Use different prompts for different reasoning depths."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def quick_response(self, question: str) -> str:
        """Quick response without full reasoning."""
        prompt = f"""Answer briefly (1-2 sentences):

Question: {question}"""
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content
    
    async def detailed_reasoning(self, question: str) -> str:
        """Detailed reasoning with working memory."""
        prompt = f"""Answer thoroughly, showing your reasoning:

Question: {question}

1. What information do I need?
2. What knowledge applies?
3. What's the best approach?
4. Final answer:"""
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content
    
    async def expert_analysis(self, question: str, context: str) -> str:
        """Expert-level analysis with full context."""
        prompt = f"""Expert analysis with full context:

Context: {context}

Question: {question}

Provide expert-level analysis covering:
1. Underlying assumptions
2. Relevant precedents
3. Trade-offs
4. Recommendations"""
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content
    
    async def choose_level(self, question: str, complexity_estimate: str) -> str:
        """Choose appropriate reasoning level."""
        if complexity_estimate == "simple":
            return await self.quick_response(question)
        elif complexity_estimate == "complex":
            return await self.detailed_reasoning(question)
        else:
            return await self.expert_analysis(question, "...")
```

**Advantages:**
- Optimized token usage for each case
- Adjustable reasoning depth
- Better performance/cost trade-off

**Disadvantages:**
- Complexity estimation required
- Different output quality levels
- Manual tuning needed

---

## Conversation History Management

Conversation history is the primary short-term memory.

### Strategy 1: Structured Conversation State

```python
from enum import Enum

class ConversationPhase(str, Enum):
    PROBLEM_UNDERSTANDING = "problem_understanding"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"

class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    phase: ConversationPhase
    task_description: str
    constraints: list[str]
    decisions_made: dict
    artifacts_created: list[str]
    errors_encountered: list[str]

def manage_conversation_state(state: ConversationState) -> ConversationState:
    """Manage conversation state across phases."""
    
    current_phase = state["phase"]
    messages = state["messages"]
    
    # Determine next phase based on conversation
    last_assistant_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
        ""
    )
    
    if "plan:" in last_assistant_msg.lower():
        next_phase = ConversationPhase.EXECUTION
    elif "verify" in last_assistant_msg.lower():
        next_phase = ConversationPhase.VERIFICATION
    elif "complete" in last_assistant_msg.lower():
        next_phase = ConversationPhase.COMPLETION
    else:
        next_phase = current_phase
    
    return {
        **state,
        "phase": next_phase
    }
```

**Advantages:**
- Structured conversation tracking
- Clear phase transitions
- Better state management

**Disadvantages:**
- Requires phase detection
- Manual state updates
- May miss state transitions

### Strategy 2: Message Compression

```python
class MessageCompressor:
    """Compress messages intelligently."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def compress_user_messages(self, messages: list[dict]) -> str:
        """Compress multiple user messages into single summary."""
        user_msgs = [m for m in messages if m["role"] == "user"]
        user_text = "\n".join([m["content"] for m in user_msgs])
        
        prompt = f"""Summarize these user messages concisely (1-2 sentences):
        
{user_text}"""
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content
    
    def compress_conversation(self, messages: list[dict], max_pairs: int = 5) -> list[dict]:
        """Keep only recent Q&A pairs."""
        # Separate system from conversation
        system_msgs = [m for m in messages if m["role"] == "system"]
        conversation = [m for m in messages if m["role"] != "system"]
        
        # Keep only recent pairs
        qa_pairs = []
        i = 0
        while i < len(conversation):
            if conversation[i]["role"] == "user":
                user_msg = conversation[i]
                assistant_msg = conversation[i+1] if i+1 < len(conversation) else None
                if assistant_msg and assistant_msg["role"] == "assistant":
                    qa_pairs.append((user_msg, assistant_msg))
                i += 2
            else:
                i += 1
        
        # Keep recent pairs
        recent = qa_pairs[-max_pairs:]
        compressed = system_msgs + [m for pair in recent for m in pair]
        
        return compressed
```

---

## Knowledge Base Integration

Integrating knowledge bases makes agents smarter.

### Pattern 1: Semantic Search + Refinement

```python
class KnowledgeBaseAgent:
    """Agent with integrated knowledge base."""
    
    def __init__(self, llm: ChatOpenAI, vectorstore: Chroma):
        self.llm = llm
        self.vectorstore = vectorstore
    
    async def answer_with_knowledge(self, question: str) -> str:
        """Answer by searching knowledge base."""
        
        # Step 1: Search knowledge base
        docs = self.vectorstore.similarity_search(question, k=3)
        
        if not docs:
            # Fallback if no results
            return await self._answer_from_reasoning(question)
        
        # Step 2: Check if knowledge is sufficient
        knowledge_text = "\n".join([doc.page_content for doc in docs])
        sufficiency_check = f"""Is this knowledge sufficient to answer the question?

Question: {question}
Knowledge: {knowledge_text}

Answer YES or NO:"""
        
        response = await self.llm.ainvoke([
            {"role": "user", "content": sufficiency_check}
        ])
        
        # Step 3: Answer based on knowledge
        if "YES" in response.content.upper():
            answer_prompt = f"""Answer based on this knowledge:

Knowledge: {knowledge_text}

Question: {question}

Answer:"""
        else:
            # Need to search more or reason
            answer_prompt = f"""This knowledge is incomplete. Use it as starting point but add your reasoning:

Knowledge: {knowledge_text}

Question: {question}

Answer:"""
        
        final_response = await self.llm.ainvoke([
            {"role": "user", "content": answer_prompt}
        ])
        
        return final_response.content
    
    async def _answer_from_reasoning(self, question: str) -> str:
        """Answer using pure reasoning when no knowledge available."""
        response = await self.llm.ainvoke([
            {"role": "user", "content": question}
        ])
        return response.content
```

---

## Performance Considerations

Memory systems have real performance implications.

### Retrieval Latency

```python
import time

class PerformanceMonitor:
    """Monitor memory system performance."""
    
    def __init__(self):
        self.metrics = {
            "semantic_search_time": [],
            "keyword_search_time": [],
            "reranking_time": [],
            "context_assembly_time": []
        }
    
    async def measure_retrieval(self, retriever, query: str):
        start = time.time()
        results = retriever.retrieve(query)
        elapsed = time.time() - start
        
        self.metrics["retrieval_time"].append(elapsed)
        return results, elapsed
    
    def get_average_latency(self, operation: str) -> float:
        times = self.metrics.get(operation, [])
        return sum(times) / len(times) if times else 0
    
    def print_stats(self):
        for operation, times in self.metrics.items():
            if times:
                avg = sum(times) / len(times)
                print(f"{operation}: {avg:.3f}s (n={len(times)})")
```

### Memory Usage

```python
class MemoryUsageMonitor:
    """Monitor memory usage."""
    
    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Rough token count estimate."""
        return len(text.split()) // 1.3  # Rough estimate
    
    @staticmethod
    def token_cost(tokens: int, model: str = "gpt-4o") -> float:
        """Estimate API cost."""
        # Approximate costs per 1M tokens
        costs = {
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "gpt-3.5": {"input": 0.5, "output": 1.5},
        }
        cost_per_k = costs.get(model, costs["gpt-4o"])
        return (tokens / 1000000) * cost_per_k["input"]
    
    @staticmethod
    def storage_size(memories: dict) -> float:
        """Estimate storage size in MB."""
        total_bytes = 0
        for name, memory in memories.items():
            if hasattr(memory, 'episodes'):
                total_bytes += len(json.dumps([e.model_dump() for e in memory.episodes]))
        return total_bytes / (1024 * 1024)
```

---

## Trade-offs Between Approaches

### Summary Table

| Strategy | Complexity | Latency | Storage | Recall Quality | Learning Capability |
|----------|-----------|---------|---------|-----------------|-----------------|
| Sliding Window | Low | Minimal | Low | Low | None |
| Summarization | Medium | Low | Low | Medium | None |
| Hierarchical Grouping | High | Low | Medium | Medium | Low |
| Episodic Memory | High | Medium | High | High | High |
| Semantic Memory | Medium | Medium | Medium | Medium | Medium |
| Procedural Memory | Medium | Low | Low | High (for known tasks) | High |
| Hybrid Search | High | Medium | Medium | High | None |
| Re-ranking | High | High | Low | Very High | None |

### Decision Matrix

```
Choose based on:

1. Task Complexity & Novelty
   - Simple, repetitive → Procedural Memory
   - Complex, novel → Episodic Memory
   - Mixed → Semantic Memory

2. User Interaction Style
   - Single-session → Sliding Window
   - Multi-session → Session Storage
   - Long-term relationships → Agentic Memory

3. Knowledge Requirements
   - Specialized domain → Knowledge Base (Semantic)
   - Procedural expertise → Procedural Memory
   - General knowledge → RAG with Retrieval

4. Performance Constraints
   - Low latency required → Sliding Window
   - Cost-sensitive → Summarization
   - Quality-first → Hybrid Search + Re-ranking

5. Learning Goals
   - Improvement across sessions → Episodic Memory
   - Pattern recognition → Semantic Memory
   - Action optimization → Procedural Memory
```

---

## References

1. **Agno Framework - Levels of Agentic Software**: https://www.agno.com/blog/the-5-levels-of-agentic-software-a-progressive-framework-for-building-reliable-ai-agents

2. **LangChain Memory Documentation**: https://python.langchain.com/docs/modules/memory/

3. **LangGraph Persistence**: https://langchain-ai.github.io/langgraph/concepts/persistence/

4. **RAG Best Practices 2026**: https://www.youngju.dev/blog/ai-platform/2026-03-14-ai-agent-multi-agent-orchestration-patterns.en

5. **Production LangChain Patterns**: https://dev.to/akisharan/building-production-ready-langchain-agents-architectural-patterns-that-work-54af

6. **OpenAI Vector Embeddings**: https://platform.openai.com/docs/guides/embeddings

7. **Chroma Vector Database**: https://docs.trychroma.com/

8. **Weaviate Documentation**: https://weaviate.io/developers/weaviate

