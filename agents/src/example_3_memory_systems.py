"""
Example 3: Memory Systems and Context Management

This example demonstrates:
- Different memory implementations
- Conversation history tracking
- Entity tracking and relationships
- Vector-based semantic search
- Memory statistics and analysis

Source: https://blog.langchain.com/how-we-built-agent-builders-memory-system/
"""

import sys
import os
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_memory_systems import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    EntityMemory,
    VectorMemory,
    MessageRole,
    MemoryFactory,
    Message,
)


def example_1_buffer_memory():
    """Demonstrate conversation buffer memory."""
    print("=" * 60)
    print("Example 1: Conversation Buffer Memory")
    print("=" * 60)

    print("\nBuffer memory stores all messages without reduction.")
    print("Best for: Short conversations, context requiring all history\n")

    # Create buffer memory
    memory = ConversationBufferMemory(max_messages=10)

    # Simulate a conversation
    conversation = [
        (
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence...",
        ),
        (
            "Can you give an example?",
            "Sure! Email spam detection is a common example...",
        ),
        ("How does it work?", "It works by training on labeled examples..."),
    ]

    print("Adding messages to memory:")
    for user_msg, assistant_msg in conversation:
        memory.add_message(MessageRole.USER, user_msg)
        memory.add_message(MessageRole.ASSISTANT, assistant_msg)
        print(f"  ✓ Added user and assistant messages")

    # Display stats
    print("\nMemory Statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get context
    print("\nMemory Context (formatted for LLM):")
    print("-" * 60)
    context = memory.get_context()
    # Show first 300 chars
    print(context[:300] + "...\n")

    # Clear memory
    print("Clearing memory...")
    memory.clear()
    print(f"✓ Cleared, messages remaining: {len(memory.get_messages())}")


def example_2_summary_memory():
    """Demonstrate summary memory with periodical summarization."""
    print("\n" + "=" * 60)
    print("Example 2: Conversation Summary Memory")
    print("=" * 60)

    print("\nSummary memory keeps recent messages and summarizes older ones.")
    print("Best for: Long conversations, cost optimization\n")

    # Create summary memory
    memory = ConversationSummaryMemory(keep_recent=3)

    # Add many messages
    messages = [
        ("What is AI?", "AI is intelligence exhibited by machines..."),
        ("What about ML?", "ML is a subset of AI focused on learning..."),
        ("Tell me about deep learning", "DL uses neural networks..."),
        ("How do neural networks work?", "They mimic biological neurons..."),
        ("What about transformers?", "Transformers revolutionized NLP..."),
        (
            "Can you explain attention?",
            "Attention allows focusing on relevant parts...",
        ),
    ]

    print("Adding multiple messages (will trigger summarization):")
    for user_msg, assistant_msg in messages:
        memory.add_message(MessageRole.USER, user_msg)
        memory.add_message(MessageRole.ASSISTANT, assistant_msg)

    print(f"✓ Added {len(messages) * 2} messages")
    print(f"  Recent messages kept: 3")
    print(f"  Older messages summarized")

    # Get context
    print("\nMemory Context:")
    print("-" * 60)
    context = memory.get_context()
    print(context)

    # Show that we kept recent
    print(f"\nRecent messages in memory: {len(memory.get_messages())}")


def example_3_entity_memory():
    """Demonstrate entity tracking memory."""
    print("\n" + "=" * 60)
    print("Example 3: Entity Memory")
    print("=" * 60)

    print("\nEntity memory tracks entities and their relationships.")
    print("Best for: Multi-user scenarios, knowledge graphs, relationship tracking\n")

    # Create entity memory
    memory = EntityMemory()

    # Simulate conversation about people
    print("Simulating conversation with entity tracking:")

    # First exchange
    memory.add_message(
        MessageRole.USER,
        "Tell me about Alice Johnson. She's a software engineer at Google, age 32.",
    )
    memory.add_entity(
        "Alice Johnson",
        {
            "role": "Software Engineer",
            "company": "Google",
            "age": 32,
            "expertise": ["Python", "Distributed Systems"],
        },
    )
    print("  ✓ Added entity: Alice Johnson")

    # Add relationships
    memory.add_relationship("Alice Johnson", "works at", "Google")
    memory.add_relationship("Alice Johnson", "specializes in", "Distributed Systems")
    print("  ✓ Added relationships")

    # Second exchange
    memory.add_message(
        MessageRole.USER, "What about Bob Smith? He manages Alice and works in product."
    )
    memory.add_entity(
        "Bob Smith",
        {"role": "Product Manager", "company": "Google", "department": "Product"},
    )
    memory.add_relationship("Bob Smith", "manages", "Alice Johnson")
    memory.add_relationship("Bob Smith", "works with", "Google")
    print("  ✓ Added entity: Bob Smith with relationships")

    # Query entities
    print("\nQuerying Entity Knowledge:")
    print("-" * 60)

    alice_info = memory.get_entity("Alice Johnson")
    print(f"\nAlice Johnson attributes:")
    for key, value in alice_info.items():
        print(f"  {key}: {value}")

    print(f"\nAll relationships:")
    for entity1, rel, entity2 in memory.get_relationships():
        print(f"  {entity1} {rel} {entity2}")

    # Display context
    print("\nEntity Context (for LLM):")
    print("-" * 60)
    print(memory.get_context())


def example_4_vector_memory():
    """Demonstrate vector-based semantic search memory."""
    print("\n" + "=" * 60)
    print("Example 4: Vector Memory (Semantic Search)")
    print("=" * 60)

    print("\nVector memory uses embeddings for semantic similarity search.")
    print("Best for: Finding relevant context, semantic search\n")

    # Create vector memory
    memory = VectorMemory(embedding_dim=384)

    # Add messages
    messages_to_add = [
        "Python is a popular programming language",
        "Machine learning requires understanding of mathematics",
        "Neural networks are inspired by biological neurons",
        "Data science combines statistics and programming",
        "Deep learning uses multiple layers of neural networks",
    ]

    print("Adding messages with embeddings:")
    for i, msg in enumerate(messages_to_add, 1):
        memory.add_message(MessageRole.USER, msg)
        print(f"  {i}. Added: {msg[:50]}...")

    # Search for similar messages
    print("\nSearching for semantically similar messages:")
    print("-" * 60)

    queries = ["programming languages", "artificial intelligence", "data analysis"]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = memory.search(query, top_k=2)

        for i, (message, similarity) in enumerate(results, 1):
            print(f"  {i}. Score: {similarity:.3f} - {message.content[:50]}...")


def example_5_memory_factory():
    """Demonstrate using the MemoryFactory."""
    print("\n" + "=" * 60)
    print("Example 5: Memory Factory Pattern")
    print("=" * 60)

    print("\nFactory pattern provides convenient pre-configured memory instances.\n")

    # Create different memory types using factory
    print("Creating memory instances with factory:")

    buffer_mem = MemoryFactory.create_buffer_memory(max_messages=50)
    print(f"✓ Buffer memory: {type(buffer_mem).__name__}")

    summary_mem = MemoryFactory.create_summary_memory(keep_recent=10)
    print(f"✓ Summary memory: {type(summary_mem).__name__}")

    entity_mem = MemoryFactory.create_entity_memory()
    print(f"✓ Entity memory: {type(entity_mem).__name__}")

    vector_mem = MemoryFactory.create_vector_memory(embedding_dim=768)
    print(f"✓ Vector memory: {type(vector_mem).__name__}")

    print("\nUse factory when you need:")
    print("  • Standard configurations")
    print("  • Consistent memory setup across agents")
    print("  • Easy switching between memory types")


def example_6_conversation_flow():
    """Show complete conversation flow with memory."""
    print("\n" + "=" * 60)
    print("Example 6: Complete Conversation Flow")
    print("=" * 60)

    print("\nDemonstrating a multi-turn conversation with memory:\n")

    # Create memory
    memory = ConversationBufferMemory(max_messages=20)

    # Simulate conversation flow
    exchanges = [
        {
            "user": "Hi! My name is Alex and I'm interested in learning Python.",
            "assistant": "Hello Alex! I'd be happy to help you learn Python. It's a great choice for beginners.",
        },
        {
            "user": "What's the best way to start?",
            "assistant": "Start with basic concepts: variables, data types, loops, and functions.",
        },
        {
            "user": "Can you recommend some resources?",
            "assistant": "Sure! Check out Python.org documentation, Codecademy, and Real Python tutorials.",
        },
    ]

    print("Conversation Flow:")
    print("-" * 60)

    for turn_num, exchange in enumerate(exchanges, 1):
        print(f"\nTurn {turn_num}:")
        print(f"User: {exchange['user']}")
        print(f"Assistant: {exchange['assistant']}")

        memory.add_message(MessageRole.USER, exchange["user"])
        memory.add_message(MessageRole.ASSISTANT, exchange["assistant"])

    # Analyze memory
    print("\n" + "-" * 60)
    print("Memory Analysis:")
    print("-" * 60)

    stats = memory.get_stats()
    print(f"Total messages: {stats['total_messages']}")
    print(f"User messages: {stats['user_messages']}")
    print(f"Assistant messages: {stats['assistant_messages']}")
    print(f"Approximate tokens: {stats['total_tokens_approx']}")


def example_7_best_practices():
    """Best practices for memory management."""
    print("\n" + "=" * 60)
    print("Example 7: Best Practices")
    print("=" * 60)

    print("""
1. CHOOSING THE RIGHT MEMORY TYPE:
   
   ┌─ Short conversations (< 10 exchanges)
   │  └─ Use: ConversationBufferMemory
   │
   ├─ Long conversations (> 50 exchanges)
   │  └─ Use: ConversationSummaryMemory
   │
   ├─ Multi-entity tracking
   │  └─ Use: EntityMemory
   │
   └─ Semantic similarity search
      └─ Use: VectorMemory

2. MEMORY SIZE MANAGEMENT:
   ✓ Set max_messages limit for buffer memory
   ✓ Use keep_recent parameter for summary memory
   ✓ Periodically archive old conversations
   ✓ Monitor memory consumption

3. TOKEN EFFICIENCY:
   ✓ Buffer memory: Simple but token-heavy
   ✓ Summary memory: Reduces tokens for long chats
   ✓ Entity memory: Efficient for structured data
   ✓ Vector memory: Selective retrieval

4. MEMORY LIFECYCLE:
   ✓ Create memory per conversation
   ✓ Clear memory when done (memory.clear())
   ✓ Archive important conversations
   ✓ Don't persist sensitive information

5. PERFORMANCE:
   ✓ Buffer memory: O(1) operations, fast
   ✓ Summary memory: O(n) summarization
   ✓ Entity memory: O(1) entity lookup
   ✓ Vector memory: O(n) similarity search

6. INTEGRATION WITH AGENTS:
   Example:
   
   from langchain_agents import BasicAgent, AgentConfig, LLMProvider
   from langchain_memory_systems import ConversationBufferMemory
   
   # Create agent and memory
   agent = BasicAgent(AgentConfig(model="gpt-4", provider=LLMProvider.OPENAI))
   memory = ConversationBufferMemory()
   
   # Conversation loop
   while True:
       user_input = input("You: ")
       memory.add_message(MessageRole.USER, user_input)
       
       context = memory.get_context()
       response = agent.invoke(context + user_input)
       memory.add_message(MessageRole.ASSISTANT, response)
       
       print(f"Agent: {response}")
""")


if __name__ == "__main__":
    print("\n")
    print("█" * 60)
    print("█  LangChain Memory Systems - Examples")
    print("█" * 60)

    example_1_buffer_memory()
    example_2_summary_memory()
    example_3_entity_memory()
    example_4_vector_memory()
    example_5_memory_factory()
    example_6_conversation_flow()
    example_7_best_practices()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  • Different memory types for different use cases")
    print("  • Buffer memory for complete conversation history")
    print("  • Summary memory for long conversations")
    print("  • Entity memory for relationship tracking")
    print("  • Vector memory for semantic search")
    print("  • Factory pattern for easy instantiation")
    print("=" * 60 + "\n")
