"""
Integration Examples - Database, API, RAG, Fine-tuned Models, and Async Workflows

This module demonstrates integrating agents with external systems and advanced patterns.

References:
- LangChain Integrations: https://docs.langchain.com/
- AGNO Production Systems: https://medium.com/data-science-collective/building-production-ready-ai-agents-with-agno-a-comprehensive-engineering-guide-22db32413fdd

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install sqlalchemy>=2.0.0
# pip install requests>=2.30.0
# pip install numpy
# pip install asyncio

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio


# ============================================================================
# INTEGRATION 1: Database Integration
# ============================================================================


@dataclass
class User:
    """User model for database storage."""

    id: int
    name: str
    email: str
    created_at: datetime = None


class DatabaseAdapter:
    """
    Adapter for integrating agents with databases.
    In production, use SQLAlchemy, async drivers, etc.
    """

    def __init__(self):
        """Initialize database adapter with simulated in-memory database."""
        self.users: Dict[int, User] = {}
        self.user_counter = 0

    def create_user(self, name: str, email: str) -> User:
        """Create a new user in the database."""
        self.user_counter += 1
        user = User(
            id=self.user_counter,
            name=name,
            email=email,
            created_at=datetime.now(),
        )
        self.users[user.id] = user
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        """Retrieve a user by ID."""
        return self.users.get(user_id)

    def get_users_by_name(self, name: str) -> List[User]:
        """Search users by name."""
        return [u for u in self.users.values() if name.lower() in u.name.lower()]

    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user fields."""
        user = self.users.get(user_id)
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
        return user

    def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False

    def list_users(self) -> List[User]:
        """List all users."""
        return list(self.users.values())


class AgentDatabaseIntegration:
    """Agent with database integration."""

    def __init__(self):
        self.db = DatabaseAdapter()

    def add_user(self, name: str, email: str) -> str:
        """Tool: Add a new user."""
        try:
            user = self.db.create_user(name, email)
            return f"✅ User created: {user.name} ({user.email}) - ID: {user.id}"
        except Exception as e:
            return f"❌ Error creating user: {str(e)}"

    def search_users(self, search_term: str) -> str:
        """Tool: Search for users."""
        users = self.db.get_users_by_name(search_term)
        if not users:
            return f"No users found matching '{search_term}'"

        result = f"Found {len(users)} users:\n"
        for user in users:
            result += f"  - {user.name} ({user.email}) - ID: {user.id}\n"
        return result

    def get_user_info(self, user_id: int) -> str:
        """Tool: Get user information."""
        user = self.db.get_user(user_id)
        if not user:
            return f"User {user_id} not found"

        return f"User: {user.name}\nEmail: {user.email}\nCreated: {user.created_at}"

    def list_all_users(self) -> str:
        """Tool: List all users."""
        users = self.db.list_users()
        if not users:
            return "No users in database"

        result = f"Total users: {len(users)}\n"
        for user in users:
            result += f"  {user.id}. {user.name} - {user.email}\n"
        return result


# ============================================================================
# INTEGRATION 2: API Integration
# ============================================================================


class APIClient:
    """
    Simulated API client for external service integration.
    In production, use requests, httpx, or async alternatives.
    """

    def __init__(self, base_url: str = "https://api.example.com"):
        self.base_url = base_url
        # Simulated API responses
        self.responses = {
            "/weather": {"city": "New York", "temp": 72, "condition": "Sunny"},
            "/news": [
                {"title": "AI Advances in 2026", "source": "TechNews"},
                {"title": "LLM Breakthroughs", "source": "AIWeekly"},
            ],
            "/stocks": {"AAPL": 150.25, "GOOGL": 140.50, "MSFT": 380.25},
        }

    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a GET request."""
        try:
            if endpoint in self.responses:
                return {"status": "success", "data": self.responses[endpoint]}
            return {"status": "error", "message": f"Endpoint not found: {endpoint}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request."""
        try:
            return {
                "status": "success",
                "message": f"Data posted to {endpoint}",
                "data": data,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class AgentAPIIntegration:
    """Agent with external API integration."""

    def __init__(self):
        self.api = APIClient()

    def get_weather(self, city: str = "New York") -> str:
        """Tool: Get weather information."""
        response = self.api.get("/weather")
        if response["status"] == "success":
            data = response["data"]
            return f"Weather in {data['city']}: {data['temp']}°F, {data['condition']}"
        return f"Error: {response['message']}"

    def get_news(self) -> str:
        """Tool: Get latest news."""
        response = self.api.get("/news")
        if response["status"] == "success":
            news = response["data"]
            result = "Latest News:\n"
            for item in news:
                result += f"  - {item['title']} ({item['source']})\n"
            return result
        return f"Error: {response['message']}"

    def get_stock_prices(self) -> str:
        """Tool: Get stock prices."""
        response = self.api.get("/stocks")
        if response["status"] == "success":
            stocks = response["data"]
            result = "Stock Prices:\n"
            for ticker, price in stocks.items():
                result += f"  {ticker}: ${price}\n"
            return result
        return f"Error: {response['message']}"


# ============================================================================
# INTEGRATION 3: RAG System Integration
# ============================================================================


class RAGSystem:
    """
    Retrieval-Augmented Generation system integration.
    Combines document retrieval with generation.
    """

    def __init__(self):
        self.documents = {
            "python_tutorial": "Python is a versatile programming language. It supports OOP, functional programming, and procedural programming.",
            "langchain_guide": "LangChain is a framework for building applications with LLMs. It provides tools for memory, agents, and chains.",
            "agent_design": "Agents are autonomous systems that can use tools, maintain state, and reason about problems.",
        }

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """Retrieve relevant documents."""
        query_words = set(query.lower().split())
        results = []

        for doc_id, content in self.documents.items():
            doc_words = set(content.lower().split())
            relevance = len(query_words & doc_words) / max(len(query_words), 1)
            if relevance > 0:
                results.append((doc_id, content, relevance))

        results.sort(key=lambda x: x[2], reverse=True)
        return [content for _, content, _ in results[:top_k]]

    def generate_answer(self, query: str) -> str:
        """Generate answer using retrieved documents."""
        documents = self.retrieve(query)
        if not documents:
            return f"No relevant documents found for: {query}"

        answer = f"Based on the knowledge base, here's information about '{query}':\n\n"
        for doc in documents:
            answer += f"- {doc[:100]}...\n"
        return answer


class AgentRAGIntegration:
    """Agent with RAG system integration."""

    def __init__(self):
        self.rag = RAGSystem()

    def answer_question(self, question: str) -> str:
        """Tool: Answer question using RAG system."""
        return self.rag.generate_answer(question)

    def retrieve_documents(self, query: str) -> str:
        """Tool: Retrieve relevant documents."""
        docs = self.rag.retrieve(query)
        if not docs:
            return f"No documents found for: {query}"

        result = f"Found {len(docs)} relevant documents:\n"
        for i, doc in enumerate(docs, 1):
            result += f"{i}. {doc[:80]}...\n"
        return result


# ============================================================================
# INTEGRATION 4: Fine-tuned Models Integration
# ============================================================================


class FineTunedModelAdapter:
    """
    Adapter for using fine-tuned models in agents.
    In production, would connect to model serving endpoints.
    """

    def __init__(self, model_name: str = "custom-classifier-v1"):
        self.model_name = model_name
        self.classes = ["positive", "negative", "neutral"]

    def predict(self, text: str) -> Dict[str, Any]:
        """Get predictions from fine-tuned model."""
        # Simulated prediction
        if any(word in text.lower() for word in ["good", "great", "excellent"]):
            return {
                "class": "positive",
                "confidence": 0.92,
                "model": self.model_name,
            }
        elif any(word in text.lower() for word in ["bad", "poor", "terrible"]):
            return {
                "class": "negative",
                "confidence": 0.88,
                "model": self.model_name,
            }
        else:
            return {
                "class": "neutral",
                "confidence": 0.75,
                "model": self.model_name,
            }

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get predictions for multiple inputs."""
        return [self.predict(text) for text in texts]


class AgentFineTunedModelIntegration:
    """Agent using fine-tuned models."""

    def __init__(self):
        self.model = FineTunedModelAdapter()

    def classify_sentiment(self, text: str) -> str:
        """Tool: Classify text sentiment using fine-tuned model."""
        result = self.model.predict(text)
        return f"Sentiment: {result['class']} (confidence: {result['confidence']:.1%})"

    def analyze_multiple(self, texts: List[str]) -> str:
        """Tool: Analyze sentiment of multiple texts."""
        results = self.model.batch_predict(texts)
        analysis = f"Analyzed {len(texts)} texts:\n"
        for text, result in zip(texts, results):
            analysis += f"- '{text[:50]}...': {result['class']}\n"
        return analysis


# ============================================================================
# INTEGRATION 5: Async Workflows
# ============================================================================


class AsyncAgent:
    """
    Agent with async/await support for concurrent operations.
    Useful for calling multiple APIs or databases in parallel.
    """

    def __init__(self):
        self.db = DatabaseAdapter()
        self.api = APIClient()

    async def async_operation(self, delay: float) -> str:
        """Simulate an async operation."""
        await asyncio.sleep(delay)
        return f"Completed async operation (waited {delay}s)"

    async def fetch_multiple_apis(self) -> Dict[str, Any]:
        """Fetch from multiple APIs concurrently."""
        tasks = [
            asyncio.create_task(self._fetch_api("/weather")),
            asyncio.create_task(self._fetch_api("/news")),
            asyncio.create_task(self._fetch_api("/stocks")),
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def _fetch_api(self, endpoint: str):
        """Async API fetch."""
        # Simulate network delay
        await asyncio.sleep(0.1)
        response = self.api.get(endpoint)
        return (endpoint, response)

    async def parallel_database_operations(self, operations: List[tuple]) -> List[str]:
        """Execute multiple database operations in parallel."""
        tasks = [self._db_operation(op) for op in operations]
        return await asyncio.gather(*tasks)

    async def _db_operation(self, operation: tuple) -> str:
        """Execute a database operation."""
        op_type, *args = operation
        await asyncio.sleep(0.05)  # Simulate DB latency

        if op_type == "create":
            user = self.db.create_user(*args)
            return f"Created user: {user.name}"
        elif op_type == "search":
            users = self.db.get_users_by_name(args[0])
            return f"Found {len(users)} users"
        return "Unknown operation"


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Integration Examples - External System Integration")
    print("=" * 70)

    # INTEGRATION 1: Database
    print("\n\n" + "=" * 70)
    print("INTEGRATION 1: Database Integration")
    print("=" * 70)

    db_agent = AgentDatabaseIntegration()

    print("\n🔨 Creating users...")
    print(db_agent.add_user("Alice Johnson", "alice@example.com"))
    print(db_agent.add_user("Bob Smith", "bob@example.com"))

    print("\n🔍 Searching for users...")
    print(db_agent.search_users("Alice"))

    print("\n📋 Listing all users...")
    print(db_agent.list_all_users())

    # INTEGRATION 2: API
    print("\n\n" + "=" * 70)
    print("INTEGRATION 2: External API Integration")
    print("=" * 70)

    api_agent = AgentAPIIntegration()

    print("\n🌤️ Getting weather...")
    print(api_agent.get_weather())

    print("\n📰 Getting news...")
    print(api_agent.get_news())

    print("\n📈 Getting stock prices...")
    print(api_agent.get_stock_prices())

    # INTEGRATION 3: RAG
    print("\n\n" + "=" * 70)
    print("INTEGRATION 3: RAG System Integration")
    print("=" * 70)

    rag_agent = AgentRAGIntegration()

    print("\n❓ Answering question...")
    print(rag_agent.answer_question("What is Python?"))

    print("\n🔎 Retrieving documents...")
    print(rag_agent.retrieve_documents("LangChain framework"))

    # INTEGRATION 4: Fine-tuned Models
    print("\n\n" + "=" * 70)
    print("INTEGRATION 4: Fine-tuned Model Integration")
    print("=" * 70)

    model_agent = AgentFineTunedModelIntegration()

    print("\n😊 Classifying sentiment...")
    print(model_agent.classify_sentiment("This is an excellent product!"))
    print(model_agent.classify_sentiment("This is terrible!"))

    print("\n📊 Batch analysis...")
    texts = ["Great job!", "Very bad", "It's okay"]
    print(model_agent.analyze_multiple(texts))

    # INTEGRATION 5: Async Workflows
    print("\n\n" + "=" * 70)
    print("INTEGRATION 5: Async Workflow Integration")
    print("=" * 70)

    async_agent = AsyncAgent()

    async def demo_async():
        print("\n⚡ Running async operations...")

        # Concurrent API calls
        print("Fetching from multiple APIs concurrently...")
        results = await async_agent.fetch_multiple_apis()
        for endpoint, response in results.items():
            print(f"  {endpoint}: {response['status']}")

        # Parallel database operations
        print("\nExecuting parallel database operations...")
        ops = [
            ("create", "Carol Davis", "carol@example.com"),
            ("create", "David Lee", "david@example.com"),
            ("search", "David"),
        ]
        results = await async_agent.parallel_database_operations(ops)
        for result in results:
            print(f"  {result}")

    # Run async demo
    asyncio.run(demo_async())

    print("\n" + "=" * 70)
    print("Integration Patterns Summary")
    print("=" * 70)
    print("""
Database Integration:
  - CRUD operations through agent tools
  - Transaction management
  - Connection pooling
  - ORM integration (SQLAlchemy)

API Integration:
  - External service calls
  - Error handling and retries
  - Rate limiting
  - Authentication (API keys, OAuth)

RAG Integration:
  - Document retrieval
  - Embedding-based search
  - Answer generation with sources
  - Document processing

Fine-tuned Models:
  - Custom model serving
  - Batch prediction
  - Model versioning
  - A/B testing

Async Workflows:
  - Concurrent API calls
  - Parallel database operations
  - Non-blocking I/O
  - Performance optimization

Best Practices:
  1. Use connection pooling for databases
  2. Implement circuit breakers for APIs
  3. Cache frequently accessed data
  4. Use async/await for I/O operations
  5. Add comprehensive error handling
  6. Monitor performance and latency
  7. Implement rate limiting
  8. Use transaction management for databases
    """)
