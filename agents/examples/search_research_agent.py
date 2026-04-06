"""
Research and Information Synthesis Agent
Author: Shuvam Banerji Seal

This module implements a specialized agent for web research and information synthesis.
The agent performs multi-source searches, tracks citations, and synthesizes information
from various sources to provide comprehensive, well-sourced answers.

Features:
- Web search tool integration
- Information synthesis from multiple sources
- Source tracking and attribution
- Result formatting with citations
- Multi-step research workflows
- Recursive search for complex queries

Source: https://www.firecrawl.dev/glossary/web-search-apis/web-search-apis-langchain-ai-frameworks-integration
Source: https://www.searchcans.com/blog/langchain-google-search-agent-tutorial/
Source: https://python.langchain.com/api_reference
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
import requests

from langchain_core.tools import Tool
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured representation of a search result."""

    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.8
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def format_citation(self) -> str:
        """Format as APA-style citation."""
        return f"[{self.source}] {self.title} - {self.url}"


@dataclass
class SynthesizedAnswer:
    """Structured representation of a synthesized research answer."""

    question: str
    summary: str
    key_points: List[str]
    sources: List[SearchResult]
    confidence_level: str  # low, medium, high
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_markdown(self) -> str:
        """Convert answer to Markdown format."""
        md = f"# Research Answer: {self.question}\n\n"
        md += f"**Summary:** {self.summary}\n\n"
        md += f"**Confidence Level:** {self.confidence_level}\n\n"

        if self.key_points:
            md += "## Key Points\n"
            for point in self.key_points:
                md += f"- {point}\n"
            md += "\n"

        if self.sources:
            md += "## Sources\n"
            for i, source in enumerate(self.sources, 1):
                md += f"{i}. {source.format_citation()}\n"
                md += f"   - {source.snippet}\n"

        return md


class ResearchTools:
    """Collection of tools for research and information gathering."""

    def __init__(self, search_api_key: Optional[str] = None):
        """
        Initialize research tools.

        Args:
            search_api_key: API key for search service (uses env var if not provided)
        """
        self.api_key = search_api_key or os.getenv("TAVILY_API_KEY")
        self.search_results_cache: Dict[str, List[SearchResult]] = {}
        logger.info("ResearchTools initialized")

    def web_search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform web search and return structured results.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects
        """
        try:
            # Check cache first
            cache_key = f"{query}:{max_results}"
            if cache_key in self.search_results_cache:
                logger.debug(f"Returning cached results for: {query}")
                return self.search_results_cache[cache_key]

            # Initialize Tavily search tool
            search_tool = TavilySearchResults(
                max_results=max_results, api_key=self.api_key
            )

            # Perform search
            raw_results = search_tool.invoke({"query": query})

            # Parse and structure results
            structured_results = []
            for result in raw_results:
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    source=self._extract_domain(result.get("url", "")),
                )
                structured_results.append(search_result)

            # Cache results
            self.search_results_cache[cache_key] = structured_results

            logger.info(
                f"Web search completed: {query} ({len(structured_results)} results)"
            )
            return structured_results

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    def deep_research(
        self, topic: str, depth: int = 2, max_results_per_query: int = 3
    ) -> List[SearchResult]:
        """
        Perform deep research on a topic with multiple related queries.

        Args:
            topic: Main research topic
            depth: Number of follow-up searches
            max_results_per_query: Max results per individual search

        Returns:
            Aggregated list of SearchResult objects
        """
        all_results = []
        searched_queries = set()

        # Initial search
        queries = [topic]

        for i in range(depth):
            for query in queries:
                if query not in searched_queries:
                    results = self.web_search(query, max_results_per_query)
                    all_results.extend(results)
                    searched_queries.add(query)

                    # Generate follow-up queries based on results
                    if i < depth - 1:
                        follow_ups = self._generate_follow_up_queries(topic, results)
                        queries.extend(follow_ups)

        # Deduplicate by URL
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result.url not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result.url)

        logger.info(f"Deep research completed: {len(unique_results)} unique results")
        return unique_results

    def compare_sources(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Compare and analyze multiple sources.

        Args:
            results: List of search results to compare

        Returns:
            Comparison analysis dictionary
        """
        if not results:
            return {"error": "No results to compare"}

        sources = [r.source for r in results]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1

        analysis = {
            "total_sources": len(set(sources)),
            "total_results": len(results),
            "source_distribution": source_counts,
            "sources_with_most_results": max(source_counts.items(), key=lambda x: x[1])[
                0
            ]
            if source_counts
            else None,
        }

        logger.info(f"Source comparison completed: {analysis}")
        return analysis

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain
        except Exception:
            return "unknown"

    def _generate_follow_up_queries(
        self, main_topic: str, results: List[SearchResult], num_queries: int = 2
    ) -> List[str]:
        """Generate follow-up search queries based on initial results."""
        # Extract potential related terms from snippets
        follow_ups = []

        # Simple keyword extraction from snippets
        all_text = " ".join([r.snippet for r in results[:2]])
        words = all_text.split()

        # Create follow-up queries
        for i in range(min(num_queries, 2)):
            if len(words) > 10 * i:
                follow_up = f"{main_topic} {words[10 * i : 10 * i + 3]}"
                follow_ups.append(follow_up.strip())

        return follow_ups


class ResearchAgent:
    """
    Intelligent research agent for information synthesis and source tracking.

    Combines web search, information synthesis, and citation management
    to provide well-researched, properly attributed answers.

    Attributes:
        llm: Language model instance
        research_tools: ResearchTools instance
        agent_executor: Agent execution engine
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.5,
        llm_api_key: Optional[str] = None,
        search_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize research agent.

        Args:
            model: LLM model name
            temperature: LLM temperature (lower = more factual)
            llm_api_key: OpenAI API key
            search_api_key: Search API key

        Example:
            >>> agent = ResearchAgent(
            ...     model="gpt-3.5-turbo",
            ...     temperature=0.3
            ... )
        """
        # Initialize LLM with lower temperature for factuality
        llm_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=llm_key,
        )

        # Initialize research tools
        self.research_tools = ResearchTools(search_api_key)

        # Setup agent tools
        self.tools = self._setup_tools()

        # Initialize agent executor
        self.agent_executor = self._create_agent_executor()

        # Research history
        self.research_history: List[SynthesizedAnswer] = []

        logger.info("ResearchAgent initialized")

    def _setup_tools(self) -> List[Tool]:
        """Setup available tools for the agent."""
        tools = [
            Tool(
                name="web_search",
                func=self._web_search_tool,
                description="Search the web for current information on a topic",
            ),
            Tool(
                name="deep_research",
                func=self._deep_research_tool,
                description="Perform comprehensive research with multiple related queries",
            ),
            Tool(
                name="compare_sources",
                func=self._compare_sources_tool,
                description="Analyze and compare reliability of different sources",
            ),
            Tool(
                name="synthesize_findings",
                func=self._synthesize_tool,
                description="Synthesize information from multiple sources into coherent answer",
            ),
        ]
        return tools

    def _web_search_tool(self, query: str) -> str:
        """Tool for web search."""
        results = self.research_tools.web_search(query, max_results=5)
        formatted = json.dumps([r.to_dict() for r in results], indent=2, default=str)
        return formatted

    def _deep_research_tool(self, topic: str) -> str:
        """Tool for deep research."""
        results = self.research_tools.deep_research(topic, depth=2)
        formatted = json.dumps([r.to_dict() for r in results], indent=2, default=str)
        return formatted

    def _compare_sources_tool(self, results_json: str) -> str:
        """Tool for comparing sources."""
        try:
            results_data = json.loads(results_json)
            results = [
                SearchResult(**r) if isinstance(r, dict) else r for r in results_data
            ]
            analysis = self.research_tools.compare_sources(results)
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return f"Error in comparison: {str(e)}"

    def _synthesize_tool(self, content: str) -> str:
        """Tool for synthesizing findings."""
        # This would normally use the LLM to synthesize
        return f"Synthesized content: {content[:200]}..."

    def _create_agent_executor(self) -> AgentExecutor:
        """Create and configure the agent executor."""
        system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert research agent. Your task is to:
1. Search for reliable information from authoritative sources
2. Synthesize information from multiple sources
3. Track and cite your sources properly
4. Provide balanced, fact-based answers
5. Highlight any conflicting information or areas of uncertainty

Always prioritize accuracy and proper attribution over speculation.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=10,
            handle_parsing_errors=True,
        )

        return executor

    def research(
        self, question: str, use_deep_research: bool = False
    ) -> SynthesizedAnswer:
        """
        Perform research on a question and synthesize findings.

        Args:
            question: Research question
            use_deep_research: Whether to use comprehensive deep research

        Returns:
            SynthesizedAnswer object with findings and sources

        Example:
            >>> agent = ResearchAgent()
            >>> answer = agent.research(
            ...     "What are the latest advances in quantum computing?",
            ...     use_deep_research=True
            ... )
            >>> print(answer.to_markdown())
        """
        try:
            # Perform research
            if use_deep_research:
                sources = self.research_tools.deep_research(question)
            else:
                sources = self.research_tools.web_search(question)

            if not sources:
                logger.warning(f"No sources found for: {question}")
                return SynthesizedAnswer(
                    question=question,
                    summary="Unable to find relevant sources.",
                    key_points=[],
                    sources=[],
                    confidence_level="low",
                )

            # Prepare content for synthesis
            source_content = "\n".join(
                [
                    f"Source: {s.source}\nTitle: {s.title}\nContent: {s.snippet}"
                    for s in sources[:5]  # Use top 5 sources
                ]
            )

            # Use agent to synthesize
            synthesis_prompt = f"""Based on these sources:
{source_content}

Please provide a comprehensive answer to: {question}

Include:
1. A clear summary
2. 3-5 key points
3. Assessment of confidence level (low/medium/high)"""

            response = self.agent_executor.invoke({"input": synthesis_prompt})

            # Parse response and create structured answer
            answer = SynthesizedAnswer(
                question=question,
                summary=response.get("output", ""),
                key_points=self._extract_key_points(response.get("output", "")),
                sources=sources[:5],  # Include top 5 sources
                confidence_level=self._assess_confidence(sources),
            )

            # Store in history
            self.research_history.append(answer)

            logger.info(f"Research completed: {question}")
            return answer

        except Exception as e:
            logger.error(f"Research error: {e}")
            return SynthesizedAnswer(
                question=question,
                summary=f"Error during research: {str(e)}",
                key_points=[],
                sources=[],
                confidence_level="low",
            )

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from synthesized text."""
        points = []
        lines = text.split("\n")

        for line in lines:
            if line.strip().startswith("-") or line.strip().startswith("•"):
                point = line.strip().lstrip("-•").strip()
                if point:
                    points.append(point)

        return points[:5]  # Return top 5 points

    def _assess_confidence(self, sources: List[SearchResult]) -> str:
        """Assess confidence level based on sources."""
        if len(sources) >= 5:
            return "high"
        elif len(sources) >= 3:
            return "medium"
        else:
            return "low"

    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get all research conducted in this session."""
        return [
            {
                "question": answer.question,
                "timestamp": answer.timestamp,
                "num_sources": len(answer.sources),
                "confidence": answer.confidence_level,
            }
            for answer in self.research_history
        ]

    def export_research(self, filepath: str) -> None:
        """
        Export all research to a file.

        Args:
            filepath: Path to export file (.md or .json)
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            if filepath.endswith(".md"):
                content = "# Research Report\n\n"
                for answer in self.research_history:
                    content += answer.to_markdown() + "\n\n---\n\n"

                with open(filepath, "w") as f:
                    f.write(content)
            else:
                # JSON export
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "research_count": len(self.research_history),
                    "research": [
                        {
                            "question": a.question,
                            "summary": a.summary,
                            "key_points": a.key_points,
                            "sources": [s.to_dict() for s in a.sources],
                            "confidence": a.confidence_level,
                            "timestamp": a.timestamp,
                        }
                        for a in self.research_history
                    ],
                }

                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            logger.info(f"Research exported to {filepath}")
        except Exception as e:
            logger.error(f"Export error: {e}")


# ============================================================================
# Usage Examples
# ============================================================================


def example_basic_research() -> None:
    """Basic research example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Research")
    print("=" * 70)

    load_dotenv()

    agent = ResearchAgent(temperature=0.3)

    question = "What are the latest developments in artificial intelligence?"
    answer = agent.research(question)

    print(answer.to_markdown())


def example_deep_research() -> None:
    """Deep research example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Deep Research")
    print("=" * 70)

    load_dotenv()

    agent = ResearchAgent(temperature=0.3)

    question = "How are LLMs improving in 2026?"
    answer = agent.research(question, use_deep_research=True)

    print(answer.to_markdown())


def example_export_research() -> None:
    """Research export example."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Export Research")
    print("=" * 70)

    load_dotenv()

    agent = ResearchAgent()

    # Conduct research
    agent.research("Python programming best practices")
    agent.research("Machine learning frameworks")

    # Export results
    agent.export_research("./research_output/findings.md")
    agent.export_research("./research_output/findings.json")

    print("Research exported to ./research_output/")


if __name__ == "__main__":
    """
    Main entry point for research agent.
    
    Required environment variables:
    - OPENAI_API_KEY: OpenAI API key
    - TAVILY_API_KEY: Tavily search API key
    
    Setup Instructions:
    1. Install dependencies:
       pip install langchain langchain-openai langchain-community python-dotenv
    2. Set up .env file with API keys
    3. Run: python search_research_agent.py
    """

    print("Research Agent - Information Synthesis Example")
    print("=" * 70)
    print("\nAvailable examples:")
    print("  - example_basic_research()")
    print("  - example_deep_research()")
    print("  - example_export_research()")
    print("\nUncomment in __main__ to run examples")
