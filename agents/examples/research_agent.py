"""
AGNO Research Agent Example

A research agent that gathers and synthesizes information from multiple sources.
Demonstrates multi-turn conversation, tool usage, and information synthesis.

Author: Shuvam Banerji Seal
Source: https://docs.agno.com/agents/tools
Source: https://medium.com/@juanc.olamendy/agno-workflow-building-intelligent-multi-agent-pipelines-for-automated-content-creation-55798e42fc5c
"""

import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    An agent that researches topics and synthesizes information.

    This agent demonstrates:
    - Information gathering from multiple sources
    - Synthesis and analysis
    - Source attribution
    - Iterative research refinement

    Example:
        >>> agent = ResearchAgent()
        >>> sources = agent.research("AI agents in 2024")
        >>> summary = agent.synthesize_findings()

    Production AGNO Code:
    ```python
    from agno.agent import Agent
    from agno.models.anthropic import Claude
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.tools.newspaper4k import Newspaper4kTools

    researcher = Agent(
        name="Researcher",
        model=Claude(id="claude-3-5-sonnet-20241022"),
        tools=[DuckDuckGoTools(), Newspaper4kTools()],
        instructions="Research the topic thoroughly and cite sources.",
    )

    researcher.print_response("Research AI agents in 2024", stream=True)
    ```
    """

    def __init__(
        self, name: str = "ResearchAgent", model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize the research agent.

        Args:
            name: Agent identifier
            model: Model to use
        """
        self.name = name
        self.model = model
        self.research_queries: List[str] = []
        self.sources: Dict[str, Any] = {}
        self.findings: List[Dict[str, Any]] = []

        logger.info(f"Initialized {name} research agent")

    def add_research_query(self, query: str) -> None:
        """
        Add a research question/query.

        Args:
            query: Research topic or question
        """
        self.research_queries.append(query)
        logger.info(f"Added research query: {query}")

    def research(self, topic: str) -> List[Dict[str, Any]]:
        """
        Research a topic.

        Args:
            topic: Topic to research

        Returns:
            List of sources with information

        AGNO Research Pattern:
        1. Search for information on the topic
        2. Retrieve and parse results
        3. Extract key information
        4. Attribute sources
        5. Return structured findings
        """
        logger.info(f"Starting research on: {topic}")

        # Add to queries
        self.add_research_query(topic)

        # Simulate search results
        sources = self._simulate_search(topic)

        self.sources[topic] = sources

        return sources

    def _simulate_search(self, topic: str) -> List[Dict[str, Any]]:
        """
        Simulate web search results.

        In production, this would use actual search tools via AGNO.
        """
        return [
            {
                "title": f"Comprehensive Guide to {topic}",
                "url": "https://example.com/guide-1",
                "source": "TechBlog",
                "date": "2026-03-15",
                "snippet": f"Learn everything about {topic}. This comprehensive guide covers...",
                "relevance": 0.95,
            },
            {
                "title": f"{topic}: Trends and Future",
                "url": "https://example.com/trends-2",
                "source": "IndustryInsights",
                "date": "2026-02-20",
                "snippet": f"Exploring the latest trends in {topic}. Current developments show...",
                "relevance": 0.88,
            },
            {
                "title": f"Practical {topic}: Implementation Guide",
                "url": "https://example.com/practical-3",
                "source": "DeveloperHub",
                "date": "2026-01-10",
                "snippet": f"How to implement {topic} in practice. Step-by-step instructions...",
                "relevance": 0.82,
            },
            {
                "title": f"Comparing {topic} Solutions",
                "url": "https://example.com/comparison-4",
                "source": "TechComparison",
                "date": "2025-12-05",
                "snippet": f"Detailed comparison of different {topic} solutions and approaches...",
                "relevance": 0.75,
            },
        ]

    def extract_findings(self) -> List[Dict[str, Any]]:
        """
        Extract key findings from research.

        Returns:
            List of key findings with sources
        """
        findings = []

        for topic, sources in self.sources.items():
            for source in sources:
                finding = {
                    "finding": source["snippet"],
                    "source": source["source"],
                    "url": source["url"],
                    "topic": topic,
                    "date": source["date"],
                    "relevance": source["relevance"],
                }
                findings.append(finding)

        self.findings = findings
        logger.info(f"Extracted {len(findings)} findings")

        return findings

    def synthesize_findings(self) -> str:
        """
        Synthesize research findings into a summary.

        AGNO Synthesis Pattern:
        Agent synthesizes information from multiple sources
        into coherent conclusions while maintaining attribution.

        Returns:
            Synthesized research summary
        """
        if not self.findings:
            self.extract_findings()

        if not self.findings:
            return "No findings to synthesize"

        # Group by topic
        by_topic = {}
        for finding in self.findings:
            topic = finding["topic"]
            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append(finding)

        # Create synthesis
        synthesis = ""

        for topic, findings_list in by_topic.items():
            synthesis += f"\n## {topic}\n\n"
            synthesis += (
                f"Research Summary (based on {len(findings_list)} sources):\n\n"
            )

            for i, finding in enumerate(findings_list[:3], 1):
                synthesis += f"{i}. {finding['finding']}\n"
                synthesis += f"   Source: {finding['source']} ({finding['date']})\n"
                synthesis += f"   URL: {finding['url']}\n\n"

        return synthesis

    def get_sources_by_relevance(self) -> List[Dict[str, Any]]:
        """Get all findings sorted by relevance."""
        return sorted(self.findings, key=lambda x: x["relevance"], reverse=True)

    def get_research_summary(self) -> Dict[str, Any]:
        """Get a summary of the research."""
        return {
            "topics_researched": len(self.research_queries),
            "total_sources": len(self.findings),
            "unique_sources": len(set(f["source"] for f in self.findings)),
            "average_relevance": (
                sum(f["relevance"] for f in self.findings) / len(self.findings)
                if self.findings
                else 0
            ),
        }


class ResearchTeam:
    """
    A team of research agents working together.

    Demonstrates multi-agent research coordination.
    """

    def __init__(self, name: str = "ResearchTeam"):
        """Initialize research team."""
        self.name = name
        self.agents: Dict[str, ResearchAgent] = {}
        self.combined_findings: List[Dict[str, Any]] = []

    def add_researcher(self, specialization: str) -> ResearchAgent:
        """
        Add a research agent to the team.

        Args:
            specialization: What the agent specializes in

        Returns:
            Created ResearchAgent
        """
        agent = ResearchAgent(
            name=f"Researcher_{specialization}", model="claude-3-5-sonnet-20241022"
        )

        self.agents[specialization] = agent
        logger.info(f"Added {specialization} researcher to team")

        return agent

    def conduct_research(self, topic: str) -> Dict[str, Any]:
        """
        Conduct research with the team.

        AGNO Multi-Agent Pattern:
        Multiple agents research different aspects in parallel.
        """
        logger.info(f"Team researching: {topic}")

        results = {}

        for specialization, agent in self.agents.items():
            # Each agent researches a specialized aspect
            specialized_query = f"{topic} - {specialization} perspective"
            sources = agent.research(specialized_query)
            results[specialization] = sources

        # Aggregate findings
        all_findings = []
        for specialization, sources in results.items():
            for source in sources:
                all_findings.append({"researcher": specialization, **source})

        self.combined_findings = all_findings

        return results


def main():
    """
    Example usage of the Research Agent.

    Reference Documentation:
    - https://docs.agno.com/agents/tools
    - https://docs.agno.com/teams/overview
    - https://medium.com/@juanc.olamendy/agno-workflow-building-intelligent-multi-agent-pipelines-for-automated-content-creation-55798e42fc5c
    """
    print("\n=== AGNO Research Agent Example ===\n")

    # 1. Single Agent Research
    print("1. Single Researcher Agent\n")
    researcher = ResearchAgent(name="TechResearcher")

    # Research a topic
    sources = researcher.research("AI agents and agentic software")

    print(f"Found {len(sources)} sources")
    for source in sources[:2]:
        print(f"  - {source['title']} (Relevance: {source['relevance']:.0%})")

    # Extract and synthesize findings
    print("\n2. Extracting Findings...")
    findings = researcher.extract_findings()
    print(f"Extracted {len(findings)} findings")

    print("\n3. Synthesized Summary:")
    summary = researcher.synthesize_findings()
    print(summary[:500] + "...")

    print("\n4. Research Summary:")
    import json

    print(json.dumps(researcher.get_research_summary(), indent=2))

    # 2. Team Research
    print("\n5. Multi-Agent Research Team\n")
    team = ResearchTeam("AIResearchTeam")

    # Add specialized researchers
    team.add_researcher("Technical")
    team.add_researcher("Business")
    team.add_researcher("Implementation")

    # Conduct team research
    team_results = team.conduct_research("Large Language Models")

    print(f"Team research complete:")
    for specialization in team_results.keys():
        print(f"  - {specialization}: {len(team_results[specialization])} sources")

    print(f"\nTotal combined findings: {len(team.combined_findings)}")

    # Show top sources by relevance
    print("\nTop 3 Most Relevant Sources:")
    top_sources = [f for f in team.combined_findings][:3]
    for i, source in enumerate(top_sources, 1):
        print(f"\n{i}. {source['title']}")
        print(f"   Researcher: {source['researcher']}")
        print(f"   Relevance: {source['relevance']:.0%}")


if __name__ == "__main__":
    main()
