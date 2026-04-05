# LLM Agents and Tool Use Patterns — Agentic Skill Prompt

ReAct pattern, structured tool definitions, agent loops, memory management, and multi-agent orchestration.

---

## 1. Identity and Mission

Build reliable, efficient LLM agents that plan, execute tools, and reason about outcomes.

---

## 2. ReAct Pattern Implementation

```python
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class ReActStep:
    """Single step in ReAct loop."""
    thought: str
    action: str
    action_input: str
    observation: str

class ReActAgent:
    """Agent following ReAct (Reasoning + Acting) pattern."""
    
    def __init__(self, llm_call_fn, tools: Dict[str, callable]):
        self.llm = llm_call_fn
        self.tools = tools
        self.max_steps = 10
    
    def build_prompt(self, query: str, history: List[ReActStep] = None) -> str:
        """Build prompt with examples."""
        if history is None:
            history = []
        
        prompt = f"""Answer the question: {query}

Use this format:
Thought: <reasoning>
Action: <tool_name>
Action Input: <json_input>
Observation: <result>
... (repeat as needed)
Final Answer: <answer>

Available tools:
"""
        for tool_name, tool_func in self.tools.items():
            doc = tool_func.__doc__ or ""
            prompt += f"- {tool_name}: {doc}\n"
        
        # Add history
        for step in history:
            prompt += f"Thought: {step.thought}\n"
            prompt += f"Action: {step.action}\n"
            prompt += f"Action Input: {step.action_input}\n"
            prompt += f"Observation: {step.observation}\n"
        
        return prompt
    
    def run(self, query: str) -> Tuple[str, List[ReActStep]]:
        """Run agent loop."""
        history = []
        
        for step_num in range(self.max_steps):
            prompt = self.build_prompt(query, history)
            
            # Get LLM response
            response = self.llm(prompt)
            
            # Parse response
            thought, action, action_input, done, final_answer = self._parse_response(response)
            
            if done:
                return final_answer, history
            
            # Execute tool
            try:
                tool_fn = self.tools.get(action)
                if tool_fn is None:
                    observation = f"Unknown tool: {action}"
                else:
                    action_args = json.loads(action_input)
                    observation = tool_fn(**action_args)
            except Exception as e:
                observation = f"Error: {str(e)}"
            
            history.append(ReActStep(thought, action, action_input, str(observation)))
        
        return "Max steps reached", history
    
    def _parse_response(self, response: str) -> Tuple[str, str, str, bool, str]:
        """Parse LLM response."""
        lines = response.split("\n")
        thought = ""
        action = ""
        action_input = ""
        final_answer = ""
        
        for line in lines:
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
            elif line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()
            elif line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
        
        done = bool(final_answer)
        return thought, action, action_input, done, final_answer

# Usage
tools = {
    "search": lambda query: f"Results for {query}",
    "calculate": lambda expr: str(eval(expr)),
}

def mock_llm(prompt: str) -> str:
    return """Thought: I need to search for information
Action: search
Action Input: "information"
Observation: Found results
Final Answer: The answer is 42"""

agent = ReActAgent(mock_llm, tools)
answer, history = agent.run("What is 2+2?")
```

---

## 3. Structured Tool Definitions

```python
from typing import Dict, Any, Callable
from enum import Enum

class ParameterType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"

class ToolParameter:
    """Define a tool parameter."""
    def __init__(
        self,
        name: str,
        type: ParameterType,
        description: str,
        required: bool = True,
    ):
        self.name = name
        self.type = type
        self.description = description
        self.required = required
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "required": self.required,
        }

class ToolDefinition:
    """Define a tool in OpenAI function format."""
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type.value,
                            "description": param.description,
                        }
                        for param in self.parameters
                    },
                    "required": [
                        param.name for param in self.parameters
                        if param.required
                    ],
                },
            },
        }
    
    def call(self, **kwargs) -> Any:
        """Execute the tool."""
        return self.func(**kwargs)

# Define a search tool
search_tool = ToolDefinition(
    name="web_search",
    description="Search the web for information",
    parameters=[
        ToolParameter("query", ParameterType.STRING, "Search query", required=True),
        ToolParameter("num_results", ParameterType.NUMBER, "Number of results", required=False),
    ],
    func=lambda query, num_results=5: f"Search results for {query}",
)

print(json.dumps(search_tool.to_openai_format(), indent=2))
```

---

## 4. Agent Memory Management

```python
from collections import deque
from typing import List, Optional

class AgentMemory:
    """Manage agent context and memory."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.step_history: deque = deque(maxlen=max_history)
        self.knowledge_base: Dict[str, Any] = {}
    
    def add_step(self, step: ReActStep) -> None:
        """Add step to history."""
        self.step_history.append(step)
    
    def store_fact(self, key: str, value: Any) -> None:
        """Store fact in knowledge base."""
        self.knowledge_base[key] = value
    
    def retrieve_fact(self, key: str) -> Optional[Any]:
        """Retrieve fact from knowledge base."""
        return self.knowledge_base.get(key)
    
    def get_context(self) -> str:
        """Get formatted context for prompt."""
        context = "Previous interactions:\n"
        for step in list(self.step_history)[-3:]:
            context += f"- {step.thought}\n"
        
        context += "\nKnown facts:\n"
        for key, val in self.knowledge_base.items():
            context += f"- {key}: {val}\n"
        
        return context

# Usage
memory = AgentMemory(max_history=5)
```

---

## 5. Multi-Agent Orchestration

```python
from typing import List

class MultiAgentOrchestrator:
    """Coordinate multiple specialized agents."""
    
    def __init__(self, agents: Dict[str, ReActAgent]):
        self.agents = agents
        self.delegation_history = []
    
    def route_task(self, task: str) -> Tuple[str, str]:
        """Route task to appropriate agent."""
        # Simple heuristic routing
        if "search" in task.lower():
            agent_name = "search_agent"
        elif "calculate" in task.lower():
            agent_name = "math_agent"
        else:
            agent_name = "general_agent"
        
        return agent_name, task
    
    def execute_multi_stage(self, task: str) -> str:
        """Execute task across multiple agents."""
        agent_name, refined_task = self.route_task(task)
        
        if agent_name not in self.agents:
            return f"No agent found for task type: {agent_name}"
        
        agent = self.agents[agent_name]
        result, history = agent.run(refined_task)
        
        self.delegation_history.append({
            "task": task,
            "agent": agent_name,
            "result": result,
        })
        
        return result
```

---

## 6. References

1. https://arxiv.org/abs/2210.03629 — "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al.)
2. https://github.com/ysymyth/ReAct — ReAct official
3. https://platform.openai.com/docs/guides/function-calling — OpenAI function calling
4. https://arxiv.org/abs/2309.10025 — "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al.)
5. https://github.com/langchain-ai/langchain — LangChain agent framework
6. https://github.com/hwchase17/langchain — LangChain official
7. https://arxiv.org/abs/2301.10160 — "Multi-Agent Collaboration for LLMs"
8. https://github.com/OpenGVLab/LLM-Agents — Agent implementations
9. https://arxiv.org/abs/2302.04761 — "Generative Agents: Interactive Simulacra of Human Behavior"
10. https://github.com/joonspk-research/generative_agents — Generative agents implementation
11. https://huggingface.co/docs/transformers/main/en/chat_interface — Chat interface patterns
12. https://arxiv.org/abs/2305.18323 — "Agent Instructed Reinforcement Learning"
13. https://github.com/microsoft/JARVIS — JARVIS multi-modal agent
14. https://arxiv.org/abs/2306.04031 — "Cooperative Agent Planning with Language Models"
15. https://github.com/IBM/nlc-omt — Multi-agent planning patterns
16. https://arxiv.org/abs/2310.07554 — "Agents that Reason and Act"

---

## 7. Uncertainty and Limitations

**Not Covered:** Adversarial attacks on agents, real-time planning optimization, autonomous agent evaluation. **Production:** Implement timeout handling, error recovery, human-in-the-loop oversight.
