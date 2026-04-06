"""
LangGraph Hello World Agent - Minimal Example

This is the simplest possible LangGraph state graph that demonstrates core concepts.
LangGraph is a framework for building stateful multi-step AI agents and workflows.

References:
- LangGraph Tutorial 2026: https://growai.in/langgraph-tutorial-stateful-ai-agents-2026/
- LangGraph 101 Building First Stateful Agent: https://abstractalgorithms.dev/langgraph-101-building-your-first-stateful-agent
- Complete Guide to LangGraph 2026: https://www.linkedin.com/pulse/complete-guide-langgraph-2026-edition-learnbay-esb7c
- LangGraph Installation & Setup: https://machinelearningplus.com/gen-ai/langgraph-installation-setup-first-graph/
- LangGraph Beginner's Guide 2026: https://langchain-tutorials.github.io/langgraph-tutorial-2026-beginners-guide/

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install langgraph>=0.1.0
# pip install langchain>=0.1.0
# pip install openai>=1.0.0

from typing import TypedDict, Literal, Optional
from enum import Enum


# ============================================================================
# State Definition
# ============================================================================


class AgentState(TypedDict):
    """
    The state that flows through the LangGraph state machine.
    This defines what information the graph passes between nodes.
    """

    input: str
    step: int
    messages: list
    result: Optional[str]
    current_action: str


# ============================================================================
# Node Functions
# ============================================================================


def process_input_node(state: AgentState) -> AgentState:
    """
    First node: Process user input.

    This node validates and processes the initial user input.
    """
    print(f"\n🔵 Node 1: Processing Input")
    print(f"   Input: {state['input']}")

    state["step"] = 1
    state["current_action"] = "processing_input"
    state["messages"].append(f"Step 1: Processing input: {state['input']}")

    return state


def analyze_node(state: AgentState) -> AgentState:
    """
    Second node: Analyze the request.

    This node analyzes what the user is asking for.
    """
    print(f"\n🟢 Node 2: Analyzing Request")

    state["step"] = 2
    state["current_action"] = "analyzing"

    analysis = f"Analyzing: '{state['input']}' - request type detected"
    state["messages"].append(f"Step 2: {analysis}")

    print(f"   {analysis}")

    return state


def execute_node(state: AgentState) -> AgentState:
    """
    Third node: Execute the action.

    This node executes the appropriate action based on the analysis.
    """
    print(f"\n🟡 Node 3: Executing Action")

    state["step"] = 3
    state["current_action"] = "executing"

    # Simulate tool execution
    execution_result = f"Executed action for: {state['input']}"
    state["messages"].append(f"Step 3: {execution_result}")
    state["result"] = execution_result

    print(f"   {execution_result}")

    return state


def generate_response_node(state: AgentState) -> AgentState:
    """
    Fourth node: Generate final response.

    This node synthesizes the results into a final response.
    """
    print(f"\n🟠 Node 4: Generating Response")

    state["step"] = 4
    state["current_action"] = "generating_response"

    response = f"Response to '{state['input']}': {state.get('result', 'No result')}"
    state["messages"].append(f"Step 4: {response}")

    print(f"   {response}")

    return state


# ============================================================================
# Edge Condition Functions
# ============================================================================


def should_continue_simple(state: AgentState) -> Literal["analyze", "end"]:
    """
    Simple routing logic: always continue to analyze.

    In more complex graphs, you might have conditional logic here.
    """
    return "analyze"


def should_execute(state: AgentState) -> Literal["execute", "end"]:
    """
    Routing logic: decide whether to execute or end.
    """
    # In a real implementation, this might check the analysis
    return "execute"


def should_respond(state: AgentState) -> Literal["generate_response", "end"]:
    """
    Routing logic: decide whether to generate response or end.
    """
    return "generate_response"


# ============================================================================
# SimpleLangGraph State Graph
# ============================================================================


class SimpleLangGraph:
    """
    A minimal LangGraph implementation demonstrating:
    - StateGraph creation
    - Node definition
    - Edge routing
    - Graph execution
    """

    def __init__(self):
        """Initialize the simple state graph."""
        self.nodes = {
            "process_input": process_input_node,
            "analyze": analyze_node,
            "execute": execute_node,
            "generate_response": generate_response_node,
        }

        self.edges = {
            "process_input": should_continue_simple,
            "analyze": should_execute,
            "execute": should_respond,
            "generate_response": None,  # End node
        }

    def create_initial_state(self, user_input: str) -> AgentState:
        """
        Create the initial state for graph execution.

        Args:
            user_input: The user's input to process

        Returns:
            Initial AgentState
        """
        return {
            "input": user_input,
            "step": 0,
            "messages": [],
            "result": None,
            "current_action": "idle",
        }

    def run(self, user_input: str) -> AgentState:
        """
        Execute the state graph with given input.

        This is a simplified synchronous execution.
        In a real LangGraph, this would use the proper framework.

        Args:
            user_input: The user's input

        Returns:
            Final state after graph execution
        """
        print("\n" + "=" * 70)
        print(f"Executing LangGraph with input: '{user_input}'")
        print("=" * 70)

        # Initialize state
        state = self.create_initial_state(user_input)

        # Execute nodes in sequence
        current_node = "process_input"
        execution_history = []

        while current_node is not None:
            # Execute the node
            if current_node in self.nodes:
                print(f"\n⚙️ Executing node: {current_node}")
                state = self.nodes[current_node](state)
                execution_history.append(current_node)

                # Determine next node
                if current_node in self.edges and self.edges[current_node]:
                    next_node = self.edges[current_node](state)
                    current_node = next_node if next_node != "end" else None
                else:
                    current_node = None
            else:
                current_node = None

        # Display execution summary
        print("\n" + "=" * 70)
        print("Graph Execution Summary")
        print("=" * 70)
        print(f"Nodes executed: {' → '.join(execution_history)}")
        print(f"Total steps: {state['step']}")
        print(f"\nConversation Flow:")
        for msg in state["messages"]:
            print(f"  {msg}")

        return state


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LangGraph Hello World Agent - Minimal Example")
    print("=" * 70)
    print("\nLangGraph is used for building stateful workflows as directed graphs.")
    print("Nodes represent processing steps, edges define the flow.")

    # Create and run the graph
    graph = SimpleLangGraph()

    # Example 1: Simple query
    print("\n\n" + "=" * 70)
    print("Example 1: Simple Query")
    print("=" * 70)
    result1 = graph.run("What is 5 plus 3?")

    # Example 2: Different input
    print("\n\n" + "=" * 70)
    print("Example 2: Information Request")
    print("=" * 70)
    result2 = graph.run("Tell me about Python programming")

    print("\n\n" + "=" * 70)
    print("Full LangGraph Implementation")
    print("=" * 70)
    print("""
To use the full LangGraph framework:

1. Install LangGraph: pip install langgraph langchain

2. Define your state as TypedDict:
   from typing import TypedDict
   
   class State(TypedDict):
       messages: list
       input: str
       output: str

3. Create a state graph:
   from langgraph.graph import StateGraph
   
   graph_builder = StateGraph(State)
   
   # Add nodes
   graph_builder.add_node("node_1", node_1_function)
   graph_builder.add_node("node_2", node_2_function)
   
   # Add edges
   graph_builder.add_edge("node_1", "node_2")
   graph_builder.set_entry_point("node_1")
   graph_builder.set_finish_point("node_2")
   
   # Compile the graph
   graph = graph_builder.compile()

4. Execute the graph:
   result = graph.invoke({"input": "user input"})

Key Concepts:
- State: The data flowing through the graph (TypedDict)
- Nodes: Functions that process state
- Edges: Routes between nodes (can be conditional)
- Entry Point: Where execution starts
- Finish Point: Where execution ends
- Recursion Limit: Prevents infinite loops

Use Cases:
- Multi-step workflows
- Conditional reasoning
- Tool use orchestration
- State machine implementations
- Complex business logic flows

See:
- https://growai.in/langgraph-tutorial-stateful-ai-agents-2026/
- https://abstractalgorithms.dev/langgraph-101-building-your-first-stateful-agent
    """)

    print("\n" + "=" * 70)
    print("State Graph Visualization")
    print("=" * 70)
    print("""
Graph Structure:
    
    [START] → [process_input] → [analyze] → [execute] → [generate_response] → [END]
    
Each node transforms the state, passing it to the next node.
Conditional edges can route to different nodes based on state.
    """)
