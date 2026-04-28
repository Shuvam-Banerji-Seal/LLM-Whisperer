# Tree-of-Thought Prompting

## Problem Statement

Chain-of-Thought prompting improves reasoning by making steps explicit, but it assumes a single, linear reasoning path. For many complex problems, this assumption is restrictive. Consider planning a trip, solving a puzzle, or debugging code - there are often multiple viable approaches, and the best path forward depends on choices made during exploration. A linear reasoning chain commits to one direction early, potentially missing better solutions or getting stuck in dead ends.

Tree-of-Thought (ToT) prompting addresses this by framing problem-solving as exploration over a tree of possible reasoning paths. At each step, the model can generate multiple candidate "thoughts," evaluate their promise, and explore the most promising branches. This deliberate search enables backtracking, parallel exploration of alternatives, and global optimization rather than greedy local decisions.

ToT is particularly powerful for problems requiring search, planning, or strategic decision-making where simple step-by-step reasoning can miss optimal or even viable solutions.

This skill covers understanding Tree-of-Thought prompting, implementing various search strategies (BFS, DFS, beam search), combining ToT with evaluation functions, and applying ToT to complex multi-step problems.

## Theory & Fundamentals

### From Chain to Tree

**Chain-of-Thought**: Single path, no backtracking
```
Start → Step 1 → Step 2 → Step 3 → ... → Answer
```

**Tree-of-Thought**: Multiple paths, branching exploration
```
                    Start
                   /  |  \
               Step1 Step1' Step1''
              / |    / \      |
           S2  S2'  S2'' S2'''  ...
```

### ToT Components

A ToT system has four key components:

**1. Thought Decomposition**: Breaking the problem into meaningful steps
$$\text{Problem} \rightarrow \{T_1, T_2, ..., T_n\} \text{ where } T_i \text{ is a thought}$$

**2. Thought Generator**: Generate candidate thoughts at each state
$$G(s, p) \rightarrow \{t_1, t_2, ..., t_k\}$$

**3. State Evaluator**: Assess the promise of each partial solution
$$V(s) \in \mathbb{R} \text{ (value score)}$$

**4. Search Algorithm**: Navigate the tree (BFS, DFS, beam, etc.)

### Search Strategies

**Breadth-First Search (BFS)**:
- Explores all nodes at depth d before depth d+1
- Guarantees finding shortest solution
- Memory-intensive for deep/large trees

**Depth-First Search (DFS)**:
- Explores one branch fully before backtracking
- Memory-efficient
- May miss better solutions in other branches

**Beam Search**:
- Keeps top-k candidates at each level
- Balances exploration and efficiency
- Prunes unlikely branches

**Monte Carlo Tree Search (MCTS)**:
- Uses random sampling and backpropagation
- Balances exploitation and exploration
- Effective for large search spaces

### Mathematical Framework

**Value Function Design**:
$$V(s) = \alpha \cdot \text{plausibility}(s) + \beta \cdot \text{progress}(s) + \gamma \cdot \text{diversity}(s)$$

Where:
- plausibility: How likely is this path to lead to a valid solution?
- progress: How much progress toward the goal?
- diversity: Is this different from other candidates?

**Upper Confidence Bound for Trees (UCT)**:
$$UCT = \frac{Q}{N} + c \sqrt{\frac{\ln N_{parent}}{N_{child}}}$$

Balances exploitation (Q/N) vs exploration (second term).

## Implementation Patterns

### Pattern 1: Basic Tree-of-Thought Implementation

```python
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import deque

class NodeState(Enum):
    PENDING = "pending"
    EXPANDED = "expanded"
    TERMINAL = "terminal"
    PRUNED = "pruned"

@dataclass
class ToTNode:
    """A node in the Tree-of-Thought."""
    id: str
    content: str
    parent: Optional['ToTNode'] = None
    children: List['ToTNode'] = field(default_factory=list)
    depth: int = 0
    value: float = 0.0
    state: NodeState = NodeState.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.value > other.value

class TreeOfThought:
    """
    Basic Tree-of-Thought implementation.
    Supports BFS, DFS, and Beam search strategies.
    """
    
    def __init__(
        self,
        llm_client,
        thought_generator: Callable[[str, int], List[str]],
        state_evaluator: Callable[[str], float],
        max_depth: int = 10,
        beam_width: int = 3
    ):
        self.llm = llm_client
        self.generate = thought_generator
        self.evaluate = state_evaluator
        self.max_depth = max_depth
        self.beam_width = beam_width
        
        self.node_counter = 0
    
    def solve(
        self,
        problem: str,
        strategy: str = "beam",
        verbose: bool = False
    ) -> Dict:
        """
        Solve problem using Tree-of-Thought.
        
        Args:
            problem: The problem to solve
            strategy: One of "bfs", "dfs", "beam", or "best"
            verbose: If True, return full search tree
        
        Returns:
            Dictionary with solution, path, and statistics
        """
        if strategy == "bfs":
            return self._solve_bfs(problem, verbose)
        elif strategy == "dfs":
            return self._solve_dfs(problem, verbose)
        elif strategy == "beam":
            return self._solve_beam(problem, verbose)
        elif strategy == "best":
            return self._solve_best_first(problem, verbose)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _solve_bfs(self, problem: str, verbose: bool) -> Dict:
        """Breadth-First Search."""
        root = self._create_node(problem, depth=0)
        root.value = self.evaluate(problem)
        
        queue = deque([root])
        nodes_expanded = 0
        best_solution = None
        best_value = float('-inf')
        
        while queue:
            current = queue.popleft()
            
            if current.depth >= self.max_depth:
                if current.value > best_value:
                    best_value = current.value
                    best_solution = current
                continue
            
            nodes_expanded += 1
            
            try:
                candidates = self._generate_candidates(current.content, current.depth)
            except Exception as e:
                continue
            
            for candidate in candidates:
                child = self._create_node(
                    content=candidate,
                    parent=current,
                    depth=current.depth + 1
                )
                child.value = self.evaluate(candidate)
                
                current.children.append(child)
                queue.append(child)
        
        return self._build_result(best_solution, nodes_expanded, verbose)
    
    def _solve_dfs(self, problem: str, verbose: bool) -> Dict:
        """Depth-First Search with pruning."""
        root = self._create_node(problem, depth=0)
        root.value = self.evaluate(problem)
        
        best_solution = [root]
        best_value = root.value
        nodes_expanded = 0
        
        def dfs(node: ToTNode):
            nonlocal best_solution, best_value, nodes_expanded
            
            if node.depth >= self.max_depth:
                if node.value > best_value:
                    best_value = node.value
                    best_solution = self._get_path(node)
                return
            
            nodes_expanded += 1
            
            if node.value < best_value * 0.5:
                return
            
            try:
                candidates = self._generate_candidates(node.content, node.depth)
            except Exception:
                return
            
            for candidate in candidates:
                child = self._create_node(
                    content=candidate,
                    parent=node,
                    depth=node.depth + 1
                )
                child.value = self.evaluate(candidate)
                
                node.children.append(child)
                
                dfs(child)
                
                if best_value >= 1.0:
                    return
        
        dfs(root)
        
        return self._build_result(best_solution[0] if best_solution else root, nodes_expanded, verbose)
    
    def _solve_beam(self, problem: str, verbose: bool) -> Dict:
        """Beam Search - keep top-k candidates at each level."""
        root = self._create_node(problem, depth=0)
        root.value = self.evaluate(problem)
        
        beam = [root]
        nodes_expanded = 0
        best_solution = root
        best_value = root.value
        
        for depth in range(self.max_depth):
            candidates = []
            
            for node in beam:
                nodes_expanded += 1
                
                try:
                    thought_candidates = self._generate_candidates(
                        node.content, depth
                    )
                except Exception:
                    continue
                
                for candidate in thought_candidates:
                    child = self._create_node(
                        content=candidate,
                        parent=node,
                        depth=depth + 1
                    )
                    child.value = self.evaluate(candidate)
                    
                    node.children.append(child)
                    candidates.append(child)
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: x.value, reverse=True)
            beam = candidates[:self.beam_width]
            
            if beam[0].value > best_value:
                best_value = beam[0].value
                best_solution = beam[0]
            
            if best_value >= 1.0:
                break
        
        return self._build_result(best_solution, nodes_expanded, verbose)
    
    def _solve_best_first(self, problem: str, verbose: bool) -> Dict:
        """Best-First Search using priority queue."""
        root = self._create_node(problem, depth=0)
        root.value = self.evaluate(problem)
        
        heap = [(~root.value, root.id, root)]
        visited = {root.id}
        nodes_expanded = 0
        best_solution = root
        best_value = root.value
        
        while heap:
            _, _, current = heapq.heappop(heap)
            
            if current.depth >= self.max_depth:
                if current.value > best_value:
                    best_value = current.value
                    best_solution = current
                continue
            
            nodes_expanded += 1
            
            try:
                candidates = self._generate_candidates(
                    current.content, current.depth
                )
            except Exception:
                continue
            
            for candidate in candidates:
                child = self._create_node(
                    content=candidate,
                    parent=current,
                    depth=current.depth + 1
                )
                child.value = self.evaluate(candidate)
                
                current.children.append(child)
                
                if child.id not in visited:
                    visited.add(child.id)
                    heapq.heappush(heap, (~child.value, child.id, child))
        
        return self._build_result(best_solution, nodes_expanded, verbose)
    
    def _generate_candidates(
        self,
        current_content: str,
        depth: int
    ) -> List[str]:
        """Generate candidate thoughts using the LLM."""
        prompt = f"""Current state:
{current_content}

Depth: {depth}/{self.max_depth}

Generate 3-5 possible next steps or alternatives. Each should be a distinct approach or direction.

Format:
1. [first alternative]
2. [second alternative]
3. [third alternative]
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=500)
        
        candidates = self._parse_candidates(response)
        
        return candidates
    
    def _parse_candidates(self, response: str) -> List[str]:
        """Parse generated candidates from LLM response."""
        lines = response.strip().split('\n')
        candidates = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = None
            for pattern in [r'^\d+[\.\)]\s*(.+)', r'^[•\-*]\s*(.+)', r'^(.+)']:
                import re
                m = re.match(pattern, line)
                if m:
                    match = m.group(1).strip()
                    break
            
            if match and len(match) > 10:
                candidates.append(match)
        
        return candidates[:5]
    
    def _create_node(
        self,
        content: str,
        parent: Optional[ToTNode] = None,
        depth: int = 0
    ) -> ToTNode:
        """Create a new node."""
        self.node_counter += 1
        
        return ToTNode(
            id=f"node_{self.node_counter}",
            content=content,
            parent=parent,
            depth=depth
        )
    
    def _get_path(self, node: ToTNode) -> List[ToTNode]:
        """Get path from root to node."""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        path.reverse()
        return path
    
    def _build_result(
        self,
        solution: ToTNode,
        nodes_expanded: int,
        verbose: bool
    ) -> Dict:
        """Build result dictionary."""
        result = {
            "solution": solution.content if solution else None,
            "value": solution.value if solution else 0,
            "nodes_expanded": nodes_expanded,
            "depth_reached": solution.depth if solution else 0
        }
        
        if verbose and solution:
            result["path"] = [
                {"content": n.content, "value": n.value}
                for n in self._get_path(solution)
            ]
        
        return result
```

### Pattern 2: MCTS-Based Tree Search

```python
import random
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MCTSNode:
    """Node for Monte Carlo Tree Search."""
    state: str
    parent: Optional['MCTSNode'] = None
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[str] = field(default_factory=list)
    depth: int = 0

class MCTS:
    """
    Monte Carlo Tree Search for Tree-of-Thought.
    Uses UCB1 for balancing exploration/exploitation.
    """
    
    def __init__(
        self,
        llm_client,
        rollout_policy: callable,
        expansion_threshold: int = 3,
        exploration_constant: float = 1.414
    ):
        self.llm = llm_client
        self.rollout = rollout_policy
        self.expansion_threshold = expansion_threshold
        self.exploration_constant = exploration_constant
    
    def search(
        self,
        initial_state: str,
        max_iterations: int = 1000,
        max_depth: int = 20
    ) -> Tuple[str, float]:
        """
        Run MCTS search.
        
        Returns:
            best_state: The highest-value state found
            best_value: Value of that state
        """
        root = MCTSNode(state=initial_state, depth=0)
        root.untried_actions = self._get_actions(initial_state)
        
        best_node = root
        best_value = 0.0
        
        for iteration in range(max_iterations):
            node = self._select(root)
            
            if node.depth < max_depth:
                node = self._expand(node)
            
            value = self._rollout(node.state, max_depth - node.depth)
            
            self._backpropagate(node, value)
            
            if node.visits > 0 and node.value / node.visits > best_value:
                best_node = node
                best_value = node.value / node.visits
        
        return self._get_best_leaf(root).state, best_value
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select child using UCB1."""
        while node.untried_actions == [] and node.children != {}:
            node = self._ucb_select(node)
        return node
    
    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB1 score."""
        def ucb1(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')
            exploitation = child.value / child.visits
            exploration = self.exploration_constant * math.sqrt(
                math.log(node.visits) / child.visits
            )
            return exploitation + exploration
        
        return max(node.children.values(), key=ucb1)
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a child."""
        if node.untried_actions == []:
            return node
        
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        new_state = self._apply_action(node.state, action)
        child = MCTSNode(
            state=new_state,
            parent=node,
            depth=node.depth + 1,
            untried_actions=self._get_actions(new_state)
        )
        
        node.children[action] = child
        return child
    
    def _rollout(self, state: str, max_depth: int) -> float:
        """Simulate random rollout to estimate value."""
        return self.rollout(state, max_depth)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Update node and ancestors with rollout result."""
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _get_best_leaf(self, node: MCTSNode) -> MCTSNode:
        """Get leaf with highest average value."""
        if node.children == {}:
            return node
        
        best_child = max(
            node.children.values(),
            key=lambda c: c.value / c.visits if c.visits > 0 else 0
        )
        
        return self._get_best_leaf(best_child)
    
    def _get_actions(self, state: str) -> List[str]:
        """Get possible actions from state."""
        prompt = f"""State:
{state}

What are 3-5 possible actions or decisions from this state? Be specific and diverse.

Format:
- Action 1: [description]
- Action 2: [description]
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=300)
        
        actions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                actions.append(line[1:].strip())
        
        return actions[:5]
    
    def _apply_action(self, state: str, action: str) -> str:
        """Apply action to state, returning new state."""
        prompt = f"""Current state:
{state}

Action to take:
{action}

What is the resulting state after taking this action? Describe the new state.
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=300)
        return response.strip()
```

### Pattern 3: Self-Evaluation in ToT

```python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import re

@dataclass
class EvaluatedThought:
    thought: str
    evaluation: str
    score: float
    strengths: List[str]
    weaknesses: List[str]

class ToTWithSelfEvaluation:
    """
    Tree-of-Thought with self-evaluation at each node.
    Model evaluates each thought before deciding to explore.
    """
    
    def __init__(
        self,
        llm_client,
        max_depth: int = 5,
        beam_width: int = 3,
        min_score_threshold: float = 0.3
    ):
        self.llm = llm_client
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_score_threshold = min_score_threshold
    
    def solve(
        self,
        problem: str,
        goal: Optional[str] = None
    ) -> Dict:
        """
        Solve with self-evaluated tree search.
        """
        if goal is None:
            goal = self._infer_goal(problem)
        
        root_evaluation = self._evaluate_state(problem, goal)
        
        if root_evaluation.score < self.min_score_threshold:
            return {
                "solution": None,
                "evaluation": root_evaluation,
                "reason": "Initial state does not meet minimum threshold"
            }
        
        beam = [(root_evaluation.score, problem, [root_evaluation])]
        best_solution = (root_evaluation.score, problem, [root_evaluation])
        
        for depth in range(self.max_depth):
            new_candidates = []
            
            for score, state, path in beam:
                if score < self.min_score_threshold:
                    continue
                
                thoughts = self._generate_alternatives(state)
                
                for thought in thoughts:
                    evaluation = self._evaluate_state(thought, goal)
                    
                    if evaluation.score >= self.min_score_threshold:
                        new_path = path + [evaluation]
                        new_candidates.append((evaluation.score, thought, new_path))
                        
                        if evaluation.score > best_solution[0]:
                            best_solution = (evaluation.score, thought, new_path)
            
            if not new_candidates:
                break
            
            new_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = new_candidates[:self.beam_width]
        
        return {
            "solution": best_solution[1],
            "evaluation_score": best_solution[0],
            "reasoning_path": [
                {"thought": e.thought, "score": e.score, "evaluation": e.evaluation}
                for e in best_solution[2]
            ]
        }
    
    def _generate_alternatives(self, state: str) -> List[str]:
        """Generate alternative thoughts for current state."""
        prompt = f"""Current thinking:
{state}

Generate 3-4 different alternative approaches or directions to continue this reasoning.
Each should take the reasoning in a distinctly different way.

Format:
1. [alternative approach]
2. [alternative approach]
3. [alternative approach]
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=400)
        
        alternatives = []
        for line in response.split('\n'):
            line = line.strip()
            match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if match:
                alternatives.append(match.group(1))
        
        return alternatives[:4]
    
    def _evaluate_state(self, state: str, goal: str) -> EvaluatedThought:
        """Evaluate how promising this state is."""
        prompt = f"""Goal: {goal}

Current state:
{state}

Evaluate this state:
1. Is it making progress toward the goal? (1-5)
2. Is the reasoning sound? (1-5)  
3. Is it likely to lead to a solution? (1-5)

Also identify:
- Strengths of this approach
- Potential weaknesses or issues

Provide your evaluation and rating.
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=300, temperature=0.3)
        
        evaluation = self._parse_evaluation(response)
        
        return evaluation
    
    def _parse_evaluation(self, response: str) -> EvaluatedThought:
        """Parse evaluation from LLM response."""
        lines = response.strip().split('\n')
        
        score_matches = re.findall(r'\b([1-5])\b', response)
        scores = [int(s) for s in score_matches[:3]]
        
        if len(scores) >= 3:
            score = np.mean(scores) / 5.0
        else:
            score = 0.5
        
        strengths = []
        weaknesses = []
        
        in_strengths = False
        in_weaknesses = False
        
        for line in lines:
            line_lower = line.lower()
            if 'strength' in line_lower:
                in_strengths = True
                in_weaknesses = False
            elif 'weakness' in line_lower or 'issue' in line_lower:
                in_weaknesses = True
                in_strengths = False
            elif line.startswith('-') or line.startswith('•'):
                content = line[1:].strip()
                if in_strengths:
                    strengths.append(content)
                elif in_weaknesses:
                    weaknesses.append(content)
        
        return EvaluatedThought(
            thought="",
            evaluation=response,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _infer_goal(self, problem: str) -> str:
        """Infer goal from problem statement."""
        prompt = f"""Problem: {problem}

What is the goal or objective to achieve? Be specific.
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=100, temperature=0)
        return response.strip()
```

### Pattern 4: ToT for Planning Tasks

```python
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: Set[str] = field(default_factory=set)
    prerequisites: Set[str] = field(default_factory=set)

class ToTPlanner:
    """
    Tree-of-Thought planning for complex multi-step tasks.
    Uses hierarchical task decomposition.
    """
    
    def __init__(
        self,
        llm_client,
        max_plan_depth: int = 5,
        max_tasks_per_level: int = 5
    ):
        self.llm = llm_client
        self.max_plan_depth = max_plan_depth
        self.max_tasks_per_level = max_tasks_per_level
    
    def create_plan(
        self,
        goal: str,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Create a plan using ToT-based hierarchical planning.
        
        Returns:
            Dictionary with plan, alternative branches, and evaluation
        """
        plan_tree = self._build_plan_tree(goal, constraints)
        
        evaluated_tree = self._evaluate_and_prune(plan_tree)
        
        best_plan = self._extract_best_plan(evaluated_tree)
        
        return {
            "plan": best_plan,
            "alternatives": self._get_alternatives(evaluated_tree),
            "tree": evaluated_tree
        }
    
    def _build_plan_tree(
        self,
        goal: str,
        constraints: Optional[Dict]
    ) -> 'PlanNode':
        """Build hierarchical plan tree."""
        root = PlanNode(
            content=f"Goal: {goal}",
            depth=0,
            node_type="goal"
        )
        
        self._expand_plan_node(root, constraints)
        
        return root
    
    def _expand_plan_node(
        self,
        node: 'PlanNode',
        constraints: Optional[Dict],
        depth: int = 0
    ):
        """Recursively expand plan node."""
        if depth >= self.max_plan_depth:
            return
        
        if node.node_type == "goal":
            prompt = f"""Goal: {node.content}

Break this goal down into 3-5 high-level tasks needed to achieve it.
Each task should be a concrete action or milestone.

Format:
1. [Task description]
2. [Task description]
"""
        elif node.node_type == "task":
            prompt = f"""Task: {node.content}

Break this task down into 3-5 specific sub-tasks or steps.
Be concrete and consider: who does what, in what order?

Format:
1. [Step description]
2. [Step description]
"""
        else:
            return
        
        response = self.llm.generate(prompt=prompt, max_tokens=400)
        
        subtasks = self._parse_tasks(response)
        
        for task_desc in subtasks[:self.max_tasks_per_level]:
            subtask_node = PlanNode(
                content=task_desc,
                depth=depth + 1,
                node_type="task" if depth < 2 else "step",
                parent=node
            )
            
            node.children.append(subtask_node)
            
            self._expand_plan_node(subtask_node, constraints, depth + 1)
    
    def _parse_tasks(self, response: str) -> List[str]:
        """Parse tasks from LLM response."""
        tasks = []
        for line in response.split('\n'):
            match = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
            if match:
                tasks.append(match.group(1))
        return tasks
    
    def _evaluate_and_prune(
        self,
        node: 'PlanNode'
    ) -> 'PlanNode':
        """Evaluate plan branches and prune low-value ones."""
        evaluation = self._evaluate_plan_node(node)
        node.value = evaluation["score"]
        
        if node.children:
            for child in node.children:
                evaluated_child = self._evaluate_and_prune(child)
            
            node.children.sort(key=lambda x: x.value, reverse=True)
            
            if len(node.children) > self.max_tasks_per_level:
                node.children = node.children[:self.max_tasks_per_level]
        
        return node
    
    def _evaluate_plan_node(self, node: 'PlanNode') -> Dict:
        """Evaluate a plan node."""
        prompt = f"""Evaluate this plan component:

{node.content}

Rate from 1-5:
1. Is this realistic and achievable? (5 = very realistic)
2. Is it well-defined and actionable? (5 = very clear)
3. Does it logically connect to the overall goal? (5 = perfect fit)

Provide your rating and brief explanation.
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=200, temperature=0.3)
        
        scores = re.findall(r'\b([1-5])\b', response)
        if len(scores) >= 3:
            score = np.mean([int(s) for s in scores[:3]]) / 5.0
        else:
            score = 0.5
        
        return {"score": score, "evaluation": response}
    
    def _extract_best_plan(self, node: 'PlanNode') -> List[str]:
        """Extract the best plan as a list of steps."""
        plan = []
        
        def traverse(n):
            if n.node_type in ["task", "step"]:
                plan.append(n.content)
            for child in n.children[:2]:
                traverse(child)
        
        traverse(node)
        return plan
    
    def _get_alternatives(self, node: 'PlanNode') -> List[List[str]]:
        """Get alternative plan branches."""
        alternatives = []
        
        if len(node.children) > 1:
            for child in node.children[1:]:
                alt = []
                for n in node.children:
                    if n != child:
                        alt.extend(self._flatten_subtree(n))
                alternatives.append([child.content] + self._flatten_subtree(child))
        
        return alternatives
    
    def _flatten_subtree(self, node: 'PlanNode') -> List[str]:
        """Flatten subtree to list of nodes."""
        result = []
        for child in node.children:
            if child.node_type in ["task", "step"]:
                result.append(child.content)
            result.extend(self._flatten_subtree(child))
        return result


@dataclass
class PlanNode:
    """Node in the plan tree."""
    content: str
    depth: int
    node_type: str  # "goal", "task", "step"
    parent: Optional['PlanNode'] = None
    children: List['PlanNode'] = field(default_factory=list)
    value: float = 0.0
```

### Pattern 5: ToT for Creative Problem Solving

```python
from typing import Dict, List, Tuple, Set
import random

class CreativeToT:
    """
    Tree-of-Thought for creative problem solving.
    Explores multiple creative directions and evaluates novelty.
    """
    
    def __init__(
        self,
        llm_client,
        novelty_weight: float = 0.4,
        quality_weight: float = 0.4,
        feasibility_weight: float = 0.2
    ):
        self.llm = llm_client
        self.weights = {
            "novelty": novelty_weight,
            "quality": quality_weight,
            "feasibility": feasibility_weight
        }
    
    def generate_diverse_solutions(
        self,
        problem: str,
        num_directions: int = 4
    ) -> Dict:
        """
        Generate diverse creative solutions using ToT.
        """
        directions = self._explore_directions(problem, num_directions)
        
        expanded_solutions = []
        
        for direction in directions:
            solutions = self._expand_direction(problem, direction)
            expanded_solutions.extend(solutions)
        
        evaluated = []
        for sol in expanded_solutions:
            eval_result = self._evaluate_solution(sol)
            evaluated.append((eval_result["total_score"], sol, eval_result))
        
        evaluated.sort(key=lambda x: x[0], reverse=True)
        
        return {
            "best_solution": evaluated[0][1] if evaluated else None,
            "top_solutions": evaluated[:3],
            "directions_explored": len(directions),
            "solutions_generated": len(expanded_solutions)
        }
    
    def _explore_directions(
        self,
        problem: str,
        num_directions: int
    ) -> List[Dict]:
        """Explore different creative directions."""
        prompt = f"""Problem: {problem}

Generate {num_directions} fundamentally different approaches or creative directions
to solve this problem. Think about different paradigms, perspectives, or
fundamentally different ways to frame the problem.

Format:
1. [Direction name]: [brief description of approach]
2. [Direction name]: [brief description of approach]
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=400)
        
        directions = []
        for line in response.split('\n'):
            match = re.match(r'^\d+[\.\)]\s*([^:]+):\s*(.+)', line.strip())
            if match:
                directions.append({
                    "name": match.group(1).strip(),
                    "description": match.group(2).strip()
                })
        
        return directions[:num_directions]
    
    def _expand_direction(
        self,
        problem: str,
        direction: Dict
    ) -> List[str]:
        """Expand a creative direction into concrete solutions."""
        prompt = f"""Problem: {problem}

Direction: {direction['name']}
Description: {direction['description']}

Develop this approach into a concrete solution or response.
Include specific details, examples, and reasoning.

Solution:
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=500)
        
        return [response.strip()]
    
    def _evaluate_solution(self, solution: str) -> Dict:
        """Evaluate solution on novelty, quality, and feasibility."""
        prompt = f"""Evaluate this solution:

{solution}

Rate 1-5 for each dimension:
1. Novelty/Originality: How creative and original is this? (5 = highly original)
2. Quality: How well-crafted and polished is it? (5 = excellent quality)
3. Feasibility: How practical and achievable is it? (5 = very practical)

Provide ratings and brief justification.
"""
        
        response = self.llm.generate(prompt=prompt, max_tokens=250, temperature=0.3)
        
        scores = re.findall(r'\b([1-5])\b', response)
        if len(scores) >= 3:
            novelty = int(scores[0]) / 5.0
            quality = int(scores[1]) / 5.0
            feasibility = int(scores[2]) / 5.0
        else:
            novelty = quality = feasibility = 0.5
        
        total = (
            self.weights["novelty"] * novelty +
            self.weights["quality"] * quality +
            self.weights["feasibility"] * feasibility
        )
        
        return {
            "novelty": novelty,
            "quality": quality,
            "feasibility": feasibility,
            "total_score": total
        }
```

## Framework Integration

### Integration with LangChain Agents

```python
from langchain.agents import Agent, Tool
from langchain.tools import BaseTool

class ToTTool(BaseTool):
    name = "tree_of_thought"
    description = "Use this to explore multiple solution paths"
    
    def _run(self, problem: str, strategy: str = "beam"):
        tot = TreeOfThought(
            llm_client=self.llm,
            thought_generator=self.generate_thoughts,
            state_evaluator=self.evaluate_state
        )
        result = tot.solve(problem, strategy=strategy)
        return result
```

### Integration with AutoGPT

```python
class ToTAutoGPT:
    def __init__(self, llm):
        self.tot = TreeOfThought(llm, ...)
    
    def execute_task(self, task: str):
        result = self.tot.solve(task, strategy="beam")
        return result["solution"]
```

## Performance Considerations

### Pruning Thresholds

Setting appropriate pruning thresholds is crucial:
- Too aggressive: May prune viable solutions
- Too conservative: Explores too many branches

### Beam Width Selection

- Larger beam: Better solutions, higher computation
- Smaller beam: Faster, may miss alternatives

### Depth Limits

- Too shallow: May not reach solution
- Too deep: Wasted computation on dead ends

## Common Pitfalls

### Pitfall 1: Not Checking for Dead Ends

**Problem**: Tree grows but all branches lead to dead ends.

**Solution**: Implement early stopping:
```python
if all(candidates_evaluate_to_zero()):
    break
```

### Pitfall 2: Infinite Loops from Circular Reasoning

**Problem**: Model loops between similar states.

**Solution**: Track visited states:
```python
visited_states = set()
if new_state in visited_states:
    continue
visited_states.add(new_state)
```

### Pitfall 3: Evaluation Bias

**Problem**: Self-evaluation consistently rates solutions too high.

**Solution**: Use multiple evaluation perspectives:
```python
evaluations = [
    self.evaluate_from_correctness(state),
    self.evaluate_from_practicality(state),
    self.evaluate_from_originality(state)
]
```

## Research References

1. **Yao et al. (2023)** - "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" - Original ToT paper.

2. **Browne et al. (2012)** - "A Survey of Monte Carlo Tree Search Methods" - MCTS comprehensive review.

3. **Kocsis & Szepesvári (2006)** - "Bandit based Monte Carlo Planning" - UCT algorithm.

4. **Silver et al. (2017)** - "Mastering the Game of Go" - AlphaGo using MCTS.

5. **Huang et al. (2023)** - "ToT for Code Generation" - ToT applied to programming.

6. **Hao et al. (2023)** - "ToT for Math Reasoning" - ToT for mathematical problem solving.

7. **Zhou et al. (2023)** - "Least-to-Most Prompting" - Hierarchical problem decomposition.

8. **Zheng et al. (2023)** - "Self-Discovery" - Self-structured ToT approaches.

9. **Wang et al. (2023)** - "Learn to Search" - ToT with learned evaluation functions.

10. **Long et al. (2023)** - "ToT for Planning" - Large-scale ToT planning.