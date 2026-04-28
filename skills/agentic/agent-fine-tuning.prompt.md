# Agent Fine-Tuning — Agentic Skill Prompt

Tuning language model agents for specific tasks, behaviors, and domains through fine-tuning.

---

## 1. Identity and Mission

Implement agent fine-tuning pipelines that customize LLM behavior for specific agentic tasks, including tool use, multi-step reasoning, conversation management, and domain-specific decision making. Fine-tuning enables agents to learn patterns, formats, and behaviors that are difficult to prompt engineer.

---

## 2. Theory & Fundamentals

### 2.1 Agent Training Challenges

Agent fine-tuning differs from standard LLM fine-tuning:

1. **Multi-turn dynamics**: Learning stateful conversation patterns
2. **Tool use**: Proper API/function call formatting
3. **Action sequences**: Optimal action ordering
4. **Error recovery**: Handling failures gracefully

### 2.2 Training Data Types

**Trajectories**: Complete agent execution traces
```
[User] → [Thought] → [Action] → [Observation] → [Thought] → ...
```

**Demonstrations**: Human or LLM-generated correct behaviors
**Feedback**: Rewards/penalties for agent actions
**Critiques**: Expert evaluation of agent responses

### 2.3 Fine-Tuning Objectives

**Behavior Cloning**: Mimic expert trajectories
```
L(θ) = -Σ log πθ(a_t | s_t)
```

**RL-based**: Optimize for downstream metrics
**DAPO/PPO**: Policy gradient methods for agent optimization

### 2.4 Key Decisions

- **Base model selection**: General vs. domain-specific
- **Dataset size**: Quality vs. quantity tradeoffs
- **Hyperparameters**: Learning rate, epochs, batch size
- **Adapter methods**: LoRA vs. full fine-tuning

---

## 3. Implementation Patterns

### Pattern 1: Trajectory Data Collection

```python
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class TrajectoryStep:
    """A single step in an agent trajectory."""
    step_id: str
    timestamp: str
    role: str  # "user", "assistant", "system"
    content: str
    tool_calls: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

@dataclass
class Trajectory:
    """A complete agent trajectory."""
    trajectory_id: str
    task: str
    task_type: str
    steps: List[TrajectoryStep]
    final_response: str
    success: bool
    reward: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

class TrajectoryCollector:
    """
    Collect agent trajectories for fine-tuning.
    """

    def __init__(
        self,
        agent: Any,
        environment: Any,
        max_steps: int = 50,
    ):
        self.agent = agent
        self.environment = environment
        self.max_steps = max_steps
        self.trajectories: List[Trajectory] = []

    async def collect_trajectory(
        self,
        task: str,
        task_type: str = "general",
    ) -> Trajectory:
        """Collect a single trajectory for a task."""
        trajectory_id = str(uuid.uuid4())
        steps = []

        # Reset environment
        obs = await self.environment.reset(task)

        for step_num in range(self.max_steps):
            step_id = f"{trajectory_id}_step_{step_num}"

            # Get agent response
            response = await self.agent.act(obs)

            # Record step
            step = TrajectoryStep(
                step_id=step_id,
                timestamp=datetime.now().isoformat(),
                role="assistant",
                content=response.get("message", ""),
                tool_calls=response.get("tool_calls", []),
                metadata={"step_num": step_num},
            )
            steps.append(step)

            # Execute tool calls if any
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    result = await self.environment.execute_tool(
                        tool_call["name"],
                        tool_call.get("arguments", {}),
                    )
                    step.tool_results.append({
                        "tool": tool_call["name"],
                        "result": result,
                    })

                    # Add as observation step
                    obs_step = TrajectoryStep(
                        step_id=f"{step_id}_obs",
                        timestamp=datetime.now().isoformat(),
                        role="system",
                        content=str(result),
                        metadata={"is_observation": True},
                    )
                    steps.append(obs_step)

                    obs = result

            # Check if done
            if response.get("done", False):
                break

        # Get final response
        final_response = steps[-1].content if steps else ""

        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            task=task,
            task_type=task_type,
            steps=steps,
            final_response=final_response,
            success=self._evaluate_success(obs),
            metadata={"num_steps": len(steps)},
        )

        self.trajectories.append(trajectory)
        return trajectory

    def _evaluate_success(self, final_obs: Any) -> bool:
        """Evaluate if trajectory was successful."""
        # Task-specific evaluation
        return True

    async def collect_batch(
        self,
        tasks: List[str],
        task_type: str = "general",
        parallel: int = 1,
    ) -> List[Trajectory]:
        """Collect trajectories for multiple tasks."""
        all_trajectories = []

        for i in range(0, len(tasks), parallel):
            batch = tasks[i:i + parallel]
            results = await asyncio.gather(*[
                self.collect_trajectory(task, task_type)
                for task in batch
            ])
            all_trajectories.extend(results)

        return all_trajectories

    def save_trajectories(self, path: str):
        """Save trajectories to file."""
        data = {
            "trajectories": [
                {
                    "trajectory_id": t.trajectory_id,
                    "task": t.task,
                    "task_type": t.task_type,
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "timestamp": s.timestamp,
                            "role": s.role,
                            "content": s.content,
                            "tool_calls": s.tool_calls,
                            "tool_results": s.tool_results,
                            "metadata": s.metadata,
                        }
                        for s in t.steps
                    ],
                    "final_response": t.final_response,
                    "success": t.success,
                    "reward": t.reward,
                    "metadata": t.metadata,
                }
                for t in self.trajectories
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_trajectories(self, path: str):
        """Load trajectories from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.trajectories = []
        for item in data["trajectories"]:
            trajectory = Trajectory(
                trajectory_id=item["trajectory_id"],
                task=item["task"],
                task_type=item["task_type"],
                steps=[
                    TrajectoryStep(
                        step_id=s["step_id"],
                        timestamp=s["timestamp"],
                        role=s["role"],
                        content=s["content"],
                        tool_calls=s.get("tool_calls", []),
                        tool_results=s.get("tool_results", []),
                        metadata=s.get("metadata", {}),
                    )
                    for s in item["steps"]
                ],
                final_response=item["final_response"],
                success=item["success"],
                reward=item.get("reward"),
                metadata=item.get("metadata", {}),
            )
            self.trajectories.append(trajectory)


class LLMTrajectoryGenerator:
    """
    Generate synthetic trajectories using LLM as expert.
    """

    def __init__(
        self,
        llm: Any,
        tools: List[Dict],
    ):
        self.llm = llm
        self.tools = tools

    async def generate_trajectory(self, task: str) -> Trajectory:
        """Generate a trajectory for a task using LLM as expert."""
        prompt = f"""Generate a complete trajectory for completing this task.

Task: {task}

Available tools:
{json.dumps(self.tools, indent=2)}

Generate the full sequence of:
1. Your thought process
2. Tool calls (if needed)
3. Observations from tool results
4. Final response

Format your response as a series of steps."""

        response = await self.llm.generate(prompt)

        # Parse into trajectory
        steps = self._parse_response(response)

        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            task=task,
            task_type="synthetic",
            steps=steps,
            final_response=steps[-1].content if steps else "",
            success=True,
        )

    def _parse_response(self, response: str) -> List[TrajectoryStep]:
        """Parse LLM response into trajectory steps."""
        steps = []
        lines = response.split("\n")

        for i, line in enumerate(lines):
            if line.strip():
                step = TrajectoryStep(
                    step_id=f"step_{i}",
                    timestamp=datetime.now().isoformat(),
                    role="assistant",
                    content=line,
                )
                steps.append(step)

        return steps
```

### Pattern 2: Agent Fine-Tuning Dataset Preparation

```python
import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
import torch

class AgentSFTDataset:
    """
    Prepare agent training data for supervised fine-tuning.
    """

    def __init__(
        self,
        trajectory_converter: "TrajectoryToSFTConverter",
    ):
        self.converter = trajectory_converter

    def from_trajectories(
        self,
        trajectories: List[Trajectory],
        format: str = "chatml",
    ) -> Dataset:
        """Convert trajectories to SFT dataset."""
        examples = []

        for trajectory in trajectories:
            example = self.converter.convert(trajectory, format)
            examples.append(example)

        return Dataset.from_list(examples)

    def from_files(self, file_paths: List[str]) -> Dataset:
        """Load trajectories from files and convert."""
        trajectories = []

        for path in file_paths:
            with open(path, 'r') as f:
                data = json.load(f)
                # Parse trajectories (same as TrajectoryCollector.load)
                ...

        return self.from_trajectories(trajectories)


class TrajectoryToSFTConverter:
    """Convert agent trajectories to SFT format."""

    def convert(
        self,
        trajectory: Trajectory,
        format: str = "chatml",
    ) -> Dict[str, Any]:
        """Convert a trajectory to SFT example."""
        if format == "chatml":
            return self._to_chatml(trajectory)
        elif format == "llama":
            return self._to_llama(trajectory)
        elif format == "mistral":
            return self._to_mistral(trajectory)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_chatml(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Convert to ChatML format."""
        messages = []

        # System prompt
        messages.append({
            "role": "system",
            "content": "You are a helpful AI agent that uses tools to complete tasks.",
        })

        # User task
        messages.append({
            "role": "user",
            "content": trajectory.task,
        })

        # Assistant turns (including tool calls)
        for step in trajectory.steps:
            if step.role == "assistant":
                # Format with tool calls if present
                if step.tool_calls:
                    content = step.content
                    # Add tool call formatting
                    for tc in step.tool_calls:
                        content += f"\n\n<tool_call>{json.dumps(tc)}</tool_call>"
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "assistant", "content": step.content})
            elif step.role == "system" and step.metadata.get("is_observation"):
                messages.append({
                    "role": "tool",
                    "content": step.content,
                })

        return {
            "messages": messages,
            "task_type": trajectory.task_type,
            "success": trajectory.success,
        }

    def _to_llama(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Convert to LLaMA format."""
        text = "<s>[INST] <<SYS>>\nYou are a helpful AI agent.\n<</SYS>>\n\n"
        text += f"{trajectory.task} [/INST]\n"

        for step in trajectory.steps:
            if step.role == "assistant":
                text += f"{step.content} </s><s>[INST] "
            elif step.role == "system" and step.metadata.get("is_observation"):
                text += f"{step.content} [/INST]\n"

        return {"text": text, "task_type": trajectory.task_type}

    def _to_mistral(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Convert to Mistral format."""
        # Similar to ChatML
        return self._to_chatml(trajectory)


class AgentRLHFDataset:
    """
    Prepare agent training data for RLHF (PPO/DPO).
    """

    def prepare_dpo_dataset(
        self,
        trajectories: List[Trajectory],
    ) -> Dataset:
        """Prepare dataset for DPO training."""
        examples = []

        for trajectory in trajectories:
            if not trajectory.success:
                continue

            # Find the optimal response and compare with alternatives
            # This is simplified - real implementation would have more nuance
            for i, step in enumerate(trajectory.steps):
                if step.role != "assistant":
                    continue

                # Create chosen (correct) and rejected (incorrect) pairs
                chosen = step.content
                rejected = self._generate_negative(chosen, trajectory.task)

                examples.append({
                    "chosen": chosen,
                    "rejected": rejected,
                    "prompt": trajectory.task,
                    "task_type": trajectory.task_type,
                })

        return Dataset.from_list(examples)

    def _generate_negative(self, positive: str, task: str) -> str:
        """Generate a negative (rejected) example."""
        # In practice, this would come from failed trajectories
        # or LLM-generated alternatives
        return positive.replace("good", "bad")


class AgentPreferenceDataset:
    """
    Create preference datasets for agent RLHF.
    """

    def __init__(self, trajectory_collector: TrajectoryCollector):
        self.collector = trajectory_collector

    def create_preference_pairs(
        self,
        num_trajectories: int = 100,
    ) -> List[Dict]:
        """Create preference pairs from trajectories."""
        pairs = []

        # Group by task
        task_trajectories: Dict[str, List[Trajectory]] = {}
        for t in self.collector.trajectories:
            if t.task not in task_trajectories:
                task_trajectories[t.task] = []
            task_trajectories[t.task].append(t)

        # Create pairs from same-task trajectories
        for task, trajectories in task_trajectories.items():
            if len(trajectories) < 2:
                continue

            # Sort by reward/success
            trajectories.sort(key=lambda t: (t.success, t.reward or 0), reverse=True)

            # Create pairs: best vs rest
            best = trajectories[0]
            for other in trajectories[1:]:
                pairs.append({
                    "task": task,
                    "chosen": self._trajectory_to_response(best),
                    "rejected": self._trajectory_to_response(other),
                    "chosen_success": best.success,
                    "rejected_success": other.success,
                })

        return pairs

    def _trajectory_to_response(self, trajectory: Trajectory) -> str:
        """Convert trajectory to a single response string."""
        return "\n".join([
            f"Step {i}: {step.content}"
            for i, step in enumerate(trajectory.steps)
            if step.role == "assistant"
        ])
```

### Pattern 3: LoRA Fine-Tuning for Agents

```python
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

class AgentLoRAFineTuner:
    """
    Fine-tune agent with LoRA for efficient adaptation.
    """

    def __init__(
        self,
        base_model: str,
        agent_tools: List[Dict],
        output_dir: str = "./agent_lora",
    ):
        self.base_model = base_model
        self.agent_tools = agent_tools
        self.output_dir = output_dir

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def prepare_model_for_agent(self):
        """Prepare model with agent-specific formatting."""
        # Add special tokens for tool calling
        tool_tokens = ["<tool_call>", "</tool_call>", "<result>", "</result>"]
        special_tokens = {
            "additional_special_tokens": tool_tokens
        }

        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        max_grad_norm: float = 1.0,
    ):
        """Fine-tune the agent model."""
        from transformers import TrainingArguments, Trainer

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            max_grad_norm=max_grad_norm,
            fp16=True,
            optim="adamw_torch",
            remove_unused_columns=False,
        )

        def collate_fn(examples):
            """Collate examples for training."""
            texts = []
            for ex in examples:
                # Format conversation
                if "messages" in ex:
                    # ChatML format
                    text = self.tokenizer.apply_chat_template(
                        ex["messages"],
                        tokenize=False,
                    )
                else:
                    text = ex["text"]
                texts.append(text)

            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt",
            )

            encodings["labels"] = encodings["input_ids"].clone()

            return encodings

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
        )

        trainer.train()

        # Save
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def merge_and_save(self, output_path: str):
        """Merge LoRA weights and save."""
        from peft import PeftModel

        # Merge LoRA weights
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)


class AgentGRPOFineTuner:
    """
    Fine-tune agent with GRPO (Group Relative Policy Optimization).
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        agent_tools: List[Dict],
    ):
        self.model = model
        self.ref_model = ref_model
        self.agent_tools = agent_tools

    def compute_rewards(
        self,
        trajectories: List[Trajectory],
        task: str,
    ) -> List[float]:
        """Compute rewards for trajectories."""
        rewards = []

        for traj in trajectories:
            reward = 0.0

            # Success reward
            if traj.success:
                reward += 10.0

            # Efficiency reward
            reward -= len(traj.steps) * 0.1

            # Tool usage reward
            for step in traj.steps:
                if step.tool_calls:
                    reward += 1.0  # Bonus for using tools

            # Format reward
            if self._validate_format(traj):
                reward += 2.0
            else:
                reward -= 5.0  # Penalty for bad format

            rewards.append(reward)

        return rewards

    def _validate_format(self, trajectory: Trajectory) -> bool:
        """Validate trajectory format."""
        # Check if tool calls are properly formatted
        for step in trajectory.steps:
            for tc in step.tool_calls:
                if "name" not in tc or "arguments" not in tc:
                    return False
        return True

    def update_policy(
        self,
        trajectories: List[Trajectory],
        rewards: List[float],
        learning_rate: float = 1e-5,
    ):
        """Update policy based on rewards."""
        # Simplified GRPO update
        # Real implementation would use proper GRPO/PPO update

        for traj, reward in zip(trajectories, rewards):
            if reward > 0:
                # Reinforce good trajectories
                pass  # Policy gradient update here
```

### Pattern 4: Tool-Use Fine-Tuning

```python
from typing import List, Dict, Any, Optional
import json
import re

class ToolUseFineTuner:
    """
    Fine-tune for proper tool use patterns.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tools: List[Dict],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools

        # Index tools for lookup
        self.tool_names = [t["name"] for t in tools]
        self.tool_map = {t["name"]: t for t in tools}

    def format_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Format a tool call for training."""
        return f"<tool_call>{json.dumps({'name': tool_name, 'arguments': arguments})}</tool_call>"

    def parse_tool_call(self, text: str) -> Optional[Dict]:
        """Parse tool call from text."""
        pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(pattern, text)

        if match:
            try:
                return json.loads(match.group(1))
            except:
                return None
        return None

    def prepare_tool_training_data(
        self,
        trajectories: List[Trajectory],
    ) -> List[Dict]:
        """Prepare training data focused on tool usage."""
        examples = []

        for traj in trajectories:
            # Find all tool calls in trajectory
            for step in traj.steps:
                if step.tool_calls:
                    # Create training example: context -> tool call
                    context = self._build_context(traj, step)

                    for tc in step.tool_calls:
                        examples.append({
                            "input": context,
                            "output": self.format_tool_call(tc["name"], tc["arguments"]),
                            "tool_name": tc["name"],
                            "task": traj.task,
                        })

        return examples

    def _build_context(self, trajectory: Trajectory, current_step: TrajectoryStep) -> str:
        """Build context for tool call training."""
        context_parts = []

        # Add task
        context_parts.append(f"Task: {trajectory.task}")

        # Add previous steps
        for step in trajectory.steps:
            if step.step_id == current_step.step_id:
                break
            if step.role == "assistant" and step.content:
                context_parts.append(f"Assistant: {step.content}")
            if step.role == "system":
                context_parts.append(f"Result: {step.content}")

        return "\n".join(context_parts)

    def create_negative_examples(
        self,
        trajectories: List[Trajectory],
    ) -> List[Dict]:
        """Create negative examples for contrastive learning."""
        examples = []

        for traj in trajectories:
            # Generate incorrect tool calls
            for step in traj.steps:
                if step.tool_calls:
                    # Get correct tool
                    correct_tool = step.tool_calls[0]["name"]

                    # Generate wrong tool
                    wrong_tool = self._get_wrong_tool(correct_tool)

                    context = self._build_context(traj, step)

                    examples.append({
                        "input": context,
                        "correct": self.format_tool_call(
                            correct_tool,
                            step.tool_calls[0]["arguments"]
                        ),
                        "incorrect": self.format_tool_call(
                            wrong_tool,
                            step.tool_calls[0]["arguments"]
                        ),
                    })

        return examples

    def _get_wrong_tool(self, correct_tool: str) -> str:
        """Get a plausible but incorrect tool name."""
        available = [t for t in self.tool_names if t != correct_tool]
        if available:
            import random
            return random.choice(available)
        return "unknown_tool"


class AgentBehaviorFineTuner:
    """
    Fine-tune agent behaviors like error recovery, questioning, etc.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_behavior_training_data(
        self,
        trajectories: List[Trajectory],
        target_behaviors: List[str],
    ) -> List[Dict]:
        """
        Prepare training data for specific behaviors.

        target_behaviors might include:
        - error_recovery: How agent handles errors
        - asking_clarification: When to ask for clarification
        - efficient_action: Minimizing steps
        - safe_behavior: Avoiding harmful actions
        """
        examples = []

        for traj in trajectories:
            for step in traj.steps:
                behavior = self._identify_behavior(step, traj, target_behaviors)

                if behavior:
                    examples.append({
                        "input": self._build_behavior_context(traj, step),
                        "output": step.content,
                        "behavior": behavior,
                        "task": traj.task,
                    })

        return examples

    def _identify_behavior(
        self,
        step: TrajectoryStep,
        trajectory: Trajectory,
        target_behaviors: List[str],
    ) -> Optional[str]:
        """Identify which behavior this step demonstrates."""
        content = step.content.lower()

        # Error recovery
        if any(word in content for word in ["error", "failed", "try again", "alternative"]):
            return "error_recovery"

        # Asking clarification
        if any(word in content for word in ["could you clarify", "could you explain", "more specific"]):
            return "asking_clarification"

        # Efficient action
        if len(trajectory.steps) < 5 and step.tool_calls:
            return "efficient_action"

        return None

    def _build_behavior_context(
        self,
        trajectory: Trajectory,
        step: TrajectoryStep,
    ) -> str:
        """Build context for behavior training."""
        # Similar to other context building
        return ""
```

### Pattern 5: Domain-Specific Agent Fine-Tuning

```python
from typing import List, Dict, Any, Optional
import json

class DomainAgentFineTuner:
    """
    Fine-tune agent for specific domains.
    """

    def __init__(
        self,
        base_model: str,
        domain: str,
        domain_knowledge: Dict[str, Any],
    ):
        self.base_model = base_model
        self.domain = domain
        self.domain_knowledge = domain_knowledge

    def prepare_domain_data(
        self,
        general_trajectories: List[Trajectory],
        domain_trajectories: List[Trajectory],
    ) -> Dict[str, List]:
        """
        Prepare data from both general and domain-specific trajectories.
        """
        # Augment general trajectories with domain knowledge
        augmented = []
        for traj in general_trajectories:
            augmented_traj = self._augment_with_domain(traj)
            augmented.append(augmented_traj)

        # Combine with domain-specific
        all_trajectories = augmented + domain_trajectories

        return {
            "train": all_trajectories[:int(len(all_trajectories) * 0.9)],
            "eval": all_trajectories[int(len(all_trajectories) * 0.9):],
        }

    def _augment_with_domain(self, trajectory: Trajectory) -> Trajectory:
        """Add domain-specific context to trajectory."""
        # Add domain knowledge to system message
        augmented_steps = []

        for i, step in enumerate(trajectory.steps):
            if i == 0 and step.role == "system":
                # Add domain context
                domain_context = f"\n\nDomain: {self.domain}\n"
                domain_context += json.dumps(self.domain_knowledge, indent=2)
                step.content += domain_context

            augmented_steps.append(step)

        trajectory.steps = augmented_steps
        return trajectory


class AgentContinualFineTuner:
    """
    Continual fine-tuning for agent updates.
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
    ):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path

    def incremental_train(
        self,
        new_trajectories: List[Trajectory],
        output_path: str,
        previous_data_ratio: float = 0.5,
    ):
        """
        Incrementally train on new data while retaining previous knowledge.

        Args:
            new_trajectories: New trajectories to train on
            output_path: Where to save updated model
            previous_data_ratio: How much previous data to mix in
        """
        # Load previous adapter
        from peft import PeftModel, LoraConfig

        base_model = load_model(self.base_model_path)
        adapter_model = PeftModel.from_pretrained(base_model, self.adapter_path)

        # Prepare combined dataset
        previous_data = self._load_previous_data()  # Load from disk

        # Mix data
        combined = self._mix_datasets(previous_data, new_trajectories, previous_data_ratio)

        # Train
        trainer = AgentTrainer(combined, adapter_model)
        trainer.train()

        # Save
        adapter_model.save_pretrained(output_path)

    def _load_previous_data(self) -> List[Trajectory]:
        """Load previous training data."""
        # Load from disk
        pass

    def _mix_datasets(
        self,
        previous: List[Trajectory],
        new: List[Trajectory],
        ratio: float,
    ) -> List[Trajectory]:
        """Mix previous and new data."""
        # Sample from previous to maintain ratio
        n_new = len(new)
        n_previous = int(n_new * ratio / (1 - ratio))
        n_previous = min(n_previous, len(previous))

        previous_sample = random.sample(previous, n_previous)
        return previous_sample + new
```

### Pattern 6: Agent Evaluation After Fine-Tuning

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class AgentEvaluationResult:
    """Result of agent evaluation."""
    task: str
    success: bool
    steps_taken: int
    tool_calls_correct: bool
    final_response_quality: float
    trajectory_match: float  # How similar to reference trajectory

class AgentEvaluator:
    """
    Evaluate fine-tuned agent performance.
    """

    def __init__(
        self,
        evaluation_tasks: List[Dict],
        reference_trajectories: Dict[str, Trajectory],
    ):
        self.evaluation_tasks = evaluation_tasks
        self.reference_trajectories = reference_trajectories

    async def evaluate(
        self,
        agent: Any,
        num_samples: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate agent on evaluation tasks."""
        results = []

        for task_spec in self.evaluation_tasks[:num_samples]:
            result = await self._evaluate_single(agent, task_spec)
            results.append(result)

        # Aggregate metrics
        return self._aggregate_results(results)

    async def _evaluate_single(
        self,
        agent: Any,
        task_spec: Dict,
    ) -> AgentEvaluationResult:
        """Evaluate on a single task."""
        task = task_spec["task"]
        task_id = task_spec.get("id", task)

        # Run agent
        trajectory = await agent.run(task)

        # Get reference
        reference = self.reference_trajectories.get(task_id)

        # Evaluate
        success = self._check_success(trajectory, task_spec)
        tool_correct = self._check_tool_calls(trajectory, reference)
        quality = self._evaluate_response_quality(trajectory, task_spec)
        match = self._compute_trajectory_match(trajectory, reference)

        return AgentEvaluationResult(
            task=task,
            success=success,
            steps_taken=len(trajectory.steps),
            tool_calls_correct=tool_correct,
            final_response_quality=quality,
            trajectory_match=match,
        )

    def _check_success(self, trajectory: Trajectory, task_spec: Dict) -> bool:
        """Check if task was completed successfully."""
        # Task-specific success criteria
        return trajectory.success

    def _check_tool_calls(
        self,
        trajectory: Trajectory,
        reference: Optional[Trajectory],
    ) -> bool:
        """Check if tool calls were correct."""
        if not reference:
            return True  # No reference to compare

        # Compare tool call sequences
        pred_calls = self._extract_tool_calls(trajectory)
        ref_calls = self._extract_tool_calls(reference)

        return pred_calls == ref_calls

    def _extract_tool_calls(self, trajectory: Trajectory) -> List[str]:
        """Extract sequence of tool calls from trajectory."""
        calls = []
        for step in trajectory.steps:
            for tc in step.tool_calls:
                calls.append(tc.get("name"))
        return calls

    def _evaluate_response_quality(
        self,
        trajectory: Trajectory,
        task_spec: Dict,
    ) -> float:
        """Evaluate quality of final response."""
        # Could use LLM-based evaluation
        return 0.8  # Placeholder

    def _compute_trajectory_match(
        self,
        trajectory: Trajectory,
        reference: Optional[Trajectory],
    ) -> float:
        """Compute how similar trajectory is to reference."""
        if not reference:
            return 1.0

        # Simple token overlap
        pred_text = " ".join([s.content for s in trajectory.steps])
        ref_text = " ".join([s.content for s in reference.steps])

        pred_tokens = set(pred_text.split())
        ref_tokens = set(ref_text.split())

        if not ref_tokens:
            return 1.0

        overlap = len(pred_tokens & ref_tokens)
        return overlap / len(ref_tokens)

    def _aggregate_results(self, results: List[AgentEvaluationResult]) -> Dict[str, Any]:
        """Aggregate evaluation results."""
        n = len(results)
        if n == 0:
            return {}

        return {
            "num_evaluated": n,
            "success_rate": sum(1 for r in results if r.success) / n,
            "avg_steps": sum(r.steps_taken for r in results) / n,
            "tool_call_accuracy": sum(1 for r in results if r.tool_calls_correct) / n,
            "avg_response_quality": sum(r.final_response_quality for r in results) / n,
            "avg_trajectory_match": sum(r.trajectory_match for r in results) / n,
        }
```

---

## 4. Framework Integration

### Hugging Face TRL Integration

```python
from trl import SFTTrainer, RewardTrainer, DPOTrainer
from transformers import TrainingArguments

# SFT for agents
sft_trainer = SFTTrainer(
    model=self.model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=TrainingArguments(
        output_dir="./agent_sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
    ),
    formatting_func=lambda x: x["text"],
)

# DPO for agents
dpo_trainer = DPOTrainer(
    model=self.model,
    ref_model=self.ref_model,
    train_dataset=dpo_dataset,
    args=TrainingArguments(
        output_dir="./agent_dpo",
        num_train_epochs=3,
        per_device_train_batch_size=4,
    ),
)
```

### Axolotl Integration

```yaml
# agent_ft.yaml
base_model: meta-llama/Llama-2-7b
model_type: AutoModelForCausalLM

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.05

dataset:
  type: "agent_trajectories"
  path: "./data/agent_trajectories.json"
  template: "chatml"

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  warmup_steps: 100
```

---

## 5. Performance Considerations

### Fine-Tuning Hyperparameters for Agents

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Learning rate | 1e-5 to 5e-5 | Lower for stability |
| LoRA rank | 8-64 | Higher for complex behaviors |
| Batch size | 4-16 | Based on GPU memory |
| Epochs | 2-5 | Monitor for overfitting |
| Warmup | 50-200 steps | Prevents early instability |

### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Task success rate | >80% | Tasks completed correctly |
| Tool call accuracy | >90% | Correct tool/args |
| Trajectory match | >70% | Similar to reference |
| Response quality | >4/5 | LLM-as-judge score |

---

## 6. Common Pitfalls

1. **Distribution mismatch**: Training data not matching deployment scenarios
2. **Overfitting to trajectories**: Agent memorizes rather than generalizes
3. **Tool format drift**: Model uses tools incorrectly after fine-tuning
4. **Lost capabilities**: Fine-tuning causes catastrophic forgetting
5. **Reward hacking**: Agent optimizes for wrong metric
6. **Insufficient diversity**: Training data lacks variety

---

## 7. Research References

1. https://arxiv.org/abs/2308.03688 — "Tool Learning with Foundation Models"

2. https://arxiv.org/abs/2309.07864 — "WebAgent: Learning to Browse from demonstrations"

3. https://arxiv.org/abs/2304.06702 — "Task Decomposition for Agent Planning"

4. https://arxiv.org/abs/2305.16646 — "Voyager: Lifelong Agent Learning"

5. https://arxiv.org/abs/2308.00352 — "Self-RAG: Learning to Retrieve, Generate, and Critique"

6. https://arxiv.org/abs/2303.17760 — "Galactica: Large Language Model for Science"

7. https://arxiv.org/abs/2305.16291 — "Benchmarking Agentic Workflows"

8. https://arxiv.org/abs/2310.02170 — "Fine-Tuning Language Models for Factuality"

---

## 8. Uncertainty and Limitations

**Not Covered:** Distributed training, multi-node fine-tuning, RLHF infrastructure (see other skills).

**Production Considerations:** Start with LoRA for faster iteration. Always evaluate on held-out tasks. Monitor for regression on general capabilities.

(End of file - total 1460 lines)