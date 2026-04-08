"""Agentic tuning core implementation."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from fine_tuning.base.core import BaseFinetuner, FinetuningMetrics
from .config import AgenticTuningConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Represents a single action in an agent trajectory."""

    tool_name: str
    tool_input: Dict[str, Any]
    reasoning: str
    observation: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "reasoning": self.reasoning,
            "observation": self.observation,
            "success": self.success,
        }


@dataclass
class AgentTrajectory:
    """Represents a sequence of agent actions (trajectory)."""

    goal: str
    actions: List[AgentAction] = field(default_factory=list)
    final_result: Optional[str] = None
    success: bool = False
    reward: float = 0.0

    def add_action(self, action: AgentAction) -> None:
        """Add action to trajectory.

        Args:
            action: Action to add
        """
        self.actions.append(action)

    def get_length(self) -> int:
        """Get trajectory length.

        Returns:
            Number of actions
        """
        return len(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "goal": self.goal,
            "actions": [a.to_dict() for a in self.actions],
            "final_result": self.final_result,
            "success": self.success,
            "reward": self.reward,
        }


class AgenticFinetuner(BaseFinetuner):
    """Fine-tuner for agentic systems.

    Specialized fine-tuning for models that need to perform tool use,
    reasoning, and sequential decision-making.
    """

    def __init__(self, config: AgenticTuningConfig):
        """Initialize agentic fine-tuner.

        Args:
            config: Agentic tuning configuration
        """
        if not isinstance(config, AgenticTuningConfig):
            raise TypeError(f"Expected AgenticTuningConfig, got {type(config)}")

        super().__init__(config)
        self.agentic_config = config
        self.trajectories: List[AgentTrajectory] = []
        self.tool_metrics: Dict[str, Dict[str, float]] = {}

    def setup_model(self) -> None:
        """Setup model for agentic fine-tuning."""
        logger.info(f"Loading model for agentic tuning: {self.config.model_name}")

        try:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float32,
                use_auth_token=self.config.use_auth_token,
                trust_remote_code=self.config.trust_remote_code,
            )
        except ImportError:
            logger.error("transformers library not installed")
            raise

        self.model.to(self.device)
        logger.info("Model setup complete for agentic tuning")

    def setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        optimizer_type = self.config.optimizer.type.value
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        logger.info("Optimizer setup complete")

    def setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")

        from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.scheduler.num_warmup_steps,
            num_training_steps=self.config.scheduler.num_training_steps,
        )

        logger.info("Scheduler setup complete")

    def add_trajectory(self, trajectory: AgentTrajectory) -> None:
        """Add a trajectory for training.

        Args:
            trajectory: Agent trajectory
        """
        self.trajectories.append(trajectory)

    def add_trajectories(self, trajectories: List[AgentTrajectory]) -> None:
        """Add multiple trajectories.

        Args:
            trajectories: List of trajectories
        """
        self.trajectories.extend(trajectories)
        logger.info(f"Added {len(trajectories)} trajectories")

    def compute_trajectory_loss(
        self,
        trajectory: AgentTrajectory,
        output_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a trajectory.

        Args:
            trajectory: Trajectory
            output_logits: Model output logits

        Returns:
            Dictionary with loss components
        """
        reasoning_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        reward_loss = torch.tensor(0.0, device=self.device)

        # Compute losses for each action in trajectory
        for i, action in enumerate(trajectory.actions):
            # Reasoning loss
            reasoning_loss += torch.tensor(
                0.1, device=self.device
            )  # Simplified: would use actual prediction

            # Action loss
            action_loss += torch.tensor(
                0.05, device=self.device
            )  # Simplified: would use actual prediction

        # Reward loss (higher reward = lower loss)
        reward_loss = -torch.tensor(trajectory.reward, device=self.device)

        return {
            "reasoning_loss": reasoning_loss,
            "action_loss": action_loss,
            "reward_loss": reward_loss,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train the agentic model.

        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader

        Returns:
            Training results
        """
        self.state.start_time = __import__("datetime").datetime.now()
        logger.info("Starting agentic fine-tuning")

        results = {
            "training_loss": [],
            "eval_loss": [],
            "reasoning_loss": [],
            "action_loss": [],
            "reward_loss": [],
        }

        self.model.train()

        for epoch in range(self.config.training.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0
            epoch_reasoning_loss = 0
            epoch_action_loss = 0
            epoch_reward_loss = 0

            for step, batch in enumerate(train_dataloader):
                self.state.step += 1

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                except Exception as e:
                    logger.error(f"Error in forward pass: {e}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

                if step % self.config.training.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    logger.info(f"Epoch {epoch}, Step {step}: Loss = {avg_loss:.4f}")

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            results["training_loss"].append(avg_epoch_loss)

            logger.info(f"Epoch {epoch} completed: Loss = {avg_epoch_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint()

        self.state.end_time = __import__("datetime").datetime.now()
        logger.info("Agentic training complete")

        return results

    def evaluate(self, eval_dataloader: DataLoader) -> FinetuningMetrics:
        """Evaluate the model.

        Args:
            eval_dataloader: Evaluation data loader

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"Error in evaluation: {e}")
                    continue

        avg_loss = total_loss / len(eval_dataloader)

        return FinetuningMetrics(
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    def compute_tool_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics per tool.

        Returns:
            Dictionary of tool metrics
        """
        tool_stats = {}

        for trajectory in self.trajectories:
            for action in trajectory.actions:
                tool = action.tool_name
                if tool not in tool_stats:
                    tool_stats[tool] = {
                        "count": 0,
                        "success_count": 0,
                        "avg_reward": 0.0,
                    }

                tool_stats[tool]["count"] += 1
                if action.success:
                    tool_stats[tool]["success_count"] += 1
                tool_stats[tool]["avg_reward"] += trajectory.reward / len(
                    trajectory.actions
                )

        # Normalize
        for tool in tool_stats:
            count = tool_stats[tool]["count"]
            tool_stats[tool]["success_rate"] = (
                tool_stats[tool]["success_count"] / count if count > 0 else 0
            )
            tool_stats[tool]["avg_reward"] /= count if count > 0 else 1

        self.tool_metrics = tool_stats
        return tool_stats

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary statistics of trajectories.

        Returns:
            Summary dictionary
        """
        if not self.trajectories:
            return {}

        lengths = [t.get_length() for t in self.trajectories]
        rewards = [t.reward for t in self.trajectories]
        successes = sum(1 for t in self.trajectories if t.success)

        return {
            "num_trajectories": len(self.trajectories),
            "avg_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "success_rate": successes / len(self.trajectories),
        }
