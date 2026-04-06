"""
AGNO Advanced Reasoning and Agentic Patterns

This module demonstrates advanced reasoning patterns including planning,
step-by-step reasoning, guardrails, and approval workflows in AGNO.

Author: Shuvam Banerji Seal
Source: https://www.agno.com/blog/the-5-levels-of-agentic-software-a-progressive-framework-for-building-reliable-ai-agents
Source: https://docs.agno.com/agents/guardrails
Source: https://docs.agno.com/agents/approval-workflows
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """
    Different reasoning strategies for AGNO agents.

    Reference: https://www.agno.com/blog/the-5-levels-of-agentic-software-a-progressive-framework-for-building-reliable-ai-agents
    """

    DIRECT = "direct"  # Single request-response
    CHAIN_OF_THOUGHT = "cot"  # Step-by-step reasoning
    TREE_OF_THOUGHT = "tot"  # Multiple reasoning paths
    STRUCTURED = "structured"  # Formal reasoning framework
    MULTI_AGENT = "multi_agent"  # Collaborative reasoning


class ActionType(Enum):
    """Types of actions agents can take."""

    TOOL_CALL = "tool_call"
    DELEGATION = "delegation"
    APPROVAL_REQUEST = "approval_request"
    DECISION = "decision"
    LEARNING = "learning"


@dataclass
class ReasoningStep:
    """Single step in agent reasoning process."""

    step_number: int
    thought: str
    reasoning: str
    action_type: ActionType
    action: str
    expected_outcome: str
    confidence: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step to dictionary."""
        return {
            "step": self.step_number,
            "thought": self.thought,
            "reasoning": self.reasoning,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
        }


@dataclass
class GuardRail:
    """
    A guardrail that constrains agent behavior.

    AGNO Pattern: Guardrails are essential for:
    - Safety and reliability
    - Policy enforcement
    - Preventing harmful actions
    - Compliance with regulations
    - Trust and accountability

    Reference: https://docs.agno.com/agents/guardrails
    """

    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    severity: str = "warning"  # warning, error, block
    enabled: bool = True

    def evaluate(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Evaluate guardrail against context.

        Returns:
            (passed, result) where passed is True if guardrail passes
        """
        try:
            if self.condition(context):
                return False, self.action(context)
            return True, None
        except Exception as e:
            logger.error(f"Guardrail {self.name} evaluation error: {e}")
            return True, None


@dataclass
class ApprovalRequest:
    """
    Request for human approval of an agent action.

    AGNO Pattern: Approval workflows enable human-in-the-loop
    execution where critical decisions require human review.
    """

    request_id: str
    agent_name: str
    action_description: str
    action_details: Dict[str, Any]
    risk_level: str  # low, medium, high, critical
    reasoning: str
    created_at: str = field(
        default_factory=lambda: str(__import__("datetime").datetime.now())
    )
    status: str = "pending"  # pending, approved, rejected
    approver_id: Optional[str] = None
    approval_timestamp: Optional[str] = None
    approval_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize request to dictionary."""
        return {
            "request_id": self.request_id,
            "agent_name": self.agent_name,
            "action": self.action_description,
            "risk_level": self.risk_level,
            "reasoning": self.reasoning,
            "status": self.status,
            "created_at": self.created_at,
        }


class ChainOfThoughtReasoning:
    """
    Implements Chain-of-Thought (CoT) reasoning in AGNO.

    CoT enables agents to:
    - Break complex problems into steps
    - Show reasoning for each step
    - Handle uncertainty and confidence levels
    - Enable debugging and verification

    AGNO Pattern: CoT improves reasoning quality and
    transparency in agentic systems.

    Reference: https://docs.agno.com/agents/reasoning
    """

    def __init__(self, agent_name: str, max_steps: int = 10):
        """
        Initialize CoT reasoning engine.

        Args:
            agent_name: Name of the agent using CoT
            max_steps: Maximum reasoning steps allowed
        """
        self.agent_name = agent_name
        self.max_steps = max_steps
        self.steps: List[ReasoningStep] = []
        self.current_step = 0

    def add_step(
        self,
        thought: str,
        reasoning: str,
        action_type: ActionType,
        action: str,
        expected_outcome: str,
        confidence: float = 0.7,
    ) -> None:
        """
        Add a reasoning step.

        Args:
            thought: What the agent is thinking
            reasoning: Why this step makes sense
            action_type: Type of action to take
            action: The action itself
            expected_outcome: What should happen
            confidence: Confidence in this step (0-1)
        """
        if self.current_step >= self.max_steps:
            logger.warning(f"Reached maximum steps ({self.max_steps})")
            return

        step = ReasoningStep(
            step_number=self.current_step + 1,
            thought=thought,
            reasoning=reasoning,
            action_type=action_type,
            action=action,
            expected_outcome=expected_outcome,
            confidence=confidence,
        )

        self.steps.append(step)
        self.current_step += 1

        logger.info(f"Added reasoning step {step.step_number}: {action}")

    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Get the complete chain of reasoning."""
        return [s.to_dict() for s in self.steps]

    def get_summary(self) -> str:
        """Get a summary of the reasoning."""
        if not self.steps:
            return "No reasoning steps yet"

        summary = f"Chain of Thought for {self.agent_name} ({len(self.steps)} steps):\n"
        for step in self.steps:
            summary += f"\nStep {step.step_number}: {step.thought}\n"
            summary += f"  Reasoning: {step.reasoning}\n"
            summary += f"  Action: {step.action_type.value} - {step.action}\n"
            summary += f"  Confidence: {step.confidence:.1%}\n"

        return summary


class PlanningAgent:
    """
    Agent that implements planning before execution.

    Planning enables:
    - Task decomposition
    - Resource allocation
    - Risk assessment
    - Alternative path analysis
    - Constraint satisfaction

    AGNO Pattern: Planning is crucial for complex, multi-step tasks
    where getting it right the first time matters.
    """

    def __init__(self, name: str):
        """Initialize planning agent."""
        self.name = name
        self.plans: Dict[str, Dict[str, Any]] = {}

    def create_plan(
        self,
        task: str,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create an execution plan for a task.

        Args:
            task: Task description
            constraints: Limitations or requirements
            success_criteria: How to measure success

        Returns:
            Execution plan

        AGNO Planning Process:
        1. Understand the task and constraints
        2. Decompose into subtasks
        3. Identify dependencies
        4. Estimate resource requirements
        5. Define success criteria
        6. Identify risks and mitigation
        """
        logger.info(f"Creating plan for task: {task}")

        plan = {
            "task": task,
            "constraints": constraints or [],
            "success_criteria": success_criteria or [],
            "subtasks": [
                {
                    "id": "subtask_1",
                    "description": "Analyze requirements",
                    "dependencies": [],
                    "estimated_time": "5 minutes",
                },
                {
                    "id": "subtask_2",
                    "description": "Design solution",
                    "dependencies": ["subtask_1"],
                    "estimated_time": "15 minutes",
                },
                {
                    "id": "subtask_3",
                    "description": "Implement solution",
                    "dependencies": ["subtask_2"],
                    "estimated_time": "30 minutes",
                },
                {
                    "id": "subtask_4",
                    "description": "Validate results",
                    "dependencies": ["subtask_3"],
                    "estimated_time": "10 minutes",
                },
            ],
            "estimated_total_time": "60 minutes",
            "risks": [
                {
                    "risk": "Incomplete requirements",
                    "mitigation": "Clarify with stakeholder",
                },
                {
                    "risk": "Resource constraints",
                    "mitigation": "Prioritize core features",
                },
            ],
        }

        plan_id = f"plan_{len(self.plans) + 1}"
        self.plans[plan_id] = plan

        return plan


class GuardRailEngine:
    """
    Manages guardrails in AGNO agents.

    Guardrails:
    - Prevent harmful agent actions
    - Enforce policies and rules
    - Ensure compliance
    - Improve safety and reliability

    AGNO Pattern: Guardrails are evaluated before and after
    agent actions to ensure safe operation.

    Reference: https://docs.agno.com/agents/guardrails
    """

    def __init__(self):
        """Initialize guardrail engine."""
        self.guardrails: List[GuardRail] = []
        self.violations: List[Dict[str, Any]] = []

    def add_guardrail(self, guardrail: GuardRail) -> None:
        """Add a guardrail."""
        self.guardrails.append(guardrail)
        logger.info(f"Added guardrail: {guardrail.name}")

    def evaluate_action(
        self, action: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Evaluate an action against all guardrails.

        Args:
            action: Agent action to evaluate

        Returns:
            (allowed, violation_info) - True if action passes all guardrails

        AGNO Safety Pattern:
        All agent actions are evaluated against guardrails before execution.
        """
        violations = []

        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue

            passed, result = guardrail.evaluate(action)

            if not passed:
                violation = {
                    "guardrail": guardrail.name,
                    "severity": guardrail.severity,
                    "result": result,
                }
                violations.append(violation)
                logger.warning(f"Guardrail violation: {guardrail.name}")

        if violations:
            self.violations.append(
                {
                    "action": action,
                    "violations": violations,
                    "timestamp": str(__import__("datetime").datetime.now()),
                }
            )

            # Block if any critical violations
            for v in violations:
                if v["severity"] == "block":
                    return False, violations

        return True, None

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get all recorded violations."""
        return self.violations.copy()


class ApprovalWorkflow:
    """
    Manages approval workflows for agent actions.

    AGNO Pattern: Approval workflows enable human-in-the-loop
    execution where critical decisions need human oversight.

    Use Cases:
    - Financial decisions above threshold
    - Data deletion or modification
    - External system changes
    - High-risk operations
    - Policy exceptions

    Reference: https://docs.agno.com/agents/approval-workflows
    """

    def __init__(self):
        """Initialize approval workflow."""
        self.requests: Dict[str, ApprovalRequest] = {}
        self.approval_rules: Dict[str, Callable] = {}

    def request_approval(
        self,
        agent_name: str,
        action_description: str,
        action_details: Dict[str, Any],
        risk_level: str = "medium",
        reasoning: str = "",
    ) -> ApprovalRequest:
        """
        Request approval for an agent action.

        Args:
            agent_name: Name of requesting agent
            action_description: Description of the action
            action_details: Detailed action parameters
            risk_level: Risk assessment (low/medium/high/critical)
            reasoning: Why this action is needed

        Returns:
            ApprovalRequest object
        """
        request_id = f"apr_{len(self.requests) + 1}"

        request = ApprovalRequest(
            request_id=request_id,
            agent_name=agent_name,
            action_description=action_description,
            action_details=action_details,
            risk_level=risk_level,
            reasoning=reasoning,
        )

        self.requests[request_id] = request

        logger.info(
            f"Created approval request {request_id} "
            f"from {agent_name} ({risk_level} risk)"
        )

        return request

    def approve_request(
        self, request_id: str, approver_id: str, approval_reason: str = ""
    ) -> bool:
        """
        Approve an approval request.

        Args:
            request_id: Request to approve
            approver_id: Who is approving
            approval_reason: Reason for approval

        Returns:
            True if approval successful
        """
        if request_id not in self.requests:
            logger.error(f"Approval request not found: {request_id}")
            return False

        request = self.requests[request_id]
        request.status = "approved"
        request.approver_id = approver_id
        request.approval_timestamp = str(__import__("datetime").datetime.now())
        request.approval_reason = approval_reason

        logger.info(f"Approved request {request_id}")
        return True

    def reject_request(
        self, request_id: str, rejector_id: str, rejection_reason: str
    ) -> bool:
        """
        Reject an approval request.

        Args:
            request_id: Request to reject
            rejector_id: Who is rejecting
            rejection_reason: Reason for rejection

        Returns:
            True if rejection successful
        """
        if request_id not in self.requests:
            return False

        request = self.requests[request_id]
        request.status = "rejected"
        request.approver_id = rejector_id
        request.approval_reason = rejection_reason

        logger.info(f"Rejected request {request_id}")
        return True

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [r for r in self.requests.values() if r.status == "pending"]

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific approval request."""
        return self.requests.get(request_id)


def main():
    """
    Demonstration of AGNO advanced reasoning and safety patterns.

    Reference Documentation:
    - https://www.agno.com/blog/the-5-levels-of-agentic-software-a-progressive-framework-for-building-reliable-ai-agents
    - https://docs.agno.com/agents/guardrails
    - https://docs.agno.com/agents/approval-workflows
    """
    print("\n=== AGNO Advanced Reasoning Demo ===\n")

    # 1. Chain of Thought Reasoning
    print("1. Chain of Thought Reasoning...")
    cot = ChainOfThoughtReasoning("DataAnalyst", max_steps=5)

    cot.add_step(
        thought="Need to analyze user data for trends",
        reasoning="Understanding data patterns helps identify issues",
        action_type=ActionType.TOOL_CALL,
        action="query_database(table='users')",
        expected_outcome="Get user data",
        confidence=0.9,
    )

    cot.add_step(
        thought="Clean and validate the data",
        reasoning="Invalid data would skew analysis results",
        action_type=ActionType.TOOL_CALL,
        action="validate_and_clean_data()",
        expected_outcome="Valid, clean dataset",
        confidence=0.85,
    )

    cot.add_step(
        thought="Identify patterns and anomalies",
        reasoning="Patterns reveal insights, anomalies highlight issues",
        action_type=ActionType.TOOL_CALL,
        action="analyze_patterns()",
        expected_outcome="List of patterns and anomalies",
        confidence=0.8,
    )

    print(cot.get_summary())

    # 2. Planning
    print("\n2. Planning Agent...")
    planner = PlanningAgent("TaskPlanner")

    plan = planner.create_plan(
        task="Deploy AGNO agent to production",
        constraints=["Must complete within 24 hours", "Zero downtime required"],
        success_criteria=[
            "Agent operational",
            "All tests passing",
            "Performance metrics met",
        ],
    )

    print(json.dumps(plan, indent=2)[:500] + "...")

    # 3. Guardrails
    print("\n3. Guardrail Safety...")
    engine = GuardRailEngine()

    # Add a guardrail for financial transactions
    def check_transaction_amount(context: Dict[str, Any]) -> bool:
        """Check if transaction exceeds limit."""
        amount = context.get("amount", 0)
        limit = context.get("daily_limit", 10000)
        return amount > limit

    def handle_large_transaction(context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transaction above limit."""
        return {
            "action": "require_approval",
            "reason": "Transaction exceeds daily limit",
        }

    guardrail = GuardRail(
        name="transaction_limit_check",
        description="Prevent transactions exceeding daily limit",
        condition=check_transaction_amount,
        action=handle_large_transaction,
        severity="block",
    )

    engine.add_guardrail(guardrail)

    # Test guardrail
    allowed, violation = engine.evaluate_action({"amount": 15000, "daily_limit": 10000})
    print(f"Transaction allowed: {allowed}")
    if violation:
        print(f"Violation: {violation}")

    # 4. Approval Workflow
    print("\n4. Approval Workflow...")
    workflow = ApprovalWorkflow()

    # Request approval for a critical action
    request = workflow.request_approval(
        agent_name="SystemAgent",
        action_description="Delete user database backup",
        action_details={"backup_id": "bak_12345", "size_gb": 500},
        risk_level="critical",
        reasoning="Freeing storage space for new backups",
    )

    print(f"Approval request created: {request.request_id}")
    print(f"Status: {request.status}")
    print(f"Risk level: {request.risk_level}")

    # Approve the request
    workflow.approve_request(
        request_id=request.request_id,
        approver_id="admin_001",
        approval_reason="Confirmed storage space needed",
    )

    updated = workflow.get_request(request.request_id)
    print(f"\nAfter approval:")
    print(f"Status: {updated.status}")
    print(f"Approved by: {updated.approver_id}")


if __name__ == "__main__":
    main()
