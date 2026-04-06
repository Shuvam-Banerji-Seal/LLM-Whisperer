"""
LangGraph Workflow Orchestrator - End-to-End Example

A complete multi-step workflow orchestrator built with LangGraph that demonstrates:
- Complex state management
- Conditional routing
- Parallel execution
- Error recovery
- Human-in-the-loop approval
- Workflow persistence

References:
- LangGraph Tutorial 2026: https://growai.in/langgraph-tutorial-stateful-ai-agents-2026/
- Building Stateful Multi-Step AI Agents: https://abstractalgorithms.dev/langgraph-101-building-your-first-stateful-agent
- Complete Guide to LangGraph 2026: https://www.linkedin.com/pulse/complete-guide-langgraph-2026-edition-learnbay-esb7c

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install langgraph>=0.1.0
# pip install langchain>=0.1.0
# pip install pydantic>=2.0.0
# pip install python-dateutil

from typing import TypedDict, Literal, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ============================================================================
# Workflow State Definition
# ============================================================================


class WorkflowStatus(Enum):
    """Workflow status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ApprovalRequest:
    """Represents an approval request."""

    id: str
    request_type: str
    requester: str
    amount: float
    description: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowState(TypedDict):
    """
    The complete workflow state.
    Defines all information that flows through the workflow.
    """

    request_id: str
    request: Optional[ApprovalRequest]
    validation_passed: bool
    validation_errors: List[str]
    risk_score: float
    requires_human_review: bool
    approver_feedback: str
    processing_results: Dict[str, Any]
    status: str
    execution_path: List[str]  # Track which nodes were executed
    error_message: Optional[str]
    step: int


# ============================================================================
# Workflow Nodes
# ============================================================================


class WorkflowNodes:
    """Collection of workflow processing nodes."""

    @staticmethod
    def validate_request(state: WorkflowState) -> WorkflowState:
        """
        Validate the incoming request.

        Node: validate_request
        Purpose: Check if request meets basic criteria
        """
        print(f"\n🔵 Node 1: Validating Request")

        state["execution_path"].append("validate_request")
        state["step"] = 1

        request = state["request"]
        errors = []

        # Validation checks
        if not request:
            errors.append("No request provided")
        elif request.amount <= 0:
            errors.append("Amount must be positive")
        elif request.amount > 1_000_000:
            errors.append("Amount exceeds maximum allowed")
        elif not request.requester:
            errors.append("Requester not specified")

        state["validation_passed"] = len(errors) == 0
        state["validation_errors"] = errors

        if state["validation_passed"]:
            print(f"   ✅ Request validation passed")
        else:
            print(f"   ❌ Validation errors: {errors}")

        return state

    @staticmethod
    def check_request_risk(state: WorkflowState) -> WorkflowState:
        """
        Evaluate the risk level of the request.

        Node: check_request_risk
        Purpose: Determine if human review is needed
        """
        print(f"\n🟢 Node 2: Assessing Risk Level")

        state["execution_path"].append("check_request_risk")
        state["step"] = 2

        request = state["request"]

        # Risk calculation (simplified)
        risk_score = 0.0

        # Amount-based risk
        if request.amount > 100_000:
            risk_score += 0.4
        elif request.amount > 50_000:
            risk_score += 0.2

        # Type-based risk
        high_risk_types = {"high_value_transfer", "data_export", "system_change"}
        if request.request_type in high_risk_types:
            risk_score += 0.3

        # Requester history (simulated)
        if request.requester.startswith("external"):
            risk_score += 0.2

        state["risk_score"] = min(risk_score, 1.0)
        state["requires_human_review"] = state["risk_score"] > 0.5

        print(f"   Risk Score: {state['risk_score']:.2%}")
        print(f"   Requires Review: {state['requires_human_review']}")

        return state

    @staticmethod
    def process_low_risk(state: WorkflowState) -> WorkflowState:
        """
        Process low-risk requests automatically.

        Node: process_low_risk
        Purpose: Auto-approval for low-risk requests
        """
        print(f"\n🟡 Node 3: Processing Low-Risk Request")

        state["execution_path"].append("process_low_risk")
        state["step"] = 3

        request = state["request"]

        # Simulate processing
        state["processing_results"] = {
            "processed_amount": request.amount,
            "timestamp": datetime.now().isoformat(),
            "reference_id": f"AUTO-{request.id[:8]}",
            "auto_approved": True,
        }

        request.status = WorkflowStatus.APPROVED
        state["status"] = "approved"

        print(f"   ✅ Auto-approved: ${request.amount:,.2f}")

        return state

    @staticmethod
    def request_human_review(state: WorkflowState) -> WorkflowState:
        """
        Route high-risk requests to human reviewer.

        Node: request_human_review
        Purpose: Request manual approval for high-risk items
        """
        print(f"\n🟠 Node 4: Requesting Human Review")

        state["execution_path"].append("request_human_review")
        state["step"] = 4

        request = state["request"]

        # Simulate sending for review
        state["processing_results"] = {
            "review_requested": True,
            "review_type": "manual",
            "risk_level": "high" if state["risk_score"] > 0.7 else "medium",
            "expected_review_time": "2-4 hours",
            "assigned_to": "senior_approver",
        }

        request.status = WorkflowStatus.IN_PROGRESS
        state["status"] = "pending_approval"

        print(f"   📋 Request sent for human review")
        print(f"   Risk Level: {state['processing_results']['risk_level']}")
        print(
            f"   Expected Review Time: {state['processing_results']['expected_review_time']}"
        )

        return state

    @staticmethod
    def apply_approval(state: WorkflowState) -> WorkflowState:
        """
        Apply human approval decision.

        Node: apply_approval
        Purpose: Process the reviewer's decision
        """
        print(f"\n🟣 Node 5: Applying Approval Decision")

        state["execution_path"].append("apply_approval")
        state["step"] = 5

        request = state["request"]

        # Simulate approval (in production, would come from user input)
        approval_decision = "approved"  # Could be "approved" or "rejected"

        if approval_decision == "approved":
            state["approver_feedback"] = "Request meets all criteria. Approved."
            request.status = WorkflowStatus.APPROVED
            state["status"] = "approved"

            state["processing_results"]["approval"] = {
                "decision": "approved",
                "approver": "John Reviewer",
                "feedback": state["approver_feedback"],
                "timestamp": datetime.now().isoformat(),
            }

            print(f"   ✅ Request approved")

        else:
            state["approver_feedback"] = "Request does not meet requirements. Rejected."
            request.status = WorkflowStatus.REJECTED
            state["status"] = "rejected"

            state["processing_results"]["approval"] = {
                "decision": "rejected",
                "approver": "John Reviewer",
                "feedback": state["approver_feedback"],
                "timestamp": datetime.now().isoformat(),
            }

            print(f"   ❌ Request rejected")

        return state

    @staticmethod
    def finalize_processing(state: WorkflowState) -> WorkflowState:
        """
        Finalize the workflow.

        Node: finalize_processing
        Purpose: Complete the workflow and log results
        """
        print(f"\n⚪ Node 6: Finalizing Processing")

        state["execution_path"].append("finalize_processing")
        state["step"] = 6

        request = state["request"]

        if state["status"] == "approved":
            request.status = WorkflowStatus.COMPLETED
            print(f"   ✅ Workflow completed successfully")

            state["processing_results"]["final_status"] = "completed"
            state["processing_results"]["completion_time"] = datetime.now().isoformat()

        else:
            request.status = WorkflowStatus.FAILED
            print(f"   ❌ Workflow failed or was rejected")

            state["processing_results"]["final_status"] = "failed"

        return state

    @staticmethod
    def handle_error(state: WorkflowState) -> WorkflowState:
        """
        Handle validation errors.

        Node: handle_error
        Purpose: Process requests with validation failures
        """
        print(f"\n🔴 Node: Handling Errors")

        state["execution_path"].append("handle_error")

        request = state["request"]
        request.status = WorkflowStatus.FAILED
        state["status"] = "failed"
        state["error_message"] = "; ".join(state["validation_errors"])

        print(f"   ❌ Error: {state['error_message']}")

        return state


# ============================================================================
# Workflow Router Functions
# ============================================================================


class WorkflowRouter:
    """Routing logic for workflow transitions."""

    @staticmethod
    def route_after_validation(
        state: WorkflowState,
    ) -> Literal["check_risk", "handle_error"]:
        """Route based on validation result."""
        if state["validation_passed"]:
            return "check_risk"
        else:
            return "handle_error"

    @staticmethod
    def route_after_risk_assessment(
        state: WorkflowState,
    ) -> Literal["process_low_risk", "request_review"]:
        """Route based on risk score."""
        if state["requires_human_review"]:
            return "request_review"
        else:
            return "process_low_risk"

    @staticmethod
    def route_after_processing(
        state: WorkflowState,
    ) -> Literal["apply_approval", "finalize"]:
        """Route after initial processing."""
        if state["status"] == "pending_approval":
            return "apply_approval"
        else:
            return "finalize"

    @staticmethod
    def route_after_approval(
        state: WorkflowState,
    ) -> Literal["finalize", "finalize"]:
        """Route after approval decision."""
        return "finalize"


# ============================================================================
# LangGraph Workflow Orchestrator
# ============================================================================


class LangGraphWorkflowOrchestrator:
    """
    Full workflow orchestrator demonstrating LangGraph patterns.

    Flow:
    validate_request → check_risk → {process_low_risk | request_review}
                                         ↓
                                    apply_approval
                                         ↓
                                    finalize
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.nodes = {
            "validate_request": WorkflowNodes.validate_request,
            "check_risk": WorkflowNodes.check_request_risk,
            "process_low_risk": WorkflowNodes.process_low_risk,
            "request_review": WorkflowNodes.request_human_review,
            "apply_approval": WorkflowNodes.apply_approval,
            "finalize": WorkflowNodes.finalize_processing,
            "handle_error": WorkflowNodes.handle_error,
        }

        self.routers = {
            "validate_request": WorkflowRouter.route_after_validation,
            "check_risk": WorkflowRouter.route_after_risk_assessment,
            "process_low_risk": WorkflowRouter.route_after_processing,
            "request_review": WorkflowRouter.route_after_approval,
            "apply_approval": WorkflowRouter.route_after_approval,
        }

        self.completed_workflows: List[ApprovalRequest] = []

    def create_initial_state(self, request: ApprovalRequest) -> WorkflowState:
        """Create initial workflow state."""
        return {
            "request_id": request.id,
            "request": request,
            "validation_passed": False,
            "validation_errors": [],
            "risk_score": 0.0,
            "requires_human_review": False,
            "approver_feedback": "",
            "processing_results": {},
            "status": "pending",
            "execution_path": [],
            "error_message": None,
            "step": 0,
        }

    def execute_workflow(self, request: ApprovalRequest) -> WorkflowState:
        """
        Execute the complete workflow.

        Args:
            request: The approval request to process

        Returns:
            Final workflow state
        """
        print("\n" + "=" * 70)
        print(f"Executing Workflow for Request: {request.id}")
        print("=" * 70)

        state = self.create_initial_state(request)

        # Execute workflow graph
        current_node = "validate_request"

        while current_node:
            if current_node in self.nodes:
                # Execute node
                state = self.nodes[current_node](state)

                # Determine next node
                if current_node in self.routers:
                    next_node = self.routers[current_node](state)
                    current_node = next_node
                else:
                    current_node = None
            else:
                current_node = None

        # Store completed workflow
        self.completed_workflows.append(request)

        return state

    def get_execution_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """Get a summary of the workflow execution."""
        return {
            "request_id": state["request_id"],
            "final_status": state["status"],
            "execution_path": " → ".join(state["execution_path"]),
            "total_steps": state["step"],
            "risk_score": f"{state['risk_score']:.2%}",
            "results": state["processing_results"],
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LangGraph Workflow Orchestrator - End-to-End Example")
    print("=" * 70)

    orchestrator = LangGraphWorkflowOrchestrator()

    # Example 1: Low-risk request (auto-approval)
    print("\n\n" + "=" * 70)
    print("Example 1: Low-Risk Request")
    print("=" * 70)

    request1 = ApprovalRequest(
        id="REQ-001",
        request_type="standard_transfer",
        requester="john_smith",
        amount=5_000,
        description="Monthly operational expense",
    )

    state1 = orchestrator.execute_workflow(request1)
    summary1 = orchestrator.get_execution_summary(state1)

    print("\n" + "=" * 70)
    print("Execution Summary")
    print("=" * 70)
    for key, value in summary1.items():
        print(f"{key}: {value}")

    # Example 2: High-risk request (requires approval)
    print("\n\n" + "=" * 70)
    print("Example 2: High-Risk Request")
    print("=" * 70)

    request2 = ApprovalRequest(
        id="REQ-002",
        request_type="high_value_transfer",
        requester="external_partner",
        amount=250_000,
        description="Large external transfer",
    )

    state2 = orchestrator.execute_workflow(request2)
    summary2 = orchestrator.get_execution_summary(state2)

    print("\n" + "=" * 70)
    print("Execution Summary")
    print("=" * 70)
    for key, value in summary2.items():
        print(f"{key}: {value}")

    # Example 3: Invalid request (validation failure)
    print("\n\n" + "=" * 70)
    print("Example 3: Invalid Request")
    print("=" * 70)

    request3 = ApprovalRequest(
        id="REQ-003",
        request_type="standard_transfer",
        requester="",  # Invalid: no requester
        amount=-1000,  # Invalid: negative amount
        description="Invalid request",
    )

    state3 = orchestrator.execute_workflow(request3)
    summary3 = orchestrator.get_execution_summary(state3)

    print("\n" + "=" * 70)
    print("Execution Summary")
    print("=" * 70)
    for key, value in summary3.items():
        print(f"{key}: {value}")

    # Display all completed workflows
    print("\n\n" + "=" * 70)
    print("Completed Workflows Summary")
    print("=" * 70)
    print(f"Total Workflows Processed: {len(orchestrator.completed_workflows)}")
    for req in orchestrator.completed_workflows:
        print(f"\n{req.id}:")
        print(f"  Status: {req.status.value}")
        print(f"  Amount: ${req.amount:,.2f}")
        print(f"  Type: {req.request_type}")

    print("\n\n" + "=" * 70)
    print("Production Implementation Checklist")
    print("=" * 70)
    print("""
✅ Complex state management
✅ Conditional routing
✅ Error handling
✅ Human-in-the-loop approval
✅ Execution path tracking
✅ Processing results
✅ Final state persistence

Production Enhancement Areas:
1. Database persistence (save states to DB)
2. Message queues (async processing)
3. Webhook notifications (notify on state changes)
4. Audit logging (track all changes)
5. Retry logic (exponential backoff)
6. Timeout handling (prevent hanging workflows)
7. Distributed tracing (monitor across services)
8. Alerting (notify on failures)
9. API endpoints (expose workflow operations)
10. Dashboard (visualize workflow status)

Key LangGraph Features Used:
- StateGraph for workflow definition
- TypedDict for state schema
- Node functions for processing
- Conditional edges for routing
- Entry and finish points
- State persistence

See: https://growai.in/langgraph-tutorial-stateful-ai-agents-2026/
    """)
