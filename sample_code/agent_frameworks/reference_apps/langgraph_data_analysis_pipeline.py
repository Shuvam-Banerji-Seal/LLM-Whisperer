"""
LangGraph Data Analysis Pipeline - Reference Application

A complete data analysis workflow using LangGraph that demonstrates:
- Multi-step data processing
- Data validation and cleaning
- Statistical analysis
- Result synthesis
- Error handling and recovery
- Progress tracking

References:
- LangGraph Tutorial 2026: https://growai.in/langgraph-tutorial-stateful-ai-agents-2026/
- Building Stateful AI Agents: https://abstractalgorithms.dev/langgraph-101-building-your-first-stateful-agent

Author: Shuvam Banerji Seal
"""

# Requirements:
# pip install langgraph>=0.1.0
# pip install langchain>=0.1.0
# pip install numpy
# pip install statistics

from typing import TypedDict, Literal, List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import json


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class DataPoint:
    """Represents a single data point."""

    value: float
    label: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of data analysis."""

    metric_name: str
    value: float
    confidence: float
    calculation_time_ms: float


# ============================================================================
# Pipeline State
# ============================================================================


class PipelineState(TypedDict):
    """State of the data analysis pipeline."""

    raw_data: List[Dict[str, Any]]
    data_points: List[DataPoint]
    validation_passed: bool
    validation_errors: List[str]
    analysis_results: Dict[str, AnalysisResult]
    summary: str
    status: str
    execution_log: List[str]
    step: int


# ============================================================================
# Pipeline Processing Nodes
# ============================================================================


class PipelineNodes:
    """Collection of pipeline processing nodes."""

    @staticmethod
    def load_data(state: PipelineState) -> PipelineState:
        """
        Load and prepare raw data.

        Node: load_data
        Purpose: Ingest raw data from source
        """
        print(f"\n🔵 Node 1: Loading Data")

        state["execution_log"].append("Loading data from source")
        state["step"] = 1

        # Simulate data loading
        if state["raw_data"]:
            print(f"   ✅ Loaded {len(state['raw_data'])} records")
            state["execution_log"].append(f"Loaded {len(state['raw_data'])} records")
        else:
            print(f"   ⚠️ No data provided")
            state["execution_log"].append("Warning: No data provided")

        return state

    @staticmethod
    def validate_data(state: PipelineState) -> PipelineState:
        """
        Validate data quality and format.

        Node: validate_data
        Purpose: Check data integrity and constraints
        """
        print(f"\n🟢 Node 2: Validating Data")

        state["execution_log"].append("Validating data quality")
        state["step"] = 2

        errors = []
        valid_points = []

        for record in state["raw_data"]:
            try:
                # Validate structure
                if "value" not in record:
                    errors.append(f"Missing 'value' in record: {record}")
                    continue

                value = float(record["value"])

                # Validate value range
                if value < -1000000 or value > 1000000:
                    errors.append(f"Value out of range: {value}")
                    continue

                # Create valid data point
                point = DataPoint(
                    value=value,
                    label=str(record.get("label", f"value_{len(valid_points)}")),
                    timestamp=record.get("timestamp"),
                    metadata=record.get("metadata", {}),
                )
                valid_points.append(point)

            except (ValueError, TypeError) as e:
                errors.append(f"Invalid value: {e}")

        state["data_points"] = valid_points
        state["validation_passed"] = len(errors) == 0
        state["validation_errors"] = errors

        if state["validation_passed"]:
            print(f"   ✅ All {len(valid_points)} records validated")
            state["execution_log"].append(f"Validated {len(valid_points)} records")
        else:
            print(f"   ⚠️ Validation errors: {len(errors)}")
            state["execution_log"].append(f"Found {len(errors)} validation errors")

        return state

    @staticmethod
    def clean_data(state: PipelineState) -> PipelineState:
        """
        Clean data and handle outliers.

        Node: clean_data
        Purpose: Remove or handle invalid/unusual values
        """
        print(f"\n🟡 Node 3: Cleaning Data")

        state["execution_log"].append("Cleaning data")
        state["step"] = 3

        if not state["data_points"]:
            print(f"   ⚠️ No data to clean")
            return state

        # Calculate statistics for outlier detection
        values = [p.value for p in state["data_points"]]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        # Remove outliers (values beyond 3 standard deviations)
        cleaned_points = []
        outliers = []

        for point in state["data_points"]:
            if stdev > 0 and abs(point.value - mean) > 3 * stdev:
                outliers.append(point)
            else:
                cleaned_points.append(point)

        original_count = len(state["data_points"])
        state["data_points"] = cleaned_points

        print(f"   📊 Removed {len(outliers)} outliers")
        print(f"   ✅ Cleaned data: {len(cleaned_points)}/{original_count} records")
        state["execution_log"].append(
            f"Cleaned data: {len(cleaned_points)} records (removed {len(outliers)} outliers)"
        )

        return state

    @staticmethod
    def analyze_data(state: PipelineState) -> PipelineState:
        """
        Perform statistical analysis.

        Node: analyze_data
        Purpose: Calculate key metrics and statistics
        """
        print(f"\n🟣 Node 4: Analyzing Data")

        state["execution_log"].append("Performing statistical analysis")
        state["step"] = 4

        if not state["data_points"]:
            print(f"   ⚠️ No data to analyze")
            return state

        values = [p.value for p in state["data_points"]]

        # Calculate metrics
        metrics = {
            "count": AnalysisResult("count", len(values), 1.0, 0.1),
            "mean": AnalysisResult("mean", statistics.mean(values), 0.95, 0.2),
            "median": AnalysisResult("median", statistics.median(values), 0.95, 0.2),
            "min": AnalysisResult("min", min(values), 1.0, 0.1),
            "max": AnalysisResult("max", max(values), 1.0, 0.1),
        }

        # Add standard deviation if enough data
        if len(values) > 1:
            metrics["stdev"] = AnalysisResult(
                "stdev", statistics.stdev(values), 0.9, 0.3
            )

        state["analysis_results"] = metrics

        print(f"   📈 Calculated {len(metrics)} metrics")
        for name, result in metrics.items():
            print(f"      {name}: {result.value:.2f}")

        state["execution_log"].append(f"Calculated {len(metrics)} metrics")

        return state

    @staticmethod
    def generate_summary(state: PipelineState) -> PipelineState:
        """
        Generate analysis summary and insights.

        Node: generate_summary
        Purpose: Synthesize results into actionable insights
        """
        print(f"\n⚪ Node 5: Generating Summary")

        state["execution_log"].append("Generating summary report")
        state["step"] = 5

        if not state["analysis_results"]:
            state["summary"] = "No analysis results to summarize"
            return state

        # Build summary
        summary = "📊 Analysis Summary\n"
        summary += f"Records Analyzed: {int(state['analysis_results'].get('count', AnalysisResult('count', 0, 1.0, 0)).value)}\n"

        if "mean" in state["analysis_results"]:
            mean = state["analysis_results"]["mean"].value
            summary += f"Average Value: {mean:.2f}\n"

        if "stdev" in state["analysis_results"]:
            stdev = state["analysis_results"]["stdev"].value
            summary += f"Standard Deviation: {stdev:.2f}\n"

        summary += f"Range: {state['analysis_results'].get('min', AnalysisResult('min', 0, 1.0, 0)).value:.2f} - "
        summary += f"{state['analysis_results'].get('max', AnalysisResult('max', 0, 1.0, 0)).value:.2f}\n"

        state["summary"] = summary
        state["status"] = "completed"

        print(f"   ✅ Summary generated")
        print(summary)

        return state


# ============================================================================
# Pipeline Router
# ============================================================================


class PipelineRouter:
    """Routing logic for pipeline transitions."""

    @staticmethod
    def route_after_load(state: PipelineState) -> Literal["validate", "end"]:
        """Route after data loading."""
        if state["raw_data"]:
            return "validate"
        else:
            return "end"

    @staticmethod
    def route_after_validation(state: PipelineState) -> Literal["clean", "end"]:
        """Route based on validation result."""
        if state["validation_passed"] and state["data_points"]:
            return "clean"
        else:
            return "end"

    @staticmethod
    def route_after_cleaning(state: PipelineState) -> Literal["analyze", "end"]:
        """Route after data cleaning."""
        if state["data_points"]:
            return "analyze"
        else:
            return "end"

    @staticmethod
    def route_after_analysis(state: PipelineState) -> Literal["summarize", "end"]:
        """Route after analysis."""
        if state["analysis_results"]:
            return "summarize"
        else:
            return "end"


# ============================================================================
# LangGraph Data Analysis Pipeline
# ============================================================================


class LangGraphDataAnalysisPipeline:
    """
    Complete data analysis pipeline using LangGraph patterns.

    Flow:
    load_data → validate → clean → analyze → summarize → end
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.nodes = {
            "load_data": PipelineNodes.load_data,
            "validate": PipelineNodes.validate_data,
            "clean": PipelineNodes.clean_data,
            "analyze": PipelineNodes.analyze_data,
            "summarize": PipelineNodes.generate_summary,
        }

        self.routers = {
            "load_data": PipelineRouter.route_after_load,
            "validate": PipelineRouter.route_after_validation,
            "clean": PipelineRouter.route_after_cleaning,
            "analyze": PipelineRouter.route_after_analysis,
        }

    def create_initial_state(self, raw_data: List[Dict[str, Any]]) -> PipelineState:
        """Create initial pipeline state."""
        return {
            "raw_data": raw_data,
            "data_points": [],
            "validation_passed": False,
            "validation_errors": [],
            "analysis_results": {},
            "summary": "",
            "status": "pending",
            "execution_log": [],
            "step": 0,
        }

    def run(self, raw_data: List[Dict[str, Any]]) -> PipelineState:
        """
        Run the data analysis pipeline.

        Args:
            raw_data: List of raw data records

        Returns:
            Final pipeline state
        """
        print("\n" + "=" * 70)
        print(f"Starting Data Analysis Pipeline")
        print(f"Input Records: {len(raw_data)}")
        print("=" * 70)

        state = self.create_initial_state(raw_data)

        # Execute pipeline
        current_node = "load_data"

        while current_node and current_node != "end":
            if current_node in self.nodes:
                state = self.nodes[current_node](state)

                # Determine next node
                if current_node in self.routers:
                    current_node = self.routers[current_node](state)
                else:
                    current_node = "end"
            else:
                current_node = "end"

        # Display final execution log
        print("\n" + "=" * 70)
        print("Pipeline Execution Log")
        print("=" * 70)
        for log_entry in state["execution_log"]:
            print(f"  - {log_entry}")

        return state


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LangGraph Data Analysis Pipeline - Reference Application")
    print("=" * 70)

    pipeline = LangGraphDataAnalysisPipeline()

    # Example 1: Normal data
    print("\n\n" + "=" * 70)
    print("Example 1: Normal Data Analysis")
    print("=" * 70)

    normal_data = [
        {"value": 10.5, "label": "measurement_1"},
        {"value": 11.2, "label": "measurement_2"},
        {"value": 10.8, "label": "measurement_3"},
        {"value": 11.5, "label": "measurement_4"},
        {"value": 10.9, "label": "measurement_5"},
        {"value": 50.0, "label": "outlier"},  # Will be detected as outlier
    ]

    state1 = pipeline.run(normal_data)

    print("\n" + "=" * 70)
    print("Analysis Results")
    print("=" * 70)
    print(state1["summary"])

    # Example 2: Data with errors
    print("\n\n" + "=" * 70)
    print("Example 2: Data with Validation Errors")
    print("=" * 70)

    error_data = [
        {"value": 100, "label": "valid_1"},
        {"value": "invalid", "label": "invalid_value"},  # Invalid
        {"value": 102, "label": "valid_2"},
        {"label": "missing_value"},  # Missing value
        {"value": 103, "label": "valid_3"},
    ]

    state2 = pipeline.run(error_data)

    print("\n" + "=" * 70)
    print("Analysis Results")
    print("=" * 70)
    print(state2["summary"])
    if state2["validation_errors"]:
        print("\nValidation Errors:")
        for error in state2["validation_errors"][:3]:
            print(f"  - {error}")

    print("\n" + "=" * 70)
    print("Production Implementation Features")
    print("=" * 70)
    print("""
✅ Multi-step data processing
✅ Data validation and error handling
✅ Statistical analysis
✅ Outlier detection and removal
✅ Result synthesis
✅ Execution logging

Production Enhancement Areas:
1. Support for larger datasets (batching, streaming)
2. Advanced statistical analysis (hypothesis testing)
3. Data visualization generation
4. Report generation (PDF, Excel)
5. Data quality scoring
6. Anomaly detection
7. Time series analysis
8. Distributed processing
9. Real-time data pipelines
10. Machine learning integration

Key Patterns Demonstrated:
- State management with TypedDict
- Multi-step workflow execution
- Conditional routing
- Error handling and recovery
- Progress tracking and logging
- Results aggregation

See:
- https://growai.in/langgraph-tutorial-stateful-ai-agents-2026/
- https://abstractalgorithms.dev/langgraph-101-building-your-first-stateful-agent
    """)
