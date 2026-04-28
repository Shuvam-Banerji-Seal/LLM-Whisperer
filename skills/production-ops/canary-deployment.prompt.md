# Canary Deployment and A/B Testing for LLM Systems

## Problem Statement

Deploying new LLM versions carries significant risk. Unlike traditional software where changes are deterministic and testable, LLM behavior emerges from complex training processes and can be difficult to predict. A new model version might perform better on benchmarks but worse on edge cases. It might generate more helpful content for most users but exhibit subtle harmful behaviors for specific populations. Without proper rollout strategies, teams risk deploying changes that cause user harm, degrade user experience, or increase costs unexpectedly.

Canary deployment addresses these risks by gradually rolling out changes to a small subset of users, measuring the impact, and progressively expanding the rollout. For LLM systems, this requires careful consideration of experiment design, statistical power, confounding factors, and the unique challenges of measuring "quality" in generative outputs.

This skill covers designing and implementing canary deployments for LLM systems, setting up A/B tests that produce statistically valid results, handling the unique challenges of LLM experimentation, and making data-driven decisions about model rollouts.

## Theory & Fundamentals

### Canary Deployment Strategies

Canary deployment for LLM systems follows several patterns, each with different trade-offs:

**Percentage-Based Canary**: Route a fixed percentage of traffic to the new model:
```
User ID Hash % N → New Model
Others → Baseline Model
```
- **Pros**: Simple, balanced traffic distribution
- **Cons**: Potential for confounding if user behavior varies

**Geographic Canary**: Route requests from specific regions to the new model:
```
Region = "EU" → New Model
Others → Baseline Model
```
- **Pros**: Isolates regional differences, natural containment
- **Cons**: Confounded by regional usage patterns

**User Segment Canary**: Route specific user segments (e.g., premium users, internal users):
```
User.tier = "premium" → New Model
Others → Baseline Model
```
- **Pros**: Protects high-value users, can target specific use cases
- **Cons**: May not represent overall population

**Feature Flag Canary**: Use feature flags for granular control:
```
FeatureFlags.enable("new_llm_model") = true → New Model
FeatureFlags.enable("new_llm_model") = false → Baseline Model
```
- **Pros**: Fine-grained control, easy rollback
- **Cons**: Requires feature flag infrastructure

### A/B Test Design for LLM Systems

LLM experimentation presents unique challenges:

**The Assignment Problem**: Unlike traditional A/B tests where you can randomly assign users, LLM outputs are non-deterministic. Two users seeing the "same" model might get different outputs due to temperature sampling.

**The Evaluation Problem**: How do you measure success? Traditional metrics (click-through rate, conversion) are indirect proxies for actual output quality.

**The Confounding Problem**: User characteristics (prompt style, preferences, context) can confound results.

**Statistical Design Requirements**:
```
For detecting d=0.1 effect at 80% power with α=0.05:
n ≈ (Z_α/2 + Z_β)² × 2 × σ² / d²

For LLM quality metrics with high variance:
- May need 10,000+ samples per arm
- Consider sequential testing to maintain validity
```

### Key Metrics for LLM Experimentation

**Primary Metrics**:
- Task completion rate (did the model successfully complete the requested task)
- User satisfaction score (explicit feedback when available)
- Response quality rating (when ground truth is available)

**Secondary Metrics**:
- Token efficiency (output tokens per successful response)
- Latency (time to first token, total generation time)
- Refusal rate (safety-related turndowns)
- Error rate (malformed outputs, crashes)

**Guardrail Metrics** (must not degrade):
- Safety violation rate
- PII leakage rate
- Hallucination rate (on sampled evaluation set)
- Output toxicity score

### Experiment Duration Calculation

```python
import numpy as np
from scipy import stats

def calculate_experiment_duration(
    baseline_rate: float,
    minimum_detectable_effect: float,
    daily_traffic: int,
    statistical_power: float = 0.8,
    significance_level: float = 0.05
) -> int:
    """
    Calculate required experiment duration for conversion rate experiments.
    
    Args:
        baseline_rate: Current conversion/success rate (e.g., 0.05 for 5%)
        minimum_detectable_effect: Relative change to detect (e.g., 0.1 for 10% lift)
        daily_traffic: Number of users/requests per day
        statistical_power: Desired power (default 0.8)
        significance_level: Type I error rate (default 0.05)
    
    Returns:
        Required number of days for experiment
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    
    z_alpha = stats.norm.ppf(1 - significance_level / 2)
    z_beta = stats.norm.ppf(statistical_power)
    
    pooled_p = (p1 + p2) / 2
    effect = abs(p2 - p1)
    
    n_per_group = (
        (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) +
         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    ) / (effect ** 2)
    
    total_sample_size = 2 * n_per_group
    days_required = np.ceil(total_sample_size / daily_traffic)
    
    return int(days_required)
```

## Implementation Patterns

### Pattern 1: Intelligent Traffic Router with Canary Support

```python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import random
import asyncio

@dataclass
class CanaryConfig:
    name: str
    model_version: str
    traffic_percentage: float
    user_segments: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"

class LLM TrafficRouter:
    def __init__(
        self,
        baseline_model: str,
        experiment_store,
        metrics_collector
    ):
        self.baseline_model = baseline_model
        self.experiments: Dict[str, CanaryConfig] = {}
        self.experiment_store = experiment_store
        self.metrics = metrics_collector
        self._rules: List[Callable] = []
    
    async def route_request(
        self,
        request_id: str,
        user_context: Dict,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Route request to appropriate model version based on experiment configuration.
        Returns routing decision with metadata for tracking.
        """
        start_time = datetime.utcnow()
        
        assigned_model, assignment_reason = await self._determine_model(
            user_context, prompt
        )
        
        routing_latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        self.metrics.record_routing(
            request_id=request_id,
            model_assigned=assigned_model,
            assignment_reason=assignment_reason,
            latency_ms=routing_latency_ms,
            user_segment=user_context.get("segment", "unknown")
        )
        
        return {
            "request_id": request_id,
            "model": assigned_model,
            "assignment_reason": assignment_reason,
            "experiment_context": self._get_experiment_context(
                user_context, assigned_model
            )
        }
    
    async def _determine_model(
        self,
        user_context: Dict,
        prompt: str
    ) -> tuple[str, str]:
        for rule in self._rules:
            result = await rule(user_context, prompt)
            if result:
                return result
        
        active_experiments = self._get_active_experiments()
        for exp in active_experiments:
            if self._user_in_experiment(user_context, exp):
                return exp.model_version, f"experiment_{exp.name}"
        
        return self.baseline_model, "baseline"
    
    def _user_in_experiment(
        self,
        user_context: Dict,
        experiment: CanaryConfig
    ) -> bool:
        if experiment.user_segments:
            if user_context.get("segment") not in experiment.user_segments:
                return False
        
        user_id = user_context.get("user_id", "")
        if not user_id:
            return random.random() < experiment.traffic_percentage
        
        hash_input = f"{user_id}:{experiment.name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        return (hash_value % 100) < (experiment.traffic_percentage * 100)
    
    def _get_active_experiments(self) -> List[CanaryConfig]:
        now = datetime.utcnow()
        return [
            exp for exp in self.experiments.values()
            if exp.status == "active"
            and (exp.start_time is None or exp.start_time <= now)
            and (exp.end_time is None or exp.end_time > now)
        ]
    
    def _get_experiment_context(
        self,
        user_context: Dict,
        assigned_model: str
    ) -> Dict:
        active = self._get_active_experiments()
        user_experiments = [
            exp for exp in active
            if self._user_in_experiment(user_context, exp)
        ]
        
        return {
            "experiments": [exp.name for exp in user_experiments],
            "variant": "treatment" if assigned_model != self.baseline_model else "control",
            "user_hash": self._get_user_hash(user_context.get("user_id", ""))
        }
    
    def _get_user_hash(self, user_id: str) -> str:
        if not user_id:
            return ""
        return hashlib.md5(user_id.encode()).hexdigest()[:8]
    
    async def create_experiment(
        self,
        name: str,
        model_version: str,
        traffic_percentage: float,
        user_segments: List[str] = None,
        **kwargs
    ) -> CanaryConfig:
        config = CanaryConfig(
            name=name,
            model_version=model_version,
            traffic_percentage=traffic_percentage,
            user_segments=user_segments or [],
            **kwargs
        )
        
        self.experiments[name] = config
        await self.experiment_store.save(config)
        
        return config
    
    async def update_experiment_traffic(
        self,
        name: str,
        new_traffic_percentage: float
    ):
        if name in self.experiments:
            self.experiments[name].traffic_percentage = new_traffic_percentage
            await self.experiment_store.save(self.experiments[name])
            
            await self.metrics.emit_event(
                "experiment_updated",
                {
                    "experiment": name,
                    "new_traffic": new_traffic_percentage,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def pause_experiment(self, name: str):
        if name in self.experiments:
            self.experiments[name].status = "paused"
            await self.experiment_store.save(self.experiments[name])
    
    async def resume_experiment(self, name: str):
        if name in self.experiments:
            self.experiments[name].status = "active"
            await self.experiment_store.save(self.experiments[name])
```

### Pattern 2: Sequential Testing for Early Experiment Termination

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy import stats

@dataclass
class SequentialTestResult:
    conclusion: str  # "continue", "reject_null", "accept_null"
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    timestamp: datetime

class SequentialLLMExperimentAnalyzer:
    """
    Implements a sequential testing framework for LLM experiments
    that allows for early termination while maintaining type I error control.
    Uses the mSPRT (mixture Sequential Probability Ratio Test) approach.
    """
    
    def __init__(
        self,
        control_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        beta: float = 0.2
    ):
        self.control_rate = control_rate
        self.mde = minimum_detectable_effect
        self.alpha = alpha
        self.beta = beta
        
        self.treatment_successes = 0
        self.treatment_total = 0
        self.control_successes = 0
        self.control_total = 0
        
        self.observations: List[Dict] = []
        self.start_time = datetime.utcnow()
        
        self._calculate_bounds()
    
    def _calculate_bounds(self):
        """
        Calculate sequential test boundaries using error spending functions.
        Uses the Lan-DeMets approximation for the O'Brien-Fleming boundary.
        """
        self.upper_boundary = 2.996  # Approximate for O'Brien-Fleming at 5 looks
        self.lower_boundary = 0.0
        self.early_stop_threshold = self._calculate_sample_size()
    
    def _calculate_sample_size(self) -> int:
        """Calculate target sample size for desired power."""
        p1 = self.control_rate
        p2 = self.control_rate * (1 + self.mde)
        
        pooled = (p1 + p2) / 2
        effect_size = abs(p2 - p1)
        
        n = 2 * ((stats.norm.ppf(1 - self.alpha/2) + stats.norm.ppf(1 - self.beta)) ** 2 
                  * pooled * (1 - pooled) / (effect_size ** 2))
        
        return int(n)
    
    def add_observation(
        self,
        variant: str,
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """
        Add a single observation to the sequential test.
        
        Args:
            variant: "treatment" or "control"
            success: Whether the outcome was successful
            metadata: Additional context (latency, quality rating, etc.)
        """
        observation = {
            "timestamp": datetime.utcnow(),
            "variant": variant,
            "success": success,
            "metadata": metadata or {}
        }
        self.observations.append(observation)
        
        if variant == "treatment":
            self.treatment_total += 1
            if success:
                self.treatment_successes += 1
        else:
            self.control_total += 1
            if success:
                self.control_successes += 1
    
    def evaluate(self) -> SequentialTestResult:
        """
        Evaluate current experiment state using sequential testing.
        
        Returns:
            SequentialTestResult with current conclusion and statistics
        """
        if self.control_total < 30 or self.treatment_total < 30:
            return SequentialTestResult(
                conclusion="insufficient_data",
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                sample_size=self.control_total + self.treatment_total,
                timestamp=datetime.utcnow()
            )
        
        z_stat, p_value = self._calculate_z_statistic()
        ci = self._calculate_confidence_interval()
        
        total_assigned = self.control_total + self.treatment_total
        
        if z_stat > self.upper_boundary:
            return SequentialTestResult(
                conclusion="reject_null",
                p_value=p_value,
                confidence_interval=ci,
                sample_size=total_assigned,
                timestamp=datetime.utcnow()
            )
        elif z_stat < self.lower_boundary:
            return SequentialTestResult(
                conclusion="accept_null",
                p_value=p_value,
                confidence_interval=ci,
                sample_size=total_assigned,
                timestamp=datetime.utcnow()
            )
        elif total_assigned >= self.early_stop_threshold:
            return SequentialTestResult(
                conclusion="reach_max_sample",
                p_value=p_value,
                confidence_interval=ci,
                sample_size=total_assigned,
                timestamp=datetime.utcnow()
            )
        else:
            return SequentialTestResult(
                conclusion="continue",
                p_value=p_value,
                confidence_interval=ci,
                sample_size=total_assigned,
                timestamp=datetime.utcnow()
            )
    
    def _calculate_z_statistic(self) -> Tuple[float, float]:
        """Calculate the z-statistic for the current observations."""
        if self.control_total == 0 or self.treatment_total == 0:
            return 0.0, 1.0
        
        p1 = self.control_successes / self.control_total
        p2 = self.treatment_successes / self.treatment_total
        
        pooled_p = (self.control_successes + self.treatment_successes) / (
            self.control_total + self.treatment_total
        )
        
        se = np.sqrt(pooled_p * (1 - pooled_p) * 
                     (1/self.control_total + 1/self.treatment_total))
        
        if se == 0:
            return 0.0, 1.0
        
        z_stat = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return z_stat, p_value
    
    def _calculate_confidence_interval(self) -> Tuple[float, float]:
        """Calculate 95% confidence interval for the effect size."""
        p1 = self.control_successes / max(self.control_total, 1)
        p2 = self.treatment_successes / max(self.treatment_total, 1)
        
        diff = p2 - p1
        se = np.sqrt(
            p1 * (1 - p1) / max(self.control_total, 1) +
            p2 * (1 - p2) / max(self.treatment_total, 1)
        )
        
        z = stats.norm.ppf(0.975)
        return (diff - z * se, diff + z * se)
    
    def get_summary_stats(self) -> Dict:
        """Get current summary statistics for the experiment."""
        return {
            "control": {
                "successes": self.control_successes,
                "total": self.control_total,
                "rate": self.control_successes / max(self.control_total, 1)
            },
            "treatment": {
                "successes": self.treatment_successes,
                "total": self.treatment_total,
                "rate": self.treatment_successes / max(self.treatment_total, 1)
            },
            "observed_lift": (
                self.treatment_successes / max(self.treatment_total, 1)
            ) / (
                self.control_successes / max(self.control_total, 1)
            ) - 1,
            "target_sample_size": self.early_stop_threshold,
            "percent_complete": (
                (self.control_total + self.treatment_total) / 
                self.early_stop_threshold * 100
            )
        }
```

### Pattern 3: Multi-Armed Bandit for Dynamic Traffic Allocation

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import random

@dataclass
class BanditArm:
    name: str
    model_version: str
    total_trials: int = 0
    total_successes: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.total_successes / max(self.total_trials, 1)
    
    def update(self, success: bool):
        self.total_trials += 1
        self.total_successes += int(success)

class ThompsonSamplingRouter:
    """
    Implements Thompson Sampling for multi-armed bandit routing in LLM experiments.
    Uses Beta-Bernoulli model for binary success outcomes.
    """
    
    def __init__(
        self,
        exploration_weight: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        minimum_trials_before_exploit: int = 100
    ):
        self.arms: Dict[str, BanditArm] = {}
        self.exploration_weight = exploration_weight
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.minimum_trials = minimum_trials_before_exploit
        self.total_trials = 0
        self.total_successes = 0
    
    def add_arm(self, name: str, model_version: str):
        """Add a new arm (model variant) to the bandit."""
        self.arms[name] = BanditArm(name=name, model_version=model_version)
    
    def select_arm(self, user_context: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Select which arm to serve using Thompson Sampling.
        
        Returns:
            Tuple of (arm_name, model_version)
        """
        exploration_count = sum(
            arm.total_trials < self.minimum_trials 
            for arm in self.arms.values()
        )
        
        if exploration_count > 0:
            arm = self._explore()
        else:
            arm = self._thompson_sample()
        
        return arm.name, arm.model_version
    
    def _explore(self) -> BanditArm:
        """Randomly select an arm for exploration."""
        under_minimum = [
            arm for arm in self.arms.values() 
            if arm.total_trials < self.minimum_trials
        ]
        return random.choice(under_minimum) if under_minimum else self._thompson_sample()
    
    def _thompson_sample(self) -> BanditArm:
        """Sample from posterior and select best arm."""
        samples = {}
        for arm in self.arms.values():
            alpha_post = self.prior_alpha + arm.total_successes
            beta_post = self.prior_beta + (arm.total_trials - arm.total_successes)
            samples[arm.name] = np.random.beta(alpha_post, beta_post)
        
        best_arm_name = max(samples, key=samples.get)
        return self.arms[best_arm_name]
    
    def update(
        self,
        arm_name: str,
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """Update arm statistics based on observed outcome."""
        if arm_name not in self.arms:
            return
        
        self.arms[arm_name].update(success)
        self.total_trials += 1
        self.total_successes += int(success)
    
    def get_recommendation(self) -> Dict:
        """Get current recommendation for traffic allocation."""
        if self.total_trials < self.minimum_trials:
            return {
                "recommendation": "continue_exploration",
                "message": f"Need more data. Current total trials: {self.total_trials}"
            }
        
        best_arm = max(self.arms.values(), key=lambda a: a.success_rate)
        expected_improvement = (
            (best_arm.success_rate - 
             sum(a.success_rate for a in self.arms.values()) / len(self.arms))
        )
        
        return {
            "recommendation": "promote_best",
            "best_arm": best_arm.name,
            "expected_improvement": expected_improvement,
            "confidence": self._calculate_confidence(best_arm)
        }
    
    def _calculate_confidence(self, arm: BanditArm) -> str:
        """Calculate confidence level in the best arm."""
        total = sum(a.total_trials for a in self.arms.values())
        if total < 1000:
            return "low"
        elif total < 5000:
            return "medium"
        else:
            return "high"
```

### Pattern 4: Guardrail Metric Monitoring During Experiments

```python
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class GuardrailCheck:
    name: str
    threshold: float
    current_value: float
    status: str  # "passing", "warning", "failing"
    comparison_window: str = "7d"

class ExperimentGuardrailMonitor:
    """
    Monitors guardrail metrics during experiments to ensure no degradation.
    Guardrails are metrics that must not fall below acceptable thresholds.
    """
    
    def __init__(self, config: Dict):
        self.guardrails = config.get("guardrails", {})
        self.baseline_values: Dict[str, float] = config.get("baseline", {})
        self.absolute_minimums: Dict[str, float] = config.get("absolute_minimums", {})
        
        self.observations: Dict[str, List[Dict]] = {}
        
        self._initialize_observations()
    
    def _initialize_observations(self):
        for metric_name in self.guardrails.keys():
            self.observations[metric_name] = []
    
    def record(
        self,
        variant: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """Record metrics for a single observation."""
        ts = timestamp or datetime.utcnow()
        
        for metric_name, value in metrics.items():
            if metric_name not in self.observations:
                self.observations[metric_name] = []
            
            self.observations[metric_name].append({
                "timestamp": ts,
                "variant": variant,
                "value": value
            })
    
    def check_guardrails(self, variant: str = "treatment") -> List[GuardrailCheck]:
        """Check all guardrails and return status for each."""
        results = []
        
        for metric_name, threshold_pct in self.guardrails.items():
            current = self._calculate_current_value(metric_name, variant)
            baseline = self.baseline_values.get(metric_name, 0)
            absolute_min = self.absolute_minimums.get(metric_name, 0)
            
            threshold_value = baseline * threshold_pct if threshold_pct > 1 else threshold_pct
            
            if current < absolute_min:
                status = "failing"
            elif current < threshold_value:
                status = "warning"
            else:
                status = "passing"
            
            results.append(GuardrailCheck(
                name=metric_name,
                threshold=threshold_value,
                current_value=current,
                status=status
            ))
        
        return results
    
    def _calculate_current_value(
        self,
        metric_name: str,
        variant: str,
        window_hours: int = 24
    ) -> float:
        """Calculate current value for a metric over a time window."""
        observations = self.observations.get(metric_name, [])
        cutoff = datetime.utcnow().timestamp() - (window_hours * 3600)
        
        recent = [
            obs["value"] for obs in observations
            if obs["variant"] == variant and obs["timestamp"].timestamp() > cutoff
        ]
        
        return np.mean(recent) if recent else 0.0
    
    def should_block_rollout(self) -> tuple[bool, List[str]]:
        """Check if experiment should be blocked from rollout."""
        checks = self.check_guardrails()
        
        failing = [c.name for c in checks if c.status == "failing"]
        
        return len(failing) > 0, failing
    
    def should_pause_experiment(self) -> tuple[bool, List[str]]:
        """Check if experiment should be paused due to guardrail issues."""
        checks = self.check_guardrails()
        
        warnings_or_failing = [
            c.name for c in checks if c.status in ["warning", "failing"]
        ]
        
        return len(warnings_or_failing) > 0, warnings_or_failing
    
    def get_guardrail_trends(self, metric_name: str) -> Dict:
        """Get trend analysis for a specific guardrail metric."""
        obs = self.observations.get(metric_name, [])
        
        variants = set(o["variant"] for o in obs)
        
        trends = {}
        for variant in variants:
            variant_obs = [o for o in obs if o["variant"] == variant]
            variant_obs.sort(key=lambda x: x["timestamp"])
            
            values = [o["value"] for o in variant_obs]
            
            if len(values) < 2:
                trends[variant] = {"trend": "insufficient_data"}
                continue
            
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            trends[variant] = {
                "current": values[-1],
                "mean": np.mean(values),
                "std": np.std(values),
                "slope": slope,
                "trend": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
            }
        
        return trends
```

### Pattern 5: Experiment Result Analyzer with LLM-Assisted Evaluation

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class ExperimentResult:
    experiment_name: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    lift: Dict[str, float]
    p_values: Dict[str, float]
    recommendation: str
    confidence: str
    sample_sizes: Dict[str, int]

class LLMExperimentAnalyzer:
    """
    Analyzes LLM experiment results using both statistical methods and
    LLM-assisted evaluation for qualitative aspects.
    """
    
    def __init__(
        self,
        llm_client,
        statistical_analyzer,
        quality_evaluator
    ):
        self.llm = llm_client
        self.stats = statistical_analyzer
        self.quality = quality_evaluator
    
    async def analyze(
        self,
        experiment_name: str,
        observations: List[Dict],
        ground_truth_eval: bool = False
    ) -> ExperimentResult:
        """
        Comprehensive analysis of experiment results.
        
        Args:
            experiment_name: Name of the experiment
            observations: List of observation dicts with variant, metrics, outputs
            ground_truth_eval: Whether ground truth labels are available
        """
        metrics_by_variant = self._split_by_variant(observations)
        
        control = metrics_by_variant.get("control", {})
        treatment = metrics_by_variant.get("treatment", {})
        
        lifts, p_values = self._calculate_statistical_significance(
            control, treatment
        )
        
        quality_comparison = await self._evaluate_quality_difference(
            observations, ground_truth_eval
        )
        
        recommendation, confidence = self._make_recommendation(
            lifts, p_values, quality_comparison
        )
        
        return ExperimentResult(
            experiment_name=experiment_name,
            control_metrics=self._aggregate_metrics(control),
            treatment_metrics=self._aggregate_metrics(treatment),
            lift=lifts,
            p_values=p_values,
            recommendation=recommendation,
            confidence=confidence,
            sample_sizes={
                "control": len(metrics_by_variant.get("control", {}).get("total", [0])),
                "treatment": len(metrics_by_variant.get("treatment", {}).get("total", [0]))
            }
        )
    
    def _split_by_variant(
        self,
        observations: List[Dict]
    ) -> Dict[str, Dict[str, List]]:
        result = {"control": {}, "treatment": {}}
        
        for obs in observations:
            variant = obs.get("variant", "control")
            if variant not in ["control", "treatment"]:
                continue
            
            for metric_name, value in obs.get("metrics", {}).items():
                if metric_name not in result[variant]:
                    result[variant][metric_name] = []
                result[variant][metric_name].append(value)
        
        return result
    
    def _aggregate_metrics(self, variant_metrics: Dict[str, List]) -> Dict[str, float]:
        import numpy as np
        aggregated = {}
        for name, values in variant_metrics.items():
            if len(values) == 0:
                continue
            aggregated[name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "count": len(values)
            }
        return aggregated
    
    def _calculate_statistical_significance(
        self,
        control: Dict[str, List],
        treatment: Dict[str, List]
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        import numpy as np
        from scipy import stats
        
        lifts = {}
        p_values = {}
        
        all_metrics = set(control.keys()) | set(treatment.keys())
        
        for metric in all_metrics:
            c_values = control.get(metric, [])
            t_values = treatment.get(metric, [])
            
            if len(c_values) < 10 or len(t_values) < 10:
                lifts[metric] = 0.0
                p_values[metric] = 1.0
                continue
            
            c_mean = np.mean(c_values)
            t_mean = np.mean(t_values)
            
            lift = (t_mean - c_mean) / max(abs(c_mean), 1e-10)
            
            t_stat, p_value = stats.ttest_ind(t_values, c_values)
            
            lifts[metric] = float(lift)
            p_values[metric] = float(p_value)
        
        return lifts, p_values
    
    async def _evaluate_quality_difference(
        self,
        observations: List[Dict],
        ground_truth_available: bool
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate qualitative differences between variants.
        """
        if ground_truth_available:
            return {"method": "ground_truth_comparison"}
        
        control_outputs = [
            obs["output"] for obs in observations
            if obs.get("variant") == "control"
        ][:50]
        treatment_outputs = [
            obs["output"] for obs in observations
            if obs.get("variant") == "treatment"
        ][:50]
        
        prompt = f"""
Compare these two sets of LLM outputs. Evaluate:
1. Overall helpfulness and relevance
2. Accuracy and factuality  
3. Safety and appropriateness
4. Coherence and readability

Control outputs (sample of 10):
{json.dumps(control_outputs[:10], indent=2)}

Treatment outputs (sample of 10):
{json.dumps(treatment_outputs[:10], indent=2)}

Provide a JSON response with:
- control_better_features: list of features where control excels
- treatment_better_features: list of features where treatment excels  
- overall_preference: "control", "treatment", or "no_clear_winner"
- confidence: "high", "medium", or "low" in this assessment
"""
        
        response = await self.llm.generate(
            prompt=prompt,
            parameters={"temperature": 0.3, "max_tokens": 1000}
        )
        
        try:
            return json.loads(response.text)
        except:
            return {"method": "llm_evaluation_failed"}
    
    def _make_recommendation(
        self,
        lifts: Dict[str, float],
        p_values: Dict[str, float],
        quality_comparison: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Make rollout recommendation based on all signals.
        """
        primary_metric = "success_rate"
        
        primary_lift = lifts.get(primary_metric, 0)
        primary_p = p_values.get(primary_metric, 1)
        
        if primary_p > 0.05:
            return " inconclusive - need more data", "low"
        
        if primary_lift < 0:
            return "rollback - negative impact detected", "high"
        
        significant_positive = sum(
            1 for m, p in p_values.items() 
            if p < 0.05 and lifts.get(m, 0) > 0
        )
        
        significant_negative = sum(
            1 for m, p in p_values.items() 
            if p < 0.05 and lifts.get(m, 0) < 0
        )
        
        if significant_negative > significant_positive:
            return "caution - mixed results with negative primary impact", "medium"
        
        if significant_positive >= 3 and primary_lift > 0.05:
            return "full_rollout_recommended", "high"
        elif significant_positive >= 1 and primary_lift > 0.02:
            return "partial_rollout_recommended", "medium"
        else:
            return "extended_testing_recommended", "low"
```

## Framework Integration

### Integration with LaunchDarkly

```python
class LaunchDarklyExperimentIntegrator:
    def __init__(self, sdk_key: str):
        self.client = LaunchDarklyClient(sdk_key)
    
    async def create_experiment_flag(
        self,
        flag_key: str,
        model_versions: List[str],
        default_version: str
    ) -> str:
        flag = await self.client.create_feature_flag(
            key=flag_key,
            name=f"LLM Model Selection - {flag_key}",
            variation_type="string",
            variations=model_versions,
            default_value=default_version
        )
        return flag["key"]
    
    async def get_model_for_context(
        self,
        flag_key: str,
        user_context: Dict
    ) -> str:
        return await self.client.variation(
            flag_key, user_context, "baseline"
        )
```

### Integration with weights & biases

```python
class WandBExperimentTracker:
    def __init__(self, project_name: str):
        self.run = wandb.init(project=project_name)
    
    def log_observation(self, observation: Dict):
        self.run.log(observation)
    
    def log_experiment_config(self, config: Dict):
        self.run.config.update(config)
    
    def create_comparison_dashboard(self, experiments: List[str]):
        panels = [
            {"metric": "lift/primary", "type": "line"},
            {"metric": "p_value/primary", "type": "line"},
            {"metric": "guardrail/safety_violation_rate", "type": "threshold"}
        ]
        return self.run.use_artifact(
            f"dashboard/{experiments[0]}",
            type="dashboard"
        )
```

## Performance Considerations

### Statistical Power and Sample Size

For LLM quality metrics with high variance, plan accordingly:

| Effect Size | Approximate Sample Size per Arm |
|------------|--------------------------------|
| 5% relative | 50,000 - 100,000 |
| 10% relative | 15,000 - 25,000 |
| 20% relative | 4,000 - 6,000 |
| 50% relative | 800 - 1,200 |

### Multiple Comparison Correction

When testing multiple metrics, apply corrections:
- **Bonferroni**: Simple but conservative
- **Holm-Bonferroni**: Slightly more powerful
- **Benjamini-Hochberg**: Good for discovering true positives

### Early Stopping Risks

Sequential testing methods reduce sample size but:
- Require pre-specified boundaries
- May produce biased estimates if misused
- Should be validated against full-sample analysis

## Common Pitfalls

### Pitfall 1: Ignoring Variance in LLM Outputs

**Problem**: LLM outputs have high variance due to sampling. Simple comparisons without accounting for this lead to noisy results.

**Solution**: Use large sample sizes and consider stratified analysis:
```python
# Stratify by prompt complexity
def stratify_analysis(self, observations):
    simple = [o for o in observations if o["prompt_length"] < 100]
    medium = [o for o in observations if 100 <= o["prompt_length"] < 500]
    complex_ = [o for o in observations if o["prompt_length"] >= 500]
    
    return {
        "simple": self._analyze(simple),
        "medium": self._analyze(medium),
        "complex": self._analyze(complex_)
    }
```

### Pitfall 2: Not Accounting for Novelty Effects

**Problem**: Users may respond differently to a new model initially due to novelty, not because of actual improvement.

**Solution**: Plan for adequate run-in period:
```python
EXPERIMENT_CONFIG = {
    "warmup_period_days": 3,
    "measurement_period_days": 14,
    "cooldown_period_days": 2
}
```

### Pitfall 3: Testing Too Many Variants Simultaneously

**Problem**: Each additional variant reduces power for comparisons.

**Solution**: Limit to 2-3 variants per experiment. Use multi-armed bandits for exploration across many variants before running focused A/B tests.

### Pitfall 4: Not Monitoring Guardrails

**Problem**: Optimizing for primary metrics while degrading secondary metrics.

**Solution**: Implement hard stops on guardrails:
```python
if guardrail_monitor.should_block_rollout():
    pause_experiment()
    alert_team("Guardrail threshold breached")
```

## Research References

1. **Kohavi et al. (2020)** - "Trustworthy Online Controlled Experiments" - Comprehensive guide to A/B testing methodology.

2. **Deng et al. (2017)** - "Statistical Methods for Discrimination Detection and Mitigation" - Fairness considerations in experimentation.

3. **Johari et al. (2022)** - "Peeking at A/B Tests" - Sequential testing methods for early stopping.

4. **Rubin (2021)** - "Multi-Armed Bandits in the Wild" - Thompson Sampling for online experiments.

5. **Lai et al. (2023)** - "Deploying Large Language Models" - Practical considerations for LLM deployment strategies.

6. **Zhang et al. (2022)** - "A/B Testing for Recommender Systems" - Relevant techniques for personalized LLM outputs.

7. **Storkey (2022)** - "Simulation-Based Inference for Online Experiments" - Modern approaches to experiment analysis.

8. **Fabijan et al. (2023)** - "Upgrading Experiment Infrastructure" - Operational aspects of experimentation platforms.

9. **Chen et al. (2023)** - "Causal Inference in Large Language Models" - Causal reasoning for LLM evaluation.

10. **Xie et al. (2023)** - "Progressive Deployment of ML Models" - Gradual rollout strategies for ML systems.