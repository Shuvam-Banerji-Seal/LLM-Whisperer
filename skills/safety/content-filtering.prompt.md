# Content Filtering and Moderation Systems

## Problem Statement

Building content moderation systems for LLM applications presents unique challenges that extend far beyond simple keyword blocking. LLMs can generate an enormous variety of content types, each with different safety considerations. A request for medical information needs different handling than a request for violent content analysis, which needs different handling than creative fiction writing. The same content may be acceptable in one context but harmful in another - a medical textbook discussing surgery is educational, while the same details in a different context could be harmful instructions.

Traditional content moderation approaches using rule-based filters fail to capture the nuances of natural language and context. A filter that blocks all mentions of self-harm might prevent a therapist from using the LLM to help clients, while a filter that's too permissive might allow harmful content to slip through. The challenge is building moderation that is both precise (minimizing false positives and negatives) and robust (hard to circumvent).

This skill covers designing multi-layered content moderation systems, implementing context-aware filtering, handling edge cases in content classification, building efficient moderation pipelines, and continuously improving moderation based on feedback.

## Theory & Fundamentals

### Content Safety Taxonomy

```
Content Categories:
├── Violence & Harm
│   ├── Physical violence (assault, murder, torture)
│   ├── Psychological harm (manipulation, coercion)
│   ├── Self-harm (suicide, self-mutilation)
│   └── Harmful instructions (weapons, drugs, hacking)
├── Sexual & Adult Content
│   ├── Explicit sexual content
│   ├── Sexual exploitation
│   ├── Adult content involving minors
│   └── Objectification
├── Hate & Discrimination
│   ├── Hate speech
│   ├── Discrimination
│   ├── Harassment
│   └── Extremist content
├── Privacy & Personal Data
│   ├── PII exposure
│   ├── Doxing
│   ├── Privacy violations
│   └── Data harvesting
├── Misinformation
│   ├── Medical misinformation
│   ├── Financial misinformation
│   ├── Political misinformation
│   └── Conspiracy content
└── Quality & Appropriateness
    ├── Profanity
    ├── Disturbing content
    ├── Spam
    └── Off-topic content
```

### Multi-Label Classification Framework

LLM content moderation requires multi-label classification because content can belong to multiple categories simultaneously:

```
Input → Feature Extraction → Multi-Label Classifier → Category Scores → Thresholding → Output
                                                              ↓
                                                      [Violence: 0.2]
                                                      [Self-harm: 0.8]  → Block + Intervention
                                                      [Medical: 0.1]
```

For multi-label classification with C categories, the probability of each label is:

$$P(y_i = 1 | x) = \sigma(w_i \cdot h(x) + b_i)$$

where $h(x)$ is the hidden representation, $w_i$ is the weight vector for category $i$, and $\sigma$ is the sigmoid function.

### Contextual Moderation Theory

Content appropriateness depends heavily on context:

$$Appropriateness(content, context) = f(content\_features, context\_features, domain\_rules)$$

Key contextual factors:
- **Intent**: Educational, harmful, entertainment, informational
- **Audience**: General public, professionals, minors, vulnerable groups
- **Setting**: Medical, legal, creative, conversational
- **Consent**: Public figure, private individual, hypothetical

### Moderation Pipeline Architecture

```
Moderation Pipeline:
├── Pre-Moderation Layer
│   ├── Input validation
│   ├── Pattern matching (fast blocklist)
│   ├── PII detection
│   └── Basic profanity filter
│
├── Primary Classification Layer
│   ├── Transformer-based classifier
│   ├── Multi-label category scoring
│   ├── Contextual enrichment
│   └── Confidence estimation
│
├── Secondary Verification Layer
│   ├── LLM-based semantic analysis
│   ├── Context-specific rules
│   ├── Edge case handling
│   └── Threshold adjustment
│
├── Action Layer
│   ├── Allow (all scores < threshold)
│   ├── Warn (some scores borderline)
│   ├── Block (high scores)
│   ├── Escalate (very high scores + specific indicators)
│   └── Custom handling per category
│
└── Post-Moderation Layer
    ├── Output validation
    ├── Log and monitor
    ├── Feedback integration
    └── Model retraining triggers
```

## Implementation Patterns

### Pattern 1: Multi-Layer Content Classifier

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import asyncio

class ContentCategory(Enum):
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    MISINFORMATION = "misinformation"
    PII = "pii"
    PROFANITY = "profanity"
    DANGEROUS_CONTENT = "dangerous_content"
    QUALITY = "quality"

@dataclass
class ClassificationResult:
    category: ContentCategory
    score: float
    confidence: float
    should_block: bool
    action: str
    details: Optional[str] = None

@dataclass  
class ModerationResult:
    overall_safe: bool
    classifications: List[ClassificationResult]
    risk_score: float
    should_allow: bool
    should_warn: bool
    escalation_needed: bool
    final_decision: str

class MultiLayerContentModerator:
    """
    Production content moderation system with multiple detection layers.
    """
    
    def __init__(
        self,
        fast_model,
        precise_model,
        context_analyzer,
        config: Dict
    ):
        self.fast_model = fast_model
        self.precise_model = precise_model
        self.context = context_analyzer
        self.config = config
        
        self.thresholds = config.get("thresholds", {
            ContentCategory.SELF_HARM: 0.5,
            ContentCategory.VIOLENCE: 0.6,
            ContentCategory.SEXUAL: 0.6,
            ContentCategory.HATE_SPEECH: 0.5,
            ContentCategory.PII: 0.5,
            ContentCategory.DANGEROUS_CONTENT: 0.5,
        })
        
        self.category_actions = config.get("actions", {
            ContentCategory.SELF_HARM: "escalate",
            ContentCategory.VIOLENCE: "block",
            ContentCategory.SEXUAL: "block",
            ContentCategory.HATE_SPEECH: "block",
            ContentCategory.PII: "redact",
            ContentCategory.DANGEROUS_CONTENT: "block",
        })
    
    async def moderate(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> ModerationResult:
        """
        Comprehensive content moderation.
        """
        if not text or len(text.strip()) == 0:
            return ModerationResult(
                overall_safe=True,
                classifications=[],
                risk_score=0.0,
                should_allow=True,
                should_warn=False,
                escalation_needed=False,
                final_decision="allow_empty"
            )
        
        fast_results = await self._fast_screening(text)
        
        high_risk_categories = [
            r for r in fast_results 
            if r.score > self.thresholds.get(r.category, 0.7)
        ]
        
        if high_risk_categories:
            precise_results = await self._precise_classification(
                text, high_risk_categories
            )
        else:
            precise_results = high_risk_categories
        
        enriched = self._enrich_with_context(
            precise_results, context or {}
        )
        
        decisions = self._make_decisions(enriched)
        
        return self._aggregate_decisions(decisions)
    
    async def _fast_screening(
        self,
        text: str
    ) -> List[ClassificationResult]:
        """
        Fast initial screening using lightweight model.
        """
        embeddings = await self.fast_model.encode(text)
        
        scores = self.fast_model.predict_categories(embeddings)
        
        results = []
        for category, score in scores.items():
            confidence = self._estimate_confidence(score, "fast")
            should_block = score > self.thresholds.get(category, 0.8)
            
            results.append(ClassificationResult(
                category=ContentCategory(category),
                score=float(score),
                confidence=confidence,
                should_block=should_block,
                action=self.category_actions.get(ContentCategory(category), "warn")
            ))
        
        return results
    
    async def _precise_classification(
        self,
        text: str,
        initial_results: List[ClassificationResult]
    ) -> List[ClassificationResult]:
        """
        Precise classification for high-risk categories.
        """
        categories_to_check = [r.category for r in initial_results]
        
        prompt = self._build_classification_prompt(text, categories_to_check)
        
        response = await self.precise_model.generate(prompt)
        
        return self._parse_classification_response(response, initial_results)
    
    def _build_classification_prompt(
        self,
        text: str,
        categories: List[ContentCategory]
    ) -> str:
        """Build prompt for precise classification."""
        category_descriptions = {
            ContentCategory.SELF_HARM: "Content about suicide, self-harm, or self-mutilation",
            ContentCategory.VIOLENCE: "Violent content, assault, torture, murder descriptions",
            ContentCategory.SEXUAL: "Explicit sexual content or sexual exploitation",
            ContentCategory.HATE_SPEECH: "Hate speech against protected groups",
            ContentCategory.DANGEROUS_CONTENT: "Instructions for harmful activities",
        }
        
        category_list = "\n".join([
            f"- {cat.value}: {category_descriptions.get(cat, 'Other')}"
            for cat in categories
        ])
        
        return f"""Analyze this text for harmful content. Rate each category from 0.0 (definitely safe) to 1.0 (definitely harmful).

Categories to check:
{category_list}

Text to analyze:
\"{text}\"

Provide ratings in JSON format:
{{"category": "score"}}
"""
    
    def _parse_classification_response(
        self,
        response: str,
        initial_results: List[ClassificationResult]
    ) -> List[ClassificationResult]:
        """Parse LLM classification response."""
        initial_map = {r.category: r for r in initial_results}
        
        try:
            import json
            scores = json.loads(response)
            
            for category_str, score in scores.items():
                category = ContentCategory(category_str)
                if category in initial_map:
                    result = initial_map[category]
                    result.score = float(score)
                    result.confidence = self._estimate_confidence(score, "precise")
                    result.should_block = score > self.thresholds.get(category, 0.5)
        
        except:
            pass
        
        return initial_results
    
    def _enrich_with_context(
        self,
        results: List[ClassificationResult],
        context: Dict
    ) -> List[ClassificationResult]:
        """
        Adjust scores based on context.
        """
        intent = context.get("intent", "unknown")
        audience = context.get("audience", "general")
        
        adjustment_factors = {
            "educational": {"violence": 0.9, "medical": 0.8},
            "therapeutic": {"self_harm": 0.7, "mental_health": 0.8},
            "creative": {"violence": 0.9, "sexual": 0.9},
            "news": {"misinformation": 0.8},
            "entertainment": {"violence": 0.95, "sexual": 0.9}
        }
        
        for result in results:
            if intent in adjustment_factors:
                factor = adjustment_factors[intent].get(result.category.value, 1.0)
                result.score *= factor
        
        return results
    
    def _make_decisions(
        self,
        results: List[ClassificationResult]
    ) -> List[ClassificationResult]:
        """Make moderation decisions for each category."""
        for result in results:
            if result.score > self.thresholds.get(result.category, 0.5):
                result.should_block = True
                result.action = self.category_actions.get(
                    result.category, "block"
                )
        
        return results
    
    def _aggregate_decisions(
        self,
        results: List[ClassificationResult]
    ) -> ModerationResult:
        """Aggregate category decisions into final moderation decision."""
        if not results:
            return ModerationResult(
                overall_safe=True,
                classifications=[],
                risk_score=0.0,
                should_allow=True,
                should_warn=False,
                escalation_needed=False,
                final_decision="no_content"
            )
        
        max_score = max(r.score for r in results)
        avg_score = np.mean([r.score for r in results])
        risk_score = max_score * 0.7 + avg_score * 0.3
        
        blocked = [r for r in results if r.should_block]
        escalated = [r for r in blocked if r.action == "escalate"]
        
        should_allow = len(blocked) == 0
        should_warn = len([r for r in results if 0.3 < r.score < 0.5]) > 2
        escalation_needed = len(escalated) > 0
        
        if escalation_needed:
            final_decision = "escalate"
        elif len(blocked) > 2:
            final_decision = "block_high_risk"
        elif blocked:
            final_decision = f"block_{blocked[0].category.value}"
        elif should_warn:
            final_decision = "allow_with_warning"
        else:
            final_decision = "allow"
        
        return ModerationResult(
            overall_safe=should_allow,
            classifications=results,
            risk_score=risk_score,
            should_allow=should_allow,
            should_warn=should_warn,
            escalation_needed=escalation_needed,
            final_decision=final_decision
        )
    
    def _estimate_confidence(
        self,
        score: float,
        model_type: str
    ) -> float:
        """Estimate confidence in classification."""
        distance_from_boundary = abs(score - 0.5) * 2
        
        if model_type == "precise":
            base_confidence = 0.85
        else:
            base_confidence = 0.7
        
        return min(base_confidence * distance_from_boundary, 0.99)
```

### Pattern 2: Context-Aware Content Rules Engine

```python
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import re

class ContentType(Enum):
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    VIOLENT = "violent"
    SEXUAL = "sexual"
    CREATIVE = "creative"
    EDUCATIONAL = "educational"
    NEWS = "news"
    GENERAL = "general"

@dataclass
class ContentRule:
    name: str
    applies_to: List[ContentType]
    condition: Callable[[str, Dict], bool]
    action: str
    priority: int = 0

class ContextualRulesEngine:
    """
    Rule-based content moderation with contextual awareness.
    """
    
    def __init__(self):
        self.rules: List[ContentRule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default moderation rules."""
        
        self.rules.append(ContentRule(
            name="medical_disclaimer",
            applies_to=[ContentType.MEDICAL],
            condition=lambda text, ctx: any(
                keyword in text.lower() 
                for keyword in ["diagnosis", "treatment", "prescribe", "medication"]
            ),
            action="add_disclaimer",
            priority=10
        ))
        
        self.rules.append(ContentRule(
            name="medical_misinformation",
            applies_to=[ContentType.MEDICAL],
            condition=self._detect_medical_misinformation,
            action="flag_for_review",
            priority=20
        ))
        
        self.rules.append(ContentRule(
            name="violent_content_minors",
            applies_to=[ContentType.VIOLENT],
            condition=lambda text, ctx: ctx.get("audience") == "minors",
            action="block",
            priority=30
        ))
        
        self.rules.append(ContentRule(
            name="creative_violence_threshold",
            applies_to=[ContentType.VIOLENT, ContentType.CREATIVE],
            condition=self._check_violence_severity,
            action="age_restrict",
            priority=15
        ))
        
        self.rules.append(ContentRule(
            name="legal_disclaimer",
            applies_to=[ContentType.LEGAL],
            condition=lambda text, ctx: any(
                keyword in text.lower()
                for keyword in ["sue", "lawsuit", "court", "legal action"]
            ),
            action="add_disclaimer",
            priority=10
        ))
        
        self.rules.append(ContentRule(
            name="financial_advice_disclaimer",
            applies_to=[ContentType.FINANCIAL],
            condition=lambda text, ctx: any(
                keyword in text.lower()
                for keyword in ["invest", "stock", "crypto", "portfolio", "trade"]
            ),
            action="add_disclaimer",
            priority=10
        ))
    
    def _detect_medical_misinformation(
        self,
        text: str,
        context: Dict
    ) -> bool:
        """Detect potential medical misinformation."""
        misinformation_indicators = [
            r"cures? (?:cancer|aids|diabetes) completely",
            r"miracle (?:cure|treatment|remedy)",
            r"doctors? (?:won't tell you|don't want you to know)",
            r"natural (?:means you can|cure that's better)",
        ]
        
        for pattern in misinformation_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _check_violence_severity(
        self,
        text: str,
        context: Dict
    ) -> bool:
        """Check if violence content exceeds threshold."""
        violence_intensity = sum([
            len(re.findall(r"(?:graphic|gory|brutal|murder|kill)", text, re.I)),
            len(re.findall(r"(?:blood|gore|trauma|injury)", text, re.I))
        ])
        
        violence_score = min(violence_intensity / 5, 1.0)
        
        return violence_score > 0.6
    
    def evaluate(
        self,
        text: str,
        content_type: ContentType,
        context: Dict
    ) -> List[Dict]:
        """
        Evaluate text against applicable rules.
        """
        applicable_rules = [
            rule for rule in self.rules
            if content_type in rule.applies_to or ContentType.GENERAL in rule.applies_to
        ]
        
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        triggered = []
        for rule in applicable_rules:
            try:
                if rule.condition(text, context):
                    triggered.append({
                        "rule": rule.name,
                        "action": rule.action,
                        "priority": rule.priority
                    })
            except Exception:
                continue
        
        return triggered
    
    def apply_actions(
        self,
        text: str,
        triggered_rules: List[Dict]
    ) -> str:
        """Apply actions from triggered rules."""
        result = text
        
        for rule in triggered_rules:
            action = rule["action"]
            
            if action == "add_disclaimer":
                result = self._add_disclaimer(result, rule["rule"])
            elif action == "age_restrict":
                result = self._add_age_restriction(result)
        
        return result
    
    def _add_disclaimer(self, text: str, rule_name: str) -> str:
        """Add appropriate disclaimer."""
        disclaimers = {
            "medical_disclaimer": "\n\n[Medical Disclaimer: This is not medical advice. Consult a healthcare professional.]\n",
            "legal_disclaimer": "\n\n[Legal Disclaimer: This is not legal advice. Consult a qualified attorney.]\n",
            "financial_advice_disclaimer": "\n\n[Financial Disclaimer: This is not financial advice. Consult a qualified financial advisor.]\n"
        }
        
        disclaimer = disclaimers.get(rule_name, "\n\n[Please verify this information independently.]\n")
        return text + disclaimer
    
    def _add_age_restriction(self, text: str) -> str:
        """Add age restriction notice."""
        return "[Age-Restricted Content]\n" + text
```

### Pattern 3: Adaptive Threshold Calibration

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import deque

@dataclass
class FeedbackDataPoint:
    timestamp: datetime
    content_hash: str
    category: str
    initial_score: float
    human_label: bool  # True if actually harmful
    feedback_source: str

class AdaptiveThresholdCalibrator:
    """
    Continuously calibrates moderation thresholds based on feedback.
    """
    
    def __init__(
        self,
        target_false_positive_rate: float = 0.05,
        target_false_negative_rate: float = 0.01
    ):
        self.target_fp = target_false_positive_rate
        self.target_fn = target_false_negative_rate
        
        self.feedback_buffer: deque = deque(maxlen=10000)
        self.thresholds: Dict[str, float] = {}
        
        self.calibration_history: Dict[str, List[Dict]] = {}
    
    def add_feedback(
        self,
        content_hash: str,
        category: str,
        model_score: float,
        human_label: bool,
        source: str
    ):
        """Add human feedback data point."""
        self.feedback_buffer.append(FeedbackDataPoint(
            timestamp=datetime.utcnow(),
            content_hash=content_hash,
            category=category,
            initial_score=model_score,
            human_label=human_label,
            feedback_source=source
        ))
    
    def calibrate_category(
        self,
        category: str,
        window_hours: int = 168
    ) -> Tuple[float, Dict]:
        """
        Calibrate threshold for a specific category.
        Uses the past week's feedback data.
        """
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        
        relevant_feedback = [
            fp for fp in self.feedback_buffer
            if fp.category == category and fp.timestamp > cutoff
        ]
        
        if len(relevant_feedback) < 50:
            return self.thresholds.get(category, 0.5), {"status": "insufficient_data"}
        
        scores = np.array([fp.initial_score for fp in relevant_feedback])
        labels = np.array([int(fp.human_label) for fp in relevant_feedback])
        
        optimal_threshold, metrics = self._find_optimal_threshold(
            scores, labels
        )
        
        self.thresholds[category] = optimal_threshold
        
        self._record_calibration(category, optimal_threshold, metrics, len(relevant_feedback))
        
        return optimal_threshold, metrics
    
    def _find_optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Find threshold that optimizes for target FP and FN rates.
        """
        threshold_range = np.linspace(0, 1, 100)
        
        best_threshold = 0.5
        best_score = float('inf')
        
        results = []
        for threshold in threshold_range:
            predictions = (scores >= threshold).astype(int)
            
            fp = np.sum((predictions == 1) & (labels == 0)) / max(np.sum(labels == 0), 1)
            fn = np.sum((predictions == 0) & (labels == 1)) / max(np.sum(labels == 1), 1)
            
            score = (fp - self.target_fp)**2 + (fn - self.target_fn)**2
            
            results.append({
                "threshold": threshold,
                "fp_rate": fp,
                "fn_rate": fn,
                "score": score
            })
            
            if score < best_score:
                best_score = score
                best_threshold = threshold
        
        best_result = min(results, key=lambda x: x["score"])
        
        metrics = {
            "optimal_threshold": best_threshold,
            "fp_rate": best_result["fp_rate"],
            "fn_rate": best_result["fn_rate"],
            "sample_size": len(scores)
        }
        
        return best_threshold, metrics
    
    def _record_calibration(
        self,
        category: str,
        threshold: float,
        metrics: Dict,
        sample_size: int
    ):
        """Record calibration history."""
        if category not in self.calibration_history:
            self.calibration_history[category] = []
        
        self.calibration_history[category].append({
            "timestamp": datetime.utcnow(),
            "threshold": threshold,
            "fp_rate": metrics["fp_rate"],
            "fn_rate": metrics["fn_rate"],
            "sample_size": sample_size
        })
        
        if len(self.calibration_history[category]) > 100:
            self.calibration_history[category].pop(0)
    
    def get_adjusted_threshold(
        self,
        category: str,
        context_adjustments: Dict
    ) -> float:
        """
        Get threshold adjusted for current context.
        """
        base_threshold = self.thresholds.get(category, 0.5)
        
        context_multipliers = context_adjustments.get("multipliers", {})
        
        adjusted = base_threshold
        for factor, multiplier in context_multipliers.items():
            if factor == "high_stakes":
                adjusted *= (1 + (1 - multiplier) * 0.2)
            elif factor == "vulnerable_population":
                adjusted *= (1 + (1 - multiplier) * 0.15)
        
        return max(0.1, min(0.95, adjusted))
    
    def get_calibration_report(self) -> Dict:
        """Generate calibration status report."""
        report = {}
        for category in set(fp.category for fp in self.feedback_buffer):
            if category in self.calibration_history:
                history = self.calibration_history[category]
                latest = history[-1]
                
                report[category] = {
                    "current_threshold": latest["threshold"],
                    "fp_rate": latest["fp_rate"],
                    "fn_rate": latest["fn_rate"],
                    "last_calibrated": latest["timestamp"].isoformat(),
                    "sample_size": latest["sample_size"]
                }
        
        return report
```

### Pattern 4: Real-time Content Stream Processing

```python
from typing import Dict, List, Optional, AsyncIterator
import asyncio
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StreamSegment:
    text: str
    start_pos: int
    end_pos: int
    is_complete: bool

class StreamingContentModerator:
    """
    Content moderation for streaming LLM outputs.
    Moderates incrementally as content is generated.
    """
    
    def __init__(
        self,
        moderator: MultiLayerContentModerator,
        buffer_size: int = 100,
        check_interval_ms: int = 500
    ):
        self.moderator = moderator
        self.buffer_size = buffer_size
        self.check_interval = check_interval_ms / 1000.0
        
        self.segment_buffer: List[StreamSegment] = []
        self.risk_threshold = 0.7
        self.blocked = False
        self.warning_issued = False
    
    async def moderate_stream(
        self,
        text_iterator: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """
        Moderate streaming content, yielding safe content and
        optionally injecting warnings or stopping on harmful content.
        """
        accumulated_text = ""
        last_check = datetime.utcnow()
        
        async for chunk in text_iterator:
            accumulated_text += chunk
            
            time_since_check = (datetime.utcnow() - last_check).total_seconds()
            
            if time_since_check >= self.check_interval or len(accumulated_text) > self.buffer_size:
                should_continue = await self._check_content(accumulated_text)
                
                if not should_continue:
                    yield "[Content moderation: Output stopped due to safety concerns]"
                    break
                
                last_check = datetime.utcnow()
            
            yield chunk
            
            if self.blocked:
                break
    
    async def _check_content(self, content: str) -> bool:
        """Check accumulated content for harmful material."""
        result = await self.moderator.moderate(content)
        
        if result.escalation_needed:
            self.blocked = True
            return False
        
        if result.risk_score > self.risk_threshold and not self.warning_issued:
            self.warning_issued = True
            return True
        
        if not result.should_allow and result.risk_score > 0.8:
            self.blocked = True
            return False
        
        return True
    
    def reset(self):
        """Reset moderation state for new stream."""
        self.segment_buffer.clear()
        self.blocked = False
        self.warning_issued = False
```

### Pattern 5: Moderation Dashboard and Analytics

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

@dataclass
class ModerationMetrics:
    total_processed: int
    blocked_count: int
    warned_count: int
    allowed_count: int
    escalation_count: int
    category_breakdown: Dict[str, int]
    avg_processing_time_ms: float
    false_positive_rate: float
    false_negative_rate: float

class ModerationAnalytics:
    """
    Analytics and reporting for content moderation system.
    """
    
    def __init__(self):
        self.metrics_history: List[ModerationMetrics] = []
        self.category_stats: Dict[str, Dict] = {}
        self.processing_times: List[float] = []
        
        self.feedback_data: List[Dict] = []
    
    def record_moderation(
        self,
        result: ModerationResult,
        processing_time_ms: float,
        context: Optional[Dict] = None
    ):
        """Record a moderation decision for analytics."""
        self.processing_times.append(processing_time_ms)
        
        for classification in result.classifications:
            cat = classification.category.value
            if cat not in self.category_stats:
                self.category_stats[cat] = {
                    "total": 0, "blocked": 0, "warned": 0
                }
            
            self.category_stats[cat]["total"] += 1
            if classification.should_block:
                self.category_stats[cat]["blocked"] += 1
            elif classification.score > 0.3:
                self.category_stats[cat]["warned"] += 1
    
    def record_feedback(
        self,
        content_id: str,
        category: str,
        model_decision: bool,
        human_decision: bool,
        feedback_source: str
    ):
        """Record human feedback for model improvement."""
        self.feedback_data.append({
            "content_id": content_id,
            "category": category,
            "model_decision": model_decision,
            "human_decision": human_decision,
            "feedback_source": feedback_source,
            "timestamp": datetime.utcnow(),
            "is_fp": model_decision and not human_decision,
            "is_fn": not model_decision and human_decision
        })
    
    def calculate_metrics(
        self,
        window_hours: int = 24
    ) -> ModerationMetrics:
        """Calculate metrics for specified time window."""
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        
        relevant_feedback = [
            f for f in self.feedback_data
            if f["timestamp"] > cutoff
        ]
        
        fp_count = sum(1 for f in relevant_feedback if f["is_fp"])
        fn_count = sum(1 for f in relevant_feedback if f["is_fn"])
        total_with_feedback = len(relevant_feedback)
        
        total = len(self.processing_times)
        blocked = sum(
            1 for cat, stats in self.category_stats.items()
            for _ in range(stats["blocked"])
        )
        
        return ModerationMetrics(
            total_processed=total,
            blocked_count=blocked,
            warned_count=0,
            allowed_count=total - blocked,
            escalation_count=0,
            category_breakdown={
                cat: stats["total"] 
                for cat, stats in self.category_stats.items()
            },
            avg_processing_time_ms=np.mean(self.processing_times[-1000:]) 
                                   if self.processing_times else 0,
            false_positive_rate=fp_count / max(total_with_feedback, 1),
            false_negative_rate=fn_count / max(total_with_feedback, 1)
        )
    
    def generate_report(self) -> Dict:
        """Generate comprehensive moderation report."""
        metrics = self.calculate_metrics()
        
        top_categories = sorted(
            self.category_stats.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )[:5]
        
        recent_feedback = [
            f for f in self.feedback_data
            if f["timestamp"] > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "summary": {
                "total_processed_24h": metrics.total_processed,
                "block_rate": metrics.blocked_count / max(metrics.total_processed, 1),
                "false_positive_rate": metrics.false_positive_rate,
                "false_negative_rate": metrics.false_negative_rate,
                "avg_latency_ms": metrics.avg_processing_time_ms
            },
            "top_categories": {
                cat: {
                    "total": stats["total"],
                    "blocked": stats["blocked"],
                    "block_rate": stats["blocked"] / max(stats["total"], 1)
                }
                for cat, stats in top_categories
            },
            "feedback_stats": {
                "total_feedback_24h": len(recent_feedback),
                "from_auto_review": sum(1 for f in recent_feedback if f["feedback_source"] == "auto"),
                "from_human_review": sum(1 for f in recent_feedback if f["feedback_source"] == "human")
            },
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: ModerationMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        if metrics.false_positive_rate > 0.1:
            recommendations.append(
                "High false positive rate detected. Consider raising thresholds "
                "or reviewing category-specific rules."
            )
        
        if metrics.false_negative_rate > 0.05:
            recommendations.append(
                "False negatives detected. Lower thresholds for high-risk categories "
                "or add additional detection layers."
            )
        
        if metrics.avg_processing_time_ms > 200:
            recommendations.append(
                "Moderation latency is high. Consider optimizing model or "
                "implementing caching for repeated content."
            )
        
        return recommendations
```

## Framework Integration

### Integration with OpenAI Moderation API

```python
class OpenAIModerationBridge:
    def __init__(self, api_key: str):
        self.client = OpenAIClient(api_key)
    
    async def check(self, text: str) -> Dict:
        response = await self.client.moderations.create(input=text)
        return response.results[0]
```

### Integration with LlamaGuard

```python
class LlamaGuardIntegrator:
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    async def moderate(self, text: str) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self._parse_output(outputs)
```

## Common Pitfalls

### Pitfall 1: Ignoring Context Leading to False Positives

**Problem**: Blocking legitimate content due to lack of context.

**Solution**: Implement context-aware rules:
```python
# Medical content in educational context vs harmful context
if context.get("intent") == "educational" and content_type == "medical":
    threshold *= 0.8  # Lower threshold = more permissive
```

### Pitfall 2: Not Handling Multilingual Content

**Problem**: Filters work for English but miss other languages.

**Solution**: Implement multilingual classifiers and translation layers:
```python
LANGUAGES = ["en", "es", "fr", "de", "zh", "ar", "ja"]
```

### Pitfall 3: Static Thresholds Not Reflecting Evolution

**Problem**: Content evolves, thresholds don't adapt.

**Solution**: Implement adaptive calibration:
```python
calibrator = AdaptiveThresholdCalibrator(target_fp=0.05, target_fn=0.01)
new_threshold = calibrator.calibrate_category(category)
```

## Research References

1. **OpenAI (2023)** - "GPT-4 Moderation System" - Production moderation architecture.

2. **Markov et al. (2023)** - "Real-Time Content Moderation" - Efficient moderation techniques.

3. **Jiang et al. (2023)** - "Multilingual Content Moderation" - Cross-lingual challenges.

4. **Google (2023)** - "Perspective API" - Industry content scoring approaches.

5. **Meta (2023)** - "BlenderBot Safety" - Open-domain conversation safety.

6. **Kosinski (2023)** - "Machine Accuracy" - On human vs. ML moderation accuracy.

7. **Microsoft (2023)** - "Azure Content Safety" - Enterprise moderation solutions.

8. **DeepMind (2023)** - "Sparrow" - Rule-following for safer AI.

9. **Buchholz & Larrondo (2023)** - "Contextual Content Moderation" - Context-aware approaches.

10. **Gorwa et al. (2023)** - "Algorithmic Content Moderation" - Academic analysis of systems.