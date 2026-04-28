# Privacy Preservation in LLM Systems

## Problem Statement

LLM systems handle vast amounts of potentially sensitive data, creating significant privacy challenges. Users may inadvertently share personal information in prompts, organizations need to process sensitive documents, and regulatory frameworks like GDPR and CCPA impose strict requirements on how personal data must be handled. A privacy breach in an LLM system can result in regulatory fines, reputational damage, and legal liability.

The challenge is that LLMs learn from training data and may inadvertently memorize and later reproduce personal information. Additionally, traditional privacy techniques like anonymization are harder to apply to free-form text, and the complexity of LLM architectures makes it difficult to prove that privacy protections are effective.

This skill covers understanding the privacy landscape for LLM systems, implementing privacy-preserving techniques, building compliant data handling pipelines, conducting privacy audits, and balancing privacy with model utility.

## Theory & Fundamentals

### Privacy Regulation Framework

```
Key Privacy Regulations:

GDPR (EU):
├── Data minimization principle
├── Purpose limitation
├── Right to erasure ("right to be forgotten")
├── Data protection impact assessment required
├── Consent requirements for data processing
└── Privacy by design requirements

CCPA (California):
├── Right to know what personal information is collected
├── Right to delete personal information
├── Right to opt-out of sale
├── Non-discrimination for exercising rights
└── Private right of action for data breaches

HIPAA (US Healthcare):
├── Protected Health Information (PHI) protection
├── Minimum necessary standard
├── Technical safeguards required
├── Business associate agreements
└── Breach notification requirements

LGPD (Brazil), PIPEDA (Canada), etc.:
├── Similar principles with regional variations
└── Often based on GDPR framework
```

### Privacy Risks in LLM Systems

**Training Data Privacy Risks**:

1. **Memorization**: LLMs can memorize and later reproduce training data verbatim
   - Risk is higher for rare patterns, repeated content, or specially crafted data
   
2. **Extraction Attacks**: Attackers can prompt models to reveal training data
   - Particularly concerning for sensitive documents in training set
   
3. **Model Inversion**: Statistical attacks to reconstruct training data from model

**Inference-Time Privacy Risks**:

1. **Prompt Leakage**: User prompts may contain sensitive information
2. **Output Contamination**: Outputs may inadvertently include PII from training
3. **Cross-User Inference**: Analyzing outputs to infer information about other users

### Privacy-Preserving ML Techniques

**Differential Privacy (DP)**:
Provides mathematical guarantee that individual data points cannot be identified:

$$P(M(D) \in S) \leq e^\epsilon \cdot P(M(D') \in S)$$

For any neighboring datasets D and D' differing in one record, and any subset S.

**Federated Learning**:
Training on decentralized data without centralizing it:

```
Client 1 → Model Update (no raw data)
Client 2 → Model Update (no raw data)
Client N → Model Update (no raw data)
                ↓
        Aggregation Server
                ↓
           Global Model
```

**Secure Multi-Party Computation (SMPC)**:
Multiple parties compute on combined data without revealing inputs

**Homomorphic Encryption**:
Compute on encrypted data directly

## Implementation Patterns

### Pattern 1: PII Detection and Redaction Pipeline

```python
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIType(Enum):
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PERSONAL_NAME = "personal_name"
    DATE_OF_BIRTH = "dob"
    ADDRESS = "address"
    MEDICAL_RECORD = "mrn"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"

@dataclass
class PIIDetectionResult:
    original_text: str
    redacted_text: str
    detected_entities: List[Dict]
    risk_score: float
    requires_audit: bool

@dataclass
class EntityReplacement:
    entity_type: PIIType
    original_value: str
    replacement_value: str
    confidence: float
    start_pos: int
    end_pos: int

class PIIRedactionPipeline:
    """
    Comprehensive PII detection and redaction pipeline.
    """
    
    def __init__(
        self,
        use_spacy: bool = True,
        use_presidio: bool = True,
        custom_patterns: Optional[Dict[str, str]] = None
    ):
        self.nlp = spacy.load("en_core_web_lg") if use_spacy else None
        self.presidio_analyzer = AnalyzerEngine() if use_presidio else None
        self.presidio_anonymizer = AnonymizerEngine()
        
        self.regex_patterns = self._build_regex_patterns()
        if custom_patterns:
            self.regex_patterns.update(custom_patterns)
        
        self.entity_replacements: Dict[str, str] = {
            PIIType.SSN: "[SSN_REDACTED]",
            PIIType.EMAIL: "[EMAIL_REDACTED]",
            PIIType.PHONE: "[PHONE_REDACTED]",
            PIIType.CREDIT_CARD: "[CARD_REDACTED]",
            PIIType.IP_ADDRESS: "[IP_REDACTED]",
            PIIType.PERSONAL_NAME: "[NAME_REDACTED]",
            PIIType.DATE_OF_BIRTH: "[DOB_REDACTED]",
            PIIType.ADDRESS: "[ADDRESS_REDACTED]",
            PIIType.MEDICAL_RECORD: "[MRN_REDACTED]",
            PIIType.PASSPORT: "[PASSPORT_REDACTED]",
            PIIType.DRIVER_LICENSE: "[DL_REDACTED]"
        }
        
        self.audit_required_patterns = {
            PIIType.SSN: 0.9,
            PIIType.CREDIT_CARD: 0.9,
            PIIType.PASSPORT: 0.9,
            PIIType.MEDICAL_RECORD: 0.95
        }
    
    def _build_regex_patterns(self) -> Dict[str, str]:
        """Build regex patterns for structured PII."""
        return {
            PIIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            PIIType.DATE_OF_BIRTH: r'\b(?:(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}|(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01]))\b',
            PIIType.PASSPORT: r'\b[A-Z]{1,2}\d{6,9}\b',
            PIIType.DRIVER_LICENSE: r'\b[A-Z]{1,2}\d{5,8}\b'
        }
    
    def detect_and_redact(
        self,
        text: str,
        return_replacements: bool = False
    ) -> PIIDetectionResult:
        """
        Detect and redact PII from text.
        """
        replacements = []
        redacted_text = text
        all_entities = []
        
        regex_entities = self._detect_regex(text)
        all_entities.extend(regex_entities)
        
        if self.presidio_analyzer:
            presidio_entities = self._detect_presidio(text)
            all_entities.extend(presidio_entities)
        
        if self.nlp:
            nlp_entities = self._detect_spacy(text)
            all_entities.extend(nlp_entities)
        
        all_entities = self._merge_overlapping_entities(all_entities)
        
        all_entities.sort(key=lambda x: x.start_pos, reverse=True)
        
        for entity in all_entities:
            if entity.start_pos < len(redacted_text):
                redacted_text = (
                    redacted_text[:entity.start_pos] +
                    entity.replacement_value +
                    redacted_text[entity.end_pos:]
                )
                replacements.append(entity)
        
        risk_score = self._calculate_risk_score(all_entities)
        requires_audit = self._check_audit_requirement(all_entities)
        
        result = PIIDetectionResult(
            original_text=text,
            redacted_text=redacted_text,
            detected_entities=[
                {
                    "type": e.entity_type.value,
                    "original": e.original_value,
                    "confidence": e.confidence
                }
                for e in replacements
            ],
            risk_score=risk_score,
            requires_audit=requires_audit
        )
        
        if return_replacements:
            return result, replacements
        
        return result
    
    def _detect_regex(self, text: str) -> List[EntityReplacement]:
        """Detect PII using regex patterns."""
        entities = []
        
        for pii_type, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, text):
                replacement_value = self.entity_replacements.get(pii_type, f"[{pii_type.value.upper()}]")
                
                entities.append(EntityReplacement(
                    entity_type=pii_type,
                    original_value=match.group(),
                    replacement_value=replacement_value,
                    confidence=0.99,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return entities
    
    def _detect_presidio(self, text: str) -> List[EntityReplacement]:
        """Detect PII using Microsoft Presidio."""
        entities = []
        
        results = self.presidio_analyzer.analyze(text=text, language='en')
        
        type_mapping = {
            "PERSON": PIIType.PERSONAL_NAME,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "DATE_TIME": PIIType.DATE_OF_BIRTH,
            "US_PASSPORT": PIIType.PASSPORT,
            "US_DRIVER_LICENSE": PIIType.DRIVER_LICENSE
        }
        
        for result in results:
            entity_type = PIIType(result.entity_type.value) if result.entity_type.value in [e.value for e in PIIType] else None
            
            if entity_type:
                mapped_type = type_mapping.get(result.entity_type.value, entity_type)
            else:
                mapped_type = PIIType.PERSONAL_NAME
            
            entities.append(EntityReplacement(
                entity_type=mapped_type,
                original_value=text[result.start:result.end],
                replacement_value=self.entity_replacements.get(mapped_type, "[PII_REDACTED]"),
                confidence=result.score,
                start_pos=result.start,
                end_pos=result.end
            ))
        
        return entities
    
    def _detect_spacy(self, text: str) -> List[EntityReplacement]:
        """Detect named entities using spaCy."""
        entities = []
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(EntityReplacement(
                    entity_type=PIIType.PERSONAL_NAME,
                    original_value=ent.text,
                    replacement_value="[NAME_REDACTED]",
                    confidence=0.85,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                ))
            elif ent.label_ == "GPE":
                entities.append(EntityReplacement(
                    entity_type=PIIType.ADDRESS,
                    original_value=ent.text,
                    replacement_value="[LOCATION_REDACTED]",
                    confidence=0.80,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char
                ))
        
        return entities
    
    def _merge_overlapping_entities(
        self,
        entities: List[EntityReplacement]
    ) -> List[EntityReplacement]:
        """Merge overlapping entity detections."""
        if not entities:
            return []
        
        entities.sort(key=lambda x: x.start_pos)
        
        merged = [entities[0]]
        
        for current in entities[1:]:
            last = merged[-1]
            
            if current.start_pos <= last.end_pos:
                merged[-1] = EntityReplacement(
                    entity_type=last.entity_type if last.confidence > current.confidence else current.entity_type,
                    original_value=last.original_value,
                    replacement_value=last.replacement_value,
                    confidence=max(last.confidence, current.confidence),
                    start_pos=min(last.start_pos, current.start_pos),
                    end_pos=max(last.end_pos, current.end_pos)
                )
            else:
                merged.append(current)
        
        return merged
    
    def _calculate_risk_score(self, entities: List[EntityReplacement]) -> float:
        """Calculate overall PII risk score."""
        if not entities:
            return 0.0
        
        weights = {
            PIIType.SSN: 1.0,
            PIIType.CREDIT_CARD: 1.0,
            PIIType.PASSPORT: 1.0,
            PIIType.MEDICAL_RECORD: 0.95,
            PIIType.DRIVER_LICENSE: 0.9,
            PIIType.SELF_HARM: 0.9,
            PIIType.EMAIL: 0.6,
            PIIType.PHONE: 0.6,
            PIIType.PERSONAL_NAME: 0.5,
            PIIType.ADDRESS: 0.7,
            PIIType.DATE_OF_BIRTH: 0.5
        }
        
        total_weight = 0.0
        max_weight = 0.0
        
        for entity in entities:
            weight = weights.get(entity.entity_type, 0.5)
            total_weight += weight * entity.confidence
            max_weight += weight
        
        return min(total_weight / max(max_weight, 1), 1.0)
    
    def _check_audit_requirement(self, entities: List[EntityReplacement]) -> bool:
        """Check if any entity requires audit."""
        for entity in entities:
            threshold = self.audit_required_patterns.get(entity.entity_type, 0.0)
            if entity.confidence >= threshold:
                return True
        return False
    
    def redact_for_storage(
        self,
        text: str,
        preserve_format: bool = True
    ) -> str:
        """
        Redact PII for safe storage.
        Optionally preserve format for certain types.
        """
        result = self.detect_and_redact(text)
        
        if preserve_format:
            redacted = result.redacted_text
            redacted = re.sub(
                r'\[SSN_REDACTED\]',
                'XXX-XX-XXXX',
                redacted
            )
            redacted = re.sub(
                r'\[PHONE_REDACTED\]',
                '(XXX) XXX-XXXX',
                redacted
            )
            redacted = re.sub(
                r'\[EMAIL_REDACTED\]',
                'XXXX@XXXX.XXX',
                redacted
            )
            return redacted
        
        return result.redacted_text
```

### Pattern 2: Differential Privacy for Fine-tuning

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class DPConfig:
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    secure_rng: bool = False

class DPFineTuner:
    """
    Differential privacy mechanisms for LLM fine-tuning.
    Implements DP-SGD with gradient clipping and noise addition.
    """
    
    def __init__(self, model: nn.Module, config: DPConfig):
        self.model = model
        self.config = config
        self.global_step = 0
        
        self.noise_generator = self._init_noise_generator()
    
    def _init_noise_generator(self):
        """Initialize noise generator for differential privacy."""
        if self.config.secure_rng:
            import secrets
            seed = secrets.randbelow(2**32)
        else:
            seed = 42
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
    
    def compute_gradient_noise(
        self,
        grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Add calibrated noise to gradients for differential privacy.
        
        Noise scale: C * sigma where C is the clipping norm and sigma
        is derived from (epsilon, delta) using the privacy accountant.
        """
        sigma = self._compute_noise_std()
        
        noise = torch.normal(
            mean=0,
            std=sigma * self.config.max_grad_norm,
            size=grad.shape,
            generator=self.noise_generator,
            device=grad.device,
            dtype=grad.dtype
        )
        
        return grad + noise
    
    def _compute_noise_std(self) -> float:
        """
        Compute noise standard deviation based on DP parameters.
        
        Uses approximate formula for Gaussian mechanism:
        sigma = c * sqrt(2 * log(1.25/delta)) / epsilon
        
        where c is the L2 sensitivity (max_grad_norm).
        """
        c = self.config.max_grad_norm
        
        if self.config.noise_multiplier > 0:
            return c * self.config.noise_multiplier
        
        return c * np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.epsilon
    
    def clip_gradients(
        self,
        grads: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Clip gradients to maximum norm.
        
        Per-sample gradient clipping for DP-SGD:
        ||grad_i|| <= C for all i
        """
        clipped = {}
        
        for name, grad in grads.items():
            if grad is None:
                continue
            
            grad_norm = torch.norm(grad)
            
            clip_factor = self.config.max_grad_norm / grad_norm
            
            if clip_factor < 1:
                clipped[name] = grad * clip_factor
            else:
                clipped[name] = grad
        
        return clipped
    
    def compute_private_gradients(
        self,
        loss: torch.Tensor,
        per_sample_grads: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute differentially private gradients.
        """
        if per_sample_grads:
            grads = self._compute_per_sample_gradients(loss)
        else:
            grads = {name: p.grad for name, p in self.model.named_parameters() if p.grad is not None}
        
        clipped_grads = self.clip_gradients(grads)
        
        noisy_grads = {
            name: self.compute_gradient_noise(grad)
            for name, grad in clipped_grads.items()
        }
        
        return noisy_grads
    
    def _compute_per_sample_gradients(
        self,
        loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-sample gradients for DP-SGD.
        
        For a batch of n samples, computes n gradients and returns
        them concatenated for clipping.
        """
        self.model.zero_grad()
        
        batch_size = loss.shape[0] if loss.dim() > 0 else 1
        
        per_sample_grads = {name: [] for name, _ in self.model.named_parameters()}
        
        for i in range(batch_size):
            self.model.zero_grad()
            
            sample_loss = loss[i] if loss.dim() > 0 else loss
            
            if sample_loss.dim() > 0:
                sample_loss = sample_loss.mean()
            
            sample_loss.backward(retain_graph=True)
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    per_sample_grads[name].append(param.grad.detach().clone())
        
        for name in per_sample_grads:
            per_sample_grads[name] = torch.stack(per_sample_grads[name])
        
        return per_sample_grads
    
    def apply_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        noisy_grads: Dict[str, torch.Tensor]
    ):
        """
        Apply noisy gradients to model parameters.
        """
        for name, param in self.model.named_parameters():
            if name in noisy_grads and param.grad is not None:
                param.grad = noisy_grads[name]
        
        optimizer.step()
        self.global_step += 1
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """
        Calculate privacy budget spent using RDP accountant.
        
        Returns epsilon spent so far.
        """
        from scipy import special
        
        q = 1.0
        
        steps = self.global_step
        sigma = self._compute_noise_std()
        
        if sigma == 0 or steps == 0:
            return {"epsilon": 0.0, "delta": self.config.delta, "steps": steps}
        
        alpha_range = np.arange(2, 256)
        
        rdp = np.zeros_like(alpha_range, dtype=float)
        
        for i, alpha in enumerate(alpha_range):
            rdp[i] = self._compute_rdp(q, sigma, alpha)
        
        eps = rdp + np.log(self.config.delta) / (alpha_range - 1)
        
        best_eps = np.min(eps)
        
        return {
            "epsilon": float(best_eps),
            "delta": self.config.delta,
            "steps": self.global_step,
            "sigma": sigma
        }
    
    def _compute_rdp(self, q: float, sigma: float, alpha: float) -> float:
        """
        Compute Renyi Differential Privacy for Gaussian mechanism.
        """
        from scipy import special
        
        if alpha == float('inf'):
            return float('inf')
        
        log_term = (alpha - 1) * (1 / (2 * sigma**2))
        
        return alpha * (1 / (2 * sigma**2)) + (alpha - 1) * log_term
```

### Pattern 3: Privacy-Preserving Prompt Handling

```python
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib
import asyncio
from datetime import datetime, timedelta

@dataclass
class PrivacyContext:
    user_id: Optional[str]
    data_classification: str  # "public", "internal", "confidential", "restricted"
    consent_level: str  # "none", "limited", "full"
    retention_period_days: int
    requires_encryption: bool
    geographic_restrictions: List[str]

class PrivacyPreservingPromptProcessor:
    """
    Handles prompts with privacy considerations.
    """
    
    def __init__(
        self,
        pii_redactor: PIIRedactionPipeline,
        consent_manager,
        encryption_service
    ):
        self.redactor = pii_redactor
        self.consent = consent_manager
        self.encryption = encryption_service
        
        self.processing_history: List[Dict] = []
        
        self.restricted_data_types = {
            "health_data": ["diagnosis", "treatment", "medication", "patient"],
            "financial_data": ["account", "ssn", "routing", "investment", "balance"],
            "biometric_data": ["fingerprint", "face", "voice", "iris"],
            "genetic_data": ["gene", "dna", "chromosome", "genetic"]
        }
        
        self.compliance_rules = {
            "gdpr": self._gdpr_compliance_check,
            "ccpa": self._ccpa_compliance_check,
            "hipaa": self._hipaa_compliance_check
        }
    
    async def process(
        self,
        prompt: str,
        context: PrivacyContext,
        framework: str = "gdpr"
    ) -> Tuple[str, Dict]:
        """
        Process prompt with privacy protections.
        """
        privacy_result = {
            "processed_at": datetime.utcnow().isoformat(),
            "data_classification": context.data_classification,
            "actions_taken": [],
            "compliance_status": {},
            "warnings": []
        }
        
        if framework in self.compliance_rules:
            compliance_result = self.compliance_rules[framework](prompt, context)
            privacy_result["compliance_status"] = compliance_result
            if not compliance_result["compliant"]:
                privacy_result["warnings"].append(f"Compliance issue: {compliance_result['reason']}")
        
        consent_ok = await self._check_consent(context)
        if not consent_ok:
            raise PrivacyViolationError("Insufficient consent for processing")
        
        pii_result = self.redactor.detect_and_redact(prompt)
        privacy_result["actions_taken"].append(f"PII_redaction: {len(pii_result.detected_entities)} entities")
        
        if pii_result.requires_audit:
            privacy_result["warnings"].append("High-risk PII detected, audit required")
        
        processed_prompt = pii_result.redacted_text
        
        restricted_found = self._check_restricted_data(processed_prompt)
        if restricted_found:
            privacy_result["warnings"].append(f"Restricted data categories: {restricted_found}")
        
        self.processing_history.append(privacy_result)
        
        return processed_prompt, privacy_result
    
    def _gdpr_compliance_check(
        self,
        prompt: str,
        context: PrivacyContext
    ) -> Dict:
        """Check GDPR compliance requirements."""
        violations = []
        
        if context.data_classification == "restricted":
            if context.consent_level not in ["limited", "full"]:
                violations.append("Restricted data requires explicit consent")
        
        if len(prompt) > 10000:
            violations.append("Data minimization: prompt exceeds necessary length")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "regulation": "gdpr"
        }
    
    def _ccpa_compliance_check(
        self,
        prompt: str,
        context: PrivacyContext
    ) -> Dict:
        """Check CCPA compliance requirements."""
        violations = []
        
        if context.requires_encryption and not context.requires_encryption:
            violations.append("Encryption required but not enabled")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "regulation": "ccpa"
        }
    
    def _hipaa_compliance_check(
        self,
        prompt: str,
        context: PrivacyContext
    ) -> Dict:
        """Check HIPAA compliance requirements."""
        violations = []
        
        medical_keywords = ["patient", "diagnosis", "treatment", "medical", "health", "clinical"]
        
        if any(kw in prompt.lower() for kw in medical_keywords):
            if context.data_classification not in ["confidential", "restricted"]:
                violations.append("PHI requires confidential classification")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "regulation": "hipaa"
        }
    
    async def _check_consent(self, context: PrivacyContext) -> bool:
        """Check if consent is sufficient for processing."""
        if context.data_classification == "public":
            return True
        
        if context.consent_level == "full":
            return True
        
        if context.data_classification == "internal":
            return context.consent_level in ["limited", "full"]
        
        return context.consent_level == "full"
    
    def _check_restricted_data(self, prompt: str) -> List[str]:
        """Check for restricted data categories."""
        found = []
        prompt_lower = prompt.lower()
        
        for category, keywords in self.restricted_data_types.items():
            if any(kw in prompt_lower for kw in keywords):
                found.append(category)
        
        return found
```

### Pattern 4: Privacy Audit Logging System

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    data_subject_id: Optional[str]
    data_categories: List[str]
    processing_purpose: str
    legal_basis: str
    data_hash: str
    action_taken: str
    retention_period: int
    metadata: Dict = field(default_factory=dict)

class PrivacyAuditLogger:
    """
    Comprehensive audit logging for privacy compliance.
    """
    
    def __init__(
        self,
        storage_backend,
        retention_days: int = 2555
    ):
        self.storage = storage_backend
        self.retention_days = retention_days
        
        self.event_types = {
            "data_access": "DATA_ACCESS",
            "data_processing": "DATA_PROCESSING",
            "data_deletion": "DATA_DELETION",
            "consent_change": "CONSENT_CHANGE",
            "pii_detected": "PII_DETECTED",
            "pii_redacted": "PII_REDACTED",
            "data_export": "DATA_EXPORT",
            "breach_detected": "BREACH_DETECTED",
            "right_to_erasure": "RIGHT_TO_ERASURE"
        }
    
    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        data_subject_id: Optional[str] = None,
        data_categories: Optional[List[str]] = None,
        processing_purpose: str = "",
        legal_basis: str = "",
        data_content: str = "",
        action_taken: str = "",
        retention_period: int = 365,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a privacy-relevant event.
        Returns an event ID for reference.
        """
        data_hash = hashlib.sha256(data_content.encode()).hexdigest()[:16] if data_content else ""
        
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=user_id,
            data_subject_id=data_subject_id,
            data_categories=data_categories or [],
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_hash=data_hash,
            action_taken=action_taken,
            retention_period=retention_period,
            metadata=metadata or {}
        )
        
        event_id = self._generate_event_id(event)
        
        self.storage.store("privacy_audit", {
            "event_id": event_id,
            **event.__dict__
        })
        
        return event_id
    
    def log_data_access(
        self,
        user_id: str,
        data_subject_id: str,
        data_categories: List[str],
        purpose: str,
        legal_basis: str
    ):
        """Log access to personal data."""
        return self.log_event(
            event_type=self.event_types["data_access"],
            user_id=user_id,
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            processing_purpose=purpose,
            legal_basis=legal_basis,
            action_taken="data_provided"
        )
    
    def log_pii_processing(
        self,
        user_id: Optional[str],
        pii_types: List[str],
        action: str,
        metadata: Optional[Dict] = None
    ):
        """Log PII detection and processing."""
        return self.log_event(
            event_type=self.event_types["pii_detected"],
            user_id=user_id,
            data_categories=pii_types,
            processing_purpose="content_moderation",
            legal_basis="legitimate_interest",
            action_taken=action,
            metadata=metadata or {}
        )
    
    def log_right_to_erasure(
        self,
        data_subject_id: str,
        data_deleted: bool,
        records_affected: int
    ):
        """Log right to erasure request fulfillment."""
        return self.log_event(
            event_type=self.event_types["right_to_erasure"],
            data_subject_id=data_subject_id,
            data_categories=["all_personal_data"],
            processing_purpose="erasure_request",
            legal_basis="gdpr_article_17",
            action_taken="data_deleted" if data_deleted else "deletion_in_progress",
            metadata={"records_affected": records_affected}
        )
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        data_subject_id: Optional[str] = None
    ) -> Dict:
        """
        Generate a compliance report for a specific time period.
        """
        events = self.storage.query(
            "privacy_audit",
            start=start_date,
            end=end_date,
            filters={"data_subject_id": data_subject_id} if data_subject_id else None
        )
        
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(events),
            "events_by_type": self._count_by_type(events),
            "data_categories_accessed": self._unique_categories(events),
            "users_who_accessed": self._unique_users(events),
            "pii_events": self._count_pii_events(events),
            "erasure_requests": self._count_erasure_requests(events)
        }
        
        return report
    
    def _generate_event_id(self, event: AuditEvent) -> str:
        """Generate unique event ID."""
        content = f"{event.timestamp}{event.event_type}{event.user_id or ''}{event.data_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:16].upper()
    
    def _count_by_type(self, events: List[Dict]) -> Dict[str, int]:
        """Count events by type."""
        counts = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
    
    def _unique_categories(self, events: List[Dict]) -> Set[str]:
        """Get unique data categories accessed."""
        categories = set()
        for event in events:
            for cat in event.get("data_categories", []):
                categories.add(cat)
        return categories
    
    def _unique_users(self, events: List[Dict]) -> Set[str]:
        """Get unique users who accessed data."""
        users = set()
        for event in events:
            if event.get("user_id"):
                users.add(event["user_id"])
        return users
    
    def _count_pii_events(self, events: List[Dict]) -> int:
        """Count PII-related events."""
        return sum(
            1 for e in events
            if e.get("event_type") in [self.event_types["pii_detected"], self.event_types["pii_redacted"]]
        )
    
    def _count_erasure_requests(self, events: List[Dict]) -> int:
        """Count right to erasure requests."""
        return sum(
            1 for e in events
            if e.get("event_type") == self.event_types["right_to_erasure"]
        )
```

### Pattern 5: Data Retention and Erasure Manager

```python
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

@dataclass
class RetentionPolicy:
    data_category: str
    retention_days: int
    deletion_method: str  # "hard", "soft", "anonymized"
    requires_audit: bool

class DataRetentionManager:
    """
    Manages data retention and erasure for privacy compliance.
    """
    
    def __init__(
        self,
        storage_backend,
        audit_logger: PrivacyAuditLogger
    ):
        self.storage = storage_backend
        self.audit = audit_logger
        
        self.default_policies = [
            RetentionPolicy("conversation_logs", 90, "soft", True),
            RetentionPolicy("prompts", 30, "anonymized", False),
            RetentionPolicy("outputs", 30, "anonymized", False),
            RetentionPolicy("pii_logs", 2555, "hard", True),
            RetentionPolicy("audit_logs", 2555, "hard", True),
            RetentionPolicy("model_interactions", 365, "anonymized", False)
        ]
    
    async def enforce_retention(
        self,
        data_category: str = None
    ) -> Dict:
        """
        Enforce retention policies, deleting expired data.
        """
        results = {
            "deleted_records": 0,
            "anonymized_records": 0,
            "errors": [],
            "policies_enforced": []
        }
        
        policies_to_enforce = (
            [p for p in self.default_policies if p.data_category == data_category]
            if data_category else self.default_policies
        )
        
        for policy in policies_to_enforce:
            try:
                result = await self._enforce_policy(policy)
                results["deleted_records"] += result.get("deleted", 0)
                results["anonymized_records"] += result.get("anonymized", 0)
                results["policies_enforced"].append(policy.data_category)
            except Exception as e:
                results["errors"].append({
                    "policy": policy.data_category,
                    "error": str(e)
                })
        
        return results
    
    async def _enforce_policy(
        self,
        policy: RetentionPolicy
    ) -> Dict:
        """Enforce a single retention policy."""
        cutoff = datetime.utcnow() - timedelta(days=policy.retention_days)
        
        expired_records = self.storage.query(
            policy.data_category,
            before=cutoff,
            deleted=False
        )
        
        if policy.deletion_method == "hard":
            count = await self._hard_delete(policy.data_category, expired_records)
            return {"deleted": count}
        
        elif policy.deletion_method == "soft":
            count = await self._soft_delete(policy.data_category, expired_records)
            return {"deleted": count}
        
        elif policy.deletion_method == "anonymized":
            count = await self._anonymize(policy.data_category, expired_records)
            return {"anonymized": count}
        
        return {}
    
    async def _hard_delete(
        self,
        category: str,
        records: List[Dict]
    ) -> int:
        """Permanently delete records."""
        count = 0
        for record in records:
            self.storage.delete(category, record["id"])
            self.audit.log_event(
                event_type="data_deletion",
                user_id=record.get("user_id"),
                data_subject_id=record.get("data_subject_id"),
                data_categories=[category],
                processing_purpose="retention_policy_enforcement",
                legal_basis="gdpr_article_5",
                action_taken="permanent_deletion"
            )
            count += 1
        
        return count
    
    async def _soft_delete(
        self,
        category: str,
        records: List[Dict]
    ) -> int:
        """Soft delete records (mark as deleted but retain)."""
        count = 0
        for record in records:
            self.storage.update(
                category,
                record["id"],
                {"deleted": True, "deleted_at": datetime.utcnow().isoformat()}
            )
            count += 1
        
        return count
    
    async def _anonymize(
        self,
        category: str,
        records: List[Dict]
    ) -> int:
        """Anonymize records by removing PII."""
        count = 0
        for record in records:
            anonymized = self._anonymize_record(record)
            self.storage.update(category, record["id"], anonymized)
            self.audit.log_event(
                event_type="data_deletion",
                user_id=None,
                data_subject_id=record.get("data_subject_id"),
                data_categories=[category],
                processing_purpose="retention_policy_enforcement",
                legal_basis="gdpr_article_5",
                action_taken="anonymization"
            )
            count += 1
        
        return count
    
    def _anonymize_record(self, record: Dict) -> Dict:
        """Anonymize a single record."""
        anonymized = record.copy()
        
        pii_fields = ["user_id", "email", "name", "phone", "ip_address"]
        for field in pii_fields:
            if field in anonymized:
                anonymized[field] = f"ANONYMIZED_{hashlib.md5(str(record['id']).encode()).hexdigest()[:8]}"
        
        return anonymized
    
    async def process_erasure_request(
        self,
        data_subject_id: str,
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        Process a right to erasure (GDPR Article 17) request.
        """
        categories = categories or [p.data_category for p in self.default_policies]
        
        results = {
            "data_subject_id": data_subject_id,
            "records_deleted": 0,
            "records_anonymized": 0,
            "categories_processed": [],
            "errors": []
        }
        
        for category in categories:
            try:
                records = self.storage.query(
                    category,
                    filters={"data_subject_id": data_subject_id}
                )
                
                deleted = await self._hard_delete(category, records)
                results["records_deleted"] += deleted
                results["categories_processed"].append(category)
                
            except Exception as e:
                results["errors"].append({
                    "category": category,
                    "error": str(e)
                })
        
        self.audit.log_right_to_erasure(
            data_subject_id=data_subject_id,
            data_deleted=results["records_deleted"] > 0,
            records_affected=results["records_deleted"]
        )
        
        return results
```

## Framework Integration

### Integration with LangChain

```python
from langchain.callbacks import CallbackManager
from langchain.prompts import PromptTemplate

class PrivacyAwarePromptTemplate(PromptTemplate):
    def __init__(self, privacy_processor, **kwargs):
        super().__init__(**kwargs)
        self.privacy = privacy_processor
    
    def format(self, **kwargs):
        formatted = super().format(**kwargs)
        processed, _ = self.privacy.process(
            formatted,
            PrivacyContext(
                user_id=kwargs.get("user_id"),
                data_classification="internal",
                consent_level="full",
                retention_period_days=30,
                requires_encryption=False,
                geographic_restrictions=[]
            )
        )
        return processed
```

## Common Pitfalls

### Pitfall 1: Incomplete PII Detection

**Problem**: Only checking structured fields, missing PII in free text.

**Solution**: Use multiple detection layers:
```python
# Combine regex, NER, and embeddings
results = (
    self._regex_detection() +
    self._ner_detection() +
    self._embedding_similarity_detection()
)
```

### Pitfall 2: Not Maintaining Audit Trails

**Problem**: Processing data without proper logging for compliance.

**Solution**: Log every data access and processing action:
```python
audit.log_data_access(user_id, subject_id, categories, purpose, legal_basis)
```

### Pitfall 3: Assuming One-Shot Consent Is Enough

**Problem**: Not handling consent withdrawal or scope changes.

**Solution**: Implement consent lifecycle management:
```python
# Check consent freshness
if consent.last_updated > timedelta(days=30):
    # Re-verify consent
    consent = await consent_manager.refresh(user_id)
```

## Research References

1. **European Commission (2018)** - "General Data Protection Regulation (GDPR)" - Primary privacy regulation.

2. **Abadi et al. (2016)** - "Deep Learning with Differential Privacy" - DP in neural networks.

3. **McMahan et al. (2018)** - "Learning Differentially Private Recurrent Language Models" - DP for language models.

4. **Nasr et al. (2019)** - "Comprehensive Privacy Analysis of Deep Learning" - memorization and extraction attacks.

5. **Carlini et al. (2019)** - "The Secret Sharer" - Memorization in RNNs.

6. **Brown et al. (2022)** - "Differential Privacy for Privacy-Preserving ML" - Comprehensive overview.

7. **Apple (2023)** - "Privacy-Preserving Machine Learning" - Industry practices.

8. **Kairouz et al. (2021)** - "The Composition of Differential Privacy" - Advanced DP theory.

9. **Rigaki & Garcia (2023)** - "A Survey of Privacy Preserving Techniques" - For ML systems.

10. **Voigt & Von dem Bussche (2017)** - "EU General Data Protection Regulation" - GDPR compliance guide.