# Behavior Tuning Module

## Overview

The `behavior_tuning` module enables fine-tuning models to exhibit specific behaviors, communication styles, and domain expertise. This is useful for customizing models for particular applications while maintaining core capabilities.

## Key Components

### Core Classes

- **`BehaviorFinetuner`**: Specialized fine-tuner for behavior training
  - Manages behavior-specific training
  - Tracks behavior consistency
  - Supports style transfer
  - Domain-aware tuning

### Configuration

- **`BehaviorTuningConfig`**: Behavior-specific configuration
  - `behavior_name`: Name of the behavior (e.g., "helpful", "creative")
  - `tone`: Communication tone ("professional", "casual", "formal", "friendly")
  - `domain`: Domain expertise ("medical", "legal", "technical", "creative")
  - `include_examples`: Use behavior examples in training
  - `behavior_consistency_weight`: Weight for consistency loss
  - `style_transfer_mode`: Enable style transfer capabilities

## Supported Behaviors

### Tone Variants

- **Professional**: Formal, accurate, business-appropriate language
- **Casual**: Friendly, conversational tone
- **Formal**: Academic, structured, detailed responses
- **Friendly**: Warm, approachable, encouraging tone

### Domain Expertise

- **Medical**: Healthcare-specific knowledge and terminology
- **Legal**: Law and contracts expertise
- **Technical**: Programming and software knowledge
- **Creative**: Creative writing and brainstorming
- **Academic**: Research and scholarly content
- **General**: Broad knowledge without specialization

## Usage

### Basic Setup

```python
from fine_tuning.behavior_tuning import BehaviorFinetuner, BehaviorTuningConfig

config = BehaviorTuningConfig(
    model_name="gpt2",
    output_dir="./behavior_output",
    behavior_name="helpful_assistant",
    tone="friendly",
    domain="technical",
    include_examples=True,
)

finetuner = BehaviorFinetuner(config)
finetuner.setup_model()
finetuner.setup_optimizer()
finetuner.setup_scheduler()
```

### Adding Behavior Examples

```python
# Define behavior examples
examples = [
    {
        "input": "How do I write a Python function?",
        "output": "Here's a simple Python function example...",
        "behavior_tag": "technical_friendly"
    },
    # More examples...
]

finetuner.add_behavior_examples(examples)
```

### Training

```python
results = finetuner.train(train_dataloader, eval_dataloader)
print(f"Final training loss: {results['training_loss'][-1]}")
```

### Evaluating Behavior Consistency

```python
test_prompts = [
    "What is machine learning?",
    "How do I debug code?",
    "Explain neural networks",
]

consistency = finetuner.evaluate_behavior_consistency(test_prompts)
print(f"Behavior consistency: {consistency:.2%}")
```

## Configuration Examples

### Professional Medical Assistant

```python
config = BehaviorTuningConfig(
    model_name="mistralai/Mistral-7B",
    behavior_name="medical_assistant",
    tone="professional",
    domain="medical",
    num_epochs=5,
    batch_size=16,
)
```

### Creative Writing Assistant

```python
config = BehaviorTuningConfig(
    model_name="mistralai/Mistral-7B",
    behavior_name="creative_writer",
    tone="friendly",
    domain="creative",
    include_examples=True,
    style_transfer_mode=True,
)
```

### Legal Document Analyzer

```python
config = BehaviorTuningConfig(
    model_name="meta-llama/Llama-2-7b",
    behavior_name="legal_advisor",
    tone="formal",
    domain="legal",
    behavior_consistency_weight=0.7,
)
```

## Training Behaviors

### Behavior Consistency Loss

Maintains consistency with target behavior throughout training:

```python
total_loss = base_loss + weight * consistency_loss
```

### Style-Specific Training

Adjust training for different communication styles:

- Casual: Higher tolerance for variation
- Professional: Strict adherence to standards
- Formal: Detailed, structured responses
- Friendly: Warm, encouraging language

## Best Practices

### 1. Use Representative Examples

```python
# Good: Diverse examples covering the behavior
examples = [
    {"input": "...", "output": "...", "tone": "friendly"},
    {"input": "...", "output": "...", "tone": "friendly"},
    # Multiple examples per style
]
```

### 2. Balance Behavior and Base Knowledge

```python
# Adjust consistency weight based on importance
config.behavior_consistency_weight = 0.5  # Balance with base knowledge
```

### 3. Monitor Consistency During Training

```python
# Evaluate behavior regularly
for epoch in range(num_epochs):
    finetuner.train_epoch()
    consistency = finetuner.evaluate_behavior_consistency(test_prompts)
    print(f"Epoch {epoch} consistency: {consistency:.2%}")
```

### 4. Validate Against Behavior Guidelines

```python
# Define what good behavior looks like
behavior_guidelines = {
    "medical_assistant": [
        "Always recommend consulting a doctor",
        "Use medical terminology accurately",
        "Avoid diagnosis without expertise",
    ]
}
```

## Common Behaviors

### Helpful Assistant

Focused on providing useful, accurate information

```python
config = BehaviorTuningConfig(
    behavior_name="helpful_assistant",
    tone="friendly",
    behavior_consistency_weight=0.6,
)
```

### Expert Advisor

Provides specialized expertise in a domain

```python
config = BehaviorTuningConfig(
    behavior_name="expert_advisor",
    tone="professional",
    domain="technical",
    behavior_consistency_weight=0.8,
)
```

### Coding Tutor

Teaches programming with patience and clarity

```python
config = BehaviorTuningConfig(
    behavior_name="coding_tutor",
    tone="friendly",
    domain="technical",
    include_examples=True,
)
```

## Advanced Features

### Style Transfer

Transfer behavior between different tones:

```python
config.style_transfer_mode = True
# Trains on transforming content between tone variants
```

### Multi-Behavior Training

Train on multiple behaviors simultaneously:

```python
config.target_behaviors = [
    "helpful",
    "accurate",
    "concise",
]
```

## Integration

- **Base**: Uses `BaseFinetuner` and configuration
- **LoRA**: Can be combined with LoRA for efficient training
- **QLoRA**: Use with quantization for large models
- **Templates**: Pre-defined behavior configurations

## Evaluation Metrics

Common metrics for behavior evaluation:

- **Consistency**: How consistently the behavior is exhibited
- **Accuracy**: Correctness of information
- **Tone Match**: Alignment with target tone
- **Domain Accuracy**: Correctness in domain-specific context

## See Also

- [Base Module](../base/README.md)
- [LoRA Module](../lora/README.md)
- [Configuration Module](../configs/README.md)
