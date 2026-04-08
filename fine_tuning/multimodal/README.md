# Multimodal Fine-tuning Module

## Overview

The `multimodal` module provides fine-tuning utilities for vision-language models, video-language models, and other multimodal architectures that process multiple input modalities (images, video, text, audio).

## Key Components

### Core Classes

- **`MultimodalFinetuner`**: Specialized fine-tuner for multimodal models
  - Handles multiple input modalities
  - Supports vision-language alignment
  - Contrastive learning support
  - Flexible fusion mechanisms

### Configuration

- **`MultimodalTuningConfig`**: Multimodal-specific configuration
  - `modality_types`: List of modalities ("vision", "language", "audio", "video")
  - `vision_encoder`: Vision encoder model (CLIP, ViT, etc.)
  - `video_encoder`: Optional video encoder
  - `align_loss_weight`: Weight for alignment loss
  - `fusion_method`: How to fuse modalities ("concat", "cross_attention", "transformer")
  - `freeze_vision_encoder`: Freeze vision encoder weights
  - `freeze_language_encoder`: Freeze language encoder weights

## Supported Modalities

### Vision-Language Models

- CLIP-based models
- BLIP models
- LLaVA models
- InstructBLIP models

### Video-Language Models

- TimeSformer-based models
- ViViT models
- CLIP-based video models

### Audio-Language Models

- Whisper + LLM combinations
- CLAP-based models

## Usage

### Basic Setup for Vision-Language

```python
from fine_tuning.multimodal import MultimodalFinetuner, MultimodalTuningConfig

config = MultimodalTuningConfig(
    model_name="openai/clip-vit-base-patch32",
    output_dir="./multimodal_output",
    modality_types=["vision", "language"],
    vision_encoder="openai/clip-vit-base-patch32",
    fusion_method="cross_attention",
)

finetuner = MultimodalFinetuner(config)
finetuner.setup_model()
finetuner.setup_optimizer()
finetuner.setup_scheduler()
```

### Training

```python
results = finetuner.train(train_dataloader, eval_dataloader)
print(f"Final training loss: {results['training_loss'][-1]}")
print(f"Final eval loss: {results['eval_loss'][-1]}")
```

## Configuration Examples

### CLIP-based Vision-Language Model

```python
config = MultimodalTuningConfig(
    model_name="openai/clip-vit-base-patch32",
    output_dir="./clip_finetuning",
    modality_types=["vision", "language"],
    vision_encoder="openai/clip-vit-base-patch32",
    align_loss_weight=0.5,
    contrastive_loss_weight=0.5,
    image_size=224,
)
```

### Video-Language Model

```python
config = MultimodalTuningConfig(
    model_name="timm/timesformer_base_patch16_224",
    modality_types=["video", "language"],
    video_encoder="timm/timesformer_base_patch16_224",
    max_video_frames=8,
    fusion_method="transformer",
)
```

### Cross-Modal Retrieval

```python
config = MultimodalTuningConfig(
    model_name="facebook/clip-vit-base-patch32",
    align_loss_weight=0.8,
    contrastive_loss_weight=0.2,
    freeze_vision_encoder=False,
    freeze_language_encoder=False,
)
```

## Fusion Methods

### Concatenation

Simple concatenation of modality embeddings:

```python
config.fusion_method = "concat"
# output = concat(vision_embed, language_embed)
```

### Cross-Attention

Cross-attention between modalities:

```python
config.fusion_method = "cross_attention"
# Uses multi-head cross-attention
```

### Transformer Fusion

Full transformer-based fusion:

```python
config.fusion_method = "transformer"
# Uses transformer encoder for fusion
```

## Loss Functions

### Alignment Loss

Aligns embeddings from different modalities:

```python
config.align_loss_weight = 0.5
# MSE or cosine distance between aligned embeddings
```

### Contrastive Loss

Contrastive learning between modalities:

```python
config.contrastive_loss_weight = 0.5
# InfoNCE or triplet loss
```

## Input Formats

### Vision-Language Batch

```python
batch = {
    "images": torch.Tensor(batch_size, 3, H, W),  # Vision input
    "input_ids": torch.LongTensor(batch_size, seq_len),  # Text tokens
    "attention_mask": torch.Tensor(batch_size, seq_len),
}
```

### Video-Language Batch

```python
batch = {
    "video": torch.Tensor(batch_size, frames, 3, H, W),  # Video frames
    "input_ids": torch.LongTensor(batch_size, seq_len),  # Text tokens
    "attention_mask": torch.Tensor(batch_size, seq_len),
}
```

## Best Practices

### 1. Balance Modality Learning

```python
# Ensure both modalities learn effectively
config.align_loss_weight = 0.5
config.contrastive_loss_weight = 0.5
```

### 2. Choose Appropriate Fusion

```python
# Simple tasks: concat
config.fusion_method = "concat"

# Complex tasks: transformer
config.fusion_method = "transformer"
```

### 3. Freeze When Appropriate

```python
# Keep pretrained vision encoder frozen
config.freeze_vision_encoder = True

# Train language encoder
config.freeze_language_encoder = False
```

### 4. Monitor Cross-Modal Alignment

```python
# Track alignment between modalities
for epoch in range(num_epochs):
    finetuner.train_epoch()
    alignment_score = finetuner.evaluate_alignment(val_dataloader)
    print(f"Alignment: {alignment_score:.4f}")
```

## Common Models

### OpenAI CLIP

```python
config = MultimodalTuningConfig(
    model_name="openai/clip-vit-large-patch14",
    modality_types=["vision", "language"],
    image_size=224,
)
```

### Salesforce BLIP

```python
config = MultimodalTuningConfig(
    model_name="Salesforce/blip-image-captioning-large",
    modality_types=["vision", "language"],
)
```

### LLaVA

```python
config = MultimodalTuningConfig(
    model_name="llava-hf/llava-1.5-7b",
    modality_types=["vision", "language"],
    image_size=336,
)
```

## Advanced Features

### Multi-Modal Pretraining

Combine multiple loss functions:

```python
config.align_loss_weight = 0.3
config.contrastive_loss_weight = 0.3
# Additional custom losses can be added
```

### Efficient Fine-tuning

Combine with LoRA for parameter efficiency:

```python
# Use LoRA for vision encoder
config.freeze_vision_encoder = True  # Or use LoRA

# Train language decoder with LoRA
config.freeze_language_encoder = False
```

## Integration

- **Base**: Uses `BaseFinetuner` and configuration
- **LoRA**: Can combine with LoRA for efficient training
- **QLoRA**: Use with quantization for large models

## See Also

- [Base Module](../base/README.md)
- [LoRA Module](../lora/README.md)
- [QLoRA Module](../qlora/README.md)
