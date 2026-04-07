# Models

Model registry, checkpoints, adapters, and merged model management.

## Overview

This module provides organization and metadata for model artifacts:
- Base model references and metadata
- Adapter and LoRA checkpoints
- Merged model manifests
- Exported models (ONNX, TensorRT, GGUF)
- Model registry and version tracking

## Structure

```
models/
├── README.md (this file)
├── base/              # Base model metadata and references
├── adapters/          # LoRA and adapter checkpoints
├── merged/            # Merged model manifests
├── exported/          # ONNX, TensorRT, GGUF exports
└── registry/          # Model registry definitions
```

## Directory Purposes

### `base/` - Base Model References

Metadata for base models without storing actual weights:

```yaml
# models/base/mistral-7b.yaml
name: Mistral-7B
model_id: mistralai/Mistral-7B-v0.1
organization: Mistral AI
size_billions: 7
parameters_billions: 7.24
context_length: 4096
vocab_size: 32000

architecture:
  type: transformer
  num_layers: 32
  hidden_size: 4096
  num_heads: 32
  num_key_value_heads: 8
  intermediate_size: 14336
  
license: Apache 2.0
paper: "https://arxiv.org/abs/2310.06825"
huggingface_url: "https://huggingface.co/mistralai/Mistral-7B-v0.1"

access:
  requires_auth: false
  download_url: "https://huggingface.co/mistralai/Mistral-7B-v0.1"
  
performance:
  mmlu: 63.9  # 5-shot
  hellaswag: 78.9
  winogrande: 77.6
  gsm8k: 27.0
  lambada: 77.5
  arc_challenge: 52.9
  arc_easy: 80.1
  truthfulqa: 42.1
  
recommended_for:
  - instruction_following
  - code_generation
  - general_purpose
  - inference_speed
  - edge_deployment

training_data:
  size_tokens: 1000000000000  # 1T tokens
  sources:
    - Common Crawl
    - Books
    - Code repositories
    - Academic papers
  cutoff_date: "2023-08-01"

quantization_support:
  - int4
  - int8
  - fp8
  - awq
  - gptq
```

### `adapters/` - LoRA and Adapter Checkpoints

Storing trained adapters and fine-tuning checkpoints:

```yaml
# models/adapters/mistral-7b-alpaca-lora.yaml
name: mistral-7b-alpaca-lora
adapter_type: lora
base_model: mistralai/Mistral-7B-v0.1
adapter_size_mb: 64

training:
  method: sft  # supervised fine-tuning
  dataset: alpaca-7k
  epochs: 3
  learning_rate: 5e-4
  batch_size: 16
  lora_rank: 64
  lora_alpha: 16
  lora_dropout: 0.05
  training_time_hours: 2
  
performance:
  alpaca_eval_win_rate: 82.5
  mmlu_improvement: +3.2
  instruction_following: +15.3
  
checkpoint:
  local_path: ./checkpoints/mistral-alpaca-lora
  huggingface_repo: username/mistral-7b-alpaca-lora
  version: 1.0
  
usage:
  from_pretrained: |
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    base_model_name = "mistralai/Mistral-7B-v0.1"
    adapter_name = "username/mistral-7b-alpaca-lora"
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_name)
```

### `merged/` - Merged Model Manifests

Metadata for models with merged adapters:

```yaml
# models/merged/mistral-7b-alpaca-merged.yaml
name: mistral-7b-alpaca-merged
type: merged_model
base_model: mistralai/Mistral-7B-v0.1
adapters:
  - mistral-7b-alpaca-lora

merging:
  method: linear  # linear, slerp, etc.
  alpha: 1.0
  merge_date: "2024-04-07"
  
checkpoint:
  local_path: ./checkpoints/mistral-alpaca-merged
  huggingface_repo: username/mistral-7b-alpaca-merged
  size_gb: 28
  
performance:
  alpaca_eval: 82.5
  inference_speed_tokens_per_second: 45
```

### `exported/` - Exported Models

Metadata for optimized exports:

```yaml
# models/exported/mistral-7b-onnx.yaml
name: mistral-7b-onnx
export_format: onnx
base_model: mistralai/Mistral-7B-v0.1

export_config:
  opset_version: 14
  optimization_level: 3
  quantization: int8
  optimize_model: true
  
artifacts:
  model_file: ./exported/mistral-7b/model.onnx
  tokenizer_file: ./exported/mistral-7b/tokenizer.json
  config_file: ./exported/mistral-7b/config.json
  size_mb: 3400
  
performance:
  inference_speed_cpu: 5  # tokens/sec on CPU
  inference_speed_gpu: 100  # tokens/sec on GPU
  memory_mb_cpu: 8192
  memory_mb_gpu: 4096
  
usage:
  example: |
    import onnxruntime as ort
    sess = ort.InferenceSession("model.onnx")
```

### `registry/` - Model Registry

Central registry of all available models:

```yaml
# models/registry/all_models.yaml
models:
  - id: mistral-7b
    name: Mistral-7B
    type: base
    source: huggingface
    url: mistralai/Mistral-7B-v0.1
    size_gb: 14
    context: 4096
    
  - id: mistral-7b-alpaca-lora
    name: Mistral-7B + Alpaca LoRA
    type: adapter
    base_model: mistral-7b
    source: local
    path: ./checkpoints/mistral-7b-alpaca-lora
    size_mb: 64
    
  - id: llama2-13b
    name: Llama-2-13B
    type: base
    source: huggingface
    url: meta-llama/Llama-2-13b-hf
    size_gb: 26
    context: 4096
    
  - id: llama2-13b-chat
    name: Llama-2-13B-Chat
    type: base
    source: huggingface
    url: meta-llama/Llama-2-13b-chat-hf
    size_gb: 26
    context: 4096
    fine_tuned_for: chat
    performance:
      mt_bench: 7.0
      alpaca_eval: 72.0
```

## Quick Start

### 1. Reference a Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### 2. Load an Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "./checkpoints/mistral-alpaca-lora")
```

### 3. Merge Adapter into Base

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./checkpoints/mistral-alpaca-merged")
```

### 4. Use Exported Model

```python
import onnxruntime as ort

sess = ort.InferenceSession("./exported/mistral-7b/model.onnx")
# Inference with ONNX Runtime
```

## Model Organization Conventions

### Base Model Naming
```
{organization}-{size}b[-{variant}].yaml
Example: mistral-7b.yaml, llama2-13b-chat.yaml
```

### Adapter Naming
```
{base_model}-{dataset}-{method}.yaml
Example: mistral-7b-alpaca-lora.yaml, llama2-13b-code-qlora.yaml
```

### Merged Model Naming
```
{base_model}-{dataset}-merged.yaml
Example: mistral-7b-alpaca-merged.yaml
```

### Exported Model Naming
```
{base_model}-{format}[-{quantization}].yaml
Example: mistral-7b-onnx.yaml, llama2-13b-tensorrt-int8.yaml
```

## Performance Metrics Reference

Standard benchmarks used in model metadata:
- **MMLU**: Multiple-choice knowledge (0-shot, 5-shot)
- **HellaSwag**: Common sense reasoning
- **WinoGrande**: Coreference resolution
- **GSM8K**: Grade school math
- **LAMBADA**: Language modeling
- **ARC**: Science QA (easy and challenge)
- **TruthfulQA**: Truthfulness measurement
- **AlpacaEval**: Instruction-following quality
- **MT-Bench**: Multi-turn conversation quality

## Size Estimation

| Model | Params | FP32 | FP16/BF16 | INT8 | INT4 | LoRA (rank 64) |
|-------|--------|------|-----------|------|------|----------------|
| 7B | 7.24B | 29 GB | 14 GB | 7 GB | 3.5 GB | 64 MB |
| 13B | 13.02B | 52 GB | 26 GB | 13 GB | 6.5 GB | 128 MB |
| 70B | 70.3B | 280 GB | 140 GB | 70 GB | 35 GB | 640 MB |

## Quantization Guide

### INT4 (Recommended for most use cases)
- 4x compression vs FP32
- ~1-2% quality loss
- Best balance of speed and quality
- Tools: BitsAndBytes, AutoGPTQ, AWQ

### INT8
- 4x compression vs FP32
- <1% quality loss
- Slower than INT4
- Native PyTorch support

### FP8
- 4x compression vs FP32
- Minimal quality loss
- Requires modern GPUs (H100, L40S)
- Best for inference

## Managing Local Models

```bash
# Set HuggingFace cache directory
export HF_HOME=./model_cache

# List downloaded models
ls -la $HF_HOME/hub/

# Clear cache
rm -rf $HF_HOME/hub/*
```

## Creating Model Metadata

```bash
cd models/base
cat > new-model.yaml << 'EOF'
name: Model Name
model_id: org/model-name
size_billions: 7
parameters_billions: 7.24
license: Apache 2.0
huggingface_url: https://huggingface.co/org/model-name
EOF
```

## References

- **Model Hub**: [HuggingFace Model Hub](https://huggingface.co/models)
- **Quantization Guide**: See `../inference/quantization/README.md`
- **LoRA Training**: See `../fine_tuning/lora/README.md`
- **Model Merging**: See `../skills/model-merging/`

## Contributing

When adding a new model:
1. Create metadata YAML in `base/`, `adapters/`, `merged/`, or `exported/`
2. Follow naming conventions
3. Include comprehensive metadata
4. Add performance benchmarks
5. Include usage examples
6. Document any special requirements

## License

Each model's metadata inherits the license of the base model.
