# Diffusers Loading and Optimization - Agentic Skill Prompt

Use this prompt for loading diffusion pipelines safely and optimizing memory and latency for image and video generation.

## 1. Mission

Build reproducible and efficient diffusion inference workflows with clear controls for scheduler choice, device placement, and precision.

## 2. Environment Baseline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U diffusers transformers accelerate huggingface_hub safetensors
pip install -U xformers bitsandbytes
hf auth login
```

## 3. Pipeline Loading Patterns

### 3.1 Load from Hub

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    device_map="cuda",
)
```

### 3.2 Load from local snapshot

```python
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline
import torch

local_path = snapshot_download("stabilityai/stable-diffusion-xl-base-1.0")
pipe = DiffusionPipeline.from_pretrained(
    local_path,
    torch_dtype=torch.float16,
    local_files_only=True,
)
```

### 3.3 Single-file load patterns

Use `from_single_file` only when pipeline and config compatibility are documented for the selected checkpoint.

## 4. Scheduler Strategy

Swap scheduler from existing config to preserve compatible parameters:

```python
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
```

Practical rule: compare quality and speed after scheduler changes, not just latency.

## 5. Memory and Speed Controls

- Start with fp16 or bf16 on modern GPUs.
- Enable model CPU offload for memory-constrained systems.
- Use VAE tiling for larger resolutions.
- Avoid stacking conflicting optimizations without profiling.

```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
# pipe.enable_sequential_cpu_offload()  # stronger memory reduction, often slower
```

## 6. Quantization Hooks

Use pipeline quantization configuration when backend support is confirmed:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

qcfg = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    components_to_quantize=["transformer", "text_encoder_2"],
)

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=qcfg,
    torch_dtype=torch.bfloat16,
)
```

## 7. Gated Models and Licensing

- Confirm access grants before automated deployment.
- Record model license and usage constraints in deployment metadata.
- Keep model revisions pinned for reproducibility.

## 8. Hardware Planning (Guidance)

| VRAM Tier | Typical Use |
|---|---|
| 8 GB | SD1.5 or reduced SDXL settings, small batch generation |
| 12 to 16 GB | SDXL with adapters and constrained video tests |
| 24 GB | Higher quality image generation and short practical video runs |
| 40 GB+ | Heavier video pipelines and larger batch experiments |

Hardware needs vary by model family, resolution, frame count, and backend kernels.

## 9. Troubleshooting Checklist

- OOM on load: lower precision, enable offload, reduce components on device.
- OOM on generation: reduce resolution, steps, or frames; enable tiling.
- Device map conflicts: reset or rebuild pipeline device mapping.
- Severe slowdown: check for sequential offload or kernel fallback.
- Auth errors: verify gated model access and token scope.

## 10. References (Fetched 2026-04-06)

1. https://huggingface.co/docs/diffusers/index - Diffusers documentation root.
2. https://huggingface.co/docs/diffusers/quicktour - Quickstart patterns and baseline usage.
3. https://huggingface.co/docs/diffusers/using-diffusers/loading - Hub and local loading workflows.
4. https://huggingface.co/docs/diffusers/using-diffusers/schedulers - Scheduler selection and configuration.
5. https://huggingface.co/docs/diffusers/optimization/memory - Memory optimization techniques.
6. https://huggingface.co/docs/diffusers/optimization/fp16 - Precision and speed guidance.
7. https://huggingface.co/docs/diffusers/quantization/overview - Diffusers quantization support.
8. https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download - Snapshot download API.
9. https://huggingface.co/docs/hub/models-gated - Gated model workflow.
10. https://huggingface.co/docs/hub/model-cards - Model card metadata and governance context.
11. https://huggingface.co/docs/hub/repositories-licenses - License policy context.
12. https://huggingface.co/stabilityai/stable-diffusion-v1-5 - Example model card and usage context.

## 11. Uncertainty Notes

- Newer pipelines and optimization flags can change across diffusers minor releases.
- Always verify exact parameter names against your installed package version.
