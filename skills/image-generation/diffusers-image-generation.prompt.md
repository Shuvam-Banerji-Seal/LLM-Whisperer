# Diffusers Image Generation and Adapters - Agentic Skill Prompt

Use this prompt for building, deploying, and optimizing practical image generation systems using the Hugging Face `diffusers` library. This skill focuses on advanced adapter stacks (LoRA, IP-Adapter, ControlNet), optimization, edge cases, and best practices.

## 1. Mission
Deliver stable, controllable, high-quality, and fast image generation pipelines. The focus should be on reproducible settings, measured tradeoffs between latency and quality, and robust error handling.

## 2. Baseline Text-to-Image (SDXL & SD 1.5)

### Best Practices:
- **Data Types:** Always use `torch.float16` or `torch.bfloat16` for inference to save VRAM and improve speed without significant quality loss.
- **Offloading:** Use `enable_model_cpu_offload()` or `enable_sequential_cpu_offload()` if VRAM is heavily constrained (e.g., < 8GB for SDXL).
- **Compilation:** Use `torch.compile` for production endpoints to reduce latency (requires PyTorch 2.0+).

```python
import torch
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler

# Document URL: https://huggingface.co/docs/diffusers/main/en/api/pipelines/auto_pipeline
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Recommended Scheduler for SDXL
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# Optimization: xFormers for memory efficient attention mapping
pipeline.enable_xformers_memory_efficient_attention()

# Generate with explicit seed for reproducibility
generator = torch.Generator(device="cuda").manual_seed(42)
image = pipeline(
    prompt="A hyper-realistic studio portrait with soft cinematic lighting, 8k, highly detailed",
    negative_prompt="blurry, poorly drawn, distorted, text, watermark",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator
).images[0]

image.save("baseline_out.png")
```

### Edge Cases to Handle:
- **CUDA Out of Memory (OOM):** If `torch.float16` + `xformers` still OOMs, fallback to `pipeline.enable_model_cpu_offload()`. Note that offloading heavily impacts inference latency.
- **Black Images Output:** This implies the NSFW safety checker triggered or precision NaN issues. If it's NaN issues, disable the VAE's upcast or use `torch.float32` for the VAE specifically (`pipeline.upcast_vae()`).

## 3. Advanced Adapter Workflows

### 3.1 LoRA (Low-Rank Adaptation)
LoRAs are small weight deltas applied to target specific concepts or styles.
**Docs:** [Diffusers LoRA Training & Inference](https://huggingface.co/docs/diffusers/main/en/training/lora)

```python
# Loading multiple LoRAs with weight adjustments
pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors", adapter_name="cereal")

# Dynamically route weights
pipeline.set_adapters(["pixel", "cereal"], adapter_weights=[0.8, 0.4])

# Edge case: To release VRAM, ALWAYS unload LoRA when switching concepts
# pipeline.unload_lora_weights() 
```

### 3.2 IP-Adapter (Image Prompt Adapter)
Allows using an image (rather than text) as the conditioning input.
**Docs:** [IP-Adapter Diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter)

```python
from diffusers.utils import load_image

# Load the IP-Adapter into the pipeline
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)
pipeline.set_ip_adapter_scale(0.6)

reference_img = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png")
image = pipeline(
    prompt="editorial fashion photography",
    ip_adapter_image=reference_img,
    num_inference_steps=25
).images[0]
```

### 3.3 ControlNet Integration
Provides spatial control (Canny edges, Depth maps, Pose).
**Docs:** [Diffusers ControlNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet)

```python
import torch
import cv2
from PIL import Image
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL

# 1. Prepare Canny edge map
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg")
init_image = np.array(init_image)
edges = cv2.Canny(init_image, 100, 200)
edges = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

# 2. Load ControlNet and Pipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

pipe_control = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Edge case: ControlNet scales can be a list if operating on MulticontrolNet (e.g. depth + canny)
image = pipe_control(
    "modern minimalist bedroom, ultra realistic",
    image=edges,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=30
).images[0]
```

## 4. Performance & Hardware Limits
- **Batching:** Generating 4 images in a batch (`prompt=["..."]*4`) is faster than a loop, but requires proportional VRAM.
- **Latency vs Quality:** Switch to `LCMScheduler` or `EulerAncestralDiscreteScheduler` for fewer steps (4-8 steps for LCM, 20 for EulerA) if latency is critical. 
- **Refiners:** For SDXL, use the Refiner model as an ensemble of experts (sharing the same prompt). Only allocate VRAM to the refiner sequentially if memory is low.

## 5. Security, Safety, and Governance
- **NSFW Checkers:** Enabled by default on base SD1.5/SD2 models. With `SDXL`, it's not embedded natively in the AutoPipeline but should be validated with [external safety moderation APIs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/safety_checker) for production.
- **Prompt Injection:** Sanitize inputs to avoid unexpected styles or offensive requests. Apply a strong base-negative prompt.
- **Watermarking:** Automatically integrated in SDXL implementations to track synthetic media. Do not remove the `Watermarker` locally for public tools.

## 6. Official Troubleshooting References
- Main Docs: https://huggingface.co/docs/diffusers/
- Out of Memory (OOM) Optimization Guidelines: https://huggingface.co/docs/diffusers/main/en/optimization/memory
- Faster inference: https://huggingface.co/docs/diffusers/main/en/optimization/fp16
