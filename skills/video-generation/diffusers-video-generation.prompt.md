# Diffusers Video Generation - Agentic Skill Prompt

Use this prompt for text-to-video and image-to-video workflows with diffusers pipelines.

## 1. Mission

Run reliable video generation with explicit control over frame count, resolution, memory strategy, and output quality.

## 2. Pipeline Families

Typical families in current ecosystem include:

- CogVideoX pipelines
- Stable Video Diffusion pipelines
- AnimateDiff-related pipelines
- Wan and other newer video-oriented pipelines

Select family based on task type (T2V, I2V, V2V), hardware budget, and model license constraints.

## 3. Baseline T2V Example

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16,
).to("cuda")

pipe.enable_model_cpu_offload()
frames = pipe(
    "A panda playing guitar in a bamboo forest",
    guidance_scale=6,
    num_inference_steps=50,
).frames[0]

export_to_video(frames, "out.mp4", fps=8)
```

## 4. Baseline I2V Example

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
image = load_image("input.png")
frames = pipe(image, num_frames=25).frames[0]
export_to_video(frames, "i2v.mp4", fps=7)
```

## 5. Memory and Throughput Strategy

- Reduce `num_frames` and resolution first when hitting OOM.
- Use model CPU offload for constrained GPUs.
- Apply VAE slicing or tiling where supported.
- Benchmark warm runs, not first-run compile time.

## 6. Hardware Planning Guidance

| VRAM Tier | Practical expectation |
|---|---|
| 8 to 12 GB | Short, low-resolution experiments |
| 16 to 24 GB | Practical short clips with careful optimization |
| 40 GB+ | Higher-quality and longer clips with fewer compromises |

Exact requirements depend on pipeline family, frame count, resolution, and precision.

## 7. Quality and Safety Controls

- Fix seed and prompt template when comparing changes.
- Store all generation parameters per run.
- Validate model license and gated access constraints before deployment.
- Add output moderation for public products.

## 8. Troubleshooting Checklist

1. OOM during decode: reduce frames and resolution; enable offload.
2. Severe slowdown: sequential offload or kernel fallback likely active.
3. Runtime mismatch errors: incompatible pipeline and checkpoint variants.
4. Low temporal consistency: adjust steps, guidance, and conditioning strategy.

## 9. References (Fetched 2026-04-06)

1. https://huggingface.co/docs/diffusers/api/pipelines/text_to_video - Text and video generation API family overview.
2. https://huggingface.co/docs/diffusers/api/pipelines/cogvideox - CogVideoX pipeline usage and constraints.
3. https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/svd - Stable Video Diffusion image-to-video API.
4. https://huggingface.co/docs/diffusers/api/pipelines/animatediff - AnimateDiff pipeline context.
5. https://huggingface.co/docs/diffusers/api/pipelines/wan - Wan pipeline reference.
6. https://huggingface.co/docs/diffusers/api/pipelines/ltx_video - LTX video pipeline reference.
7. https://huggingface.co/docs/diffusers/api/pipelines/hunyuan_video - Hunyuan video pipeline reference.
8. https://huggingface.co/docs/diffusers/api/pipelines/mochi - Mochi pipeline reference.
9. https://huggingface.co/docs/diffusers/optimization/memory - Memory optimization techniques.
10. https://huggingface.co/docs/diffusers/optimization/fp16 - Precision and speed tuning.
11. https://huggingface.co/docs/diffusers/quantization/overview - Quantization support and caveats.
12. https://huggingface.co/docs/hub/models-gated - Gated model access workflow.

## 10. Uncertainty Notes

- Video pipeline APIs evolve quickly; verify arguments against your installed diffusers version.
- Published VRAM expectations are highly workload dependent.
