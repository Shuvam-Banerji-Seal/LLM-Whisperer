# Master Agentic Skill: Diffusers Video Generation (SVD, Wan, CogVideoX, Memory and Quality)

## 1. Mission
Ship high-quality video generation systems that balance visual fidelity, temporal consistency, and memory efficiency under real hardware constraints.

## 2. Principles
- Prioritize reproducibility over one-off wins.
- Log every configuration that can alter behavior.
- Validate quality and latency together; never optimize one blindly.
- Keep rollback paths documented and tested.
- Treat safety and governance checks as first-class production requirements.

## 3. Source Index (Docs and Blogs)
1. https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid
2. https://huggingface.co/docs/diffusers/main/en/optimization/memory
3. https://huggingface.co/docs/diffusers/main/en/quantization/overview
4. https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/svd
5. https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan
6. https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox
7. https://stability.ai/news/stable-video-diffusion-open-ai-video-model
8. https://huggingface.co/blog/stable-video-diffusion

## 4. Fast Documentation Fetch Commands
Use these commands when someone reports issues and you need to verify behavior against upstream docs quickly.

```bash
mkdir -p /tmp/skill_refs
curl -L "https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid" -o /tmp/skill_refs/huggingface.co_docs_diffusers_main_en_using-diffusers_text-img2vid.html
curl -L "https://huggingface.co/docs/diffusers/main/en/optimization/memory" -o /tmp/skill_refs/huggingface.co_docs_diffusers_main_en_optimization_memory.html
curl -L "https://huggingface.co/docs/diffusers/main/en/quantization/overview" -o /tmp/skill_refs/huggingface.co_docs_diffusers_main_en_quantization_overview.html
curl -L "https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/svd" -o /tmp/skill_refs/huggingface.co_docs_diffusers_main_en_api_pipelines_stable_diffusion_svd.html
curl -L "https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan" -o /tmp/skill_refs/huggingface.co_docs_diffusers_main_en_api_pipelines_wan.html
curl -L "https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox" -o /tmp/skill_refs/huggingface.co_docs_diffusers_main_en_api_pipelines_cogvideox.html
curl -L "https://stability.ai/news/stable-video-diffusion-open-ai-video-model" -o /tmp/skill_refs/stability.ai_news_stable-video-diffusion-open-ai-video-model.html
curl -L "https://huggingface.co/blog/stable-video-diffusion" -o /tmp/skill_refs/huggingface.co_blog_stable-video-diffusion.html
ls -lh /tmp/skill_refs
```

## 5. Operational Policies
Use this section as the mandatory baseline policy set for Diffusers video generation.

### 5.1 Metrics that must always be tracked
- frame_consistency_score
- flicker_rate
- artifact_rate
- fps_output
- generation_time_seconds
- vram_peak_gb
- decode_failure_rate
- prompt_adherence_score

### 5.2 Guardrails
- Always lock seed when comparing parameter changes.
- Keep prompt, steps, and scheduler fixed during A/B tests.
- Track per-frame artifacts, not just first-frame quality.
- Require negative prompt baselines for public-facing generation.
- Fail release if OOM occurs on target hardware profile.
- Validate export codecs and FPS expectations before publish.

## 6. Codebook
Each recipe is production-oriented and intentionally explicit.

### Recipe 01: Stable Video Diffusion image-to-video baseline
Use this as the initial quality baseline for image-conditioned clips.

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
).resize((1024, 576))

frames = pipe(
    image,
    num_frames=25,
    decode_chunk_size=8,
    motion_bucket_id=127,
    noise_aug_strength=0.1,
    generator=torch.manual_seed(42),
).frames[0]

export_to_video(frames, "svd-output.mp4", fps=7)
```

Notes:
- decode_chunk_size trades speed for memory footprint.
- motion_bucket_id strongly affects perceived motion dynamics.

### Recipe 02: Wan pipeline with group offloading
Use this for large video models on constrained VRAM.

```python
import torch
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
)
vae = AutoModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="vae",
    torch_dtype=torch.float32,
)
transformer = AutoModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

apply_group_offloading(
    text_encoder,
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=4,
)
```

Notes:
- Group offloading can lower VRAM but increases transfer overhead.
- Tune offload granularity for latency-sensitive use cases.

### Recipe 03: CogVideoX compilation for inference speed
Use this where repeated runs amortize compile cost.

```python
import torch
from diffusers import CogVideoXPipeline

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16,
).to("cuda")

pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="max-autotune",
    fullgraph=True,
)
```

Notes:
- Changing static shapes can trigger recompilation.
- Benchmark warm vs cold start separately.

### Recipe 04: Video generation with explicit parameter controls
Use explicit controls to isolate quality regressions quickly.

```python
prompt = "A handheld camera follows a red kite over a foggy valley at sunrise"
negative_prompt = "blurry, washed out, low detail, artifacts, subtitles"

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=40,
).frames[0]
```

Notes:
- Increase guidance cautiously; too high may create artifacts.
- Track temporal consistency when changing num_frames.

### Recipe 05: Quantized video pipeline components
Use this when deploying large video models on mid-tier hardware.

```python
from diffusers import WanPipeline, AutoModel
from diffusers.quantizers import PipelineQuantizationConfig

qcfg = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True},
    components_to_quantize=["transformer", "text_encoder"],
)

vae = AutoModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="vae",
    torch_dtype="float32",
)
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    quantization_config=qcfg,
    torch_dtype="bfloat16",
)
```

Notes:
- Quantization may affect temporal smoothness; validate on motion-heavy prompts.
- Keep a full-precision fallback profile for critical output quality.

## 7. Failure and Recovery Matrix
This matrix is intentionally exhaustive. Follow one case at a time and log every change.

### Case 001: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 002: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 003: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 004: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 005: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 006: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 007: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 008: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 009: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 010: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 011: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 012: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 013: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 014: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 015: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 016: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 017: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 018: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: artifact rate remains within acceptance threshold

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 019: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 020: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: prompt adherence score improves on regression prompts

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 021: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 022: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: clip fps and frame count match expected outputs

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 023: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: p95 generation latency meets SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 024: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: artifact rate remains within acceptance threshold

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 025: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 026: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 027: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 028: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 029: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 030: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 031: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 032: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 033: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 034: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 035: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 036: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 037: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 038: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 039: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 040: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 041: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 042: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: artifact rate remains within acceptance threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 043: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 044: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: prompt adherence score improves on regression prompts

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 045: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 046: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: clip fps and frame count match expected outputs

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 047: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: p95 generation latency meets SLO

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 048: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: artifact rate remains within acceptance threshold

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 049: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 050: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 051: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 052: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 053: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 054: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 055: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 056: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 057: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 058: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 059: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 060: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 061: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 062: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 063: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 064: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 065: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 066: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: artifact rate remains within acceptance threshold

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 067: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 068: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: prompt adherence score improves on regression prompts

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 069: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 070: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: clip fps and frame count match expected outputs

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 071: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: p95 generation latency meets SLO

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 072: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: artifact rate remains within acceptance threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 073: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 074: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 075: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 076: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 077: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 078: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 079: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 080: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 081: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 082: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 083: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 084: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 085: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 086: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 087: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 088: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 089: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 090: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: artifact rate remains within acceptance threshold

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 091: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 092: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: prompt adherence score improves on regression prompts

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 093: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 094: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: clip fps and frame count match expected outputs

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 095: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: p95 generation latency meets SLO

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 096: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: artifact rate remains within acceptance threshold

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 097: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 098: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 099: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 100: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 101: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 102: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 103: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 104: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 105: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 106: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 107: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 108: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 109: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 110: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 111: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 112: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 113: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 114: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: artifact rate remains within acceptance threshold

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 115: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 116: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: prompt adherence score improves on regression prompts

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 117: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 118: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: clip fps and frame count match expected outputs

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 119: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: p95 generation latency meets SLO

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 120: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: artifact rate remains within acceptance threshold

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 121: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 122: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 123: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 124: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 125: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 126: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 127: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 128: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 129: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 130: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 131: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 132: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 133: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 134: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 135: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 136: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 137: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 138: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: artifact rate remains within acceptance threshold

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 139: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 140: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: prompt adherence score improves on regression prompts

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 141: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 142: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: clip fps and frame count match expected outputs

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 143: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: p95 generation latency meets SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 144: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: artifact rate remains within acceptance threshold

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 145: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 146: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 147: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 148: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 149: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 150: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 151: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 152: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 153: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 154: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 155: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 156: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 157: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 158: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 159: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 160: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 161: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 162: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: artifact rate remains within acceptance threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 163: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 164: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: prompt adherence score improves on regression prompts

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 165: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 166: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: clip fps and frame count match expected outputs

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 167: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: p95 generation latency meets SLO

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 168: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: artifact rate remains within acceptance threshold

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 169: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: flicker rate decreases after parameter tuning

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 170: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: prompt adherence score improves on regression prompts

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 171: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: OOM events drop to zero on target hardware

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 172: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: clip fps and frame count match expected outputs

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 173: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: p95 generation latency meets SLO

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 174: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: artifact rate remains within acceptance threshold

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 175: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: flicker rate decreases after parameter tuning

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 176: first frame quality is good but later frames degrade
- Signal: first frame quality is good but later frames degrade
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: prompt adherence score improves on regression prompts

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 177: inference is too slow for SLA
- Signal: inference is too slow for SLA
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: OOM events drop to zero on target hardware

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 178: prompt adherence is inconsistent
- Signal: prompt adherence is inconsistent
- Likely cause: insufficient memory optimizations for selected model size
- Immediate action: adjust guidance_scale in small increments and compare clips
- Verification metric: clip fps and frame count match expected outputs

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 179: negative prompt over-suppresses visual details
- Signal: negative prompt over-suppresses visual details
- Likely cause: decode chunk size too high for hardware budget
- Immediate action: add explicit camera-motion and scene continuity cues in prompt
- Verification metric: p95 generation latency meets SLO

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 180: exported file has codec or playback issues
- Signal: exported file has codec or playback issues
- Likely cause: prompt not specific enough for temporal dynamics
- Immediate action: apply group offloading or model CPU offload
- Verification metric: artifact rate remains within acceptance threshold

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

### Case 181: flicker appears between adjacent frames
- Signal: flicker appears between adjacent frames
- Likely cause: quantization side effects on temporal coherence
- Immediate action: evaluate fp16 vs quantized output on same seed and prompt
- Verification metric: flicker rate decreases after parameter tuning

```bash
python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'
```

### Case 182: motion does not follow prompt intent
- Signal: motion does not follow prompt intent
- Likely cause: scheduler mismatch with model recommendation
- Immediate action: lock scheduler and seed for controlled comparison
- Verification metric: prompt adherence score improves on regression prompts

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 183: generated clip has severe artifacts
- Signal: generated clip has severe artifacts
- Likely cause: offloading configuration not tuned for workload
- Immediate action: run frame-by-frame artifact analysis
- Verification metric: OOM events drop to zero on target hardware

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### Case 184: VRAM OOM occurs during decode
- Signal: VRAM OOM occurs during decode
- Likely cause: export pipeline mismatched fps or codec settings
- Immediate action: validate video export settings with target playback environment
- Verification metric: clip fps and frame count match expected outputs

```bash
ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'
```

### Case 185: output duration does not match expected frames/fps
- Signal: output duration does not match expected frames/fps
- Likely cause: num_frames, guidance, and step settings are unbalanced
- Immediate action: reduce decode_chunk_size and retest memory profile
- Verification metric: p95 generation latency meets SLO

```bash
python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json
```

## 8. Validation Drills
Complete every drill before promoting a change to production.

- [ ] Drill 001: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 002: Measure memory with and without group offloading on same prompt.
- [ ] Drill 003: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 004: Validate export with ffprobe and playback on target clients.
- [ ] Drill 005: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 006: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 007: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 008: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 009: Measure memory with and without group offloading on same prompt.
- [ ] Drill 010: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 011: Validate export with ffprobe and playback on target clients.
- [ ] Drill 012: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 013: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 014: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 015: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 016: Measure memory with and without group offloading on same prompt.
- [ ] Drill 017: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 018: Validate export with ffprobe and playback on target clients.
- [ ] Drill 019: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 020: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 021: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 022: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 023: Measure memory with and without group offloading on same prompt.
- [ ] Drill 024: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 025: Validate export with ffprobe and playback on target clients.
- [ ] Drill 026: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 027: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 028: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 029: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 030: Measure memory with and without group offloading on same prompt.
- [ ] Drill 031: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 032: Validate export with ffprobe and playback on target clients.
- [ ] Drill 033: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 034: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 035: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 036: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 037: Measure memory with and without group offloading on same prompt.
- [ ] Drill 038: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 039: Validate export with ffprobe and playback on target clients.
- [ ] Drill 040: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 041: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 042: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 043: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 044: Measure memory with and without group offloading on same prompt.
- [ ] Drill 045: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 046: Validate export with ffprobe and playback on target clients.
- [ ] Drill 047: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 048: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 049: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 050: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 051: Measure memory with and without group offloading on same prompt.
- [ ] Drill 052: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 053: Validate export with ffprobe and playback on target clients.
- [ ] Drill 054: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 055: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 056: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 057: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 058: Measure memory with and without group offloading on same prompt.
- [ ] Drill 059: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 060: Validate export with ffprobe and playback on target clients.
- [ ] Drill 061: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 062: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 063: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 064: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 065: Measure memory with and without group offloading on same prompt.
- [ ] Drill 066: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 067: Validate export with ffprobe and playback on target clients.
- [ ] Drill 068: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 069: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 070: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 071: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 072: Measure memory with and without group offloading on same prompt.
- [ ] Drill 073: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 074: Validate export with ffprobe and playback on target clients.
- [ ] Drill 075: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 076: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 077: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 078: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 079: Measure memory with and without group offloading on same prompt.
- [ ] Drill 080: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 081: Validate export with ffprobe and playback on target clients.
- [ ] Drill 082: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 083: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 084: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 085: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 086: Measure memory with and without group offloading on same prompt.
- [ ] Drill 087: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 088: Validate export with ffprobe and playback on target clients.
- [ ] Drill 089: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 090: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 091: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 092: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 093: Measure memory with and without group offloading on same prompt.
- [ ] Drill 094: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 095: Validate export with ffprobe and playback on target clients.
- [ ] Drill 096: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 097: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 098: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 099: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 100: Measure memory with and without group offloading on same prompt.
- [ ] Drill 101: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 102: Validate export with ffprobe and playback on target clients.
- [ ] Drill 103: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 104: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 105: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 106: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 107: Measure memory with and without group offloading on same prompt.
- [ ] Drill 108: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 109: Validate export with ffprobe and playback on target clients.
- [ ] Drill 110: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 111: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 112: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 113: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 114: Measure memory with and without group offloading on same prompt.
- [ ] Drill 115: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 116: Validate export with ffprobe and playback on target clients.
- [ ] Drill 117: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 118: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 119: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 120: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 121: Measure memory with and without group offloading on same prompt.
- [ ] Drill 122: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 123: Validate export with ffprobe and playback on target clients.
- [ ] Drill 124: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 125: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 126: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 127: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 128: Measure memory with and without group offloading on same prompt.
- [ ] Drill 129: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 130: Validate export with ffprobe and playback on target clients.
- [ ] Drill 131: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 132: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 133: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 134: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 135: Measure memory with and without group offloading on same prompt.
- [ ] Drill 136: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 137: Validate export with ffprobe and playback on target clients.
- [ ] Drill 138: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 139: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 140: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 141: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 142: Measure memory with and without group offloading on same prompt.
- [ ] Drill 143: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 144: Validate export with ffprobe and playback on target clients.
- [ ] Drill 145: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 146: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 147: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 148: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 149: Measure memory with and without group offloading on same prompt.
- [ ] Drill 150: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 151: Validate export with ffprobe and playback on target clients.
- [ ] Drill 152: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 153: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 154: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 155: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 156: Measure memory with and without group offloading on same prompt.
- [ ] Drill 157: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 158: Validate export with ffprobe and playback on target clients.
- [ ] Drill 159: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 160: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 161: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 162: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 163: Measure memory with and without group offloading on same prompt.
- [ ] Drill 164: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 165: Validate export with ffprobe and playback on target clients.
- [ ] Drill 166: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 167: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 168: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 169: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 170: Measure memory with and without group offloading on same prompt.
- [ ] Drill 171: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 172: Validate export with ffprobe and playback on target clients.
- [ ] Drill 173: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 174: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 175: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 176: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 177: Measure memory with and without group offloading on same prompt.
- [ ] Drill 178: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 179: Validate export with ffprobe and playback on target clients.
- [ ] Drill 180: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 181: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 182: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 183: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 184: Measure memory with and without group offloading on same prompt.
- [ ] Drill 185: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 186: Validate export with ffprobe and playback on target clients.
- [ ] Drill 187: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 188: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 189: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 190: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 191: Measure memory with and without group offloading on same prompt.
- [ ] Drill 192: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 193: Validate export with ffprobe and playback on target clients.
- [ ] Drill 194: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 195: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 196: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 197: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 198: Measure memory with and without group offloading on same prompt.
- [ ] Drill 199: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 200: Validate export with ffprobe and playback on target clients.
- [ ] Drill 201: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 202: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 203: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 204: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 205: Measure memory with and without group offloading on same prompt.
- [ ] Drill 206: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 207: Validate export with ffprobe and playback on target clients.
- [ ] Drill 208: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 209: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 210: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 211: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 212: Measure memory with and without group offloading on same prompt.
- [ ] Drill 213: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 214: Validate export with ffprobe and playback on target clients.
- [ ] Drill 215: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 216: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 217: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 218: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 219: Measure memory with and without group offloading on same prompt.
- [ ] Drill 220: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 221: Validate export with ffprobe and playback on target clients.
- [ ] Drill 222: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 223: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 224: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 225: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 226: Measure memory with and without group offloading on same prompt.
- [ ] Drill 227: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 228: Validate export with ffprobe and playback on target clients.
- [ ] Drill 229: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 230: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 231: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 232: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 233: Measure memory with and without group offloading on same prompt.
- [ ] Drill 234: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 235: Validate export with ffprobe and playback on target clients.
- [ ] Drill 236: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 237: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 238: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 239: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 240: Measure memory with and without group offloading on same prompt.
- [ ] Drill 241: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 242: Validate export with ffprobe and playback on target clients.
- [ ] Drill 243: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 244: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 245: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 246: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 247: Measure memory with and without group offloading on same prompt.
- [ ] Drill 248: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 249: Validate export with ffprobe and playback on target clients.
- [ ] Drill 250: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 251: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 252: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 253: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 254: Measure memory with and without group offloading on same prompt.
- [ ] Drill 255: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 256: Validate export with ffprobe and playback on target clients.
- [ ] Drill 257: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 258: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 259: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 260: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 261: Measure memory with and without group offloading on same prompt.
- [ ] Drill 262: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 263: Validate export with ffprobe and playback on target clients.
- [ ] Drill 264: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 265: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 266: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 267: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 268: Measure memory with and without group offloading on same prompt.
- [ ] Drill 269: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 270: Validate export with ffprobe and playback on target clients.
- [ ] Drill 271: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 272: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 273: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.
- [ ] Drill 274: Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.
- [ ] Drill 275: Measure memory with and without group offloading on same prompt.
- [ ] Drill 276: Evaluate temporal consistency on at least 20 motion-heavy prompts.
- [ ] Drill 277: Validate export with ffprobe and playback on target clients.
- [ ] Drill 278: Run prompt adherence scoring before release candidate tagging.
- [ ] Drill 279: Benchmark cold-start and warm-start latency separately.
- [ ] Drill 280: Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.

