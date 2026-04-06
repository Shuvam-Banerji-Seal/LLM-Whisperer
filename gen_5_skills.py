import os

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# 1. LLM Fine-Tuning
llm_ft = """# Master Agentic Skill: LLM Fine-Tuning (SFT, RLHF, DPO, QLoRA)
**Version**: 2.0 (Production Level)

## 1. Mission and High-Level Strategy
This skill defines the rigorous, production-grade methodology for fine-tuning Large Language Models (LLMs). The goal is to align base/instruct models using PEFT (Parameter-Efficient Fine-Tuning) such as LoRA/QLoRA, or Full Parameter Fine-Tuning via FSDP/DeepSpeed.

## 2. Environment and Library Stack
- `transformers`, `peft`, `trl` (Transformer Reinforcement Learning), `bitsandbytes`, `accelerate`

### 2.1 QLoRA Initialization Best Practices
Always use nested quantization and bfloat16 compute dtype to retain numeric stability during QLoRA.
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False # Crucial for training
)
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target ALL linear layers
)
model = get_peft_model(model, peft_config)
```

## 3. Data Packing and Formatting (SFTTrainer)
To maximize GPU compute, do not pad heavily. Use `ConstantLengthDataset` or `packing=True` in SFTTrainer.
- Always apply Chat Templates (e.g. ChatML).
- Mask prompt labels! Only train on the assistant's completions using `DataCollatorForCompletionOnlyLM`.

```python
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fixes weird FP16 overflow issues during training

instruction_template = "<|im_start|>user"
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template, 
    response_template=response_template, 
    tokenizer=tokenizer, 
    mlm=False
)
```

## 4. Alignment: DPO (Direct Preference Optimization)
If preference pairs (Chosen vs Rejected) exist, bypass PPO and use DPO.
```python
from trl import DPOTrainer
# DPO requires a reference model (frozen). If PEFT is used, DPOTrainer handles this internally!
dpo_trainer = DPOTrainer(
    model,
    ref_model=None, # TRL handles adapter toggling for reference
    args=training_args,
    beta=0.1, # KL penalty
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

## 5. DeepSpeed Zero-3 / FSDP Edge Cases
- QLoRA does NOT natively work with DeepSpeed Zero-3 because 4-bit weights cannot be partitioned cleanly. Use FSDP with QLoRA via `accelerate` instead, or stick to DeepSpeed Zero-2.

## 6. Official Docs and Troubleshooting
- Loss goes to 0 / NaN: Check learning rate (should be ~2e-5 or 2e-4 for LoRA, much lower 1e-6 for full FT).
- RAM OOM during save: Set `safe_serialization=True` and `save_safetensors=True`.
"""

# 2. LLM Quantization
llm_quant = """# Master Agentic Skill: LLM Quantization
**Version**: 2.0 (Production Level)

## 1. High-Level Strategy
Quantization reduces VRAM usage and increases memory bandwidth throughput. We focus on Post-Training Quantization (PTQ) techniques: AWQ, GPTQ, EXL2, and GGUF (llama.cpp).

## 2. AutoAWQ (Activation-aware Weight Quantization)
AWQ observes activations over a calibration dataset, protecting the top 1% of salient weights to prevent degradation natively.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-chat-hf"
quant_path = "llama-2-7b-chat-awq"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoAWQForCausalLM.from_pretrained(
    model_path, 
    low_cpu_mem_usage=True, 
    use_cache=False
)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}
# Calibrate and Quantize
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

## 3. GGUF / Llama.cpp Quantization
For edge devices (Mac, Windows, Mobile), GGUF is the undisputed king.

**Workflow via Makefile**:
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# Convert to GGUF FP16
python convert.py /path/to/hf/model --outfile model-fp16.gguf --outtype f16

# Quantize to Q4_K_M (Recommended balance of speed/quality)
./quantize model-fp16.gguf model-q4_k_m.gguf Q4_K_M
```

## 4. Edge Cases: GPTQ vs AWQ
- **GPTQ**: Better if heavily calibrated with high-quality domain datasets. Uses `AutoGPTQ` library.
- **Troubleshooting Perplexity Spikes**: Llama-3 models suffer severe perplexity degradation below 4.5 bpw. Mixed precision quantization (EXL2) or high group sizes (group_size=64) are required.

## 5. Deployment
- Use `vLLM` for AWQ/GPTQ.
- Use `Ollama` / `llama.cpp` for GGUF.
"""

# 3. Mixture of Experts Loading
moe_loading = """# Master Agentic Skill: Mixture of Experts (MoE) 
**Version**: 2.0 (Production Level)

## 1. Overview
Models like Mixtral 8x7B or DeepSeek-V2 use Sparse MoE layers. Most parameters are cold during any single token's forward pass.

## 2. Hardware Considerations
- **VRAM**: An 8x7B model requires 47GB of VRAM in FP16 (equivalent to a dense 47B model).
- **Compute**: An 8x7B model requires the compute of a ~13B model since only 2 experts are active.
This means MoE is highly Memory Bandwidth bound.

## 3. Loading in Transformers
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Using device_map="auto" with load_in_4bit allows 8x7B to fit on a single 24GB GPU (RTX 3090)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16, # Standard for base
    load_in_4bit=True # Optional bitsandbytes
)
```

## 4. Expert Offloading (Edge Case)
If VRAM is extremely limited (e.g., 16GB), use custom layers to offload inactive experts to CPU RAM natively. `llama.cpp` manages MoE expert mmap offloading phenomenally.

## 5. Load Balancing & Z-Loss
When fine-tuning an MoE model, you MUST add a Router Aux Loss (load balancing loss) to prevent expert collapse (all tokens routing to expert 0).
```python
# In training arguments
training_args = TrainingArguments(
    ...
    output_router_logits=True # CRITICAL for Custom MoE Training
)
```
"""

# 4. Diffusers Video Generation
video_gen = """# Master Agentic Skill: Video Generation (Diffusers)
**Version**: 2.0 (Production Level)

## 1. High-Level Strategy
Video generation requires expanding 2D spatial U-Nets to 3D spatio-temporal architectures. Expect intense VRAM consumption.

## 2. Stable Video Diffusion (SVD)
SVD is image-to-video. Frame interpolation and motion buckets are key.

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe.enable_model_cpu_offload() # Mandatory for < 24GB VRAM

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(
    image, 
    decode_chunk_size=8, # Decoding VAE frame by frame to save memory
    generator=generator,
    motion_bucket_id=127,
    noise_aug_strength=0.1
).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

## 3. AnimateDiff (Text-to-Video via SD 1.5)
AnimateDiff injects a motion module into a standard text-to-image model.
- Always use `FreeInit` or `LCMScheduler` for speed.
- Context length defaults to 16 frames. 

## 4. Memory Optimizations for Video
- **VAE Slicing**: `pipe.enable_vae_slicing()`
- **VAE Tiling**: `pipe.enable_vae_tiling()` (Prevents CUDA OOM on massive resolution frames).
- **Chunked Decode**: Keep `decode_chunk_size` small (under 8).
"""

# 5. Token Management
token_mgmt = """# Master Agentic Skill: Token Management & Tokenizers
**Version**: 2.0 (Production Level)

## 1. Core Principles
Tokenization logic defines how an LLM inherently "sees" the world. Misalignment between training and inference tokenization causes instantaneous gibberish generation.

## 2. Chat Templates (Jinja2)
Modern models (Llama-3, Mistral, Qwen) use custom chat templates with internal special tokens (`<|im_start|>`, `<|eot_id|>`). Always use the tokenizer's apply method.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a pirate."},
    {"role": "user", "content": "Hello!"}
]

# apply_chat_template natively handles BOS, EOS, and special separator tokens.
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
```

## 3. Padding Strategies
- **Left Padding**: MUST be used for decoder-only batched inference (causal generation).
- **Right Padding**: Typically used for Training (SFT) and Masked Language Models.

```python
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token # If no pad token exists
```

## 4. Vocab Resizing
When adding new special tokens (e.g., for JSON keys), you must resize the embedding matrix!

```python
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['<json_start>', '<json_end>']})
model.resize_token_embeddings(len(tokenizer))
```

## 5. Edge Cases
- **Llama 3 EOT ID**: Llama 3 uses `<|eot_id|>` instead of typical EOS. Ensure generation endpoints set `stop_token_ids=[128009]`.
- **Fast vs Slow Tokenizers**: `use_fast=True` leverages the fully Rust-based `tokenizers` library. Always enforce it.
"""

files = {
    "skills/fine-tuning/llm-finetuning.prompt.md": llm_ft,
    "skills/quantization/llm-quantization.prompt.md": llm_quant,
    "skills/moe/mixture-of-experts-loading.prompt.md": moe_loading,
    "skills/video-generation/diffusers-video-generation.prompt.md": video_gen,
    "skills/huggingface/token-management.prompt.md": token_mgmt
}

for path, content in files.items():
    full_path = os.path.join("/home/shuvam/codes/LLM-Whisperer", path)
    ensure_dir(full_path)
    with open(full_path, "w") as f:
        f.write(content)
    print(f"Written {full_path}")
