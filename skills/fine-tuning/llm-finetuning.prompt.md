# Master Agentic Skill: LLM Fine-Tuning (SFT, DPO, QLoRA, FSDP)

## 1. Mission
Deliver robust, reproducible, and safe post-training pipelines for instruction tuning and preference alignment, with explicit safeguards for memory, stability, and evaluation drift.

## 2. Principles
- Prioritize reproducibility over one-off wins.
- Log every configuration that can alter behavior.
- Validate quality and latency together; never optimize one blindly.
- Keep rollback paths documented and tested.
- Treat safety and governance checks as first-class production requirements.

## 3. Source Index (Docs and Blogs)
1. https://huggingface.co/docs/trl/sft_trainer
2. https://huggingface.co/docs/trl/dpo_trainer
3. https://huggingface.co/docs/trl/dataset_formats
4. https://huggingface.co/docs/peft/developer_guides/quantization
5. https://huggingface.co/docs/accelerate/en/usage_guides/fsdp
6. https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
7. https://huggingface.co/papers/2305.14314
8. https://huggingface.co/papers/2305.18290
9. https://huggingface.co/docs/transformers/chat_templating
10. https://huggingface.co/docs/transformers/main_classes/trainer
11. https://huggingface.co/docs/transformers/main/en/tasks/language_modeling
12. https://github.com/huggingface/alignment-handbook

## 4. Fast Documentation Fetch Commands
Use these commands when someone reports issues and you need to verify behavior against upstream docs quickly.

```bash
mkdir -p /tmp/skill_refs
curl -L "https://huggingface.co/docs/trl/sft_trainer" -o /tmp/skill_refs/huggingface.co_docs_trl_sft_trainer.html
curl -L "https://huggingface.co/docs/trl/dpo_trainer" -o /tmp/skill_refs/huggingface.co_docs_trl_dpo_trainer.html
curl -L "https://huggingface.co/docs/trl/dataset_formats" -o /tmp/skill_refs/huggingface.co_docs_trl_dataset_formats.html
curl -L "https://huggingface.co/docs/peft/developer_guides/quantization" -o /tmp/skill_refs/huggingface.co_docs_peft_developer_guides_quantization.html
curl -L "https://huggingface.co/docs/accelerate/en/usage_guides/fsdp" -o /tmp/skill_refs/huggingface.co_docs_accelerate_en_usage_guides_fsdp.html
curl -L "https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/" -o /tmp/skill_refs/pytorch.org_blog_introducing-pytorch-fully-sharded-data-parallel-api_.html
curl -L "https://huggingface.co/papers/2305.14314" -o /tmp/skill_refs/huggingface.co_papers_2305.14314.html
curl -L "https://huggingface.co/papers/2305.18290" -o /tmp/skill_refs/huggingface.co_papers_2305.18290.html
curl -L "https://huggingface.co/docs/transformers/chat_templating" -o /tmp/skill_refs/huggingface.co_docs_transformers_chat_templating.html
curl -L "https://huggingface.co/docs/transformers/main_classes/trainer" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_classes_trainer.html
curl -L "https://huggingface.co/docs/transformers/main/en/tasks/language_modeling" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_tasks_language_modeling.html
curl -L "https://github.com/huggingface/alignment-handbook" -o /tmp/skill_refs/github.com_huggingface_alignment-handbook.html
ls -lh /tmp/skill_refs
```

## 5. Operational Policies
Use this section as the mandatory baseline policy set for LLM fine-tuning.

### 5.1 Metrics that must always be tracked
- tokens_per_second
- train_loss
- eval_loss
- mean_token_accuracy
- grad_norm
- gpu_memory_reserved_gb
- oom_event_count
- rewards_chosen
- rewards_rejected
- reward_margin
- reward_accuracy
- checkpoint_restore_success_rate

### 5.2 Guardrails
- Freeze experiment seeds and dataloader order before A/B comparisons.
- Do not change tokenizer, template, and stop tokens in the same experiment run.
- Fail CI if train and eval tokenization policies diverge.
- Reject training runs with missing dataset lineage metadata.
- Always keep a reference checkpoint for DPO and recovery.
- Require at least one held-out domain eval set before merge.
- Require a rollback checkpoint from the previous release.

## 6. Codebook
Each recipe is production-oriented and intentionally explicit.

### Recipe 01: QLoRA model bootstrap with NF4 and double quant
Use this baseline when GPU memory is constrained and you need stable adapter tuning.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "mistralai/Mistral-7B-v0.1"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
    use_cache=False,
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
```

Notes:
- Prefer target_modules='all-linear' for QLoRA-style coverage.
- Set use_cache=False during training to avoid gradient checkpointing conflicts.

### Recipe 02: SFTTrainer with assistant-only loss and packing
Use this for conversational instruction tuning where you only want to optimize assistant tokens.

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

train_ds = load_dataset("trl-lib/Capybara", split="train")

config = SFTConfig(
    output_dir="runs/sft-capybara",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    bf16=True,
    packing=True,
    assistant_only_loss=True,
    max_length=2048,
)

trainer = SFTTrainer(
    model=model_id,
    args=config,
    train_dataset=train_ds,
    processing_class=tokenizer,
)
trainer.train()
```

Notes:
- Use assistant_only_loss only with compatible chat templates.
- Always inspect a few tokenized samples before long runs.

### Recipe 03: DPOTrainer baseline with explicit preference format
Use this when preference pairs are available and you want simpler alignment than PPO.

```python
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
cfg = DPOConfig(
    output_dir="runs/dpo-ultrafeedback",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    beta=0.1,
    max_length=2048,
    logging_steps=10,
    save_steps=200,
    bf16=True,
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=cfg,
    train_dataset=ds,
)
trainer.train()
```

Notes:
- Confirm chosen/rejected quality manually on a random sample.
- Track reward margin and reward accuracy trends, not just loss.

### Recipe 04: Accelerate FSDP configuration for large full fine-tuning
Use this for full-parameter tuning where model state sharding is required.

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_processes: 8
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
```

Notes:
- When fsdp_cpu_ram_efficient_loading=true, fsdp_sync_module_states must also be true.
- Merge sharded checkpoints before downstream handoff if required by tooling.

### Recipe 05: Dataset preprocessing with chat templates
Use this to make train-time formatting deterministic and reproducible.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
ds = load_dataset("trl-lib/Capybara", split="train")

def format_chat(row):
    messages = row["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

ds = ds.map(format_chat)
print(ds[0]["text"][:400])
```

Notes:
- Use add_generation_prompt=False for training preprocessing.
- If tokenizing later, set add_special_tokens=False to avoid duplication.

### Recipe 06: Checkpoint save and resume health check
Use this to verify that checkpoints are resumable before long jobs continue.

```bash
set -euo pipefail
RUN_DIR=runs/sft-capybara
python train.py --output_dir "$RUN_DIR" --max_steps 100
test -d "$RUN_DIR/checkpoint-100"
python train.py --output_dir "$RUN_DIR" --resume_from_checkpoint "$RUN_DIR/checkpoint-100" --max_steps 120
python - <<'PY'
import json, pathlib
s = pathlib.Path("runs/sft-capybara/trainer_state.json")
data = json.loads(s.read_text())
print("global_step", data.get("global_step"))
PY
```

Notes:
- Block release if resume fails.
- Retain at least one older checkpoint for rollback.

## 7. Failure and Recovery Matrix
This matrix is intentionally exhaustive. Follow one case at a time and log every change.

### Case 001: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: train_loss decreases without NaN

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 002: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 003: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 004: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 005: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: GPU memory peak remains under planned threshold

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 006: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 007: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: token length distribution is stable across splits

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 008: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 009: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 010: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 011: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: reward_margin remains positive and stable

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 012: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 013: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: resume run reaches expected global_step

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 014: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 015: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 016: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 017: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: grad_norm remains bounded

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 018: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 019: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: GPU memory peak remains under planned threshold

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 020: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 021: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 022: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 023: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: eval loss and downstream metric move in same direction

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 024: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 025: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: reward_margin remains positive and stable

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 026: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 027: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 028: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 029: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: train_loss decreases without NaN

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 030: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 031: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: grad_norm remains bounded

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 032: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 033: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 034: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 035: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: token length distribution is stable across splits

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 036: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 037: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: eval loss and downstream metric move in same direction

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 038: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 039: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 040: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 041: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: resume run reaches expected global_step

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 042: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 043: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: train_loss decreases without NaN

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 044: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 045: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 046: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 047: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: GPU memory peak remains under planned threshold

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 048: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 049: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: token length distribution is stable across splits

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 050: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 051: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 052: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 053: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: reward_margin remains positive and stable

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 054: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 055: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: resume run reaches expected global_step

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 056: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 057: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 058: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 059: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: grad_norm remains bounded

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 060: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 061: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: GPU memory peak remains under planned threshold

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 062: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 063: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 064: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 065: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: eval loss and downstream metric move in same direction

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 066: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 067: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: reward_margin remains positive and stable

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 068: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 069: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 070: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 071: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: train_loss decreases without NaN

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 072: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 073: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: grad_norm remains bounded

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 074: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 075: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 076: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 077: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: token length distribution is stable across splits

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 078: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 079: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: eval loss and downstream metric move in same direction

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 080: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 081: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 082: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 083: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: resume run reaches expected global_step

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 084: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 085: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: train_loss decreases without NaN

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 086: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 087: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 088: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 089: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: GPU memory peak remains under planned threshold

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 090: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 091: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: token length distribution is stable across splits

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 092: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 093: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 094: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 095: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: reward_margin remains positive and stable

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 096: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 097: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: resume run reaches expected global_step

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 098: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 099: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 100: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 101: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: grad_norm remains bounded

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 102: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 103: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: GPU memory peak remains under planned threshold

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 104: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 105: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 106: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 107: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: eval loss and downstream metric move in same direction

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 108: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 109: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: reward_margin remains positive and stable

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 110: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 111: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 112: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 113: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: train_loss decreases without NaN

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 114: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 115: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: grad_norm remains bounded

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 116: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 117: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 118: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 119: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: token length distribution is stable across splits

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 120: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 121: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: eval loss and downstream metric move in same direction

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 122: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 123: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 124: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 125: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: resume run reaches expected global_step

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 126: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 127: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: train_loss decreases without NaN

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 128: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 129: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 130: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 131: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: GPU memory peak remains under planned threshold

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 132: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 133: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: token length distribution is stable across splits

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 134: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 135: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 136: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 137: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: reward_margin remains positive and stable

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 138: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 139: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: resume run reaches expected global_step

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 140: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 141: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 142: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 143: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: grad_norm remains bounded

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 144: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 145: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: GPU memory peak remains under planned threshold

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 146: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 147: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 148: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 149: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: eval loss and downstream metric move in same direction

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 150: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 151: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: reward_margin remains positive and stable

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 152: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 153: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 154: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 155: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: train_loss decreases without NaN

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 156: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 157: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: grad_norm remains bounded

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 158: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 159: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 160: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 161: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: token length distribution is stable across splits

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 162: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 163: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: eval loss and downstream metric move in same direction

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 164: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 165: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 166: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 167: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: resume run reaches expected global_step

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 168: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 169: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: train_loss decreases without NaN

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 170: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 171: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 172: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 173: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: GPU memory peak remains under planned threshold

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 174: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 175: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: token length distribution is stable across splits

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 176: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 177: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 178: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: grad_norm remains bounded

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 179: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: reward_margin remains positive and stable

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 180: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: GPU memory peak remains under planned threshold

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 181: training loss is NaN
- Signal: training loss is NaN
- Likely cause: learning rate too high for active trainable parameter count
- Immediate action: reduce learning rate and rerun a 100-step sanity sweep
- Verification metric: resume run reaches expected global_step

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 182: eval loss improves but downstream task fails
- Signal: eval loss improves but downstream task fails
- Likely cause: incompatible chat template or duplicated special tokens
- Immediate action: inspect tokenized samples and compare against expected chat template
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 183: gradient norm spikes unpredictably
- Signal: gradient norm spikes unpredictably
- Likely cause: mixed precision instability due to unsupported kernels
- Immediate action: enable gradient clipping and verify grad_norm trend
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 184: DPO reward margin collapses
- Signal: DPO reward margin collapses
- Likely cause: incorrect preference pair formatting
- Immediate action: validate chosen/rejected pair mapping and prompt boundaries
- Verification metric: eval loss and downstream metric move in same direction

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

### Case 185: OOM during backward pass
- Signal: OOM during backward pass
- Likely cause: sequence lengths exceed planned memory envelope
- Immediate action: lower max_length and activate packing or checkpointing
- Verification metric: grad_norm remains bounded

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

### Case 186: checkpoint resume diverges
- Signal: checkpoint resume diverges
- Likely cause: optimizer state corruption from interrupted writes
- Immediate action: run checkpoint integrity check and restore from prior checkpoint
- Verification metric: reward_margin remains positive and stable

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('runs/sft-capybara/trainer_state.json')
print('exists', p.exists())
print(p.read_text()[:200] if p.exists() else '')
PY
```

### Case 187: tokenization mismatch between train and eval
- Signal: tokenization mismatch between train and eval
- Likely cause: padding-side mismatch between training and evaluation
- Immediate action: set tokenizer.padding_side explicitly and rerun evaluation
- Verification metric: GPU memory peak remains under planned threshold

```bash
python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'
```

### Case 188: assistant responses become truncated
- Signal: assistant responses become truncated
- Likely cause: stop token IDs not aligned with model family
- Immediate action: align eos and stop token IDs with model tokenizer config
- Verification metric: resume run reaches expected global_step

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 189: learning stalls after first epoch
- Signal: learning stalls after first epoch
- Likely cause: frozen modules are accidentally unfrozen
- Immediate action: rebuild optimizer with only trainable parameters
- Verification metric: token length distribution is stable across splits

```bash
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

### Case 190: reference model drift in preference runs
- Signal: reference model drift in preference runs
- Likely cause: dataset leakage from train into eval
- Immediate action: recreate deterministic split with hashed IDs
- Verification metric: train_loss decreases without NaN

```bash
python - <<'PY'
from datasets import load_dataset
ds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')
print(ds[0].keys())
PY
```

## 8. Validation Drills
Complete every drill before promoting a change to production.

- [ ] Drill 001: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 002: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 003: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 004: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 005: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 006: Compute and store train/eval token length histograms.
- [ ] Drill 007: Run a held-out task benchmark before and after tuning.
- [ ] Drill 008: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 009: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 010: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 011: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 012: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 013: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 014: Compute and store train/eval token length histograms.
- [ ] Drill 015: Run a held-out task benchmark before and after tuning.
- [ ] Drill 016: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 017: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 018: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 019: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 020: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 021: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 022: Compute and store train/eval token length histograms.
- [ ] Drill 023: Run a held-out task benchmark before and after tuning.
- [ ] Drill 024: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 025: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 026: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 027: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 028: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 029: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 030: Compute and store train/eval token length histograms.
- [ ] Drill 031: Run a held-out task benchmark before and after tuning.
- [ ] Drill 032: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 033: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 034: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 035: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 036: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 037: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 038: Compute and store train/eval token length histograms.
- [ ] Drill 039: Run a held-out task benchmark before and after tuning.
- [ ] Drill 040: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 041: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 042: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 043: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 044: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 045: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 046: Compute and store train/eval token length histograms.
- [ ] Drill 047: Run a held-out task benchmark before and after tuning.
- [ ] Drill 048: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 049: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 050: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 051: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 052: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 053: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 054: Compute and store train/eval token length histograms.
- [ ] Drill 055: Run a held-out task benchmark before and after tuning.
- [ ] Drill 056: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 057: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 058: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 059: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 060: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 061: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 062: Compute and store train/eval token length histograms.
- [ ] Drill 063: Run a held-out task benchmark before and after tuning.
- [ ] Drill 064: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 065: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 066: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 067: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 068: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 069: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 070: Compute and store train/eval token length histograms.
- [ ] Drill 071: Run a held-out task benchmark before and after tuning.
- [ ] Drill 072: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 073: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 074: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 075: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 076: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 077: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 078: Compute and store train/eval token length histograms.
- [ ] Drill 079: Run a held-out task benchmark before and after tuning.
- [ ] Drill 080: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 081: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 082: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 083: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 084: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 085: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 086: Compute and store train/eval token length histograms.
- [ ] Drill 087: Run a held-out task benchmark before and after tuning.
- [ ] Drill 088: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 089: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 090: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 091: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 092: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 093: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 094: Compute and store train/eval token length histograms.
- [ ] Drill 095: Run a held-out task benchmark before and after tuning.
- [ ] Drill 096: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 097: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 098: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 099: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 100: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 101: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 102: Compute and store train/eval token length histograms.
- [ ] Drill 103: Run a held-out task benchmark before and after tuning.
- [ ] Drill 104: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 105: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 106: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 107: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 108: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 109: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 110: Compute and store train/eval token length histograms.
- [ ] Drill 111: Run a held-out task benchmark before and after tuning.
- [ ] Drill 112: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 113: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 114: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 115: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 116: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 117: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 118: Compute and store train/eval token length histograms.
- [ ] Drill 119: Run a held-out task benchmark before and after tuning.
- [ ] Drill 120: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 121: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 122: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 123: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 124: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 125: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 126: Compute and store train/eval token length histograms.
- [ ] Drill 127: Run a held-out task benchmark before and after tuning.
- [ ] Drill 128: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 129: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 130: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 131: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 132: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 133: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 134: Compute and store train/eval token length histograms.
- [ ] Drill 135: Run a held-out task benchmark before and after tuning.
- [ ] Drill 136: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 137: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 138: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 139: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 140: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 141: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 142: Compute and store train/eval token length histograms.
- [ ] Drill 143: Run a held-out task benchmark before and after tuning.
- [ ] Drill 144: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 145: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 146: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 147: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 148: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 149: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 150: Compute and store train/eval token length histograms.
- [ ] Drill 151: Run a held-out task benchmark before and after tuning.
- [ ] Drill 152: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 153: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 154: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 155: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 156: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 157: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 158: Compute and store train/eval token length histograms.
- [ ] Drill 159: Run a held-out task benchmark before and after tuning.
- [ ] Drill 160: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 161: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 162: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 163: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 164: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 165: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 166: Compute and store train/eval token length histograms.
- [ ] Drill 167: Run a held-out task benchmark before and after tuning.
- [ ] Drill 168: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 169: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 170: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 171: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 172: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 173: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 174: Compute and store train/eval token length histograms.
- [ ] Drill 175: Run a held-out task benchmark before and after tuning.
- [ ] Drill 176: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 177: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 178: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 179: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 180: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 181: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 182: Compute and store train/eval token length histograms.
- [ ] Drill 183: Run a held-out task benchmark before and after tuning.
- [ ] Drill 184: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 185: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 186: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 187: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 188: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 189: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 190: Compute and store train/eval token length histograms.
- [ ] Drill 191: Run a held-out task benchmark before and after tuning.
- [ ] Drill 192: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 193: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 194: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 195: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 196: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 197: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 198: Compute and store train/eval token length histograms.
- [ ] Drill 199: Run a held-out task benchmark before and after tuning.
- [ ] Drill 200: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 201: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 202: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 203: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 204: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 205: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 206: Compute and store train/eval token length histograms.
- [ ] Drill 207: Run a held-out task benchmark before and after tuning.
- [ ] Drill 208: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 209: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 210: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 211: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 212: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 213: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 214: Compute and store train/eval token length histograms.
- [ ] Drill 215: Run a held-out task benchmark before and after tuning.
- [ ] Drill 216: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 217: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 218: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 219: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 220: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 221: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 222: Compute and store train/eval token length histograms.
- [ ] Drill 223: Run a held-out task benchmark before and after tuning.
- [ ] Drill 224: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 225: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 226: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 227: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 228: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 229: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 230: Compute and store train/eval token length histograms.
- [ ] Drill 231: Run a held-out task benchmark before and after tuning.
- [ ] Drill 232: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 233: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 234: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 235: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 236: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 237: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 238: Compute and store train/eval token length histograms.
- [ ] Drill 239: Run a held-out task benchmark before and after tuning.
- [ ] Drill 240: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 241: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 242: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 243: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 244: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 245: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 246: Compute and store train/eval token length histograms.
- [ ] Drill 247: Run a held-out task benchmark before and after tuning.
- [ ] Drill 248: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 249: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 250: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 251: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 252: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 253: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 254: Compute and store train/eval token length histograms.
- [ ] Drill 255: Run a held-out task benchmark before and after tuning.
- [ ] Drill 256: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 257: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 258: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 259: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 260: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 261: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 262: Compute and store train/eval token length histograms.
- [ ] Drill 263: Run a held-out task benchmark before and after tuning.
- [ ] Drill 264: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 265: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 266: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 267: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 268: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 269: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 270: Compute and store train/eval token length histograms.
- [ ] Drill 271: Run a held-out task benchmark before and after tuning.
- [ ] Drill 272: Confirm model card metadata includes dataset lineage and license notes.
- [ ] Drill 273: Run a 200-step smoke training and validate no NaN values in metrics.
- [ ] Drill 274: Validate exactly 20 random tokenized rows against intended chat template.
- [ ] Drill 275: A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.
- [ ] Drill 276: Verify resume-from-checkpoint consistency at step boundaries.
- [ ] Drill 277: Confirm assistant-only masking excludes user and system spans.
- [ ] Drill 278: Compute and store train/eval token length histograms.
- [ ] Drill 279: Run a held-out task benchmark before and after tuning.
- [ ] Drill 280: Confirm model card metadata includes dataset lineage and license notes.

