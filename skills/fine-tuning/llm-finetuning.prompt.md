# LLM Fine-Tuning (Full FT, SFT, PEFT, LoRA, QLoRA) - Agentic Skill Prompt

Use this prompt when planning or executing supervised fine-tuning workflows for causal language models.

## 1. Mission

Select and run the right fine-tuning method for the task, while balancing quality, compute cost, and deployment constraints.

## 2. Method Selection

- Full fine-tuning: choose when task shift is large and compute budget is high.
- SFT (instruction tuning): choose for behavior shaping and chat alignment.
- LoRA or QLoRA: choose for compute-efficient adaptation with minimal trainable parameters.

Practical default for limited hardware: QLoRA with robust evaluation.

## 3. Environment Baseline

```bash
pip install -U transformers accelerate datasets trl peft bitsandbytes evaluate
accelerate config
```

## 4. Reproducibility Baseline

- Set seeds.
- Pin package versions and model revisions.
- Log git SHA and full training config.
- Save tokenizer and chat template with checkpoints.

```python
from transformers import set_seed
set_seed(42)
```

## 5. Full Fine-Tuning Recipe (Trainer)

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("your/dataset", split="train")

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=4096)

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
splits = dataset.train_test_split(test_size=0.02)

model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

args = TrainingArguments(
    output_dir="out/fullft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.1,
    bf16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=splits["train"],
    eval_dataset=splits["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
```

## 6. SFT Recipe (TRL SFTTrainer)

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

train = load_dataset("trl-lib/Capybara", split="train")

cfg = SFTConfig(
    output_dir="out/sft",
    learning_rate=2e-5,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=4096,
    packing=True,
    assistant_only_loss=True,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B-Base",
    args=cfg,
    train_dataset=train,
)
trainer.train()
```

## 7. LoRA and QLoRA Recipe (PEFT)

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb,
)
base = prepare_model_for_kbit_training(base)

lora = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base, lora)
model.print_trainable_parameters()
```

## 8. Hyperparameter Starting Points

| Model Class | Method | LR Range | Sequence Length | Notes |
|---|---|---|---|---|
| 7B | Full FT | 1e-5 to 2e-5 | 4k to 8k | Strong quality if compute allows |
| 7B | QLoRA | 1e-4 to 2e-4 | 4k to 8k | Good quality-cost tradeoff |
| 13B | Full FT | 8e-6 to 1.5e-5 | 4k to 8k | Often needs distributed setup |
| 13B | QLoRA | 8e-5 to 2e-4 | 4k to 8k | Start with q or k or v or o or all-linear targeting |
| 70B | Full FT | 3e-6 to 8e-6 | 4k to 8k | Strong infra and checkpoint strategy required |
| 70B | QLoRA | 5e-5 to 1e-4 | 4k to 8k | Most practical with constrained budget |

Treat these as initialization ranges, not universal optima.

## 9. Validation and Evaluation

- During training: eval loss, token-level metrics, gradient stability.
- Post training: benchmark on representative tasks and safety slices.
- Keep baseline comparison before and after adaptation.

```bash
pip install -U "lm_eval[hf,vllm]"
lm_eval --model hf --model_args pretrained=your/model --tasks hellaswag,arc_easy,mmlu --batch_size auto
```

## 10. Data Quality and Safety Checklist

- Confirm dataset license and allowed use.
- Deduplicate and remove leakage across splits.
- Normalize schema and chat roles.
- Scrub PII and secrets.
- Inspect harmful or illegal content handling and policy alignment.

## 11. References (Fetched 2026-04-06)

1. https://huggingface.co/docs/transformers/training - Transformers training overview.
2. https://huggingface.co/docs/transformers/main_classes/trainer - Trainer API and training controls.
3. https://huggingface.co/docs/transformers/chat_templating - Chat template correctness for SFT workflows.
4. https://huggingface.co/docs/trl/sft_trainer - TRL SFTTrainer reference.
5. https://huggingface.co/docs/trl/index - TRL project documentation.
6. https://huggingface.co/docs/peft/index - PEFT methods overview.
7. https://huggingface.co/docs/peft/package_reference/lora - LoRA configuration options.
8. https://huggingface.co/docs/peft/developer_guides/quantization - PEFT quantization guidance.
9. https://huggingface.co/papers/2305.14314 - QLoRA paper summary.
10. https://huggingface.co/docs/bitsandbytes/main/en/index - bitsandbytes documentation.
11. https://huggingface.co/docs/accelerate/index - Accelerate fundamentals.
12. https://huggingface.co/docs/accelerate/usage_guides/deepspeed - DeepSpeed integration via Accelerate.
13. https://huggingface.co/docs/accelerate/usage_guides/fsdp - FSDP integration guidance.
14. https://huggingface.co/docs/accelerate/usage_guides/checkpoint - Checkpointing and resume in Accelerate.
15. https://pytorch.org/docs/stable/fsdp.html - PyTorch FSDP API reference.
16. https://deepspeed.readthedocs.io/en/latest/index.html - DeepSpeed official docs.
17. https://github.com/EleutherAI/lm-evaluation-harness - Evaluation harness for LLMs.
18. https://huggingface.co/docs/lighteval/index - Hugging Face LightEval docs.
19. https://huggingface.co/docs/evaluate/index - Evaluation library docs.

## 12. Uncertainty Notes

- Best hyperparameters vary by model family and dataset entropy.
- Distributed strategy preference (DeepSpeed vs FSDP) depends on hardware topology and workload.
