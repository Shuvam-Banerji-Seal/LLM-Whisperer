# Fine-Tuning Evaluation and Operations - Agentic Skill Prompt

Use this prompt to harden fine-tuning workflows with robust evaluation, checkpointing, distributed training controls, and release criteria.

## 1. Mission

Ensure training outputs are reproducible, measurable, and production-safe before release.

## 2. Evaluation Layers

1. In-training metrics: loss, token accuracy, gradient norms.
2. Task metrics: benchmark suites and domain-specific test sets.
3. Safety metrics: harmful content tendencies, refusal balance, policy compliance.

## 3. Benchmark Execution Baseline

```bash
pip install -U "lm_eval[hf,vllm]"
lm_eval \
  --model hf \
  --model_args pretrained=your/model \
  --tasks hellaswag,arc_easy,mmlu \
  --batch_size auto \
  --output_path eval_out \
  --log_samples
```

Add domain-specific eval suites before production sign-off.

## 4. Checkpoint Strategy

- Save frequent incremental checkpoints.
- Keep `save_total_limit` to bound storage.
- Validate resume logic before long runs.

Accelerate state example:

```python
from accelerate import Accelerator

acc = Accelerator(project_dir="runs/exp1")
acc.save_state()
acc.load_state("runs/exp1/checkpoints/checkpoint_3")
```

## 5. Distributed Training Guidance

- Use DeepSpeed ZeRO stages for larger model footprints.
- Use FSDP full-shard for strong memory savings where stable.
- Validate communication overhead versus throughput gains.

Launch patterns:

```bash
accelerate config
accelerate launch --config_file ds_zero3.yaml train.py
accelerate launch --config_file fsdp_full_shard.yaml train.py
```

## 6. Release Gates

A candidate is release-ready only if:

- Core task metrics meet target deltas versus baseline.
- Safety and policy checks pass on required slices.
- Throughput and latency meet serving SLO assumptions.
- Rollback artifacts are available and validated.

## 7. Data and Safety Controls

- Track provenance and licensing for every dataset split.
- Ensure train and eval split integrity and leakage checks.
- Scrub sensitive data and known secrets.
- Keep red-team prompts for recurring regression checks.

## 8. Incident and Rollback Protocol

If release regression appears:

1. Freeze rollout.
2. Revert to last validated checkpoint.
3. Compare diff in data, hyperparameters, and code revision.
4. Re-run blocked benchmark and safety slices.

## 9. References (Fetched 2026-04-06)

1. https://github.com/EleutherAI/lm-evaluation-harness - Standard LLM benchmark execution framework.
2. https://huggingface.co/docs/lighteval/index - Hugging Face evaluation toolkit.
3. https://huggingface.co/docs/evaluate/index - General metric tooling for custom tasks.
4. https://huggingface.co/docs/accelerate/index - Distributed and mixed-precision training framework.
5. https://huggingface.co/docs/accelerate/usage_guides/checkpoint - Checkpoint lifecycle and restoration.
6. https://huggingface.co/docs/accelerate/usage_guides/deepspeed - DeepSpeed integration guide.
7. https://huggingface.co/docs/accelerate/usage_guides/fsdp - FSDP integration guide.
8. https://pytorch.org/docs/stable/fsdp.html - FSDP internals and API options.
9. https://deepspeed.readthedocs.io/en/latest/index.html - DeepSpeed documentation.
10. https://huggingface.co/docs/transformers/main_classes/trainer - Trainer checkpoint and eval controls.
11. https://huggingface.co/docs/trl/sft_trainer - SFT training instrumentation and logging.
12. https://huggingface.co/docs/peft/index - PEFT adapters and trainable-parameter operations.
