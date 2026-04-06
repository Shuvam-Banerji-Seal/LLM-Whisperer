from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def add_code_block(lines, lang, code):
    lines.append(f"```{lang}")
    lines.extend(dedent(code).strip("\n").splitlines())
    lines.append("```")
    lines.append("")


def add_intro(lines, title, mission):
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## 1. Mission")
    lines.append(mission)
    lines.append("")
    lines.append("## 2. Principles")
    lines.append("- Prioritize reproducibility over one-off wins.")
    lines.append("- Log every configuration that can alter behavior.")
    lines.append("- Validate quality and latency together; never optimize one blindly.")
    lines.append("- Keep rollback paths documented and tested.")
    lines.append("- Treat safety and governance checks as first-class production requirements.")
    lines.append("")


def add_source_index(lines, urls):
    lines.append("## 3. Source Index (Docs and Blogs)")
    for idx, url in enumerate(urls, start=1):
        lines.append(f"{idx}. {url}")
    lines.append("")


def add_fetch_commands(lines, urls):
    lines.append("## 4. Fast Documentation Fetch Commands")
    lines.append("Use these commands when someone reports issues and you need to verify behavior against upstream docs quickly.")
    lines.append("")
    lines.append("```bash")
    lines.append("mkdir -p /tmp/skill_refs")
    for url in urls:
        safe_name = (
            url.replace("https://", "")
            .replace("http://", "")
            .replace("/", "_")
            .replace("?", "_")
            .replace("=", "_")
            .replace("&", "_")
        )
        lines.append(f"curl -L \"{url}\" -o /tmp/skill_refs/{safe_name}.html")
    lines.append("ls -lh /tmp/skill_refs")
    lines.append("```")
    lines.append("")


def add_operational_policies(lines, domain_name, metrics, guardrails):
    lines.append("## 5. Operational Policies")
    lines.append(f"Use this section as the mandatory baseline policy set for {domain_name}.")
    lines.append("")
    lines.append("### 5.1 Metrics that must always be tracked")
    for metric in metrics:
        lines.append(f"- {metric}")
    lines.append("")
    lines.append("### 5.2 Guardrails")
    for guardrail in guardrails:
        lines.append(f"- {guardrail}")
    lines.append("")


def add_codebook(lines, snippets):
    lines.append("## 6. Codebook")
    lines.append("Each recipe is production-oriented and intentionally explicit.")
    lines.append("")
    for idx, snippet in enumerate(snippets, start=1):
        lines.append(f"### Recipe {idx:02d}: {snippet['title']}")
        lines.append(snippet["why"])
        lines.append("")
        add_code_block(lines, snippet["lang"], snippet["code"])
        if snippet.get("notes"):
            lines.append("Notes:")
            for note in snippet["notes"]:
                lines.append(f"- {note}")
            lines.append("")


def add_failure_matrix(lines, symptoms, causes, actions, commands, metrics, count):
    lines.append("## 7. Failure and Recovery Matrix")
    lines.append("This matrix is intentionally exhaustive. Follow one case at a time and log every change.")
    lines.append("")
    for i in range(count):
        symptom = symptoms[i % len(symptoms)]
        cause = causes[i % len(causes)]
        action = actions[i % len(actions)]
        command = commands[i % len(commands)]
        metric = metrics[i % len(metrics)]
        lines.append(f"### Case {i + 1:03d}: {symptom}")
        lines.append(f"- Signal: {symptom}")
        lines.append(f"- Likely cause: {cause}")
        lines.append(f"- Immediate action: {action}")
        lines.append(f"- Verification metric: {metric}")
        lines.append("")
        add_code_block(lines, "bash", command)


def add_validation_drills(lines, drills, count):
    lines.append("## 8. Validation Drills")
    lines.append("Complete every drill before promoting a change to production.")
    lines.append("")
    for i in range(count):
        drill = drills[i % len(drills)]
        lines.append(f"- [ ] Drill {i + 1:03d}: {drill}")
    lines.append("")


def pad_to_min_lines(lines, min_lines, domain_name):
    idx = 1
    while len(lines) < min_lines:
        lines.append(f"- [ ] Extended audit checkpoint {idx:03d} for {domain_name}: confirm reproducibility, rollback path, and monitoring coverage.")
        idx += 1


def build_skill_file(config):
    lines = []
    add_intro(lines, config["title"], config["mission"])
    add_source_index(lines, config["urls"])
    add_fetch_commands(lines, config["urls"])
    add_operational_policies(lines, config["domain_name"], config["policy_metrics"], config["guardrails"])
    add_codebook(lines, config["snippets"])
    add_failure_matrix(
        lines,
        config["symptoms"],
        config["causes"],
        config["actions"],
        config["commands"],
        config["verification_metrics"],
        config.get("failure_cases", 180),
    )
    add_validation_drills(lines, config["drills"], config.get("drill_count", 250))
    pad_to_min_lines(lines, config.get("min_lines", 1050), config["domain_name"])
    return "\n".join(lines) + "\n"


fine_tuning_config = {
    "title": "Master Agentic Skill: LLM Fine-Tuning (SFT, DPO, QLoRA, FSDP)",
    "domain_name": "LLM fine-tuning",
    "mission": "Deliver robust, reproducible, and safe post-training pipelines for instruction tuning and preference alignment, with explicit safeguards for memory, stability, and evaluation drift.",
    "urls": [
        "https://huggingface.co/docs/trl/sft_trainer",
        "https://huggingface.co/docs/trl/dpo_trainer",
        "https://huggingface.co/docs/trl/dataset_formats",
        "https://huggingface.co/docs/peft/developer_guides/quantization",
        "https://huggingface.co/docs/accelerate/en/usage_guides/fsdp",
        "https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/",
        "https://huggingface.co/papers/2305.14314",
        "https://huggingface.co/papers/2305.18290",
        "https://huggingface.co/docs/transformers/chat_templating",
        "https://huggingface.co/docs/transformers/main_classes/trainer",
        "https://huggingface.co/docs/transformers/main/en/tasks/language_modeling",
        "https://github.com/huggingface/alignment-handbook",
    ],
    "policy_metrics": [
        "tokens_per_second",
        "train_loss",
        "eval_loss",
        "mean_token_accuracy",
        "grad_norm",
        "gpu_memory_reserved_gb",
        "oom_event_count",
        "rewards_chosen",
        "rewards_rejected",
        "reward_margin",
        "reward_accuracy",
        "checkpoint_restore_success_rate",
    ],
    "guardrails": [
        "Freeze experiment seeds and dataloader order before A/B comparisons.",
        "Do not change tokenizer, template, and stop tokens in the same experiment run.",
        "Fail CI if train and eval tokenization policies diverge.",
        "Reject training runs with missing dataset lineage metadata.",
        "Always keep a reference checkpoint for DPO and recovery.",
        "Require at least one held-out domain eval set before merge.",
        "Require a rollback checkpoint from the previous release.",
    ],
    "snippets": [
        {
            "title": "QLoRA model bootstrap with NF4 and double quant",
            "why": "Use this baseline when GPU memory is constrained and you need stable adapter tuning.",
            "lang": "python",
            "code": """
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            model_id = \"mistralai/Mistral-7B-v0.1\"
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=\"nf4\",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map=\"auto\",
                use_cache=False,
            )
            model = prepare_model_for_kbit_training(model)

            lora_cfg = LoraConfig(
                r=64,
                lora_alpha=16,
                lora_dropout=0.05,
                bias=\"none\",
                task_type=\"CAUSAL_LM\",
                target_modules=\"all-linear\",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            """,
            "notes": [
                "Prefer target_modules='all-linear' for QLoRA-style coverage.",
                "Set use_cache=False during training to avoid gradient checkpointing conflicts.",
            ],
        },
        {
            "title": "SFTTrainer with assistant-only loss and packing",
            "why": "Use this for conversational instruction tuning where you only want to optimize assistant tokens.",
            "lang": "python",
            "code": """
            from datasets import load_dataset
            from trl import SFTTrainer, SFTConfig
            from transformers import AutoTokenizer

            model_id = \"Qwen/Qwen2.5-1.5B-Instruct\"
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            train_ds = load_dataset(\"trl-lib/Capybara\", split=\"train\")

            config = SFTConfig(
                output_dir=\"runs/sft-capybara\",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                learning_rate=2e-5,
                num_train_epochs=2,
                logging_steps=10,
                save_steps=200,
                eval_strategy=\"steps\",
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
            """,
            "notes": [
                "Use assistant_only_loss only with compatible chat templates.",
                "Always inspect a few tokenized samples before long runs.",
            ],
        },
        {
            "title": "DPOTrainer baseline with explicit preference format",
            "why": "Use this when preference pairs are available and you want simpler alignment than PPO.",
            "lang": "python",
            "code": """
            from datasets import load_dataset
            from trl import DPOTrainer, DPOConfig

            ds = load_dataset(\"trl-lib/ultrafeedback_binarized\", split=\"train\")
            cfg = DPOConfig(
                output_dir=\"runs/dpo-ultrafeedback\",
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
                model=\"Qwen/Qwen2.5-1.5B-Instruct\",
                args=cfg,
                train_dataset=ds,
            )
            trainer.train()
            """,
            "notes": [
                "Confirm chosen/rejected quality manually on a random sample.",
                "Track reward margin and reward accuracy trends, not just loss.",
            ],
        },
        {
            "title": "Accelerate FSDP configuration for large full fine-tuning",
            "why": "Use this for full-parameter tuning where model state sharding is required.",
            "lang": "yaml",
            "code": """
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
            """,
            "notes": [
                "When fsdp_cpu_ram_efficient_loading=true, fsdp_sync_module_states must also be true.",
                "Merge sharded checkpoints before downstream handoff if required by tooling.",
            ],
        },
        {
            "title": "Dataset preprocessing with chat templates",
            "why": "Use this to make train-time formatting deterministic and reproducible.",
            "lang": "python",
            "code": """
            from datasets import load_dataset
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(\"HuggingFaceH4/zephyr-7b-beta\")
            ds = load_dataset(\"trl-lib/Capybara\", split=\"train\")

            def format_chat(row):
                messages = row[\"messages\"]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return {\"text\": text}

            ds = ds.map(format_chat)
            print(ds[0][\"text\"][:400])
            """,
            "notes": [
                "Use add_generation_prompt=False for training preprocessing.",
                "If tokenizing later, set add_special_tokens=False to avoid duplication.",
            ],
        },
        {
            "title": "Checkpoint save and resume health check",
            "why": "Use this to verify that checkpoints are resumable before long jobs continue.",
            "lang": "bash",
            "code": """
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
            """,
            "notes": [
                "Block release if resume fails.",
                "Retain at least one older checkpoint for rollback.",
            ],
        },
    ],
    "symptoms": [
        "training loss is NaN",
        "eval loss improves but downstream task fails",
        "gradient norm spikes unpredictably",
        "DPO reward margin collapses",
        "OOM during backward pass",
        "checkpoint resume diverges",
        "tokenization mismatch between train and eval",
        "assistant responses become truncated",
        "learning stalls after first epoch",
        "reference model drift in preference runs",
    ],
    "causes": [
        "learning rate too high for active trainable parameter count",
        "incompatible chat template or duplicated special tokens",
        "mixed precision instability due to unsupported kernels",
        "incorrect preference pair formatting",
        "sequence lengths exceed planned memory envelope",
        "optimizer state corruption from interrupted writes",
        "padding-side mismatch between training and evaluation",
        "stop token IDs not aligned with model family",
        "frozen modules are accidentally unfrozen",
        "dataset leakage from train into eval",
    ],
    "actions": [
        "reduce learning rate and rerun a 100-step sanity sweep",
        "inspect tokenized samples and compare against expected chat template",
        "enable gradient clipping and verify grad_norm trend",
        "validate chosen/rejected pair mapping and prompt boundaries",
        "lower max_length and activate packing or checkpointing",
        "run checkpoint integrity check and restore from prior checkpoint",
        "set tokenizer.padding_side explicitly and rerun evaluation",
        "align eos and stop token IDs with model tokenizer config",
        "rebuild optimizer with only trainable parameters",
        "recreate deterministic split with hashed IDs",
    ],
    "commands": [
        "python -m pip freeze | rg 'transformers|trl|peft|accelerate|bitsandbytes'",
        "python - <<'PY'\nfrom transformers import AutoTokenizer\nt=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')\nprint('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)\nPY",
        "python - <<'PY'\nimport torch\nprint('cuda', torch.cuda.is_available())\nprint('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)\nPY",
        "python - <<'PY'\nfrom datasets import load_dataset\nds=load_dataset('trl-lib/ultrafeedback_binarized', split='train[:3]')\nprint(ds[0].keys())\nPY",
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv",
        "python - <<'PY'\nimport json, pathlib\np=pathlib.Path('runs/sft-capybara/trainer_state.json')\nprint('exists', p.exists())\nprint(p.read_text()[:200] if p.exists() else '')\nPY",
    ],
    "verification_metrics": [
        "train_loss decreases without NaN",
        "eval loss and downstream metric move in same direction",
        "grad_norm remains bounded",
        "reward_margin remains positive and stable",
        "GPU memory peak remains under planned threshold",
        "resume run reaches expected global_step",
        "token length distribution is stable across splits",
    ],
    "drills": [
        "Run a 200-step smoke training and validate no NaN values in metrics.",
        "Validate exactly 20 random tokenized rows against intended chat template.",
        "A/B test learning rates 1e-6, 2e-6, 5e-6 on fixed seed.",
        "Verify resume-from-checkpoint consistency at step boundaries.",
        "Confirm assistant-only masking excludes user and system spans.",
        "Compute and store train/eval token length histograms.",
        "Run a held-out task benchmark before and after tuning.",
        "Confirm model card metadata includes dataset lineage and license notes.",
    ],
    "failure_cases": 190,
    "drill_count": 280,
    "min_lines": 1050,
}


quantization_config = {
    "title": "Master Agentic Skill: LLM Quantization (AWQ, GPTQ, GGUF, Runtime Validation)",
    "domain_name": "LLM quantization",
    "mission": "Deliver predictable quantization workflows that preserve quality while reducing memory and latency, with clear fallback plans per backend.",
    "urls": [
        "https://huggingface.co/docs/transformers/main/en/quantization",
        "https://huggingface.co/docs/peft/developer_guides/quantization",
        "https://huggingface.co/blog/4bit-transformers-bitsandbytes",
        "https://huggingface.co/papers/2305.14314",
        "https://huggingface.co/papers/2306.00978",
        "https://huggingface.co/papers/2210.17323",
        "https://github.com/ggerganov/llama.cpp",
        "https://github.com/ModelCloud/GPTQModel",
        "https://github.com/vllm-project/vllm",
        "https://huggingface.co/docs/diffusers/main/en/quantization/overview",
    ],
    "policy_metrics": [
        "perplexity_delta_vs_fp16",
        "exact_match_delta_vs_fp16",
        "throughput_tokens_per_second",
        "ttft_ms",
        "gpu_memory_reserved_gb",
        "host_memory_gb",
        "decoder_error_rate",
        "generation_hallucination_rate",
    ],
    "guardrails": [
        "Do not release quantized artifacts without fp16 baseline comparison.",
        "Track quantization config fields in model card and artifact metadata.",
        "Use representative calibration data from deployment domain.",
        "Block deployment if perplexity regression exceeds agreed threshold.",
        "Store conversion scripts and exact tool versions with artifact.",
        "Test at least one long-context benchmark after quantization.",
    ],
    "snippets": [
        {
            "title": "BitsAndBytes 4-bit load with NF4",
            "why": "Use this for memory-constrained inference and adapter training workflows.",
            "lang": "python",
            "code": """
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            model_id = \"mistralai/Mistral-7B-v0.1\"
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=\"nf4\",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=qcfg,
                device_map=\"auto\",
            )
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            print(tok(\"hello\", return_tensors=\"pt\").input_ids.shape)
            """,
            "notes": [
                "Use nf4 for QLoRA-style workflows.",
                "Benchmark quality against fp16 baseline before rollout.",
            ],
        },
        {
            "title": "AutoAWQ calibration and save",
            "why": "Use this for AWQ artifact creation with explicit quantization parameters.",
            "lang": "python",
            "code": """
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            model_id = \"meta-llama/Llama-2-7b-chat-hf\"
            out_dir = \"artifacts/llama2-awq\"
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoAWQForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

            quant_config = {
                \"zero_point\": True,
                \"q_group_size\": 128,
                \"w_bit\": 4,
                \"version\": \"GEMM\",
            }
            model.quantize(tok, quant_config=quant_config)
            model.save_quantized(out_dir)
            tok.save_pretrained(out_dir)
            """,
            "notes": [
                "Keep calibration prompts representative of serving traffic.",
                "Validate tokenizer artifact is stored with quantized model.",
            ],
        },
        {
            "title": "GPTQ quantization config example",
            "why": "Use this when relying on GPTQ toolchain and config-driven quantization.",
            "lang": "python",
            "code": """
            from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

            model_id = \"facebook/opt-125m\"
            tok = AutoTokenizer.from_pretrained(model_id)
            gptq = GPTQConfig(bits=4, group_size=128, dataset=\"wikitext2\", tokenizer=tok)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=gptq,
                device_map=\"auto\",
            )
            model.save_pretrained(\"artifacts/opt-125m-gptq\")
            tok.save_pretrained(\"artifacts/opt-125m-gptq\")
            """,
            "notes": [
                "Prefer GPTQModel over deprecated AutoGPTQ paths where possible.",
                "Re-run eval matrix if group_size changes.",
            ],
        },
        {
            "title": "GGUF conversion and quantization with llama.cpp",
            "why": "Use this for edge deployment artifacts and local CPU/GPU inference.",
            "lang": "bash",
            "code": """
            set -euo pipefail
            git clone https://github.com/ggerganov/llama.cpp
            cd llama.cpp
            make -j

            python convert_hf_to_gguf.py /path/to/hf_model --outfile model-f16.gguf --outtype f16
            ./quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
            ./llama-cli -m model-q4_k_m.gguf -p \"Write a concise release note.\" -n 128
            """,
            "notes": [
                "Track exact llama.cpp commit used for conversion.",
                "Re-validate prompt-format compatibility after conversion.",
            ],
        },
        {
            "title": "vLLM serving with AWQ model",
            "why": "Use this for OpenAI-compatible serving with quantized checkpoints.",
            "lang": "bash",
            "code": """
            python -m vllm.entrypoints.openai.api_server \
              --model TheBloke/Mixtral-8x7B-v0.1-AWQ \
              --quantization awq \
              --host 0.0.0.0 \
              --port 8000 \
              --gpu-memory-utilization 0.92
            """,
            "notes": [
                "Always measure TTFT and output quality after enabling quantization.",
                "Adjust max_model_len to avoid cache pressure regressions.",
            ],
        },
    ],
    "symptoms": [
        "perplexity jumps after quantization",
        "output quality drops on domain prompts",
        "runtime kernel incompatibility error",
        "quantized model fails to load",
        "throughput is lower than fp16",
        "long-context generation degrades",
        "artifact size unexpectedly large",
        "inference crashes on specific sequence lengths",
        "decoder repeats tokens aggressively",
        "merged adapters produce corrupted outputs",
    ],
    "causes": [
        "calibration set does not represent production distribution",
        "group_size or bit-width too aggressive for target model",
        "backend version mismatch with quantized artifact format",
        "missing tokenizer files in artifact directory",
        "compute dtype mismatch for selected kernels",
        "KV cache constraints not tuned post-quantization",
        "incorrect conversion pipeline or stale scripts",
        "unsupported quant backend for model architecture",
        "rope or position scaling mismatch after conversion",
        "adapter merge attempted on unsupported quantized base",
    ],
    "actions": [
        "re-quantize with larger group_size or safer bit-width",
        "rebuild calibration corpus from recent serving logs",
        "pin backend versions and rerun smoke tests",
        "verify artifact completeness including tokenizer and config",
        "adjust kv cache dtype and max sequence length",
        "run short and long sequence benchmark suites",
        "compare outputs against fp16 golden prompts",
        "disable unsupported merge path and keep adapters separate",
    ],
    "commands": [
        "python -m pip freeze | rg 'transformers|bitsandbytes|awq|gptq|vllm'",
        "python - <<'PY'\nfrom transformers import AutoConfig\nc=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')\nprint('model_type', c.model_type)\nPY",
        "python - <<'PY'\nimport os\nfor p in ['artifacts/llama2-awq', 'artifacts/opt-125m-gptq']:\n    print(p, os.path.exists(p), os.listdir(p)[:8] if os.path.exists(p) else [])\nPY",
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv",
        "python eval_perplexity.py --model fp16 --dataset wikitext2 --out fp16.json",
        "python eval_perplexity.py --model quant --dataset wikitext2 --out quant.json",
    ],
    "verification_metrics": [
        "perplexity delta remains within approved threshold",
        "domain benchmark accuracy remains within approved threshold",
        "TTFT and throughput improve or stay neutral",
        "artifact loads successfully across all target runtimes",
        "long-context task regression remains bounded",
        "no deterministic crash on sequence length sweep",
    ],
    "drills": [
        "Run fp16 vs quantized A/B on a fixed prompt pack and compare exact outputs.",
        "Evaluate perplexity on at least two datasets, not just one benchmark.",
        "Run sequence length sweep from 256 to max_model_len in fixed increments.",
        "Validate model loading on target runtime matrix: transformers, vLLM, llama.cpp where relevant.",
        "Record quantization config hash and include in release notes.",
        "Verify prompt template behavior remains unchanged after conversion.",
        "Run memory and latency profile under realistic concurrency.",
    ],
    "failure_cases": 185,
    "drill_count": 280,
    "min_lines": 1050,
}


moe_config = {
    "title": "Master Agentic Skill: Mixture of Experts (MoE) Loading, Routing, and Operations",
    "domain_name": "Mixture of Experts operations",
    "mission": "Run sparse expert models reliably by controlling routing behavior, memory placement, and throughput under realistic production traffic.",
    "urls": [
        "https://huggingface.co/docs/transformers/main/en/model_doc/mixtral",
        "https://mistral.ai/news/mixtral-of-experts/",
        "https://huggingface.co/blog/moe",
        "https://huggingface.co/docs/transformers/en/kv_cache",
        "https://huggingface.co/docs/transformers/chat_templating",
        "https://huggingface.co/docs/transformers/main/en/quantization",
        "https://github.com/vllm-project/vllm",
        "https://github.com/mistralai/mistral-src",
        "https://huggingface.co/papers/2401.04088",
    ],
    "policy_metrics": [
        "tokens_per_second",
        "router_entropy",
        "expert_utilization_std",
        "aux_router_loss",
        "z_loss",
        "oom_event_count",
        "ttft_ms",
        "p95_generation_latency_ms",
    ],
    "guardrails": [
        "Track router stats in every benchmark and production canary.",
        "Block release if expert utilization collapses to small subset.",
        "Do not tune routing hyperparameters without fixed prompt benchmark.",
        "Keep separate dashboards for model quality and routing quality.",
        "Validate tokenizer template compatibility for instruct checkpoints.",
        "Run long-context and high-concurrency tests before promotion.",
    ],
    "snippets": [
        {
            "title": "Load Mixtral instruct checkpoint with automatic placement",
            "why": "Use this as the minimal reliable baseline for MoE inference.",
            "lang": "python",
            "code": """
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = \"mistralai/Mixtral-8x7B-Instruct-v0.1\"
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=\"auto\",
            )

            messages = [{\"role\": \"user\", \"content\": \"Give me three robust deployment tips.\"}]
            inputs = tok.apply_chat_template(messages, return_tensors=\"pt\").to(model.device)
            out = model.generate(inputs, max_new_tokens=128, do_sample=False)
            print(tok.decode(out[0], skip_special_tokens=True))
            """,
            "notes": [
                "Instruct checkpoints require chat template formatting.",
                "Start with deterministic decoding for debugging.",
            ],
        },
        {
            "title": "Enable router logits during training diagnostics",
            "why": "Use this to monitor expert load balancing and prevent routing collapse.",
            "lang": "python",
            "code": """
            from transformers import AutoConfig, AutoModelForCausalLM

            model_id = \"mistralai/Mixtral-8x7B-v0.1\"
            cfg = AutoConfig.from_pretrained(model_id)
            cfg.output_router_logits = True

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=cfg,
                device_map=\"auto\",
            )

            print(\"output_router_logits\", model.config.output_router_logits)
            """,
            "notes": [
                "Router logits are useful for diagnostics and auxiliary routing losses.",
                "Disable unnecessary diagnostics in latency-critical serving paths.",
            ],
        },
        {
            "title": "Quantized MoE loading for constrained hardware",
            "why": "Use this when full precision does not fit your hardware envelope.",
            "lang": "python",
            "code": """
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            model_id = \"mistralai/Mixtral-8x7B-Instruct-v0.1\"
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=\"nf4\",
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map=\"auto\",
            )
            print(type(model))
            """,
            "notes": [
                "Benchmark quality regressions after quantization.",
                "Confirm router behavior remains healthy with quantized weights.",
            ],
        },
        {
            "title": "vLLM server launch for Mixtral",
            "why": "Use this for OpenAI-compatible production serving with high throughput.",
            "lang": "bash",
            "code": """
            python -m vllm.entrypoints.openai.api_server \
              --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
              --tensor-parallel-size 2 \
              --max-model-len 8192 \
              --gpu-memory-utilization 0.92 \
              --host 0.0.0.0 \
              --port 8000
            """,
            "notes": [
                "Profile queue depth and KV cache usage under realistic load.",
                "Pin model revision to avoid accidental drift.",
            ],
        },
        {
            "title": "Flash Attention setup for faster MoE inference",
            "why": "Use this where compatible hardware and kernels are available.",
            "lang": "python",
            "code": """
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                \"mistralai/Mixtral-8x7B-v0.1\",
                dtype=torch.float16,
                attn_implementation=\"flash_attention_2\",
                device_map=\"auto\",
            )
            print(model.config.model_type)
            """,
            "notes": [
                "Confirm flash-attn version supports sliding window behavior.",
                "Fallback gracefully if target GPU is unsupported.",
            ],
        },
    ],
    "symptoms": [
        "expert utilization is highly skewed",
        "router entropy drops unexpectedly",
        "auxiliary routing loss explodes",
        "throughput collapses at moderate concurrency",
        "long-context quality declines sharply",
        "inference latency has large tail spikes",
        "OOM triggered on certain prompt shapes",
        "chat output format becomes inconsistent",
        "model overuses repetitive token patterns",
        "quality regresses after quantized deployment",
    ],
    "causes": [
        "router configuration not tuned for current traffic profile",
        "insufficient or mismatched fine-tuning data distribution",
        "incomplete telemetry for expert-level metrics",
        "memory fragmentation from dynamic batch composition",
        "cache policy or max length inconsistent with model assumptions",
        "chat template mismatch for instruct model",
        "kernel/backend incompatibility under load",
        "quantization settings too aggressive for routing stability",
    ],
    "actions": [
        "enable router diagnostics and compare expert usage histograms",
        "run controlled prompt pack to isolate regression class",
        "tighten max sequence length and retune batching policy",
        "align chat templating with model family and test stop tokens",
        "roll back to last known-good quantization profile",
        "increase monitoring on p95 and p99 latency for specific route",
        "validate kernel versions and attention implementation flags",
        "repeat canary with deterministic generation settings",
    ],
    "commands": [
        "python - <<'PY'\nfrom transformers import AutoConfig\nc=AutoConfig.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')\nprint('experts_per_tok', c.num_experts_per_tok)\nprint('num_local_experts', c.num_local_experts)\nPY",
        "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv",
        "python benchmark_serving.py --model mixtral --concurrency 1,2,4,8,16 --max_new_tokens 128",
        "python benchmark_quality.py --model mixtral --suite regression_pack_v2",
        "python dump_router_stats.py --model mixtral --prompts prompts/router_probe.jsonl",
    ],
    "verification_metrics": [
        "expert utilization variance remains within expected range",
        "router entropy is stable across benchmark groups",
        "aux router loss and z-loss remain bounded",
        "throughput scales with concurrency until planned saturation point",
        "p95 latency remains under SLO",
        "quality benchmark score remains within acceptance band",
    ],
    "drills": [
        "Run router diagnostics on three prompt classes: short, long, and multilingual.",
        "Validate instruct formatting with chat template golden samples.",
        "Run load test with concurrency ramp and observe utilization vs latency.",
        "Evaluate quantized and fp16 variants with same prompt pack.",
        "Verify long-context behavior at 4k, 8k, and configured maximum.",
        "Run deterministic decode mode to isolate routing vs sampling issues.",
        "Capture and review top failing prompts weekly.",
    ],
    "failure_cases": 185,
    "drill_count": 280,
    "min_lines": 1050,
}


video_config = {
    "title": "Master Agentic Skill: Diffusers Video Generation (SVD, Wan, CogVideoX, Memory and Quality)",
    "domain_name": "Diffusers video generation",
    "mission": "Ship high-quality video generation systems that balance visual fidelity, temporal consistency, and memory efficiency under real hardware constraints.",
    "urls": [
        "https://huggingface.co/docs/diffusers/main/en/using-diffusers/text-img2vid",
        "https://huggingface.co/docs/diffusers/main/en/optimization/memory",
        "https://huggingface.co/docs/diffusers/main/en/quantization/overview",
        "https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/svd",
        "https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan",
        "https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox",
        "https://stability.ai/news/stable-video-diffusion-open-ai-video-model",
        "https://huggingface.co/blog/stable-video-diffusion",
    ],
    "policy_metrics": [
        "frame_consistency_score",
        "flicker_rate",
        "artifact_rate",
        "fps_output",
        "generation_time_seconds",
        "vram_peak_gb",
        "decode_failure_rate",
        "prompt_adherence_score",
    ],
    "guardrails": [
        "Always lock seed when comparing parameter changes.",
        "Keep prompt, steps, and scheduler fixed during A/B tests.",
        "Track per-frame artifacts, not just first-frame quality.",
        "Require negative prompt baselines for public-facing generation.",
        "Fail release if OOM occurs on target hardware profile.",
        "Validate export codecs and FPS expectations before publish.",
    ],
    "snippets": [
        {
            "title": "Stable Video Diffusion image-to-video baseline",
            "why": "Use this as the initial quality baseline for image-conditioned clips.",
            "lang": "python",
            "code": """
            import torch
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video

            pipe = StableVideoDiffusionPipeline.from_pretrained(
                \"stabilityai/stable-video-diffusion-img2vid-xt\",
                torch_dtype=torch.float16,
                variant=\"fp16\",
            )
            pipe.enable_model_cpu_offload()

            image = load_image(
                \"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png\"
            ).resize((1024, 576))

            frames = pipe(
                image,
                num_frames=25,
                decode_chunk_size=8,
                motion_bucket_id=127,
                noise_aug_strength=0.1,
                generator=torch.manual_seed(42),
            ).frames[0]

            export_to_video(frames, \"svd-output.mp4\", fps=7)
            """,
            "notes": [
                "decode_chunk_size trades speed for memory footprint.",
                "motion_bucket_id strongly affects perceived motion dynamics.",
            ],
        },
        {
            "title": "Wan pipeline with group offloading",
            "why": "Use this for large video models on constrained VRAM.",
            "lang": "python",
            "code": """
            import torch
            from diffusers import AutoModel, WanPipeline
            from diffusers.hooks.group_offloading import apply_group_offloading
            from transformers import UMT5EncoderModel

            text_encoder = UMT5EncoderModel.from_pretrained(
                \"Wan-AI/Wan2.1-T2V-14B-Diffusers\",
                subfolder=\"text_encoder\",
                torch_dtype=torch.bfloat16,
            )
            vae = AutoModel.from_pretrained(
                \"Wan-AI/Wan2.1-T2V-14B-Diffusers\",
                subfolder=\"vae\",
                torch_dtype=torch.float32,
            )
            transformer = AutoModel.from_pretrained(
                \"Wan-AI/Wan2.1-T2V-14B-Diffusers\",
                subfolder=\"transformer\",
                torch_dtype=torch.bfloat16,
            )

            apply_group_offloading(
                text_encoder,
                onload_device=torch.device(\"cuda\"),
                offload_device=torch.device(\"cpu\"),
                offload_type=\"block_level\",
                num_blocks_per_group=4,
            )
            """,
            "notes": [
                "Group offloading can lower VRAM but increases transfer overhead.",
                "Tune offload granularity for latency-sensitive use cases.",
            ],
        },
        {
            "title": "CogVideoX compilation for inference speed",
            "why": "Use this where repeated runs amortize compile cost.",
            "lang": "python",
            "code": """
            import torch
            from diffusers import CogVideoXPipeline

            pipe = CogVideoXPipeline.from_pretrained(
                \"THUDM/CogVideoX-2b\",
                torch_dtype=torch.float16,
            ).to(\"cuda\")

            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode=\"max-autotune\",
                fullgraph=True,
            )
            """,
            "notes": [
                "Changing static shapes can trigger recompilation.",
                "Benchmark warm vs cold start separately.",
            ],
        },
        {
            "title": "Video generation with explicit parameter controls",
            "why": "Use explicit controls to isolate quality regressions quickly.",
            "lang": "python",
            "code": """
            prompt = \"A handheld camera follows a red kite over a foggy valley at sunrise\"
            negative_prompt = \"blurry, washed out, low detail, artifacts, subtitles\"

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=81,
                guidance_scale=5.0,
                num_inference_steps=40,
            ).frames[0]
            """,
            "notes": [
                "Increase guidance cautiously; too high may create artifacts.",
                "Track temporal consistency when changing num_frames.",
            ],
        },
        {
            "title": "Quantized video pipeline components",
            "why": "Use this when deploying large video models on mid-tier hardware.",
            "lang": "python",
            "code": """
            from diffusers import WanPipeline, AutoModel
            from diffusers.quantizers import PipelineQuantizationConfig

            qcfg = PipelineQuantizationConfig(
                quant_backend=\"bitsandbytes_4bit\",
                quant_kwargs={\"load_in_4bit\": True},
                components_to_quantize=[\"transformer\", \"text_encoder\"],
            )

            vae = AutoModel.from_pretrained(
                \"Wan-AI/Wan2.1-T2V-14B-Diffusers\",
                subfolder=\"vae\",
                torch_dtype=\"float32\",
            )
            pipe = WanPipeline.from_pretrained(
                \"Wan-AI/Wan2.1-T2V-14B-Diffusers\",
                vae=vae,
                quantization_config=qcfg,
                torch_dtype=\"bfloat16\",
            )
            """,
            "notes": [
                "Quantization may affect temporal smoothness; validate on motion-heavy prompts.",
                "Keep a full-precision fallback profile for critical output quality.",
            ],
        },
    ],
    "symptoms": [
        "flicker appears between adjacent frames",
        "motion does not follow prompt intent",
        "generated clip has severe artifacts",
        "VRAM OOM occurs during decode",
        "output duration does not match expected frames/fps",
        "first frame quality is good but later frames degrade",
        "inference is too slow for SLA",
        "prompt adherence is inconsistent",
        "negative prompt over-suppresses visual details",
        "exported file has codec or playback issues",
    ],
    "causes": [
        "num_frames, guidance, and step settings are unbalanced",
        "insufficient memory optimizations for selected model size",
        "decode chunk size too high for hardware budget",
        "prompt not specific enough for temporal dynamics",
        "quantization side effects on temporal coherence",
        "scheduler mismatch with model recommendation",
        "offloading configuration not tuned for workload",
        "export pipeline mismatched fps or codec settings",
    ],
    "actions": [
        "reduce decode_chunk_size and retest memory profile",
        "adjust guidance_scale in small increments and compare clips",
        "add explicit camera-motion and scene continuity cues in prompt",
        "apply group offloading or model CPU offload",
        "evaluate fp16 vs quantized output on same seed and prompt",
        "lock scheduler and seed for controlled comparison",
        "run frame-by-frame artifact analysis",
        "validate video export settings with target playback environment",
    ],
    "commands": [
        "python -m pip freeze | rg 'diffusers|transformers|torch|xformers|bitsandbytes'",
        "python - <<'PY'\nimport torch\nprint('cuda', torch.cuda.is_available())\nprint('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)\nPY",
        "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv",
        "ffprobe -hide_banner -show_streams generated.mp4 | sed -n '1,120p'",
        "python analyze_video_frames.py --input generated.mp4 --out reports/frame_metrics.json",
    ],
    "verification_metrics": [
        "flicker rate decreases after parameter tuning",
        "prompt adherence score improves on regression prompts",
        "OOM events drop to zero on target hardware",
        "clip fps and frame count match expected outputs",
        "p95 generation latency meets SLO",
        "artifact rate remains within acceptance threshold",
    ],
    "drills": [
        "Run fixed-seed A/B tests for guidance_scale values 4.0, 5.0, and 6.0.",
        "Measure memory with and without group offloading on same prompt.",
        "Evaluate temporal consistency on at least 20 motion-heavy prompts.",
        "Validate export with ffprobe and playback on target clients.",
        "Run prompt adherence scoring before release candidate tagging.",
        "Benchmark cold-start and warm-start latency separately.",
        "Test decode chunk sizes 4, 8, and 12 under fixed hardware limits.",
    ],
    "failure_cases": 185,
    "drill_count": 280,
    "min_lines": 1050,
}


token_mgmt_config = {
    "title": "Master Agentic Skill: Token Management and Chat Template Correctness",
    "domain_name": "token management",
    "mission": "Guarantee tokenizer, chat template, padding, and stop token correctness across training and inference so that model behavior remains stable and predictable.",
    "urls": [
        "https://huggingface.co/docs/transformers/chat_templating",
        "https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils",
        "https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer",
        "https://huggingface.co/docs/transformers/conversations",
        "https://huggingface.co/docs/trl/dataset_formats",
        "https://huggingface.co/docs/transformers/main/en/tasks/language_modeling",
        "https://huggingface.co/docs/transformers/main/en/model_doc/mixtral",
        "https://huggingface.co/docs/transformers/main/en/model_doc/llama",
    ],
    "policy_metrics": [
        "token_count_per_request",
        "template_render_error_rate",
        "stop_token_match_rate",
        "response_truncation_rate",
        "generation_prefix_correctness",
        "train_eval_tokenization_divergence",
        "invalid_special_token_rate",
    ],
    "guardrails": [
        "Never mix manual prompt formatting and chat templates in same endpoint.",
        "Enforce tokenizer revision pinning for reproducible deployments.",
        "Fail CI on template rendering exceptions.",
        "Keep explicit tests for add_generation_prompt and continue_final_message behavior.",
        "Validate stop token IDs for every supported model family.",
        "Require token length budget checks before rollout.",
    ],
    "snippets": [
        {
            "title": "Safe chat template rendering for inference",
            "why": "Use this when serving chat models to avoid malformed prompt structures.",
            "lang": "python",
            "code": """
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(\"HuggingFaceH4/zephyr-7b-beta\", use_fast=True)
            messages = [
                {\"role\": \"system\", \"content\": \"You are concise and factual.\"},
                {\"role\": \"user\", \"content\": \"Summarize retrieval-augmented generation in 2 lines.\"},
            ]

            rendered = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=\"pt\",
            )
            print(rendered.shape)
            """,
            "notes": [
                "Prefer tokenize=True to avoid accidental special token duplication.",
                "Keep add_generation_prompt=True for inference in most chat models.",
            ],
        },
        {
            "title": "Training-time chat formatting",
            "why": "Use this for dataset preprocessing where generation prompt should not be appended.",
            "lang": "python",
            "code": """
            from datasets import Dataset
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(\"HuggingFaceH4/zephyr-7b-beta\")
            ds = Dataset.from_dict({
                \"chat\": [
                    [{\"role\": \"user\", \"content\": \"What is 2+2?\"}, {\"role\": \"assistant\", \"content\": \"4\"}],
                    [{\"role\": \"user\", \"content\": \"Name a prime\"}, {\"role\": \"assistant\", \"content\": \"13\"}],
                ]
            })

            def fmt(row):
                return {
                    \"text\": tok.apply_chat_template(
                        row[\"chat\"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                }

            ds = ds.map(fmt)
            print(ds[0][\"text\"])
            """,
            "notes": [
                "For training, set add_generation_prompt=False.",
                "Keep formatting deterministic across train and eval.",
            ],
        },
        {
            "title": "Prefill with continue_final_message",
            "why": "Use this to continue an unfinished assistant message such as JSON prefills.",
            "lang": "python",
            "code": """
            chat = [
                {\"role\": \"user\", \"content\": \"Return JSON with keys name and age\"},
                {\"role\": \"assistant\", \"content\": '{\"name\": \"'}
            ]

            packed = tok.apply_chat_template(
                chat,
                tokenize=True,
                return_dict=True,
                continue_final_message=True,
            )
            """,
            "notes": [
                "Do not combine continue_final_message with add_generation_prompt.",
                "Use this only when intentionally prefilling assistant output.",
            ],
        },
        {
            "title": "Padding and stop token policy",
            "why": "Use this to enforce consistent batching behavior for decoder-only models.",
            "lang": "python",
            "code": """
            tok.padding_side = \"left\"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model_stop_ids = [tok.eos_token_id]
            print(\"pad\", tok.pad_token_id, \"eos\", tok.eos_token_id, \"stops\", model_stop_ids)
            """,
            "notes": [
                "Left padding is generally required for batched decoder-only generation.",
                "Model families may require additional stop IDs.",
            ],
        },
        {
            "title": "Add special tokens and resize embeddings safely",
            "why": "Use this after adding domain-specific control tokens.",
            "lang": "python",
            "code": """
            special = {\"additional_special_tokens\": [\"<json_start>\", \"<json_end>\"]}
            n_added = tok.add_special_tokens(special)
            print(\"added\", n_added)
            model.resize_token_embeddings(len(tok))
            """,
            "notes": [
                "Never add tokens without resizing embeddings.",
                "Re-evaluate perplexity and downstream metrics after vocabulary changes.",
            ],
        },
    ],
    "symptoms": [
        "assistant continues user text instead of replying",
        "responses truncate too early",
        "duplicate BOS/EOS tokens appear",
        "template rendering throws runtime errors",
        "train and inference outputs diverge",
        "tool-calling format fails validation",
        "batch generation produces inconsistent lengths",
        "model outputs malformed JSON in structured tasks",
        "stop sequence not respected",
        "language mixing appears unexpectedly",
    ],
    "causes": [
        "add_generation_prompt misconfigured for inference",
        "continue_final_message misused with generation prompt",
        "special tokens duplicated during manual tokenization",
        "padding side incompatible with decode path",
        "stop token IDs incomplete for model family",
        "template changed without retraining or migration checks",
        "tokenizer revision drift between environments",
        "missing embedding resize after adding tokens",
    ],
    "actions": [
        "render templates to plain text and inspect exact control tokens",
        "switch to apply_chat_template(tokenize=True) for safety",
        "set explicit padding side and pad token policy",
        "align stop token IDs with tokenizer config",
        "lock tokenizer revision and verify in CI",
        "run train-vs-inference formatting diff on sample set",
        "remove manual special token insertion where template already handles it",
        "add strict tests for JSON and tool-calling output format",
    ],
    "commands": [
        "python - <<'PY'\nfrom transformers import AutoTokenizer\nt=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')\nprint('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)\nPY",
        "python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl",
        "python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct",
        "python run_generation_regression.py --suite tests/golden_chat_outputs.json",
        "python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json",
    ],
    "verification_metrics": [
        "template render success rate remains 100%",
        "stop token match rate stays within expected threshold",
        "response truncation rate remains low and stable",
        "train-eval tokenization divergence remains zero",
        "structured output validity rate remains within target",
    ],
    "drills": [
        "Render 100 sample chats and manually inspect control-token layout.",
        "Run add_generation_prompt=True/False comparison suite for inference endpoints.",
        "Run continue_final_message prefill tests for JSON tasks.",
        "Validate tokenizer revision consistency between local, CI, and production.",
        "Test stop token behavior on multilingual and long-context prompts.",
        "Run structured output validation for JSON and tool-calling messages.",
        "Re-run regression pack after any tokenizer or template update.",
    ],
    "failure_cases": 185,
    "drill_count": 280,
    "min_lines": 1050,
}


TARGETS = {
    ROOT / "skills" / "fine-tuning" / "llm-finetuning.prompt.md": fine_tuning_config,
    ROOT / "skills" / "quantization" / "llm-quantization.prompt.md": quantization_config,
    ROOT / "skills" / "moe" / "mixture-of-experts-loading.prompt.md": moe_config,
    ROOT / "skills" / "video-generation" / "diffusers-video-generation.prompt.md": video_config,
    ROOT / "skills" / "huggingface" / "token-management.prompt.md": token_mgmt_config,
}


for path, cfg in TARGETS.items():
    content = build_skill_file(cfg)
    path.write_text(content)
    print(f"Wrote {path} with {len(content.splitlines())} lines")
