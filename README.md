# LLM Whisperer

LLM Whisperer is the central repository for reusable skills, agent workflows, fine-tuning recipes, RAG systems, inference acceleration patterns, and practical sample code.

Author: Shuvam Banerji Seal

## Vision

This repository is designed to be the go-to base where you can:

- Pick a minimal working template and extend it quickly.
- Reuse battle-tested skills and prompts across agents.
- Build and evaluate multiple fine-tuning styles (LoRA, QLoRA, behavior, RAG, agentic).
- Compare fast inference backends (including DLL-driven/runtime-optimized paths).
- Keep implementation knowledge, experiments, and production-ready patterns in one place.

## Core Principles

- Start minimal, then scale: every area should have a runnable "small" example before complex variants.
- Separate concerns: skills, agents, tuning, inference, evaluation, and infra should live in dedicated domains.
- Reproducibility first: every workflow should include configs, scripts, and clear run instructions.
- Production-aware: include both research notebooks and hardened service patterns.
- Keep the repo navigable: clear naming, predictable folder contracts, and focused READMEs.

## Recommended Repository Structure

Use this as the canonical structure as the repository grows:

```text
LLM-Whisperer/
├── LICENSE
├── README.md
├── docs/
│   ├── architecture/               # System diagrams, module boundaries, design docs
│   ├── guides/                     # Setup, environment, troubleshooting, onboarding
│   ├── playbooks/                  # Repeatable workflows (train, eval, deploy)
│   └── decisions/                  # ADR-style architecture and tooling decisions
│
├── skills/
│   ├── foundational/               # Universal software and engineering skills
│   ├── llm-engineering/            # Prompting, context strategy, eval, safety patterns
│   ├── rag/                        # Retrieval and grounding specific skills
│   ├── agentic/                    # Tool-use, planning, memory, orchestration skills
│   ├── inference/                  # Runtime optimization and serving skills
│   ├── fine-tuning/                # Tuning strategy and quality skills
│   └── general-python-development.prompt.md
│
├── agents/
│   ├── prompts/                    # System prompts for dedicated agents
│   ├── configs/                    # Agent runtime configs, policy constraints
│   ├── workflows/                  # Multi-agent workflows and routing maps
│   └── evaluation/                 # Agent behavior tests and benchmark suites
│
├── fine_tuning/
│   ├── base/                       # Full fine-tune baselines and reference loops
│   ├── lora/                       # LoRA recipes and hyperparameter variants
│   ├── qlora/                      # QLoRA recipes and quantized training paths
│   ├── rag_tuning/                 # Retrieval-augmented fine-tuning workflows
│   ├── behavior_tuning/            # Style/policy/persona/task behavior tuning
│   ├── agentic_tuning/             # Tool use, planning, multi-step tuning data
│   ├── multimodal/                 # Text + image/audio/video tuning workflows
│   ├── reward_modeling/            # Preference/reward model training recipes
│   ├── configs/                    # Shared train configs and sweep templates
│   └── templates/                  # Starter script templates for new experiments
│
├── rag/
│   ├── ingestion/                  # Loaders/connectors and document normalization
│   ├── chunking/                   # Chunking and segmentation strategies
│   ├── embeddings/                 # Embedding model wrappers and adapters
│   ├── indexing/                   # Vector index creation and maintenance
│   ├── retrieval/                  # Retriever strategies and fusion methods
│   ├── reranking/                  # Cross-encoder/reranker stages
│   ├── generation/                 # Prompt assembly and grounded generation
│   └── eval/                       # Retrieval/generation quality and faithfulness tests
│
├── inference/
│   ├── engines/
│   │   ├── vllm/                   # vLLM serving recipes
│   │   ├── tensorrt/               # TensorRT optimization and serving
│   │   ├── onnxruntime/            # ONNX export and runtime execution
│   │   ├── llama_cpp/              # llama.cpp CPU/GPU deployment patterns
│   │   ├── triton/                 # Triton inference server integration
│   │   └── custom_dll/             # DLL/runtime-specific acceleration paths
│   ├── quantization/               # INT8/INT4/AWQ/GPTQ and tradeoff studies
│   ├── serving/                    # API servers, batching, streaming, scaling
│   └── benchmarking/               # Latency/throughput/memory benchmark harnesses
│
├── datasets/
│   ├── raw/                        # Raw source snapshots (metadata only in git)
│   ├── interim/                    # Cleaned but not final datasets
│   ├── processed/                  # Training-ready datasets
│   ├── synthetic/                  # Generated instruction or preference data
│   ├── prompt_sets/                # Prompt corpora and task collections
│   └── eval_sets/                  # Golden datasets for regression and benchmark tests
│
├── models/
│   ├── base/                       # Base model metadata and references
│   ├── adapters/                   # LoRA/adapter metadata and checkpoints refs
│   ├── merged/                     # Merged model manifests
│   ├── exported/                   # ONNX/TensorRT/GGUF export metadata
│   └── registry/                   # Model registry definitions and version maps
│
├── evaluation/
│   ├── task_benchmarks/            # Domain/task-specific benchmark suites
│   ├── llm_as_judge/               # Rubrics and evaluator pipelines
│   ├── safety/                     # Harms, jailbreak, policy, toxicity checks
│   ├── latency/                    # Runtime and SLA-oriented performance checks
│   └── regression/                 # CI-compatible quality regression suites
│
├── experiments/
│   ├── tracking/                   # Experiment metadata schemas and exporters
│   ├── ablations/                  # Controlled experiments and comparison scripts
│   └── reports/                    # Result summaries and analysis artifacts
│
├── sample_code/
│   ├── minimal/                    # Tiny, single-purpose examples
│   ├── end_to_end/                 # Full workflows from data to deployment
│   └── reference_apps/             # Larger exemplar applications
│
├── tools/
│   ├── dragonborn/                 # New tools and integrations (including Dragonborn)
│   ├── cli/                        # Utility command-line tools
│   └── automation/                 # Helpers for setup, checks, and release tasks
│
├── notebooks/
│   ├── exploration/                # Research and quick hypothesis notebooks
│   ├── tutorials/                  # Learning-focused notebooks
│   └── reports/                    # Analysis notebooks for experiments/eval
│
├── pipelines/
│   ├── data/                       # Data preparation and validation pipelines
│   ├── training/                   # Training/fine-tuning orchestrations
│   ├── evaluation/                 # Automated eval pipelines
│   └── deployment/                 # Packaging and deployment pipelines
│
├── infra/
│   ├── docker/                     # Container definitions
│   ├── kubernetes/                 # K8s manifests/helm charts
│   ├── terraform/                  # IaC modules
│   └── monitoring/                 # Observability dashboards/alerts definitions
│
├── configs/
│   ├── environments/               # Local/dev/staging/prod overlays
│   ├── models/                     # Model and tokenizer configs
│   ├── datasets/                   # Dataset processing and split configs
│   └── runtime/                    # Serving/runtime system configs
│
├── scripts/                        # Reusable scripts (bootstrap, lint, release, migrate)
└── tests/
    ├── unit/                       # Fast, isolated tests
    ├── integration/                # Cross-module tests
    └── e2e/                        # End-to-end scenario tests
```

## Folder Contract (Important)

For any new major module (for example a new fine-tuning method or inference backend), keep the same internal layout:

```text
<module>/
├── README.md           # What it does, when to use it, how to run
├── src/                # Implementation code
├── configs/            # Reproducible configs and presets
├── scripts/            # Launch, preprocess, and utility scripts
├── tests/              # Unit/integration tests for that module
├── examples/           # Minimal usage examples
└── artifacts/          # Local outputs (should usually be gitignored)
```

## What to Keep in Git vs External Storage

Keep in git:

- Code, configs, prompts, docs, evaluation definitions, metadata manifests.
- Small sample datasets and tiny toy checkpoints only when useful for tests.

Keep external (object storage / model registry):

- Large raw datasets.
- Full checkpoints, merged weights, model binaries.
- Large benchmark artifacts.

Store references in repo using manifests (paths, hashes, versions, provenance).

## Suggested Workflow for Adding New Capabilities

1. Create the domain folder (or choose an existing one).
2. Add a focused README for that module with purpose and expected inputs/outputs.
3. Add at least one minimal runnable example before advanced variants.
4. Add configs and tests alongside code from day one.
5. Add a simple evaluation script and baseline report.
6. Link the module from this root README.

## Priority Build Order (Recommended)

If you want to evolve this repo in phases:

1. Skills + Agents foundation (shared prompts, routing, evaluation).
2. Fine-tuning baselines (base, LoRA, QLoRA).
3. RAG baseline + retrieval evaluation.
4. Inference engines + benchmark harness.
5. Dragonborn and other tool integrations.
6. CI pipelines, regression gates, and deployment templates.

## Naming Conventions

- Use snake_case for new top-level engineering folders.
- Keep skills subfolders in kebab-case to match existing convention.
- Keep module names explicit (for example `behavior_tuning`, not `behavior`).
- Keep one concern per folder.
- Prefer small READMEs across modules instead of one giant document.

## Status

This repository is currently in active build-out mode.

The structure above is the source of truth for scaling LLM Whisperer into a complete engineering hub for skills, fine-tuning, RAG, fast inference, and agent systems.