# LLM-Whisperer

**A comprehensive, production-ready repository for LLM engineering, skills, agents, fine-tuning, RAG, inference acceleration, and evaluation.**

Author: Shuvam Banerji Seal | Updated: April 2026

---

## Table of Contents

- [Vision & Mission](#vision--mission)
- [What Is LLM-Whisperer?](#what-is-llm-whisperer)
- [Repository Architecture](#repository-architecture)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Support & Community](#support--community)

---

## Vision & Mission

LLM-Whisperer is designed to be **the central hub** for production-grade LLM engineering knowledge and implementation patterns. It serves as a comprehensive resource where teams can:

- **Pick a minimal template and extend it quickly** into production systems
- **Reuse battle-tested skills and prompts** across multiple agents and applications
- **Build and evaluate multiple fine-tuning strategies** (LoRA, QLoRA, RAG-based, behavior-driven, agentic)
- **Compare fast inference backends** with standardized benchmarking and optimization patterns
- **Keep implementation knowledge, experiments, and production-ready patterns in one place** with clear documentation and examples

### Core Principles

1. **Start minimal, then scale** - Every capability includes a runnable "small" example before complex variants
2. **Separate concerns** - Skills, agents, fine-tuning, inference, evaluation, and infrastructure live in dedicated domains
3. **Reproducibility first** - Every workflow includes configs, scripts, and clear execution instructions
4. **Production-aware** - Both research notebooks and hardened service patterns are included
5. **Navigable repository** - Clear naming conventions, predictable folder contracts, and focused READMEs

---

## What Is LLM-Whisperer?

LLM-Whisperer is a **monorepo** containing:

### 🎯 **Skills Library** (100+ reusable skills)
- Foundational software engineering patterns
- LLM-specific techniques (prompting, evaluation, safety)
- RAG strategies and optimization patterns
- Agentic system design and orchestration
- Inference optimization and serving patterns
- Fine-tuning methodologies and recipes
- Advanced architectures (MoE, attention variants, etc.)
- Production operations and monitoring

### 🤖 **Agent Framework**
- System prompts and evaluation suites for specialized agents
- Multi-agent orchestration and routing patterns
- Workflow definitions for common tasks
- Tool integration and execution frameworks

### 🔧 **Fine-Tuning Suite**
- Base model fine-tuning baselines
- LoRA and QLoRA implementations
- RAG-augmented fine-tuning
- Behavior and style tuning
- Agentic system tuning
- Multimodal fine-tuning recipes
- Reward model training

### 📚 **RAG System**
- Document ingestion and normalization
- Intelligent chunking strategies
- Embedding models and adapters
- Vector indexing and retrieval
- Re-ranking and fusion methods
- Context-aware generation
- Evaluation harnesses for RAG quality

### ⚡ **Inference Acceleration**
- vLLM, TensorRT, ONNX, llama.cpp, Triton integrations
- Quantization strategies (INT8, INT4, AWQ, GPTQ)
- Batching, streaming, and scaling patterns
- Comprehensive benchmarking harnesses
- Latency/throughput optimization

### 📊 **Evaluation Framework** (5 categories)
- **Task Benchmarks**: MMLU, GSM8K, HumanEval, SWE-bench
- **LLM-as-Judge**: Standardized rubrics and multi-judge consensus
- **Safety**: Toxicity, bias, jailbreak, and PII detection
- **Latency**: TTFT, TPOT, throughput, and SLA monitoring
- **Regression**: Golden datasets, quality gates, and drift detection

### 🏗️ **Production Infrastructure**
- Docker containers and Kubernetes manifests
- Terraform IaC modules
- CI/CD pipelines and automation
- Monitoring, logging, and alerting configurations
- Deployment patterns and disaster recovery

---

## Repository Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM-Whisperer Monorepo                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Skills     │  │    Agents    │  │    Fine-     │      │
│  │   Library    │  │  Framework   │  │   Tuning     │      │
│  │ (100+ skills)│  │  Multi-agent │  │   Suite      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     RAG      │  │  Inference   │  │ Evaluation   │      │
│  │    System    │  │ Acceleration │  │ Framework    │      │
│  │  (E2E flow)  │  │  (5 engines) │  │  (5 types)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Production Infrastructure & DevOps           │  │
│  │    Docker, K8s, Terraform, CI/CD, Monitoring        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Datasets, Models, Configs, Utilities         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

The repository follows a **modular, domain-driven structure** with consistent folder contracts:

```
LLM-Whisperer/
├── README.md                          # You are here
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
│
├── docs/                              # Documentation hub
│   ├── README.md                      # Docs index
│   ├── architecture/                  # System design & diagrams
│   ├── guides/                        # Setup, troubleshooting, onboarding
│   ├── playbooks/                     # Repeatable workflows (train, eval, deploy)
│   ├── decisions/                     # Architecture Decision Records (ADRs)
│   ├── ORGANIZATION_STATUS.md         # Repository organization status
│   ├── 2026_LLM_RESEARCH_MASTER_INDEX.md  # Comprehensive research index
│   └── PROJECT_COMPLETION_SUMMARY.md  # Project completion details
│
├── skills/                            # 100+ reusable skills library (3.5K docs)
│   ├── README.md                      # Skills catalog and navigation
│   ├── foundational/                  # Core software engineering skills
│   ├── llm-engineering/               # Prompting, context, evaluation, safety
│   ├── rag/                           # Retrieval and grounding patterns
│   ├── agentic/                       # Tool-use, planning, memory, orchestration
│   ├── inference/                     # Runtime optimization and serving
│   ├── fine-tuning/                   # Tuning strategy and quality patterns
│   ├── advanced-llm-architectures/    # MoE, attention variants, transformers
│   ├── code-generation/               # LLM-based code synthesis
│   ├── advanced-reasoning/            # CoT, ToT, multi-step reasoning
│   ├── infrastructure-deployment/     # K8s, cost optimization, production patterns
│   ├── production-ops/                # Monitoring, observability, LLMOps
│   ├── research-archive/              # Consolidated research documentation (744 KB)
│   └── general-python-development.prompt.md
│
├── agents/                            # Multi-agent framework & orchestration
│   ├── README.md                      # Agent framework overview
│   ├── prompts/                       # System prompts for specialized agents
│   ├── src/                           # Agent execution engines
│   ├── configs/                       # Runtime configurations and policies
│   ├── workflows/                     # Multi-agent workflows and routing
│   └── evaluation/                    # Agent behavior tests and benchmarks
│
├── fine_tuning/                       # Fine-tuning recipes & experiments
│   ├── README.md                      # Fine-tuning overview
│   ├── base/                          # Full fine-tune baselines
│   ├── lora/                          # LoRA recipes and hyperparameter variants
│   ├── qlora/                         # QLoRA and quantized training
│   ├── rag_tuning/                    # Retrieval-augmented fine-tuning
│   ├── behavior_tuning/               # Style/policy/persona tuning
│   ├── agentic_tuning/                # Tool-use and planning tuning
│   ├── multimodal/                    # Text + image/audio/video tuning
│   ├── reward_modeling/               # Preference learning
│   ├── configs/                       # Shared training configurations
│   └── templates/                     # Starter templates for new experiments
│
├── rag/                               # Retrieval-Augmented Generation system
│   ├── README.md                      # RAG system overview
│   ├── ingestion/                     # Document loaders and normalization
│   ├── chunking/                      # Chunking strategies
│   ├── embeddings/                    # Embedding model wrappers
│   ├── indexing/                      # Vector index operations
│   ├── retrieval/                     # Retriever implementations
│   ├── reranking/                     # Cross-encoder reranking
│   ├── generation/                    # Prompt assembly and generation
│   └── eval/                          # Retrieval and generation quality tests
│
├── inference/                         # Inference engines & acceleration
│   ├── README.md                      # Inference overview
│   ├── engines/
│   │   ├── vllm/                      # vLLM serving recipes
│   │   ├── tensorrt/                  # TensorRT optimization
│   │   ├── onnxruntime/               # ONNX export and runtime
│   │   ├── llama_cpp/                 # llama.cpp CPU/GPU deployment
│   │   ├── triton/                    # Triton inference server
│   │   └── custom_dll/                # DLL/runtime acceleration
│   ├── quantization/                  # INT8, INT4, AWQ, GPTQ
│   ├── serving/                       # API servers, batching, scaling
│   └── benchmarking/                  # Latency/throughput harnesses
│
├── evaluation/                        # Comprehensive evaluation framework
│   ├── README.md                      # Evaluation overview
│   ├── src/                           # Shared evaluation utilities
│   │   ├── base.py                    # Base evaluator interface
│   │   ├── metrics.py                 # Common metric computations
│   │   └── __init__.py                # Module initialization
│   ├── task_benchmarks/               # Standard benchmarks (MMLU, GSM8K, etc.)
│   │   ├── README.md                  # Benchmark documentation
│   │   ├── src/                       # Benchmark implementations
│   │   └── configs/                   # Benchmark configurations
│   ├── llm_as_judge/                  # Judge-based evaluation
│   │   ├── README.md                  # Judge framework documentation
│   │   ├── src/                       # Judge implementations and rubrics
│   │   └── configs/                   # Rubric configurations
│   ├── safety/                        # Safety evaluation
│   │   ├── README.md                  # Safety checks documentation
│   │   ├── src/                       # Safety evaluators
│   │   └── configs/                   # Safety rules
│   ├── latency/                       # Performance & SLA monitoring
│   │   ├── README.md                  # Latency measurement documentation
│   │   ├── src/                       # Latency measurement tools
│   │   └── configs/                   # SLA configurations
│   └── regression/                    # Quality regression testing
│       ├── README.md                  # Regression testing documentation
│       ├── src/                       # Golden dataset and quality gate implementations
│       └── configs/                   # Quality gate thresholds
│
├── datasets/                          # Data curation and management
│   ├── README.md                      # Dataset organization guide
│   ├── raw/                           # Raw source snapshots
│   ├── interim/                       # Cleaned intermediate data
│   ├── processed/                     # Training-ready data
│   ├── synthetic/                     # Generated instruction/preference data
│   ├── prompt_sets/                   # Prompt corpora
│   └── eval_sets/                     # Golden test sets
│
├── models/                            # Model metadata and references
│   ├── README.md                      # Model registry guide
│   ├── base/                          # Base model metadata
│   ├── adapters/                      # LoRA adapter references
│   ├── merged/                        # Merged model manifests
│   ├── exported/                      # Export format metadata
│   └── registry/                      # Model version registry
│
├── experiments/                       # Experiment tracking and analysis
│   ├── README.md                      # Experiment management guide
│   ├── tracking/                      # Metadata schemas
│   ├── ablations/                     # Controlled experiments
│   └── reports/                       # Analysis artifacts
│
├── sample_code/                       # Example implementations
│   ├── README.md                      # Sample code index
│   ├── minimal/                       # Single-purpose examples
│   ├── end_to_end/                    # Complete workflows
│   └── reference_apps/                # Larger exemplar applications
│
├── tools/                             # Utility tools and integrations
│   ├── cli/                           # Command-line utilities
│   ├── automation/                    # Setup and release helpers
│   └── dragonborn/                    # Dragonborn tools integration
│
├── notebooks/                         # Jupyter notebooks
│   ├── README.md                      # Notebook index
│   ├── exploration/                   # Research and hypothesis notebooks
│   ├── tutorials/                     # Learning-focused notebooks
│   └── reports/                       # Analysis notebooks
│
├── pipelines/                         # Data and ML pipelines
│   ├── README.md                      # Pipeline overview
│   ├── data/                          # Data preparation pipelines
│   ├── training/                      # Training orchestration
│   ├── evaluation/                    # Automated evaluation
│   └── deployment/                    # Deployment pipelines
│
├── infra/                             # Production infrastructure
│   ├── README.md                      # Infrastructure overview
│   ├── docker/                        # Container definitions
│   ├── kubernetes/                    # K8s manifests and Helm charts
│   ├── terraform/                     # IaC modules
│   └── monitoring/                    # Observability configurations
│
├── configs/                           # Global configurations
│   ├── README.md                      # Config management guide
│   ├── environments/                  # Local/dev/staging/prod overlays
│   ├── models/                        # Model and tokenizer configs
│   ├── datasets/                      # Dataset processing configs
│   └── runtime/                       # Runtime system configs
│
├── scripts/                           # Reusable utility scripts
│   ├── README.md                      # Script inventory
│   ├── bootstrap.sh                   # Initial setup
│   ├── lint.sh                        # Code quality checks
│   ├── release.sh                     # Release automation
│   └── migrate.sh                     # Data/config migration
│
├── tests/                             # Test suites
│   ├── README.md                      # Testing guide
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   └── e2e/                           # End-to-end scenario tests
│
└── .gitignore                         # Git ignore patterns
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer.git
cd LLM-Whisperer

# Install dependencies (optional, depends on what you're using)
pip install -r requirements.txt

# Explore the skills library
ls skills/

# Check documentation
cat docs/README.md
```

### Pick a Domain and Get Started

**Want to fine-tune a model?**
```bash
cd fine_tuning/lora
cat README.md
python scripts/train_lora.py --config configs/base.yaml
```

**Want to build a RAG system?**
```bash
cd rag
cat README.md
python -m rag.ingestion.loader --data-path your_documents/
```

**Want to evaluate your model?**
```bash
cd evaluation
cat README.md
python -m evaluation.task_benchmarks --benchmark mmlu --model-path your_model/
```

**Want to deploy with Kubernetes?**
```bash
cd infra/kubernetes
cat README.md
kubectl apply -f manifests/
```

### Explore Skills

Browse 100+ reusable skills:
```bash
cd skills
cat README.md              # See all skill categories
cat foundational/README.md # Check specific category
```

Each skill is documented with:
- Clear purpose and use cases
- When to apply the pattern
- Implementation examples
- Relevant research papers
- Code snippets and configurations

---

## Core Components

### 1. Skills Library

A comprehensive collection of **100+ reusable, battle-tested skills** organized by domain:

- **Foundational**: Core software engineering patterns
- **LLM Engineering**: Prompting techniques, context management, evaluation
- **RAG**: Retrieval, chunking, reranking strategies
- **Agentic Systems**: Tool-use, planning, memory management
- **Inference**: Optimization, quantization, serving patterns
- **Fine-Tuning**: LoRA, QLoRA, behavior tuning recipes
- **Advanced**: MoE architectures, attention variants, reasoning techniques
- **Production**: Kubernetes, monitoring, cost optimization

**Where**: `skills/` directory
**How to use**: Pick a skill, read its README, apply the pattern to your project

### 2. Evaluation Framework

Production-ready evaluation across 5 categories:

| Category | Purpose | Includes |
|----------|---------|----------|
| **Task Benchmarks** | Measure core capabilities | MMLU, GSM8K, HumanEval, SWE-bench |
| **LLM-as-Judge** | Semantic quality evaluation | Standard rubrics, multi-judge consensus |
| **Safety** | Detect harmful behaviors | Toxicity, bias, jailbreak, PII detection |
| **Latency** | Monitor performance | TTFT, TPOT, throughput, SLA checks |
| **Regression** | Prevent degradation | Golden datasets, quality gates, drift detection |

**Where**: `evaluation/` directory
**Documentation**: `evaluation/README.md`

### 3. Fine-Tuning Suite

Complete recipes for all tuning methodologies:

- Full fine-tune baselines
- LoRA (Parameter-Efficient Fine-Tuning)
- QLoRA (Quantized LoRA)
- RAG-augmented tuning
- Behavior and style tuning
- Agentic system tuning
- Reward model training

**Where**: `fine_tuning/` directory
**Quick Start**: `fine_tuning/lora/README.md`

### 4. RAG System

End-to-end retrieval-augmented generation:

- Document ingestion and chunking
- Embedding and indexing
- Dense and hybrid retrieval
- Re-ranking strategies
- Context-aware generation
- Quality evaluation

**Where**: `rag/` directory
**Documentation**: `rag/README.md`

### 5. Inference Acceleration

Multiple backend support with benchmarking:

- vLLM (highest throughput)
- TensorRT (NVIDIA optimization)
- ONNX Runtime (cross-platform)
- llama.cpp (CPU inference)
- Triton (production serving)

**Where**: `inference/` directory
**Documentation**: `inference/README.md`

---

## Getting Started

### For Researchers

1. **Explore skills library**: `cd skills && cat README.md`
2. **Review research archive**: `cat docs/2026_LLM_RESEARCH_MASTER_INDEX.md`
3. **Check latest evaluations**: `cat evaluation/README.md`
4. **Run benchmarks**: `python -m evaluation.task_benchmarks`

### For ML Engineers

1. **Pick a fine-tuning approach**: `cd fine_tuning && ls`
2. **Prepare your data**: `cat datasets/README.md`
3. **Configure training**: Edit configs in `fine_tuning/configs/`
4. **Run experiments**: Follow README in chosen fine-tuning subdirectory
5. **Evaluate results**: Use `evaluation/` framework

### For DevOps/MLOps

1. **Understand infrastructure**: `cat infra/README.md`
2. **Check Kubernetes configs**: `ls infra/kubernetes/`
3. **Setup monitoring**: `cat infra/monitoring/README.md`
4. **Deploy with pipelines**: `cat pipelines/README.md`
5. **Scale with automation**: `cat tools/cli/README.md`

### For Application Builders

1. **Start with sample code**: `cat sample_code/README.md`
2. **Pick a framework**: Agents, RAG, or fine-tuned models
3. **Follow the end-to-end examples**: `sample_code/end_to_end/`
4. **Integrate with your stack**: Use provided recipes
5. **Deploy to production**: Follow infra templates

---

## Contributing

LLM-Whisperer is **open for contributions** from the community. We welcome:

### 1. New Skills & Patterns

Have a battle-tested LLM engineering pattern? Share it!

- **Prompting techniques** that work well
- **Fine-tuning strategies** for specific use cases
- **RAG optimization patterns**
- **Agentic orchestration approaches**
- **Inference optimization tricks**
- **Production deployment patterns**

**How to contribute:**
1. Create a new directory in `skills/[category]/`
2. Add a clear README explaining the pattern
3. Include implementation examples
4. Link relevant research papers
5. Submit a pull request

### 2. Better Evaluation Suites

Help us evaluate LLM systems more comprehensively:

- Custom benchmark implementations
- New judge rubrics for evaluation
- Safety detection improvements
- Latency profiling tools
- Regression test datasets

**Where**: `evaluation/` directory
**How**: Create a pull request with your evaluation code

### 3. New Fine-Tuning Recipes

Contribute tuning methodologies you've refined:

- Novel LoRA configurations
- Multi-task training strategies
- Curriculum learning approaches
- Preference learning techniques
- Domain-specific tuning recipes

**Where**: `fine_tuning/` directory
**How**: Add your recipe with configs and documentation

### 4. Infrastructure & Deployment

Improve production deployment:

- Kubernetes manifests
- Terraform modules
- Docker optimizations
- Monitoring dashboards
- CI/CD pipeline improvements

**Where**: `infra/` and `pipelines/` directories
**How**: Submit templates with documentation

### 5. Documentation & Examples

Help others learn:

- Tutorials and guides
- Code examples and notebooks
- Architecture decision records
- Troubleshooting guides
- Best practices documentation

**Where**: `docs/` and `notebooks/` directories
**How**: Clear, well-commented contributions

### 6. Research & Analysis

Share your findings:

- Comparative studies of approaches
- Benchmark results on new models
- Performance analysis reports
- Techniques you've discovered
- Production lessons learned

**Where**: `experiments/reports/` or `docs/playbooks/`
**How**: Document your methodology and findings

### Contribution Guidelines

Please follow these guidelines when contributing:

1. **Create a feature branch**: `git checkout -b feature/your-contribution`
2. **Follow the folder contract**: Each module should have:
   - `README.md` (what, why, how)
   - `src/` or implementation code
   - `configs/` (YAML configurations)
   - `examples/` or `scripts/` (usage examples)
   - `tests/` (unit/integration tests)

3. **Write clear documentation**:
   - Explain the pattern and when to use it
   - Include code examples
   - Link to relevant papers/resources
   - Add quick-start instructions

4. **Test your code**:
   - Include unit tests
   - Verify examples run
   - Check for breaking changes

5. **Submit a pull request** with:
   - Clear title and description
   - Reference to any related issues
   - Changes made and why
   - Examples of usage

6. **Respond to feedback**:
   - Address review comments
   - Update documentation if needed
   - Ensure CI/CD checks pass

### Code of Conduct

We're committed to providing a welcoming and inclusive environment. Please:

- Be respectful of different viewpoints
- Provide constructive feedback
- Help others learn and grow
- Give credit appropriately
- Follow all applicable laws and regulations

### Recognition

Contributors are recognized in:
- Git commit history
- `CONTRIBUTORS.md` (coming soon)
- Repository documentation
- Release notes

---

## License

LLM-Whisperer is released under the **MIT License**, a permissive open-source license that allows:

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ✅ Using parts in proprietary projects

**With the requirement to:**
- ⚖️ Include license and copyright notice
- ⚖️ State changes made to the code

See `LICENSE` file for full details.

### Third-Party Attribution

This repository includes or references:

- **Research papers** from academic institutions
- **Code patterns** from open-source projects
- **Configuration templates** from community projects
- **Benchmarks** from benchmark suites like MMLU, GSM8K, HumanEval, SWE-bench

Each component includes proper attribution. When using specific components, check their own licenses.

---

## Support & Community

### Getting Help

1. **Check the documentation**: Start with relevant README in `docs/` or the module you're using
2. **Search existing issues**: Look for similar problems on GitHub
3. **Ask a question**: Create a GitHub Discussion (coming soon)
4. **Report a bug**: Open a GitHub Issue with reproduction steps

### Learning Resources

- **Skills Library**: `skills/README.md` - Browse 100+ patterns
- **Guides**: `docs/guides/` - Setup and troubleshooting
- **Sample Code**: `sample_code/README.md` - Runnable examples
- **Notebooks**: `notebooks/tutorials/` - Interactive learning
- **Architecture Docs**: `docs/architecture/` - System design

### Research & References

- **Master Research Index**: `docs/2026_LLM_RESEARCH_MASTER_INDEX.md` - Comprehensive research compilation
- **Organization Status**: `docs/ORGANIZATION_STATUS.md` - Repository organization details
- **Project Summary**: `docs/PROJECT_COMPLETION_SUMMARY.md` - What's been delivered

### Staying Updated

- **Watch** the repository for updates
- **Star** to show support
- **Follow** releases for new features
- **Join discussions** for feature requests

---

## Repository Statistics

| Metric | Count |
|--------|-------|
| Skills Library | 100+ reusable patterns |
| Documentation | 500+ markdown files |
| Code Examples | 300+ runnable examples |
| Research Papers Referenced | 100+ papers |
| GitHub Repositories Analyzed | 150+ repos |
| Evaluation Categories | 5 types |
| Fine-Tuning Methods | 8+ approaches |
| Inference Engines | 5+ backends |
| Test Coverage | Unit, integration, E2E |

---

## Roadmap & Future

**Planned additions:**

- [ ] Interactive skill browser web interface
- [ ] Automated benchmarking CI/CD pipeline
- [ ] Model registry with versioning
- [ ] Community-contributed skills showcase
- [ ] Video tutorials for major components
- [ ] Integration with Hugging Face Hub
- [ ] SDK for easy project bootstrapping

---

## Citation

If you use LLM-Whisperer in your research or project, please cite:

```bibtex
@misc{llmwhisperer2026,
  title={LLM-Whisperer: A Comprehensive Repository for Production LLM Engineering},
  author={Seal, Shuvam Banerji},
  year={2026},
  howpublished={\url{https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer}},
  note={Open-source repository with 100+ skills and production-ready patterns}
}
```

---

## Frequently Asked Questions

**Q: Where should I start?**
A: Check the Quick Start section above, then browse skills in `skills/` that match your use case.

**Q: Can I use this in production?**
A: Yes! The repository includes production-ready patterns, infrastructure templates, and hardened deployment examples.

**Q: How do I contribute?**
A: See the Contributing section. We welcome new skills, examples, improvements, and documentation.

**Q: Is this a framework?**
A: No, it's a comprehensive monorepo of patterns, skills, and examples. Mix and match what you need.

**Q: Can I copy code from here into my project?**
A: Yes, everything is MIT licensed. Include the license notice and you're good to go.

**Q: How is this different from other LLM repos?**
A: We focus on **reusable skills and patterns** rather than a single framework. We cover the full LLM stack (skills, agents, fine-tuning, RAG, inference, evaluation) with production maturity.

---

## Acknowledgments

LLM-Whisperer builds on research, insights, and patterns from:

- The open-source AI community
- Academic research institutions
- Production deployment experiences
- Community feedback and contributions

Special thanks to all contributors and users who help improve this repository.

---

## Contact & Social

- **Author**: Shuvam Banerji Seal
- **GitHub**: [Shuvam-Banerji-Seal](https://github.com/Shuvam-Banerji-Seal)
- **Repository**: [LLM-Whisperer](https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer)

---

**Last Updated**: April 2026

For the latest updates, check the [GitHub repository](https://github.com/Shuvam-Banerji-Seal/LLM-Whisperer).
