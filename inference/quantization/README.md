# Quantization Toolkit

Author: Shuvam Banerji Seal

This module standardizes quantization workflows used by the inference stack.
It includes profiles, conversion scripts, and a scorecard template to compare
memory/latency/quality impact.

## Supported Paths

- bitsandbytes (INT8 and NF4 variants)
- GPTQ (post-training quantization profile)
- AWQ runtime loading/fusion configuration
- GGUF quantization via llama.cpp toolchain

## Key References

- https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
- https://huggingface.co/docs/transformers/main/en/quantization/gptq
- https://huggingface.co/docs/transformers/main/en/quantization/awq
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/quantize/README.md
- https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md

## Workflow

1. Select a profile in `configs/quantization_profiles.yaml`.
2. Run the associated script from `scripts/`.
3. Benchmark with `../benchmarking/scripts/run_openai_benchmark.py`.
4. Record quality and performance in `comparison/scorecard_template.csv`.

## Important Caveats

- AWQ fused modules and FlashAttention2 are not always compatible.
- AutoAWQ installation may pin/downgrade Transformers versions.
- Re-quantization from already quantized weights can degrade quality significantly.
- Always validate tokenizer/chat-template parity after conversion.
