# Quantization Implementation

Complete guide to model compression through quantization techniques.

## Overview

This implementation covers all major quantization methods:
- **BitsAndBytes** - Easy 4-bit and 8-bit quantization
- **AutoAWQ** - Activation-aware weight quantization (4-bit, high quality)
- **GPTQ** - Gradient-informed post-training quantization (fastest inference)
- **GGUF** - CPU-friendly format for llama.cpp
- **QAT** - Quantization-aware training (best quality)
- **PTQ** - Post-training quantization (simplest)

## Files Included

```
quantization/
├── quantization-complete.py    # Complete implementation (768 lines)
├── README.md                   # This file
└── Examples:
    ├── BitsAndBytes quantization
    ├── AutoAWQ quantization
    ├── GPTQ with calibration
    ├── GGUF conversion
    ├── QAT training
    ├── Benchmarking & evaluation
    └── Production deployment
```

## Compression Overview

### Memory Impact (7B Model)

| Format | Size | Compression | Quality | Inference |
|--------|------|-------------|---------|-----------|
| FP32 | 28 GB | 1x | 100% (baseline) | CPU/GPU |
| FP16 | 14 GB | 2x | 99.5% | Fast GPU |
| INT8 | 7 GB | 4x | 96-98% | Fast GPU |
| INT4 | 3.5 GB | 8x | 90-96% | Fast GPU |
| GGUF (Q4) | 3.5 GB | 8x | 90-96% | Fast CPU |

### Trade-offs by Bitwidth

| Bitwidth | Size | Speed | Quality | Use Case |
|----------|------|-------|---------|----------|
| FP32 | 100% | 1.0x | 100% | Training, reference |
| FP16 | 50% | 2.0x | 99% | Fine-tuning, inference |
| INT8 | 25% | 2.5x | 96% | Edge inference |
| INT4 | 12.5% | 3.5x | 93% | Mobile, consumer GPU |
| INT3 | 9.4% | 4.0x | 88% | Extreme compression |
| INT2 | 6.25% | 4.5x | 80% | Specialized hardware |

## Key Components

### 1. BitsAndBytes Quantization
Easiest quantization method, supports both 4-bit and 8-bit:

```python
from quantization_complete import BitsAndBytesQuantizer

# 4-bit quantization
quantizer = BitsAndBytesQuantizer(
    model_name="meta-llama/Llama-2-7b-hf",
    bits=4,
    compute_dtype="bfloat16",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 (better than INT4)
    bnb_4bit_use_double_quant=True
)

quantizer.quantize()
quantizer.save_quantized("./quantized_model")

# 8-bit quantization (faster, higher quality)
quantizer_8bit = BitsAndBytesQuantizer(
    model_name="meta-llama/Llama-2-7b-hf",
    bits=8
)
```

**When to use**:
- ✅ Quickest to implement
- ✅ Works with HuggingFace Transformers
- ✅ Good quality-speed trade-off
- ❌ Slower inference than GPTQ

**Performance**:
- Memory: 7B model → 3.5-7 GB
- Speed: 2.5x faster than FP32
- Quality: 93-96% of original

### 2. AutoAWQ (Activation-Aware)
Quantization that considers activation patterns:

```python
from quantization_complete import AutoAWQQuantizer

quantizer = AutoAWQQuantizer(
    model_name="meta-llama/Llama-2-7b-hf",
    bits=4,
    group_size=128
)

# Quantize and calibrate on sample data
quantizer.quantize(
    calibration_data=calibration_texts,  # Sample inputs
    num_samples=128
)

quantizer.save_quantized("./awq_model")

# Benchmark quality
metrics = quantizer.benchmark()
print(f"Perplexity: {metrics['perplexity']}")
```

**Advantages**:
- ✅ Higher quality than INT4 (93-95% vs 90-92%)
- ✅ Activation-aware (considers actual data patterns)
- ✅ Well-supported by vLLM
- ❌ Requires calibration data
- ❌ Slower quantization process

**Quality**: 95-96% of FP32 (comparable to INT8)

### 3. GPTQ (Gradient-Informed)
Post-training quantization using Hessian information:

```python
from quantization_complete import GPTQQuantizer

quantizer = GPTQQuantizer(
    model_name="meta-llama/Llama-2-7b-hf",
    bits=4,
    group_size=128,
    desc_act=True,  # Describe activation order
    use_double_quant=True
)

# Calibrate on representative data
quantizer.quantize(
    calibration_data=calibration_texts,
    num_samples=256
)

quantizer.save_quantized("./gptq_model")

# Benchmark
metrics = quantizer.benchmark()
```

**Why GPTQ**:
- ✅ Fastest inference (optimized kernels)
- ✅ Minimal quality loss (using Hessian)
- ✅ Well-optimized in inference engines
- ❌ Slower quantization (requires Hessian computation)
- ❌ Larger calibration overhead

**Performance**:
- Inference: 3-4x faster than FP32
- Quality: 92-94% of original (good for most tasks)

### 4. GGUF Format
CPU-friendly quantization for llama.cpp:

```python
from quantization_complete import GGUFConverter

converter = GGUFConverter(
    model_name="meta-llama/Llama-2-7b-hf",
    output_format="gguf_q4_0"  # Q4_0, Q4_1, Q5_0, Q6_K, etc.
)

# Convert to GGUF
converter.convert("./llama-2-7b.gguf")

# Use with llama.cpp
# ./main -m ./llama-2-7b.gguf -n 256 -p "Hello"
```

**GGUF Quantization Types**:
- **Q4_0**: Fast, 60-70% size (4-bit with 32 values per group)
- **Q4_1**: 60-70% size, better accuracy
- **Q5_0**: 70-80% size, better quality than Q4
- **Q6_K**: 80-85% size, minimal quality loss

**Use Cases**:
- CPU inference on laptops/servers
- Edge deployment without GPU
- Multi-model serving (low memory footprint)

### 5. QAT (Quantization-Aware Training)
Train with quantization to improve quality:

```python
from quantization_complete import QATTrainer

trainer = QATTrainer(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="instructions.jsonl",
    quantization_config={
        "bits": 4,
        "group_size": 128,
        "fake_quant": True  # Simulate quantization during training
    }
)

metrics = trainer.train(
    num_epochs=1,
    learning_rate=1e-5,  # Lower LR for fine-tuning
    batch_size=8
)
```

**When to use**:
- ✅ Best quality (95-97% of original)
- ✅ Fine-tuning minimizes quantization impact
- ❌ Requires training (1-2 days for 7B)
- ❌ Higher computational cost

**Quality**: 95-97% (comparable to FP16)

### 6. PTQ (Post-Training Quantization)
Simplest method - quantize without retraining:

```python
from quantization_complete import PTQOptimizer

optimizer = PTQOptimizer(
    model_name="meta-llama/Llama-2-7b-hf",
    quantization_method="int8",
    calibration_method="percentile"  # or "entropy", "kl"
)

optimizer.quantize(
    calibration_data=calibration_texts,
    num_samples=100
)
```

**Pros & Cons**:
- ✅ Quickest to implement
- ✅ No training required
- ❌ Lower quality (90-94%)
- ❌ Less control over accuracy

## Quick Start

### Recommendation by Use Case

**For General Deployment (Recommended)**
```python
# Use AutoAWQ: Best quality-speed trade-off
from quantization_complete import AutoAWQQuantizer

quantizer = AutoAWQQuantizer(
    model_name="meta-llama/Llama-2-7b-hf",
    bits=4
)
quantizer.quantize(calibration_data=sample_texts)
quantizer.save_quantized("./model")
```

**For Fastest Inference**
```python
# Use GPTQ: Optimized kernels
from quantization_complete import GPTQQuantizer

quantizer = GPTQQuantizer(
    model_name="meta-llama/Llama-2-7b-hf",
    bits=4
)
quantizer.quantize(calibration_data=sample_texts)
```

**For CPU Inference**
```python
# Use GGUF with Q4_0
from quantization_complete import GGUFConverter

converter = GGUFConverter(
    model_name="meta-llama/Llama-2-7b-hf",
    output_format="gguf_q4_0"
)
converter.convert("./model.gguf")
```

**For Highest Quality**
```python
# Use QAT: Fine-tuning improves accuracy
from quantization_complete import QATTrainer

trainer = QATTrainer(
    model_name="meta-llama/Llama-2-7b",
    dataset_path="instructions.jsonl"
)
trainer.train(num_epochs=1)
```

## Benchmarking & Evaluation

```python
from quantization_complete import QuantizationBenchmark

benchmark = QuantizationBenchmark(
    original_model="meta-llama/Llama-2-7b",
    quantized_model="./quantized_model"
)

# Evaluate quality
results = benchmark.evaluate(
    test_dataset="wikitext",
    metrics=["perplexity", "accuracy", "f1"]
)

print(f"Perplexity: {results['perplexity']}")
print(f"Memory Savings: {results['memory_savings']}x")
print(f"Speedup: {results['speedup']}x")
```

## Common Patterns

### Production Deployment Pipeline
1. Use AutoAWQ or GPTQ for quantization
2. Benchmark on your domain data
3. Deploy with vLLM or TensorRT
4. Monitor quality metrics in production

### Quality-Critical Applications
1. Use QAT for fine-tuning
2. Validate on task-specific test set
3. Compare with FP32 baseline
4. Deploy if quality ≥95% of baseline

### Mobile/Edge Deployment
1. Convert to GGUF format
2. Use Q4_0 or Q5_0 quantization
3. Test on target hardware
4. Profile memory and latency

## Troubleshooting

**Q: Large quality drop after quantization?**
- Try higher bitwidth (INT8 instead of INT4)
- Use QAT to fine-tune quantized model
- Check if model is complex (large → more sensitive)

**Q: OOM during quantization?**
- Reduce calibration batch size
- Use smaller calibration dataset
- Quantize layer-by-layer instead

**Q: Inference slow after quantization?**
- Check if inference engine supports quantized format
- Ensure GPU kernels are optimized (GPTQ better than AWQ on some hardware)
- Try GGUF if CPU inference acceptable

## References

- **BitsAndBytes**: [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339)
- **AutoAWQ**: [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- **GPTQ**: [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- **GGUF**: [GGML Format Specification](https://github.com/ggerganov/ggml)
- **QAT**: [Quantization-Aware Training](https://arxiv.org/abs/1609.07061)

## Integration with Other Skills

- **Fast Inference**: Combine with KV-cache for 10-15x speedup
- **Fine-Tuning**: Use QLoRA (QAT + LoRA) for efficient tuning
- **Infrastructure**: Deploy quantized models with vLLM
- **Monitoring**: Track quality metrics post-quantization

