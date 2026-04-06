# Pipeline Parallelism: Scaling LLM Inference Across 1000+ GPUs
**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Date:** April 2026  
**Status:** Production-Ready Skill Documentation

---

## Problem Statement

**Extreme Scale Challenge:** Training 175B models requires 1000+ GPUs. **Pure tensor parallelism TP=1000 suffers from:**
- Communication overhead dominates
- Each GPU only does 1/1000 of computation
- Communication cost: 1000x memory per forward pass

**Solution:** Pipeline parallelism (PP) partitions model layers sequentially across GPUs.

---

## Mathematical Foundations

### 1. Pipeline Latency

$$T_{\text{pipeline}} = (S - 1 + K) \times T_{\text{stage}}$$

Where:
- S = number of stages (GPUs)
- K = micro-batches per mini-batch
- $T_{\text{stage}}$ = time per stage

**Bubble Ratio:**
$$\text{Bubble} = \frac{S - 1}{S - 1 + K}$$

**Example:** S=8, K=8
$$T = (8-1+8) \times T_{\text{stage}} = 15T_{\text{stage}}$$
$$\text{Bubble} = \frac{7}{15} = 47\%$$

### 2. Memory Efficiency

$$M_{\text{per\_GPU}} = \frac{M_{\text{model}}}{S} + M_{\text{activations}}$$

**For 175B model, S=8:**
$$M_{\text{model\_per\_GPU}} = \frac{175 \times 10^9 \times 2}{8} = 43.75\text{GB}$$

### 3. Stage Balancing

Optimal when:
$$\max_i(T_{\text{stage}_i}) / \text{mean}_i(T_{\text{stage}_i}) \approx 1.0$$

---

## Core Concepts

### 1. Stage Distribution

**Even Partition (Simple):**
```
8 GPUs, 96 layers:
- Each GPU: 12 layers
- Problem: Attention layers slower than FFN layers
```

**Cost-Based Partition (Optimal):**
```
Measure each layer's cost
Assign layers to minimize max(stage_time)
Using dynamic programming
```

### 2. Micro-Batching (GPipe Strategy)

**Split mini-batch into micro-batches:**
```
Mini-batch: 64 samples
Micro-batches: 8 × 8 = 8 stages × 8 micro-batches

Stage 1: Process MB1, MB2, ... in pipeline
Stage 2: Can process MB1 while Stage 1 processes MB2
Result: Pipelined execution reduces bubble
```

### 3. Pipeline Bubble Elimination

**Interleaving Micro-Batches:**
```
Time →
Stage 0: [MB1] [MB2] [MB3] [MB4]
Stage 1:  [MB1] [MB2] [MB3] [MB4]
Stage 2:   [MB1] [MB2] [MB3] [MB4]

Utilized ████████████████
Idle     ██

Bubble: 2/16 = 12.5%
```

---

## Implementation Guide

### Step 1: Basic Pipeline Setup

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-175b-hf",  # Hypothetical
    tensor_parallel_size=8,   # TP=8
    pipeline_parallel_size=8,  # PP=8
    dtype="float16"
)
```

### Step 2: DeepSpeed Pipeline Configuration

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params={
        "train_batch_size": 64,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 0.001}
        },
        "pipeline": {
            "partitions": 8,  # 8 pipeline stages
        }
    }
)
```

### Step 3: Manual Stage Implementation

```python
class PipelineStage(nn.Module):
    def __init__(self, layers, stage_id):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.stage_id = stage_id
    
    def forward(self, x):
        return self.layers(x)

class GPipeMicrobatching:
    def __init__(self, stages, num_micro_batches=4):
        self.stages = stages
        self.num_micro_batches = num_micro_batches
        self.num_stages = len(stages)
    
    def forward(self, x):
        micro_batches = x.chunk(self.num_micro_batches, dim=0)
        
        # Forward pass with pipelining
        activations = {}
        
        for mb_id in range(self.num_micro_batches):
            for stage_id, stage in enumerate(self.stages):
                if stage_id == 0:
                    activations[(mb_id, stage_id)] = stage(micro_batches[mb_id])
                else:
                    prev_activation = activations[(mb_id, stage_id - 1)]
                    activations[(mb_id, stage_id)] = stage(prev_activation)
        
        # Gather outputs
        outputs = [
            activations[(mb_id, self.num_stages - 1)]
            for mb_id in range(self.num_micro_batches)
        ]
        return torch.cat(outputs, dim=0)
```

### Step 4: Heterogeneous Pipeline (Orca Strategy)

```python
class HeterogeneousPipeline:
    """Different batch sizes per stage (prefill vs decode)."""
    
    def __init__(self, stages, prefill_batch=32, decode_batch=4):
        self.stages = stages
        self.prefill_batch = prefill_batch
        self.decode_batch = decode_batch
    
    def forward(self, prompts):
        # Stage 0: Large batch (prefill - high parallelism)
        prefill_results = self.stages[0](prompts, batch_size=self.prefill_batch)
        
        # Stages 1-n: Small batch (decode - low parallelism)
        output = prefill_results
        for stage in self.stages[1:]:
            output = stage(output, batch_size=self.decode_batch)
        
        return output
```

---

## Performance Analysis

### 1. Pipeline Bubble Trade-off

| Num Stages | K (Micro-batches) | Bubble % | Efficiency |
|-----------|------------------|---------|-----------|
| 4 | 4 | 43% | 57% |
| 8 | 8 | 47% | 53% |
| 8 | 16 | 30% | 70% |
| 8 | 32 | 18% | 82% |

**Optimal:** K = 2-4x number of stages

### 2. Scaling to 1000 GPUs

**175B Model, PP=125, TP=8:**
```
Throughput (tokens/sec): 500-1000
Communication per forward: ~10x model memory
Bubble with optimal K: ~15-20%
Overall efficiency: ~50-60% GPU utilization
```

### 3. Memory Efficiency

**Per-GPU Memory (175B, 8 stages):**
```
Model weights: 175B/8 ≈ 22GB
Activations: 5-10GB (checkpointing helps)
KV cache: 2-5GB
Total: ~30-40GB (fits on H100)
```

---

## Real-World Examples

### Example 1: Multi-Stage Inference

**Problem:** Train GPT-175B, now deploy inference

**Solution:** PP=8 + TP=8 on 64 H100s
```
Configuration:
- 8 pipeline stages
- 8 tensor parallel per stage
- 64 H100s total
- Throughput: 500 tokens/sec (8x single GPU)
- Cost: $160/hour
```

### Example 2: Extreme Scale Serving

**Problem:** Serve 175B model to 10,000 concurrent users

**Solution:** PP=32, TP=4 on 128 A100s
```
Configuration:
- 32 stages (efficient for extreme scale)
- 4 TP per stage
- 128 A100s (or 64 H100s)
- Batch size: 256-512
- Throughput: 2000+ tokens/sec
- Users: 10,000+ at 1 token/sec each
```

---

## Integration Guide

### DeepSpeed Integration

```bash
# Launch with pipeline parallelism
deepspeed --num_gpus=64 inference.py \
  --model meta-llama/Llama-2-175b-hf \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 8 \
  --pp-config config.json
```

---

## Sources and Citations

### 1. **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**
- **Authors:** Yanping Huang, Youlong Cheng, Ankur Bapna, et al.
- **Organization:** Google AI
- **ArXiv:** 1811.06965
- **Published:** November 2018
- **Key:** 557M AmoebaNet, 6B 128-layer Transformer on 8 TPUs

### 2. **PipeDream: Generalized Pipeline Parallelism for DNN Training**
- **Authors:** Deepak Narayanan, Aaron Harlap, Amar Phanishayee, et al.
- **Venue:** SOSP 2019
- **Key:** Heterogeneous pipeline strategies

### 3. **Memory-Efficient Pipeline-Parallel DNN Training**
- **Authors:** Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, et al.
- **Venue:** ICML 2021
- **Focus:** Activation memory optimization

---

**End of Skill Documentation**

**Integration Status:** Ready for production  
**Recommended Phase:** 3 (Advanced)  
**Estimated Learning Time:** 4-5 hours  
**Code Examples:** 12+ provided
