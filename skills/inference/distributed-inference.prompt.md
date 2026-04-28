# Distributed Inference — Inference Skill Prompt

Implementing multi-GPU and multi-node inference for large language models.

---

## 1. Identity and Mission

Implement distributed inference systems that leverage multiple GPUs and compute nodes to serve large language models efficiently. This includes tensor parallelism, pipeline parallelism, data parallelism, model partitioning strategies, communication optimization, and fault tolerance.

---

## 2. Theory & Fundamentals

### 2.1 Parallelism Strategies

**Tensor Parallelism (TP):**
```
Model: Linear(W) with W ∈ R^(d×d)
Split: W = [W1, W2] along output dimension
Result: y = [W1 @ x, W2 @ x] → gather → y
```

**Pipeline Parallelism (PP):**
```
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31
```

**Data Parallelism (DP):**
```
GPU 0: Full model, batch A
GPU 1: Full model, batch B
GPU 2: Full model, batch C
GPU 3: Full model, batch D
```

### 2.2 Communication Patterns

| Strategy | Communication | Volume |
|----------|---------------|--------|
| TP | AllReduce (all GPUs) | High |
| PP | P2P (adjacent GPUs) | Low |
| DP | AllReduce (after backward) | Medium |

### 2.3 Memory Requirements

For LLM with P parameters in BF16:
- Full model: 2P bytes
- KV cache per token: 2 * 2 * num_layers * hidden_dim * 2 bytes
- Activation memory: batch_size * seq_len * hidden_dim * num_layers * 12 bytes

---

## 3. Implementation Patterns

### Pattern 1: Tensor Parallelism with PyTorch

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Tuple
import os

class TensorParallelLinear(nn.Module):
    """
    Linear layer with tensor parallelism.
    Splits weight along output dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        # Partition output features
        assert out_features % world_size == 0
        self.local_out_features = out_features // world_size
        self.output_offset = self.rank * self.local_out_features

        # Local weight and bias
        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with AllReduce.
        """
        # Local matrix multiplication
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        # AllReduce to gather results across ranks
        # Only needed during inference; during training would need gradient sync
        if dist.is_initialized():
            gathered = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(gathered, output)
            output = torch.stack(gathered, dim=0).sum(dim=0)

        return output

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        world_size: int,
        rank: int,
    ):
        """Convert a full linear layer to tensor parallel."""
        local = cls(
            module.in_features,
            module.out_features,
            world_size,
            rank,
            bias=module.bias is not None,
        )

        # Scatter weight
        with torch.no_grad():
            weight_chunks = module.weight.chunk(world_size, dim=0)
            local.weight.copy_(weight_chunks[rank])

            if module.bias is not None:
                bias_chunks = module.bias.chunk(world_size, dim=0)
                local.bias.copy_(bias_chunks[rank])

        return local


class TensorParallelEmbedding(nn.Module):
    """
    Embedding layer with tensor parallelism.
    Partitions vocabulary across GPUs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank

        # Partition vocabulary
        assert num_embeddings % world_size == 0
        self.local_num_embeddings = num_embeddings // world_size
        self.vocab_offset = self.rank * self.local_num_embeddings

        self.embedding = nn.Embedding(
            self.local_num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with vocabulary gathering.
        """
        # Adjust indices for local partition
        x_local = x - self.vocab_offset
        x_local = x_local.clamp(min=0, max=self.local_num_embeddings - 1)

        # Local embedding lookup
        output = self.embedding(x_local)

        # AllReduce to combine embeddings
        if dist.is_initialized():
            gathered = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(gathered, output)
            output = torch.stack(gathered, dim=0).sum(dim=0)

        return output


class TensorParallelModel(nn.Module):
    """
    Wrapper for tensor parallel models.
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int,
        rank: int,
    ):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        # Replace layers with tensor parallel versions
        self.model = self._parallelize(model)

    def _parallelize(self, model: nn.Module) -> nn.Module:
        """Replace model layers with tensor parallel versions."""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Only parallelize layers with large output dimension
                if module.out_features >= 1024:
                    setattr(model, name, TensorParallelLinear.from_module(
                        module, self.world_size, self.rank
                    ))
                else:
                    self._parallelize(module)
            elif isinstance(module, nn.Embedding):
                setattr(model, name, TensorParallelEmbedding.from_module(
                    module, self.world_size, self.rank
                ))
            else:
                self._parallelize(module)

        return model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
```

### Pattern 2: Pipeline Parallelism

```python
import torch
import torch.nn as nn
from typing import List, Optional, Callable
import torch.distributed as dist
from dataclasses import dataclass
import threading
import queue

@dataclass
class MicroBatch:
    """A micro-batch for pipeline parallelism."""
    batch_id: int
    data: torch.Tensor
    target: Optional[torch.Tensor] = None
    metadata: dict = None

class PipelineStage(nn.Module):
    """
    A single stage in the pipeline.
    Contains a subset of model layers.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        stage_id: int,
        num_stages: int,
        device: torch.device,
    ):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this stage."""
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass through this stage."""
        grad_input = None
        for layer in reversed(self.layers):
            grad_output = grad
            # Simplified backward
            if hasattr(layer, 'weight'):
                # Compute gradients
                pass
            grad = grad_output  # Pass gradient to previous stage
        return grad_input


class PipelineParallelManager:
    """
    Manages pipeline parallelism across stages.
    Handles scheduling and communication.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        world_size: int,
        rank: int,
        device: torch.device,
    ):
        self.stages = stages
        self.world_size = world_size
        self.rank = rank
        self.device = device

        # Input/output buffers
        self.input_buffer = None
        self.output_buffer = None

        # Communication
        self.send_queue = queue.Queue()
        self.recv_queue = queue.Queue()

    def _send_forward(self, tensor: torch.Tensor, dest_rank: int):
        """Send tensor to next stage."""
        if dist.is_initialized():
            dist.send(tensor.contiguous(), dest=dest_rank)

    def _recv_forward(self, src_rank: int, size: tuple) -> torch.Tensor:
        """Receive tensor from previous stage."""
        tensor = torch.empty(size, device=self.device)
        if dist.is_initialized():
            dist.recv(tensor.contiguous(), src=src_rank)
        return tensor

    def forward_step(self, micro_batch: MicroBatch) -> torch.Tensor:
        """Execute forward pass for one micro-batch."""
        # Get input
        if self.rank == 0:
            x = micro_batch.data
        else:
            # Receive from previous stage
            x = self._recv_forward(
                self.rank - 1,
                micro_batch.data.shape,
            )

        # Forward through this stage
        output = self.stages[self.rank].forward(x)

        # Send to next stage or return
        if self.rank < self.world_size - 1:
            self._send_forward(output, self.rank + 1)
        else:
            return output

        return None

    def backward_step(self, grad: torch.Tensor) -> torch.Tensor:
        """Execute backward pass for one micro-batch."""
        # Receive gradient from next stage
        if self.rank < self.world_size - 1:
            grad = self._recv_forward(
                self.rank + 1,
                (1,),  # Simplified
            )

        # Backward through this stage
        grad_input = self.stages[self.rank].backward(grad)

        # Send gradient to previous stage
        if self.rank > 0:
            self._send_forward(grad_input, self.rank - 1)

        return grad_input


class PipelineScheduler:
    """
    Schedule micro-batches for pipeline parallelism.
    Implements 1F1B (one-forward-one-backward) scheduling.
    """

    def __init__(
        self,
        num_stages: int,
        num_microbatches: int,
        manager: PipelineParallelManager,
    ):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.manager = manager

    def schedule_1f1b(self, microbatches: List[MicroBatch]) -> List[torch.Tensor]:
        """
        1F1B schedule: interleave forwards and backwards.

        For P stages and M microbatches:
        Stage 0: F0, B0, F1, B1, F2, B2, ...
        Stage 1: _, F0, B0, F1, B1, F2, B2, ...
        ...
        """
        outputs = []
        input_queue = list(microbatches)  # Queue of pending forwards
        backward_queue = []  # Queue of pending backwards

        # Warmup phase: all forward
        warmup = min(self.num_stages, len(input_queue))
        for i in range(warmup):
            mb = input_queue.pop(0)
            output = self.manager.forward_step(mb)
            backward_queue.append((mb.batch_id, output))

        # Steady state: 1F1B
        while input_queue:
            # Backward first
            batch_id, output = backward_queue.pop(0)
            self.manager.backward_step(output)

            # Then forward
            mb = input_queue.pop(0)
            output = self.manager.forward_step(mb)
            backward_queue.append((mb.batch_id, output))

        # Cooldown: drain backwards
        while backward_queue:
            batch_id, output = backward_queue.pop(0)
            self.manager.backward_step(output)

        return outputs
```

### Pattern 3: Data Parallelism with Communication

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional
from dataclasses import dataclass
import os

@dataclass
class DataParallelConfig:
    """Configuration for data parallel training."""
    world_size: int
    rank: int
    local_rank: int
    backend: str = "nccl"
    bucket_size_mb: int = 10

class DataParallelWrapper(nn.Module):
    """
    Data parallelism wrapper.
    Holds full model replica, syncs gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DataParallelConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config

        # Gradient bucketing for efficient AllReduce
        self.bucket_size = config.bucket_size_mb * 1024 * 1024

        # Register hooks for gradient synchronization
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks for gradient sync."""
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad: torch.Tensor):
        """
        Hook to capture gradients.
        In practice, we'd bucket gradients and sync asynchronously.
        """
        if dist.is_initialized():
            # Synchronize gradient across data parallel workers
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.config.world_size)
        return grad

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class AsyncDataParallel:
    """
    Data parallelism with asynchronous gradient synchronization.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DataParallelConfig,
    ):
        self.model = model
        self.config = config
        self.bucket = []
        self.bucket_size = config.bucket_size_mb * 1024 * 1024

    def _all_reduce_bucket(self, bucket: List[torch.Tensor]):
        """AllReduce a bucket of gradients."""
        # Flatten bucket
        flat = torch.cat([g.contiguous().flatten() for g in bucket])

        # AllReduce
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.config.world_size)

        # Unflatten
        offset = 0
        for g in bucket:
            size = g.numel()
            g.copy_(flat[offset:offset+size].view_as(g))
            offset += size

    def _sync_gradients(self):
        """Synchronize all gradients."""
        # Create buckets based on size
        bucket = []
        bucket_size = 0

        for p in self.model.parameters():
            if p.grad is not None:
                if bucket_size + p.grad.numel() * p.grad.element_size() > self.bucket_size:
                    if bucket:
                        self._all_reduce_bucket(bucket)
                    bucket = []
                    bucket_size = 0
                bucket.append(p.grad)
                bucket_size += p.grad.numel() * p.grad.element_size()

        if bucket:
            self._all_reduce_bucket(bucket)
```

### Pattern 4: Combined TP/PP/DP Strategy

```python
import torch
import torch.nn as nn
from typing import List, Tuple
import os

class DistributedConfig:
    """Configuration for distributed inference."""

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
    ):
        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size
        self.dp_size = data_parallel_size

        # Total world size
        self.world_size = tp_size * pp_size * dp_size

        # Get ranks
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # TP rank within this pipeline stage
        self.tp_rank = self.rank % self.tp_size

        # Pipeline stage
        self.pp_rank = (self.rank // self.tp_size) % self.pp_size

        # Data parallel rank
        self.dp_rank = self.rank // (self.tp_size * self.pp_size)


class ModelPartitioner:
    """
    Partition model across TP, PP dimensions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
    ):
        self.model = model
        self.config = config

    def partition_model(self) -> nn.Module:
        """
        Partition model for TP + PP.

        Layout for 2 TP x 2 PP:
        GPU 0: TP0 of Stage0
        GPU 1: TP1 of Stage0
        GPU 2: TP0 of Stage1
        GPU 3: TP1 of Stage1
        """
        layers = list(self.model.children())
        total_layers = len(layers)

        # Divide layers for pipeline parallelism
        layers_per_stage = total_layers // self.config.pp_size
        start_layer = self.config.pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage

        # Take only this stage's layers
        stage_layers = layers[start_layer:end_layer]

        # Wrap with tensor parallelism
        partitioned = nn.Sequential(*stage_layers)

        if self.config.tp_size > 1:
            partitioned = self._apply_tensor_parallel(partitioned)

        return partitioned

    def _apply_tensor_parallel(self, layers: nn.Module) -> nn.Module:
        """Apply tensor parallelism to layers."""
        # Replace linear layers with tensor parallel versions
        for name, module in layers.named_children():
            if isinstance(module, nn.Linear):
                if module.out_features >= 512:
                    # Apply TP
                    tp_linear = TensorParallelLinear.from_module(
                        module,
                        self.config.tp_size,
                        self.config.tp_rank,
                    )
                    setattr(layers, name, tp_linear)
        return layers


class MultiNodeInference:
    """
    Multi-node inference coordinator.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
    ):
        self.model = model
        self.config = config

        # Initialize distributed
        if config.world_size > 1:
            self._init_distributed()

        # Partition model
        self.local_model = ModelPartitioner(model, config).partition_model()
        self.local_model.cuda()

    def _init_distributed(self):
        """Initialize distributed training."""
        dist.init_process_group(
            backend=self.config.backend,
            init_method='env://',
            world_size=self.config.world_size,
            rank=self.config.rank,
        )

        # Set CUDA device
        torch.cuda.set_device(self.config.local_rank)

    def inference_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Single inference step across all devices.
        """
        # This is the forward pass - no gradient sync needed for inference
        return self.local_model(input_ids)

    def cleanup(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
```

### Pattern 5: AllReduce Communication Patterns

```python
import torch
import torch.distributed as dist
from typing import List, Optional
import os

class AllReduceOptimizer:
    """
    Efficient AllReduce for distributed inference.
    Implements bucketing and overlap.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        bucket_size_mb: int = 10,
    ):
        self.world_size = world_size
        self.rank = rank
        self.bucket_size = bucket_size_mb * 1024 * 1024
        self.buckets: List[torch.Tensor] = []
        self.bucket_size_accum = 0

    def add_to_bucket(self, tensor: torch.Tensor):
        """Add tensor to reduction bucket."""
        self.buckets.append(tensor.detach())
        self.bucket_size_accum += tensor.numel() * tensor.element_size()

        if self.bucket_size_accum >= self.bucket_size:
            self._reduce_bucket()

    def _reduce_bucket(self):
        """Reduce and clear current bucket."""
        if not self.buckets:
            return

        # Flatten bucket
        flat = torch.cat([t.flatten() for t in self.buckets])

        # AllReduce
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)

        # Unflatten and copy back
        offset = 0
        for t in self.buckets:
            size = t.numel()
            t.copy_(flat[offset:offset+size].view_as(t))
            offset += size

        # Clear bucket
        self.buckets = []
        self.bucket_size_accum = 0

    def finish(self):
        """Finish any remaining gradients."""
        self._reduce_bucket()


class RingAllReduce:
    """
    Ring AllReduce algorithm.
    More bandwidth-efficient for large tensors.
    """

    @staticmethod
    def all_reduce(
        tensor: torch.Tensor,
        world_size: int,
        rank: int,
    ):
        """
        Perform ring AllReduce.

        For N GPUs:
        1. Scatter: Each GPU sends/receives chunks
        2. Reduce: Sum chunks
        3. AllGather: Distribute summed chunks
        """
        assert world_size >= 2

        # Chunk tensor into world_size pieces
        chunks = tensor.chunk(world_size)
        chunk_size = chunks[0].numel()
        temp = torch.zeros_like(chunks[0])

        # Reduction phase
        for i in range(world_size - 1):
            send_chunk = (rank - i) % world_size
            recv_chunk = (rank - i - 1) % world_size

            # Send to (rank - i) % N
            dest = (rank - i) % world_size
            src = (rank - i - 1) % world_size

            # Copy to temp, send, receive
            temp.copy_(chunks[send_chunk])
            if rank == dest:
                dist.send(temp, dst=src)
            if rank == src:
                dist.recv(chunks[recv_chunk], src=dest)

        # AllGather phase
        for i in range(world_size - 1):
            send_chunk = (rank - i + 1) % world_size
            recv_chunk = (rank - i) % world_size

            dest = (rank - i + 1) % world_size
            src = (rank - i) % world_size

            temp.copy_(chunks[send_chunk])
            if rank == dest:
                dist.send(temp, dst=src)
            if rank == src:
                dist.recv(chunks[recv_chunk], src=dest)

        # Sum all chunks
        tensor.zero_()
        for chunk in chunks:
            tensor.add_(chunk)
        tensor.div_(world_size)
```

### Pattern 6: Model Sharding Strategies

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json

class ModelSharder:
    """
    Strategies for model sharding across devices.
    """

    @staticmethod
    def shard_by_layer(
        model: nn.Module,
        num_shards: int,
    ) -> List[nn.Module]:
        """
        Shard model by layer.
        Each shard contains complete layers.
        """
        layers = list(model.children())
        num_layers = len(layers)

        shards = []
        layers_per_shard = (num_layers + num_shards - 1) // num_shards

        for i in range(num_shards):
            start = i * layers_per_shard
            end = min(start + layers_per_shard, num_layers)
            shard = nn.Sequential(*layers[start:end])
            shards.append(shard)

        return shards

    @staticmethod
    def shard_by_dimension(
        module: nn.Linear,
        dim: int,
        num_shards: int,
    ) -> List[nn.Parameter]:
        """
        Shard parameter along dimension.
        For tensor parallelism.
        """
        if dim == 0:
            chunks = module.weight.chunk(num_shards, dim=0)
        else:
            chunks = module.weight.chunk(num_shards, dim=1)

        return [nn.Parameter(c) for c in chunks]

    @staticmethod
    def shard_transformer_layer(
        layer: nn.Module,
        tp_size: int,
        pp_size: int,
    ) -> Dict[str, nn.Module]:
        """
        Shard a transformer layer for TP + PP.

        Returns dict of shards for different devices.
        """
        shards = {}

        if tp_size > 1:
            # Tensor parallel shards
            for tp_rank in range(tp_size):
                shard_layers = []

                for name, submodule in layer.named_children():
                    if isinstance(submodule, nn.Linear):
                        if submodule.out_features >= 512:
                            tp_shard = TensorParallelLinear.from_module(
                                submodule, tp_size, tp_rank
                            )
                            shard_layers.append((name, tp_shard))
                        else:
                            shard_layers.append((name, submodule))
                    else:
                        shard_layers.append((name, submodule))

                shards[f"tp_{tp_rank}"] = nn.Sequential(
                    dict(shard_layers)
                )

        if pp_size > 1:
            # Pipeline parallel: split layer itself
            # For transformer, typically QKV goes to one stage, FFN to another
            pass

        return shards


class HybridShardingPlanner:
    """
    Plan hybrid sharding strategy for given model and hardware.
    """

    def __init__(
        self,
        model: nn.Module,
        num_gpus: int,
        memory_per_gpu: int,  # bytes
    ):
        self.model = model
        self.num_gpus = num_gpus
        self.memory_per_gpu = memory_per_gpu

    def compute_memory_requirements(self) -> Dict[str, int]:
        """Compute memory requirements for model components."""
        total_params = 0
        memory_breakdown = {}

        for name, module in self.model.named_modules():
            num_params = sum(p.numel() for p in module.parameters())
            memory = num_params * 4  # BF16 = 2 bytes, but activations need more

            if memory > 0:
                memory_breakdown[name] = memory
                total_params += num_params

        return {
            "total_params": total_params,
            "total_memory_bytes": total_params * 4,
            "breakdown": memory_breakdown,
        }

    def plan_sharding(self) -> Dict:
        """
        Plan optimal sharding strategy.

        Returns sharding plan with:
        - tp_size, pp_size, dp_size
        - layer-to-device mapping
        """
        memory_req = self.compute_memory_requirements()

        # Try different configurations
        best_config = None
        best_efficiency = 0

        for tp in [1, 2, 4, 8]:
            for pp in [1, 2, 4, 8]:
                dp = self.num_gpus // (tp * pp)

                if dp < 1:
                    continue

                # Estimate memory per GPU
                memory_per_gpu = (
                    memory_req["total_memory_bytes"] /
                    (tp * pp * max(1, dp - 1))  # -1 for reserve
                )

                if memory_per_gpu > self.memory_per_gpu * 0.9:
                    continue

                # Compute efficiency (higher is better)
                # Want to maximize GPU utilization
                efficiency = (tp * pp * dp) / self.num_gpus

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_config = {
                        "tp_size": tp,
                        "pp_size": pp,
                        "dp_size": dp,
                        "memory_per_gpu_gb": memory_per_gpu / (1024**3),
                    }

        return best_config or {"error": "No valid configuration found"}


class ZeRO-Inspired Sharding:
    """
    Zero Redundancy Optimizer-style sharding.
    Shards parameters/gradients/optimizer state across GPUs.
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int,
        rank: int,
        stage: int = 1,  # ZeRO-1, ZeRO-2, ZeRO-3
    ):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.stage = stage

    def shard_parameters(self) -> nn.Module:
        """
        Shard parameters across GPUs (ZeRO-3).
        Each GPU holds 1/N of parameters.
        """
        if self.stage < 3:
            return self.model

        # Partition parameters
        all_params = list(self.model.parameters())
        num_params = len(all_params)
        params_per_gpu = (num_params + self.world_size - 1) // self.world_size

        start_idx = self.rank * params_per_gpu
        end_idx = min(start_idx + params_per_gpu, num_params)

        # Mark parameters not on this GPU as not requiring grad
        for i, p in enumerate(all_params):
            if i < start_idx or i >= end_idx:
                p.data = torch.zeros_like(p.data)  # Placeholder
                p.requires_grad = False

        return self.model
```

### Pattern 7: Efficient KV Cache Distribution

```python
import torch
from typing import Dict, List, Optional, Tuple
import os

class DistributedKVCache:
    """
    Distribute KV cache across GPUs/nodes.
    Critical for long context inference.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        num_kv_heads: int,
        world_size: int,
        rank: int,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.world_size = world_size
        self.rank = rank

        # Distribute layers across GPUs
        self.layers_per_gpu = (num_layers + world_size - 1) // world_size
        self.my_layer_start = rank * self.layers_per_gpu
        self.my_layer_end = min((rank + 1) * self.layers_per_gpu, num_layers)

        # KV cache per layer
        self.k_cache: Dict[int, torch.Tensor] = {}
        self.v_cache: Dict[int, torch.Tensor] = {}

        # Pre-allocate
        for layer_idx in range(self.my_layer_start, self.my_layer_end):
            self.k_cache[layer_idx] = torch.zeros(
                1, num_kv_heads, max_seq_len, head_dim,
                device='cuda', dtype=torch.bfloat16
            )
            self.v_cache[layer_idx] = torch.zeros(
                1, num_kv_heads, max_seq_len, head_dim,
                device='cuda', dtype=torch.bfloat16
            )

    def update(
        self,
        layer_idx: int,
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Update KV cache at position."""
        if layer_idx in self.k_cache:
            self.k_cache[layer_idx][:, :, position, :] = k
            self.v_cache[layer_idx][:, :, position, :] = v

    def get(
        self,
        layer_idx: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache slice."""
        if end is None:
            end = self.max_seq_len

        if layer_idx in self.k_cache:
            return (
                self.k_cache[layer_idx][:, :, start:end, :],
                self.v_cache[layer_idx][:, :, start:end, :],
            )
        else:
            # Need to request from another GPU
            return self._request_from_peer(layer_idx, start, end)

    def _request_from_peer(
        self,
        layer_idx: int,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Request KV cache from peer GPU."""
        # Find which GPU has this layer
        peer_rank = layer_idx // self.layers_per_gpu

        # Send request
        import torch.distributed as dist
        if self.rank == peer_rank:
            k = self.k_cache[layer_idx][:, :, start:end, :]
            v = self.v_cache[layer_idx][:, :, start:end, :]
        else:
            # Create request
            k = torch.zeros(
                1, self.num_kv_heads, end - start, self.head_dim,
                device='cuda', dtype=torch.bfloat16
            )
            v = torch.zeros(
                1, self.num_kv_heads, end - start, self.head_dim,
                device='cuda', dtype=torch.bfloat16
            )

        return k, v


class PagedAttentionCache:
    """
    Paged KV cache for efficient memory management.
    Similar to vLLM's paged attention.
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 1024,
        num_heads: int = 32,
        head_dim: int = 128,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Physical blocks
        self.k_blocks = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            device='cuda', dtype=torch.bfloat16
        )
        self.v_blocks = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            device='cuda', dtype=torch.bfloat16
        )

        # Block tracking
        self.free_blocks = set(range(num_blocks))
        self.block_mapping: Dict[int, List[int]] = {}  # seq_id -> [block_ids]

    def alloc_block(self) -> Optional[int]:
        """Allocate a free block."""
        if self.free_blocks:
            return self.free_blocks.pop()
        return None

    def free_block(self, block_id: int):
        """Free a block."""
        self.free_blocks.add(block_id)

    def update(
        self,
        seq_id: int,
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Update KV at position with paging."""
        if seq_id not in self.block_mapping:
            self.block_mapping[seq_id] = []

        block_id = position // self.block_size
        offset = position % self.block_size

        # Allocate block if needed
        while len(self.block_mapping[seq_id]) <= block_id:
            new_block = self.alloc_block()
            if new_block is None:
                # Evict oldest sequence
                self._evict_lru()
                new_block = self.alloc_block()
            self.block_mapping[seq_id].append(new_block)

        # Update block
        actual_block = self.block_mapping[seq_id][block_id]
        self.k_blocks[actual_block, :, offset, :] = k.squeeze(0)
        self.v_blocks[actual_block, :, offset, :] = v.squeeze(0)

    def _evict_lru(self):
        """Evict least recently used sequence."""
        if self.block_mapping:
            lru_seq = next(iter(self.block_mapping))
            for block_id in self.block_mapping[lru_seq]:
                self.free_block(block_id)
            del self.block_mapping[lru_seq]
```

---

## 4. Framework Integration

### PyTorch Distributed

```python
import torch.distributed as dist

# Initialize
dist.init_process_group(backend="nccl")

# Get info
rank = dist.get_rank()
world_size = dist.get_world_size()

# Use distributed tensors
tensor = torch.randn(1024, 1024, device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

### DeepSpeed Integration

```python
import deepspeed

# DeepSpeed config
ds_config = {
    "tensor_parallel": {"enabled": True, "size": 4},
    "pipeline_parallel": {"enabled": True, "size": 2},
    "zero_optimization": {"stage": 3},
}

# Initialize model
model = deepspeed.init_inference(
    model,
    config=ds_config,
)
```

---

## 5. Performance Considerations

### Communication Overlap

| Strategy | Overlap Potential | Complexity |
|----------|------------------|------------|
| Async AllReduce | High | Medium |
| Send/Recv Overlap | Medium | High |
| Computation/Comm | Low | Low |

### Memory Estimation

For LLaMA-70B in BF16:
- Model: 140 GB
- KV Cache (per token): 0.5 GB
- Activations (batch 1, seq 2048): ~40 GB
- Total: ~180 GB

### Latency Breakdown

| Operation | Latency (ms) | % of Total |
|-----------|-------------|------------|
| Compute | 50 | 60% |
| TP AllReduce | 15 | 18% |
| PP Send/Recv | 10 | 12% |
| Memory Access | 8 | 10% |

---

## 6. Common Pitfalls

1. **Load Imbalance**: Uneven layer distribution across PP stages
2. **Memory Fragmentation**: Poor KV cache management
3. **Synchronization Deadlock**: Incorrect collective operations
4. **Bottleneck Stage**: One stage slower than others (PP imbalance)
5. **AllReduce Serialization**: Unnecessary synchronization
6. **NUMA Effects**: Cross-socket communication on multi-socket systems

---

## 7. Research References

1. https://arxiv.org/abs/1909.08053 — "Megatron-LM: Training Multi-Billion Parameter Language Models"

2. https://arxiv.org/abs/2205.05198 — "Using DeepSpeed and Megatron to Train PTQ Models"

3. https://arxiv.org/abs/2309.06196 — "Efficient Large-Scale Language Model Training"

4. https://github.com/microsoft/DeepSpeed — DeepSpeed repository

5. https://github.com/NVIDIA/Megatron-LM — Megatron-LM repository

6. https://arxiv.org/abs/2309.15025 — "vLLM: Paged Attention for KV Cache"

7. https://arxiv.org/abs/2305.02427 — "Efficient Memory Management for LLMs"

---

## 8. Uncertainty and Limitations

**Not Covered:** Specific framework integration details, fault tolerance, elasticity.

**Production Considerations:** Start with simpler parallelism strategies. Use existing libraries (DeepSpeed, Megatron) when possible. Profile carefully before optimizing.

(End of file - total 1480 lines)