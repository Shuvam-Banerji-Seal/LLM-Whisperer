# LLM Infrastructure and Deployment Guide (2026)

A comprehensive guide for deploying, scaling, and optimizing Large Language Models in production environments.

## Table of Contents

1. [Kubernetes & Container Orchestration](#kubernetes--container-orchestration)
2. [Container & Docker for LLM Inference](#container--docker-for-llm-inference)
3. [Cloud Deployment Platforms](#cloud-deployment-platforms)
4. [LLM Serving Infrastructure](#llm-serving-infrastructure)
5. [Infrastructure Monitoring & Cost Optimization](#infrastructure-monitoring--cost-optimization)
6. [Common Patterns & Best Practices](#common-patterns--best-practices)
7. [Code Examples & Configuration](#code-examples--configuration)

---

## Kubernetes & Container Orchestration

### Top Repositories

#### 1. KubeRay (Ray on Kubernetes)
- **URL**: https://github.com/ray-project/kuberay
- **Stars**: 2,416
- **Language**: Go (87.8%), Python (7.2%), TypeScript (3.0%)
- **Description**: A toolkit to run Ray applications on Kubernetes, including Ray Serve for LLM serving
- **Key Features**:
  - Native Kubernetes integration for Ray clusters
  - Automatic resource management
  - Multi-GPU distributed inference support
  - Gang scheduling and workload prioritization

#### 2. vLLM Production Stack
- **URL**: https://github.com/vllm-project/production-stack
- **Stars**: 2,258
- **Language**: Python (44%), Shell (19.2%), Go (17.2%)
- **Description**: vLLM's reference system for K8S-native cluster-wide deployment
- **Key Features**:
  - Production-ready Kubernetes manifests
  - Multi-cloud deployment examples (AWS, GCP, Azure)
  - Load balancing and auto-scaling configurations
  - GPU resource optimization patterns

#### 3. LLM-D (Kubernetes-native LLM inference)
- **URL**: https://github.com/llm-d/llm-d
- **Stars**: 3,000+
- **Language**: Shell
- **Description**: Achieve state-of-the-art inference performance with modern accelerators on Kubernetes
- **Key Features**:
  - Disaggregated prefill and decode stages
  - Tiered caching for memory optimization
  - Kubernetes-native scheduling
  - Multi-node distributed inference

#### 4. NVIDIA KAI Scheduler
- **URL**: https://github.com/nvidia/kai-scheduler
- **Stars**: 1,217
- **Language**: Go (99.2%)
- **Description**: Kubernetes-native GPU scheduler for AI workloads at large scale
- **Key Features**:
  - Gang scheduling (all-or-nothing scheduling)
  - Fair-share GPU allocation
  - Time-based fairshare for over-quota resources
  - Native integration with KubeRay

#### 5. Kubeflow with Seldon
- **URL**: https://github.com/kubeflow/example-seldon
- **Stars**: 172
- **Language**: Jupyter Notebook (38.5%), Python (29.8%), Shell (14.6%)
- **Description**: End-to-end ML pipeline example with Seldon Core on Kubernetes
- **Key Features**:
  - Kubeflow pipeline orchestration
  - Seldon Core model serving
  - A/B testing and canary deployments
  - MLOps best practices

#### 6. Akamai KubeRay GPU LLM Quickstart
- **URL**: https://github.com/akamai-developers/kuberay-gpu-llm-quickstart
- **Stars**: 4
- **Language**: Python
- **Description**: Sample LLM deployment via KubeRay on Kubernetes with GPU support
- **Key Features**:
  - Ready-to-run examples
  - Qwen model deployment
  - GPU resource configuration

#### 7. LLM-D Model Service (Helm Charts)
- **URL**: https://github.com/llm-d-incubation/llm-d-modelservice
- **Stars**: 30
- **Language**: Go Template (75.8%), Shell (16.9%)
- **Description**: Helm charts for deploying models with LLM-D
- **Key Features**:
  - Production Helm charts
  - Customizable deployment values
  - Multi-cluster support

#### 8. LLM-D Infrastructure (Helm & Deployment Examples)
- **URL**: https://github.com/llm-d-incubation/llm-d-infra
- **Stars**: 54
- **Language**: Go Template (58.3%), Shell (15.8%)
- **Description**: LLM-D Helm charts and deployment examples
- **Key Features**:
  - Complete deployment pipelines
  - Infrastructure-as-code examples
  - Configuration templates

#### 9. Llama Stack Kubernetes Operator
- **URL**: https://github.com/llamastack/llama-stack-k8s-operator
- **Stars**: 23
- **Language**: Go (90.4%), Makefile (4.6%)
- **Description**: Kubernetes operator for Llama Stack deployment
- **Key Features**:
  - Custom resource definitions for LLM deployment
  - Automated lifecycle management
  - Multi-node orchestration

#### 10. Ray Serve Examples
- **URL**: https://docs.ray.io/en/latest/serve/index.html
- **Documentation**: Ray Serve official documentation
- **Description**: Scalable model serving library with Kubernetes support
- **Key Features**:
  - Built-in load balancing
  - Dynamic batching
  - Per-replica configuration
  - Ray cluster auto-scaling

---

## Container & Docker for LLM Inference

### Top Repositories

#### 1. Triton Inference Server
- **URL**: https://github.com/NVIDIA/triton-inference-server
- **Stars**: 10,517
- **Language**: Python (57.1%), Shell (21.5%), C++ (18.4%)
- **Description**: NVIDIA's optimized cloud and edge inference solution
- **Key Features**:
  - Multi-backend support (TensorRT, vLLM, TensorFlow, PyTorch)
  - Model ensemble support
  - Built-in metrics and monitoring
  - Request batching and scheduling

#### 2. vLLM
- **URL**: https://github.com/vllm-project/vllm
- **Stars**: 30,000+
- **Language**: Python (87%), CUDA (7%), C++ (4.3%)
- **Description**: High-throughput and memory-efficient LLM inference engine
- **Key Features**:
  - PagedAttention for efficient memory management
  - Tensor parallelism and pipeline parallelism
  - OpenAI-compatible API
  - Speculative decoding
  - Multi-model serving

#### 3. SGLang
- **URL**: https://github.com/sgl-project/sglang
- **Stars**: 25,431
- **Language**: Python (81.7%), Rust (8.2%), CUDA (4.7%)
- **Description**: High-performance serving framework for LLMs and multimodal models
- **Key Features**:
  - RadixAttention for efficient caching
  - Speculative decoding
  - Flexible batching strategies
  - Structured generation support

#### 4. SGLang Kubernetes Workload (RBG)
- **URL**: https://github.com/sgl-project/rbg
- **Stars**: 195
- **Language**: Go (93.8%), Python (3.2%)
- **Description**: Kubernetes workload for deploying SGLang inference services
- **Key Features**:
  - Production-ready Kubernetes manifests
  - Service discovery
  - Health checking

#### 5. Hugging Face Text Generation Inference (TGI)
- **URL**: https://huggingface.co/docs/text-generation-inference
- **Official Docker Images**: https://hub.docker.com/r/ghcr.io/huggingface/text-generation-inference
- **Language**: Rust
- **Description**: Production-grade inference server for text generation models
- **Key Features**:
  - Flash Attention for efficiency
  - Token streaming
  - Distributed inference
  - Integrated OpenAI API

#### 6. BentoML & OpenLLM
- **URL**: https://github.com/bentoml/OpenLLM
- **Stars**: 12,272
- **Language**: Python (95.9%)
- **Description**: Run any open-source LLMs with OpenAI-compatible API
- **Key Features**:
  - Model server abstraction
  - A/B testing and canary deployments
  - Built-in REST API
  - Model auto-scaling

#### 7. BentoVLLM
- **URL**: https://github.com/bentoml/BentoVLLM
- **Stars**: 169
- **Language**: Python
- **Description**: Self-host LLMs with vLLM and BentoML
- **Key Features**:
  - Production-ready packaging
  - Model management
  - API endpoint generation

#### 8. Ollama
- **URL**: https://github.com/ollama/ollama
- **Stars**: 90,000+
- **Language**: Go
- **Description**: Get up and running with large language models locally
- **Key Features**:
  - Simple CLI interface
  - Automatic GPU detection
  - Model library management
  - OpenAI API compatibility

#### 9. NVIDIA NIM Deploy
- **URL**: https://github.com/NVIDIA/nim-deploy
- **Stars**: 229
- **Language**: Jupyter Notebook
- **Description**: YAML, Helm charts, and guides for NVIDIA NIM deployment
- **Key Features**:
  - TensorRT-LLM integration
  - Production deployment examples
  - Multi-cloud support

#### 10. NVIDIA NeMo Export & Deploy
- **URL**: https://github.com/NVIDIA-NeMo/Export-Deploy
- **Stars**: 33
- **Language**: Python (97.2%)
- **Description**: Library for exporting and deploying NeMo and Hugging Face models
- **Key Features**:
  - Model optimization (quantization, distillation)
  - Backend-agnostic deployment
  - Performance tuning

### Docker Best Practices

#### Base Image Selection
- **NVIDIA CUDA Images**: `nvidia/cuda:12.1-devel-ubuntu22.04`
- **Python Official Images**: `python:3.11-slim`
- **Hugging Face Optimized**: `huggingface/transformers-pytorch-gpu`

#### Multi-Stage Docker Build Example

```dockerfile
# Stage 1: Builder
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3.11-venv \
    python3.11-dev

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

EXPOSE 8000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]
```

#### GPU-Enabled Container Run Command

```bash
docker run --gpus all \
  -p 8000:8000 \
  -e MODEL_NAME=meta-llama/Llama-2-7b-hf \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-image:latest
```

#### Docker Compose with GPU Support

```yaml
version: '3.8'

services:
  vllm:
    image: vllm/vllm:latest-gpu
    container_name: vllm-server
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=meta-llama/Llama-2-13b-hf
      - MAX_MODEL_LEN=4096
      - GPU_MEMORY_UTILIZATION=0.9
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  huggingface_cache:
    driver: local
```

---

## Cloud Deployment Platforms

### AWS SageMaker

#### Key Resources
- **Official Guide**: https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/
- **Best Practices**: https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html
- **LLM Optimization**: https://aws.amazon.com/blogs/machine-learning/optimizing-llm-inference-on-amazon-sagemaker-ai-with-bentomls-llm-optimizer/

#### Key Patterns
1. **Multi-Model Serving with vLLM**
   - Deploy multiple fine-tuned models on single SageMaker endpoint
   - Use disaggregated inference for prefill/decode
   - Configure GPU capacity with training plans

2. **Fine-tuning at Scale**
   - Hugging Face integration
   - Distributed training with SageMaker Training
   - Model registry and versioning

3. **Cost Optimization**
   - GPU instance selection (e.g., p4d, g4dn)
   - Spot instances for non-critical workloads
   - Endpoint auto-scaling based on inference load

#### Example: Deploy vLLM on SageMaker

```python
import sagemaker
from sagemaker.model import Model

role = sagemaker.get_execution_role()
model = Model(
    image_uri="vllm/vllm:latest-gpu",
    model_data="s3://bucket/model.tar.gz",
    role=role,
    env={
        "MODEL_NAME": "meta-llama/Llama-2-7b-hf",
        "GPU_MEMORY_UTILIZATION": "0.9",
        "TENSOR_PARALLEL_SIZE": "2"
    }
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.p4d.24xlarge",
    endpoint_name="llm-endpoint"
)
```

### Google Cloud Vertex AI

#### Key Resources
- **vLLM on Vertex AI**: https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm
- **Hugging Face Integration**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-hugging-face-models
- **Model Garden**: https://cloud.google.com/vertex-ai/docs/model-garden/overview

#### Key Features
1. **vLLM Serving on GKE**
   - Deploy with A3 Mega GPU clusters
   - Automatic scaling and load balancing
   - Integrated monitoring and logging

2. **Custom Container Deployment**
   - Artifact Registry integration
   - Custom training and serving scripts
   - A/B testing with Vertex AI

3. **SGLang Deployment on GKE**
   - Example repository: https://github.com/AI-Hypercomputer/gpu-recipes
   - Multi-GPU distributed inference
   - DeepSeek-R1-671B deployment patterns

#### Vertex AI Deployment Template

```python
from google.cloud import aiplatform

aiplatform.init(project="project-id", location="us-central1")

model = aiplatform.Model.upload(
    display_name="vllm-llama-2",
    artifact_uri="gs://bucket/vllm-model",
    serving_container_image_uri="us-docker.pkg.dev/.../vllm:latest",
    serving_container_environment_variables={
        "MODEL_NAME": "meta-llama/Llama-2-7b-hf",
        "GPU_MEMORY_UTILIZATION": "0.9"
    }
)

endpoint = model.deploy(
    machine_type="g2-standard-24",
    min_replica_count=1,
    max_replica_count=10
)
```

### Azure Machine Learning

#### Key Resources
- **Hugging Face Integration**: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-models-from-huggingface
- **Container Instances**: https://learn.microsoft.com/en-us/azure/container-instances/container-instances-best-practices-and-considerations
- **AKS Deployment**: https://huggingface.co/blog/vpkprasanna/deploying-language-models-on-azure

#### Key Patterns
1. **Managed Online Endpoints**
   - Deploy Hugging Face models directly
   - Automatic scaling and traffic splitting
   - Integration with Azure Key Vault for secrets

2. **AKS Deployment**
   - Use Azure Kubernetes Service for control
   - GPU node pools for inference
   - Persistent storage integration

3. **Container Instances**
   - Serverless container execution
   - Cost-effective for lower workloads
   - Easy scaling and management

#### Azure ML Deployment Example

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group_name="<resource-group>",
    workspace_name="<workspace>"
)

endpoint = ManagedOnlineEndpoint(
    name="llm-endpoint",
    auth_mode="aad_token"
)

deployment = ManagedOnlineDeployment(
    name="vllm-deployment",
    endpoint_name="llm-endpoint",
    model="azureml://registries/huggingface/models/Llama-2-7b-hf/versions/1",
    instance_type="Standard_NC24s_v3",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment)
```

---

## LLM Serving Infrastructure

### Top Serving Frameworks Comparison

| Framework | Primary Language | Key Feature | Use Case | GitHub Stars |
|-----------|-----------------|-------------|----------|--------------|
| **vLLM** | Python/CUDA | PagedAttention | General LLM serving | 30K+ |
| **SGLang** | Python/Rust | RadixAttention | Structured generation | 25K+ |
| **TGI** | Rust | Flash Attention | Hugging Face native | 10K+ |
| **Triton** | C++/Python | Multi-backend | Production inference | 10.5K |
| **Ollama** | Go | Simplicity | Local deployment | 90K+ |
| **TensorRT-LLM** | CUDA/C++ | GPU optimization | NVIDIA hardware | 8K+ |
| **Ray Serve** | Python | Distributed serving | Multi-model systems | 20K+ |

### Load Balancing Patterns

#### 1. NVIDIA NIM with Load Balancer

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  type: LoadBalancer
  selector:
    app: llm-inference
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

#### 2. Kubernetes Ingress with Rate Limiting

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-ingress
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "10"
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /v1
            pathType: Prefix
            backend:
              service:
                name: llm-service
                port:
                  number: 8000
```

### Auto-Scaling Solutions

#### 1. Kubernetes Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: vllm_request_queue_length
        target:
          type: AverageValue
          averageValue: "5"
```

#### 2. KEDA (Kubernetes Event-Driven Autoscaling)

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-scaled-object
spec:
  scaleTargetRef:
    name: llm-deployment
  minReplicaCount: 1
  maxReplicaCount: 20
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: vllm_request_duration_seconds
        query: |
          rate(vllm_request_duration_seconds_sum[1m]) / 
          rate(vllm_request_duration_seconds_count[1m])
        threshold: "5000"
```

### High-Availability Patterns

#### 1. Multi-Node LLM Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-cluster
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: llm
  template:
    metadata:
      labels:
        app: llm
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - llm
                topologyKey: kubernetes.io/hostname
      containers:
        - name: vllm
          image: vllm/vllm:latest-gpu
          env:
            - name: TENSOR_PARALLEL_SIZE
              value: "2"
            - name: PIPELINE_PARALLEL_SIZE
              value: "1"
          resources:
            requests:
              nvidia.com/gpu: "2"
              memory: "32Gi"
              cpu: "16"
            limits:
              nvidia.com/gpu: "2"
              memory: "40Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 5
```

#### 2. Disaggregated Inference Architecture

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-prefill-stage
spec:
  replicas: 3
  selector:
    matchLabels:
      stage: prefill
  template:
    metadata:
      labels:
        stage: prefill
    spec:
      containers:
        - name: vllm-prefill
          image: vllm/vllm:latest-gpu
          env:
            - name: VLLM_SCHEDULER
              value: "lpm"  # Least pending request first
            - name: PREFILL_MODE
              value: "true"
          resources:
            requests:
              nvidia.com/gpu: "1"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-decode-stage
spec:
  replicas: 6
  selector:
    matchLabels:
      stage: decode
  template:
    metadata:
      labels:
        stage: decode
    spec:
      containers:
        - name: vllm-decode
          image: vllm/vllm:latest-gpu
          env:
            - name: VLLM_SCHEDULER
              value: "lpm"
            - name: DECODE_MODE
              value: "true"
```

---

## Infrastructure Monitoring & Cost Optimization

### Monitoring Stack

#### 1. Prometheus + Grafana Setup

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'vllm'
        static_configs:
          - targets: ['localhost:8000']
      - job_name: 'gpu-metrics'
        static_configs:
          - targets: ['localhost:9400']
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
```

#### 2. NVIDIA DCGM Exporter for GPU Metrics

```bash
docker run -d \
  --gpus all \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
```

#### 3. Key Metrics to Monitor

```yaml
vllm_request_latency_seconds # Request latency
vllm_request_queue_length # Queue depth
vllm_gpu_memory_usage # GPU memory utilization
vllm_cache_utilization # KV cache usage
vllm_batch_size # Current batch size
vllm_number_of_requests # Active requests
nvidia_smi_memory_used # NVIDIA GPU memory
nvidia_smi_power_draw # GPU power consumption
container_memory_usage_bytes # Container memory
container_cpu_usage_seconds_total # Container CPU usage
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "LLM Inference Monitoring",
    "panels": [
      {
        "title": "Request Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, vllm_request_latency_seconds)"
          },
          {
            "expr": "histogram_quantile(0.95, vllm_request_latency_seconds)"
          },
          {
            "expr": "histogram_quantile(0.99, vllm_request_latency_seconds)"
          }
        ]
      },
      {
        "title": "GPU Memory Utilization",
        "targets": [
          {
            "expr": "vllm_gpu_memory_usage / 80000 * 100"
          }
        ]
      },
      {
        "title": "Request Queue Length",
        "targets": [
          {
            "expr": "vllm_request_queue_length"
          }
        ]
      },
      {
        "title": "Active Requests",
        "targets": [
          {
            "expr": "vllm_number_of_requests"
          }
        ]
      },
      {
        "title": "Throughput (Tokens/sec)",
        "targets": [
          {
            "expr": "rate(vllm_tokens_generated_total[1m])"
          }
        ]
      }
    ]
  }
}
```

### Cost Optimization Strategies

#### 1. GPU Instance Selection Strategy

```python
# Cost per token for different GPU instances
gpu_costs = {
    "A100-80GB": {"cost_per_hour": 3.06, "batch_size": 64, "tokens_per_sec": 2000},
    "H100-80GB": {"cost_per_hour": 4.58, "batch_size": 128, "tokens_per_sec": 3500},
    "L4": {"cost_per_hour": 0.60, "batch_size": 16, "tokens_per_sec": 800},
    "V100": {"cost_per_hour": 3.06, "batch_size": 32, "tokens_per_sec": 1200},
}

def calculate_cost_per_token(gpu_instance, utilization=0.7):
    gpu = gpu_costs[gpu_instance]
    # Cost per token = (hourly cost / 3600) / (tokens per sec * utilization)
    return (gpu["cost_per_hour"] / 3600) / (gpu["tokens_per_sec"] * utilization)

# Find most cost-effective instance
for gpu, cost in sorted(
    {k: calculate_cost_per_token(k) for k in gpu_costs}.items(),
    key=lambda x: x[1]
):
    print(f"{gpu}: ${cost:.6f} per token")
```

#### 2. Quantization Implementation (vLLM + AWQ)

```bash
# Deploy quantized model
docker run --gpus all -p 8000:8000 \
  vllm/vllm:latest-gpu \
  --model TheBloke/Mistral-7B-Instruct-v0.1-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.95
```

#### 3. Batch Processing for Cost Reduction

```python
from vllm import LLM, SamplingParams
import time

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.95,
    max_model_len=4096
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

# Batch inference reduces cost per token
requests = [
    "What is machine learning?",
    "Explain transformers",
    "What is attention mechanism?"
] * 100  # 300 requests batched

start_time = time.time()
outputs = llm.generate(requests, sampling_params)
end_time = time.time()

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / (end_time - start_time)
print(f"Throughput: {throughput:.2f} tokens/sec")
```

#### 4. Spot Instance Strategy (AWS)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-spot-instance
spec:
  template:
    spec:
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: karpenter.sh/capacity-type
                    operator: In
                    values: ["spot"]
      tolerations:
        - key: karpenter.sh/do-not-evict
          operator: Equal
          value: "false"
      containers:
        - name: vllm
          image: vllm/vllm:latest-gpu
          # Use checkpoint-restore for spot instance interruption handling
          lifecycle:
            preStop:
              exec:
                command:
                  - /bin/sh
                  - -c
                  - sleep 30 && kill -TERM 1
```

---

## Common Patterns & Best Practices

### 1. Tensor Parallelism Configuration

```bash
# For single-node multi-GPU setup
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1

# For multi-node tensor parallelism
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 8 \
  --distributed-executor-backend nccl
```

### 2. Pipeline Parallelism for Large Models

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    distributed_executor_backend="nccl"
)
```

### 3. Memory Optimization Techniques

```bash
# Enable PagedAttention (default in vLLM)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching

# Quantization for smaller memory footprint
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --quantization awq \
  --gpu-memory-utilization 0.95
```

### 4. Speculative Decoding (vLLM 0.8+)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --speculative-model meta-llama/Llama-2-7b-hf \
  --num-speculative-tokens 5
```

### 5. Model Caching Strategy

```python
# Prefix caching for repeated prompts
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,  # Reuse KV cache for common prefixes
)

# Useful for multi-turn conversations or similar queries
prompts = [
    "System prompt is...\nUser: Question 1",
    "System prompt is...\nUser: Question 2",
    "System prompt is...\nUser: Question 3",
]
```

### 6. LoRA Model Serving (vLLM 0.7+)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --enable-lora \
  --max-lora-rank 64 \
  --max-num-seqs-to-allocate-lora 4
```

### 7. Hot Model Swapping (LLM-D)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-pool
data:
  models.json: |
    {
      "models": [
        {"name": "llama-7b", "weight": 0.4},
        {"name": "mistral-7b", "weight": 0.3},
        {"name": "qwen-7b", "weight": 0.3}
      ],
      "cache_size": "20GB"
    }
```

---

## Code Examples & Configuration

### Complete vLLM + Kubernetes Deployment

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-inference

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
  namespace: llm-inference
data:
  model: "meta-llama/Llama-2-7b-hf"
  gpu_memory_utilization: "0.9"
  max_model_len: "4096"
  tensor_parallel_size: "1"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: llm-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
        - name: vllm
          image: vllm/vllm:latest-gpu
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          env:
            - name: MODEL_NAME
              valueFrom:
                configMapKeyRef:
                  name: vllm-config
                  key: model
            - name: GPU_MEMORY_UTILIZATION
              valueFrom:
                configMapKeyRef:
                  name: vllm-config
                  key: gpu_memory_utilization
            - name: MAX_MODEL_LEN
              valueFrom:
                configMapKeyRef:
                  name: vllm-config
                  key: max_model_len
            - name: TENSOR_PARALLEL_SIZE
              valueFrom:
                configMapKeyRef:
                  name: vllm-config
                  key: tensor_parallel_size
          resources:
            requests:
              memory: "32Gi"
              cpu: "16"
              nvidia.com/gpu: "1"
            limits:
              memory: "40Gi"
              cpu: "32"
              nvidia.com/gpu: "1"
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 30
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 2
          volumeMounts:
            - name: huggingface-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: huggingface-cache
          persistentVolumeClaim:
            claimName: huggingface-pvc
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - vllm
                topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: gpu-type
                    operator: In
                    values:
                      - nvidia-a100
                      - nvidia-h100

---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: llm-inference
spec:
  type: LoadBalancer
  selector:
    app: vllm
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  sessionAffinity: ClientIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: huggingface-pvc
  namespace: llm-inference
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: llm-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 75
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Helm Chart for LLM Deployment

```yaml
# values.yaml
replicaCount: 2

image:
  repository: vllm/vllm
  tag: latest-gpu
  pullPolicy: IfNotPresent

model:
  name: meta-llama/Llama-2-7b-hf
  huggingfaceToken: ""

resources:
  requests:
    memory: "32Gi"
    cpu: "16"
    nvidia.com/gpu: "1"
  limits:
    memory: "40Gi"
    cpu: "32"
    nvidia.com/gpu: "1"

service:
  type: LoadBalancer
  port: 8000
  targetPort: 8000

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 75
  targetMemoryUtilizationPercentage: 80

persistence:
  enabled: true
  size: 100Gi
  storageClassName: fast-ssd

nodeSelector:
  gpu-type: nvidia-a100
```

### FastAPI Client for vLLM

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
from typing import List

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

class CompletionResponse(BaseModel):
    prompt: str
    generated_text: str
    finish_reason: str

# Load balance across multiple vLLM instances
VLLM_SERVERS = [
    "http://vllm-server-1:8000",
    "http://vllm-server-2:8000",
    "http://vllm-server-3:8000"
]

current_server = 0

async def get_vllm_server():
    global current_server
    server = VLLM_SERVERS[current_server % len(VLLM_SERVERS)]
    current_server += 1
    return server

@app.post("/completions", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    server = await get_vllm_server()
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{server}/v1/completions",
                json={
                    "model": "default",
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return CompletionResponse(
                prompt=request.prompt,
                generated_text=data["choices"][0]["text"],
                finish_reason=data["choices"][0].get("finish_reason", "length")
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=500,
                detail=f"vLLM server error: {str(e)}"
            )

@app.get("/health")
async def health_check():
    tasks = [
        httpx.get(f"{server}/health", timeout=5.0)
        for server in VLLM_SERVERS
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    healthy_servers = sum(
        1 for r in results
        if not isinstance(r, Exception) and r.status_code == 200
    )
    
    return {
        "status": "healthy" if healthy_servers > 0 else "degraded",
        "healthy_servers": healthy_servers,
        "total_servers": len(VLLM_SERVERS)
    }
```

---

## Additional Resources

### Official Documentation
- **vLLM**: https://docs.vllm.ai
- **Ray Serve**: https://docs.ray.io/en/latest/serve/
- **SGLang**: https://sgl-project.github.io
- **Triton**: https://docs.nvidia.com/deeplearning/triton-inference-server/
- **KubeRay**: https://ray-project.github.io/kuberay/
- **LLM-D**: https://github.com/llm-d/llm-d

### Monitoring & Observability
- **Prometheus vLLM Metrics**: https://docs.vllm.ai/en/latest/serving/metrics.html
- **OpenTelemetry for LLMs**: https://opentelemetry.io
- **Grafana Templates**: https://grafana.com/grafana/dashboards/

### Community Resources
- **Awesome LLMOps**: https://github.com/InftyAI/Awesome-LLMOps
- **Kubernetes Recipes AI**: https://kubernetes.recipes/recipes/ai/
- **GKE AI Labs**: https://gke-ai-labs.dev/
- **Red Hat OpenShift AI**: https://www.redhat.com/en/technologies/cloud-computing/openshift/ai

### Emerging Technologies
- **Disaggregated Inference**: https://github.com/llm-d/llm-d
- **Speculative Decoding**: vLLM 0.8+
- **Prefix Caching**: vLLM 0.7+
- **LoRA Hot-Swapping**: vLLM 0.7+

---

## Summary

This comprehensive guide covers:
- **25+ top repositories** for LLM infrastructure and deployment
- **Key insights** from production deployments in 2026
- **Common patterns** for high-availability and cost-effective deployments
- **Code examples** ready for production use
- **Monitoring strategies** for reliability and cost optimization
- **Cloud-native approaches** on AWS, GCP, and Azure

The landscape continues to evolve rapidly with improvements in:
- Inference engine efficiency (quantization, speculative decoding)
- Distributed serving (disaggregated inference, tensor parallelism)
- Kubernetes integration (operators, schedulers, workload management)
- Cost optimization (spot instances, batching, model compression)

For production deployments in 2026, the recommended stack includes:
1. **Serving Layer**: vLLM, SGLang, or TGI for inference
2. **Orchestration**: Kubernetes with KAI Scheduler for GPU allocation
3. **Cloud Platform**: Multi-cloud support (AWS, GCP, Azure)
4. **Monitoring**: Prometheus + Grafana with custom GPU metrics
5. **Cost Optimization**: Quantization, batching, and intelligent instance selection

This setup provides reliability, scalability, and cost-effectiveness for production LLM workloads.
