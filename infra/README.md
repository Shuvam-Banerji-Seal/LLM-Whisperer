# Infrastructure

Containerization, orchestration, and infrastructure-as-code for deploying LLM systems.

## Overview

This module provides production-ready infrastructure templates:
- Docker containerization
- Kubernetes deployments
- Infrastructure-as-Code (Terraform)
- Monitoring and observability
- Deployment automation

## Structure

```
infra/
├── README.md (this file)
├── docker/           # Docker images and compose files
├── kubernetes/       # K8s manifests and Helm charts
├── terraform/        # IaC modules for cloud providers
└── monitoring/       # Observability dashboards and alerts
```

## Directory Purposes

### `docker/` - Containerization

Docker images for all major components:

```
docker/
├── base/             # Base Python images
├── inference/        # Inference server images
├── training/         # Training job images
├── rag/              # RAG pipeline images
├── monitoring/       # Monitoring stack images
└── docker-compose.yml
```

**Key Images**:
- `llm-inference:latest` - vLLM/Triton inference server
- `llm-training:latest` - PyTorch training environment
- `rag-pipeline:latest` - RAG ingestion and retrieval
- `monitoring-stack:latest` - Prometheus + Grafana

### `kubernetes/` - Orchestration

Kubernetes manifests for production deployment:

```
kubernetes/
├── base/             # Base configurations
├── overlays/         # Environment-specific overlays (prod, dev)
├── helm/             # Helm charts
├── ingress/          # Ingress configuration
├── storage/          # PersistentVolume configurations
└── monitoring/       # Prometheus/Grafana manifests
```

**Key Patterns**:
- Kustomize for deployment management
- Helm for templating
- StatefulSets for model serving
- HorizontalPodAutoscaler for scaling

### `terraform/` - Infrastructure as Code

Terraform modules for cloud infrastructure:

```
terraform/
├── aws/              # AWS modules
│   ├── eks/          # Elastic Kubernetes Service
│   ├── rds/          # Managed databases
│   ├── s3/           # Object storage
│   └── networking/   # VPC, subnets
├── gcp/              # GCP modules
│   ├── gke/          # Google Kubernetes Engine
│   ├── cloudsql/     # Managed databases
│   └── networking/   # VPC, firewall
└── modules/          # Shared modules
```

### `monitoring/` - Observability

Monitoring and alerting setup:

```
monitoring/
├── prometheus/       # Prometheus config
├── grafana/          # Dashboard definitions
├── alertmanager/     # Alert rules
└── loki/             # Log aggregation
```

## Quick Start

### 1. Local Development with Docker Compose

```bash
cd infra/docker
docker-compose up -d
# Services:
# - http://localhost:8000 (API)
# - http://localhost:3000 (Grafana)
```

### 2. Deploy to Kubernetes

```bash
cd infra/kubernetes
kubectl apply -k overlays/dev
# Check deployment
kubectl get pods -l app=llm-server
```

### 3. Deploy to AWS

```bash
cd infra/terraform/aws
terraform init
terraform plan
terraform apply
```

## Configuration Files

### Docker Compose

```yaml
# infra/docker/docker-compose.yml
version: '3.8'

services:
  # Inference server
  inference:
    image: llm-inference:latest
    ports:
      - "8000:8000"
    environment:
      MODEL_NAME: mistralai/Mistral-7B-v0.1
      TENSOR_PARALLEL_SIZE: 1
    volumes:
      - model_cache:/root/.cache
    gpu: true

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  model_cache:
```

### Kubernetes Deployment

```yaml
# infra/kubernetes/base/inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: inference
        image: llm-inference:latest
        resources:
          requests:
            memory: "16Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: "24Gi"
            nvidia.com/gpu: "1"
        ports:
        - containerPort: 8000
```

### Terraform Configuration

```hcl
# infra/terraform/aws/main.tf
module "eks" {
  source = "./modules/eks"
  
  cluster_name = "llm-serving"
  region = "us-east-1"
  
  node_groups = {
    gpu_nodes = {
      instance_types = ["g4dn.2xlarge"]
      min_size = 2
      max_size = 10
      disk_size = 100
    }
  }
}
```

## Production Checklist

### Security
- [ ] Network policies (ingress/egress)
- [ ] RBAC for Kubernetes
- [ ] Secrets management (Vault, AWS Secrets Manager)
- [ ] TLS certificates
- [ ] Container image scanning

### Reliability
- [ ] Health checks (liveness, readiness probes)
- [ ] Pod disruption budgets
- [ ] Backup and disaster recovery
- [ ] Load balancing
- [ ] Auto-scaling policies

### Monitoring
- [ ] Prometheus metrics scraping
- [ ] Grafana dashboards
- [ ] Alert rules
- [ ] Log aggregation (Loki)
- [ ] Distributed tracing

### Performance
- [ ] Resource requests/limits
- [ ] GPU sharing/isolation
- [ ] Network policies
- [ ] Storage optimization
- [ ] Cache strategies

## Deployment Patterns

### Pattern 1: Single-Node Serving
```bash
# For development/testing
docker run -d \
  -p 8000:8000 \
  --gpus all \
  llm-inference:latest
```

### Pattern 2: Multi-GPU Single Machine
```bash
# Using tensor parallelism
docker run -d \
  -p 8000:8000 \
  --gpus all \
  -e TENSOR_PARALLEL_SIZE=4 \
  llm-inference:latest
```

### Pattern 3: Kubernetes Cluster
```bash
kubectl apply -k infra/kubernetes/overlays/prod
```

### Pattern 4: Serverless (AWS Lambda)
```bash
# Using container images on Lambda
aws lambda create-function \
  --function-name llm-inference \
  --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --code ImageUri=ACCOUNT.dkr.ecr.REGION.amazonaws.com/llm-inference:latest
```

## Scaling Guide

### Vertical Scaling
- Increase GPU memory per pod
- Use larger GPU models (A100 vs V100)
- Enable tensor parallelism

### Horizontal Scaling
- Replicate pods across nodes
- Load balance traffic
- Use auto-scaling policies

### Optimization
- Model quantization (4-bit)
- Prefix caching
- Batching
- Speculative decoding

## Cost Optimization

### Compute
- Use spot instances for non-critical workloads
- Right-size GPU allocation
- Enable auto-scaling down
- Consider cheaper alternatives (TPU, CPU-only)

### Storage
- Use S3/GCS for model caching
- Compress model checkpoints
- Clean up old artifacts

### Networking
- Minimize data transfer
- Use region-local resources
- CDN for static assets

## Troubleshooting

### GPU Memory Issues
```bash
# Check GPU usage
nvidia-smi

# Reduce batch size
export BATCH_SIZE=1
```

### Network Issues
```bash
# Check connectivity
kubectl logs -f deployment/llm-inference

# Check network policies
kubectl describe networkpolicy
```

### Storage Issues
```bash
# Check PV status
kubectl get pv
kubectl describe pv <pv-name>
```

## References

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [vLLM Serving Guide](https://docs.vllm.ai/en/latest/serving/serving_guide.html)

## Contributing

When adding infrastructure:
1. Follow existing naming conventions
2. Document all variables and outputs
3. Include example values
4. Test in dev environment first
5. Document security implications
6. Add monitoring/alerting

## License

See LICENSE file in repository root.
