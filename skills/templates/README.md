# Infrastructure Templates

Production-ready configuration templates for deploying LLM systems.

## Overview

This directory contains ready-to-use templates for:
- **Docker** - Containerized LLM inference servers
- **Kubernetes** - Orchestrated deployment with high availability
- **Helm** - Package management for Kubernetes deployments
- **Cloud** - AWS, GCP, Azure specific configurations

## Files Included

```
templates/
├── docker/
│   ├── Dockerfile.llm              # Multi-stage Docker build
│   ├── docker-compose.yml          # Single-node stack
│   └── .dockerignore               # Exclude large files
├── kubernetes/
│   ├── llm-inference-deployment.yaml   # Deployment manifest
│   ├── service.yaml                    # Kubernetes Service
│   ├── hpa.yaml                        # Horizontal Pod Autoscaling
│   ├── configmap.yaml                  # Configuration
│   └── pvc.yaml                        # Persistent Volume
├── helm/
│   ├── values.yaml                 # Helm chart values
│   ├── Chart.yaml                  # Chart metadata
│   └── templates/                  # Chart templates
└── README.md                       # This file
```

## Docker Deployment

### Quick Start

```bash
# Build image
docker build -f docker/Dockerfile.llm -t llm-server:latest .

# Run container
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME="meta-llama/Llama-2-7b-hf" \
  llm-server:latest

# Test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

### Docker Compose (Full Stack)

```bash
# Start full stack (server + prometheus + grafana)
docker-compose -f docker/docker-compose.yml up -d

# Services:
# - LLM Server: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### Dockerfile Breakdown

```dockerfile
# Stage 1: Build
FROM nvidia/cuda:12.0-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip

WORKDIR /build
COPY . .
RUN pip install -r requirements.txt

# Stage 2: Runtime (smaller image)
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/
COPY --from=builder /build /app

WORKDIR /app
EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.api_server", ...]
```

**Image Optimization**:
- Multi-stage build (reduces final size)
- Only runtime dependencies in final image
- CUDA slim base (2.5GB vs 5GB full)
- No model weights in image (downloaded at runtime)

## Kubernetes Deployment

### Quick Start

```bash
# Apply manifests
kubectl apply -f kubernetes/

# Check status
kubectl get deployments
kubectl get pods
kubectl get services

# Port forward for testing
kubectl port-forward service/llm-server 8000:8000

# Test
curl http://localhost:8000/v1/completions
```

### Manifest Files

#### Deployment (llm-inference-deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-server
spec:
  replicas: 3  # HA setup
  selector:
    matchLabels:
      app: llm-server
  template:
    spec:
      containers:
      - name: vllm
        image: llm-server:latest
        resources:
          requests:
            nvidia.com/gpu: 1  # Request 1 GPU
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-hf"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

**Key Features**:
- 3 replicas for high availability
- Health checks (liveness + readiness probes)
- Resource requests/limits (prevent node overload)
- Environment configuration via ConfigMap

#### Service (service.yaml)

Exposes deployment internally and externally:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-server
spec:
  type: LoadBalancer  # Expose externally
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: llm-server
```

**Service Types**:
- **ClusterIP** - Internal only (default)
- **NodePort** - External via node port
- **LoadBalancer** - Cloud provider LB (recommended)
- **ExternalName** - Route to external service

#### HPA (hpa.yaml) - Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Auto-Scaling Logic**:
- Minimum 3 replicas (always available)
- Maximum 10 replicas (cost limit)
- Scale up if CPU >70% or Memory >80%
- Scale down if <30% utilization

## Helm Deployment

### Quick Start

```bash
# Install
helm install llm-server ./helm \
  --set model.name="meta-llama/Llama-2-7b-hf" \
  --set replicaCount=3

# Upgrade
helm upgrade llm-server ./helm --set model.name="new-model"

# Uninstall
helm uninstall llm-server
```

### Chart Structure

```yaml
Chart.yaml:
  name: llm-inference
  version: 1.0.0
  description: LLM inference server

values.yaml:
  replicaCount: 3
  image:
    repository: llm-server
    tag: latest
  model:
    name: "meta-llama/Llama-2-7b-hf"
    cache: /models
  resources:
    gpu: 1
    memory: 16Gi
```

**Helm Benefits**:
- ✅ Template reuse across environments
- ✅ Version control for configs
- ✅ Easy rollbacks
- ✅ Package management

### Custom Values per Environment

```bash
# Development
helm install llm-dev ./helm -f values-dev.yaml

# Staging
helm install llm-staging ./helm -f values-staging.yaml

# Production
helm install llm-prod ./helm -f values-prod.yaml
```

## Configuration Patterns

### Development (Single GPU)

```yaml
replicas: 1
resources:
  gpu: 1
  memory: 16Gi
autoscaling:
  enabled: false
monitoring:
  enabled: false
```

### Staging (3 GPUs, Monitoring)

```yaml
replicas: 2
resources:
  gpu: 1
  memory: 24Gi
autoscaling:
  minReplicas: 2
  maxReplicas: 5
monitoring:
  enabled: true
  prometheus: true
```

### Production (Full HA)

```yaml
replicas: 4
resources:
  gpu: 2  # Multiple GPUs per pod
  memory: 40Gi
autoscaling:
  minReplicas: 4
  maxReplicas: 20
monitoring:
  enabled: true
  prometheus: true
  alerting: true
affinity:
  podAntiAffinity: required  # Spread across nodes
```

## Networking & Load Balancing

### Service Mesh (Istio)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-server
spec:
  hosts:
  - llm-server
  http:
  - match:
    - uri:
        prefix: /fast
    route:
    - destination:
        host: llm-server-fast
      weight: 100
    timeout: 5s
  - route:
    - destination:
        host: llm-server-accurate
      weight: 100
    timeout: 30s
```

### Ingress (API Gateway)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-ingress
spec:
  ingressClassName: nginx
  rules:
  - host: llm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-server
            port:
              number: 8000
```

## Storage & Models

### Persistent Volume for Models

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadOnlyMany  # Multiple pods can read
  resources:
    requests:
      storage: 100Gi  # 7B model + overhead
  storageClassName: fast-ssd
```

### Download Models at Startup

```bash
# In Dockerfile or startup script
RUN python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/models')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/models')
"
```

## Security Best Practices

### Security Context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL
  allowPrivilegeEscalation: false
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-llm
spec:
  podSelector:
    matchLabels:
      app: llm-server
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: production
```

### Image Security

```yaml
image:
  repository: llm-server
  tag: latest
  pullPolicy: Always  # Always verify latest
imagePullSecrets:
  - name: docker-registry  # Private registry credentials
```

## Monitoring & Observability

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llm-server
spec:
  selector:
    matchLabels:
      app: llm-server
  endpoints:
  - port: metrics
    interval: 30s
```

### Logging with ELK

```yaml
env:
- name: LOG_LEVEL
  value: "INFO"
- name: ELASTICSEARCH_HOST
  value: "elasticsearch:9200"
volumeMounts:
- name: logs
  mountPath: /var/log/llm
```

## Cost Optimization

### Resource Limits

```yaml
# Request: minimum required
# Limit: maximum allowed
resources:
  requests:
    gpu: 1
    cpu: 4
    memory: 16Gi
  limits:
    gpu: 1
    cpu: 8
    memory: 32Gi
```

### Spot Instances (for non-critical)

```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: cloud.google.com/gke-nodepool
          operator: In
          values:
          - spot-pool
```

## Troubleshooting

**Q: Pods not getting GPU?**
- Check GPU availability: `kubectl describe node`
- Verify nvidia-device-plugin running
- Check resource requests/limits

**Q: Service not accessible?**
- Check service type (LoadBalancer vs ClusterIP)
- Verify port forwarding
- Check firewall rules

**Q: Model downloading too slow?**
- Use persistent volume cache
- Pre-warm model before deploying
- Check network bandwidth

**Q: High latency in production?**
- Check pod affinity (spread across nodes)
- Verify GPU utilization
- Monitor batch queue depth

## References

- **Kubernetes**: [Official Docs](https://kubernetes.io/docs/)
- **Helm**: [Getting Started](https://helm.sh/docs/)
- **Docker**: [Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- **vLLM**: [Deployment Guide](https://docs.vllm.ai/en/latest/)

## Integration with Other Skills

- **Infrastructure** - Deploy servers (main integration)
- **Fast Inference** - Optimize with vLLM
- **Monitoring** - Add Prometheus + Grafana
- **Fine-Tuning** - Serve LoRA adapters
- **Quantization** - Deploy quantized models

