# Agent Deployment Guide: Production-Ready AI Agents in 2026

**Author:** Shuvam Banerji Seal  
**Last Updated:** April 2026

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [Pre-Deployment Preparation](#pre-deployment-preparation)
3. [Containerization Strategies](#containerization-strategies)
4. [API Deployment Options](#api-deployment-options)
5. [Scaling Considerations](#scaling-considerations)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Error Handling and Recovery](#error-handling-and-recovery)
9. [Version Management](#version-management)
10. [Cost Optimization](#cost-optimization)
11. [Security Considerations](#security-considerations)
12. [Infrastructure Recommendations](#infrastructure-recommendations)

---

## Executive Overview

Deploying AI agents to production represents one of the most complex system integration challenges in modern software engineering (2026). Unlike traditional software, agent deployment involves:

- **Multi-layered systems**: Models, prompts, data pipelines, guardrails
- **Non-deterministic behavior**: Same input can produce different outputs
- **State persistence**: Long-running workflows that must survive failures
- **Resource constraints**: Significant computational and memory costs
- **Safety requirements**: Preventing hallucinations, prompt injections, tool misuse
- **Compliance obligations**: Audit trails, data privacy, explainability

This guide covers the complete production deployment lifecycle for AI agents.

---

## Pre-Deployment Preparation

### 1. Production Readiness Checklist

Before deploying any agent to production:

```yaml
Code Quality:
  ✓ Code review completed
  ✓ Type hints present (mypy/pyright passing)
  ✓ No hardcoded secrets or API keys
  ✓ Error handling for all external calls
  ✓ Logging at debug/info levels

Testing:
  ✓ Unit tests (>80% coverage)
  ✓ Integration tests for tools
  ✓ Semantic evaluation for LLM outputs
  ✓ Regression tests for prompt changes
  ✓ Load tests for expected throughput

Documentation:
  ✓ Architecture diagram
  ✓ Tool/capability documentation
  ✓ Known limitations documented
  ✓ Runbook for common issues
  ✓ SLA/performance targets defined

Dependencies:
  ✓ Version pins (no floating versions)
  ✓ Dependency security scan (no CVEs)
  ✓ License compliance check
  ✓ Memory/compute requirements documented
  ✓ External service health checks

Monitoring:
  ✓ Key metrics identified
  ✓ Alert thresholds defined
  ✓ Logging infrastructure ready
  ✓ Tracing setup complete
  ✓ Fallback strategies documented
```

### 2. Agent Architecture Validation

```python
from typing import List
from dataclasses import dataclass

@dataclass
class DeploymentChecklist:
    """Validate agent deployment readiness."""
    
    model_specified: bool  # e.g., "gpt-4o"
    tools_validated: bool  # All tools tested
    guardrails_enabled: bool  # Input/output validation
    memory_configured: bool  # Persistence mechanism
    error_handlers: bool  # Retry logic, fallbacks
    monitoring_setup: bool  # Logs, metrics, traces
    secrets_managed: bool  # No hardcoded values
    performance_baseline: bool  # Load test results
    
    def is_production_ready(self) -> bool:
        """Check if all requirements are met."""
        return all([
            self.model_specified,
            self.tools_validated,
            self.guardrails_enabled,
            self.memory_configured,
            self.error_handlers,
            self.monitoring_setup,
            self.secrets_managed,
            self.performance_baseline,
        ])

# Usage
checklist = DeploymentChecklist(
    model_specified=True,
    tools_validated=True,
    guardrails_enabled=True,
    memory_configured=True,
    error_handlers=True,
    monitoring_setup=True,
    secrets_managed=True,
    performance_baseline=True,
)

assert checklist.is_production_ready(), "Not ready for production!"
```

---

## Containerization Strategies

### 1. Docker Setup for AGNO Agents

```dockerfile
# Dockerfile for AGNO agent
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY src/ ./src/
COPY config/ ./config/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run agent API
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose for Multi-Service Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent:
    build: .
    container_name: ai-agent
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    networks:
      - agent-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: agent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - agent-network
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: agent-postgres
    environment:
      POSTGRES_DB: agent_db
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - agent-network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    container_name: agent-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - agent-network
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
  prometheus-data:

networks:
  agent-network:
    driver: bridge
```

### 3. Requirements.txt for Production

```
# Core dependencies
agno==2.2.6
langchain==1.2.7
langgraph==1.0.7

# API framework
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.8.0
pydantic-settings==2.4.0

# Database
sqlalchemy==2.0.24
psycopg2-binary==2.9.9
alembic==1.13.1

# Caching
redis==5.1.0

# Monitoring
prometheus-client==0.20.0
opentelemetry-api==1.24.0
opentelemetry-sdk==1.24.0
opentelemetry-exporter-prometheus==0.45b0

# Logging
python-json-logger==2.0.7

# Async support
aiohttp==3.9.3
httpx==0.26.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Testing
pytest==7.4.3
pytest-asyncio==0.23.2
pytest-cov==4.1.0

# Code quality
black==24.1.0
mypy==1.8.0
ruff==0.2.2
```

---

## API Deployment Options

### 1. FastAPI Agent Service

```python
# src/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent API", version="1.0.0")

# Initialize agent
agent = Agent(
    name="Production Agent",
    model=OpenAIChat(id="gpt-4"),
)

# Request/Response models
class AgentRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    metadata: Optional[dict] = None

class AgentResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: int
    processing_time_ms: float

# Health check endpoint
@app.get("/health")
async def health_check():
    """Verify service is running."""
    return {
        "status": "healthy",
        "version": "1.0.0",
    }

# Ready check endpoint
@app.get("/ready")
async def ready_check():
    """Check if agent is ready to process requests."""
    try:
        # Test LLM connectivity
        response = agent.run("Test")
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Service not ready: {e}")
        return {"status": "not_ready", "error": str(e)}, 503

# Main agent endpoint
@app.post("/chat", response_model=AgentResponse)
async def chat(request: AgentRequest):
    """Process user message and return agent response."""
    try:
        import time
        start_time = time.time()
        
        # Add user context
        context = f"User: {request.user_id}, Session: {request.session_id}"
        full_message = f"{context}\n{request.message}"
        
        # Run agent
        response = agent.run(full_message)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AgentResponse(
            response=response,
            session_id=request.session_id or "new",
            tokens_used=0,  # Track from actual LLM call
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming endpoint
@app.post("/chat-stream")
async def chat_stream(request: AgentRequest):
    """Stream agent response token by token."""
    async def generate():
        try:
            # For streaming support
            response = agent.run(request.message)
            for chunk in response.split(" "):
                yield chunk + " "
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    from prometheus_client import generate_latest
    return generate_latest()

# Structured output endpoint
@app.post("/analyze")
async def analyze(request: AgentRequest):
    """Get structured output from agent."""
    try:
        # Use agent with structured output
        result = agent.run(request.message)
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-deployment
  namespace: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
        version: v1
    spec:
      serviceAccountName: agent-sa
      
      # Init container for migrations
      initContainers:
      - name: migrate
        image: agent:1.0.0
        command: ["python", "-m", "alembic", "upgrade", "head"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
      
      containers:
      - name: agent
        image: agent:1.0.0
        imagePullPolicy: IfNotPresent
        
        ports:
        - containerPort: 8000
          name: http
        
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        
        # Resource requests and limits
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
        
        # Volume mounts
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /app/logs
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}
      
      # Affinity rules for high availability
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
                  - agent
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: agent
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http

---
apiVersion: autoscaling.k8s.io/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-deployment
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

### 3. AWS Deployment (ECS + Lambda)

```python
# AWS ECS Task Definition (Fargate)
import json

task_definition = {
    "family": "agent-task",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "containerDefinitions": [
        {
            "name": "agent",
            "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/agent:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "hostPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "essential": True,
            "environment": [
                {"name": "ENVIRONMENT", "value": "production"},
                {"name": "LOG_LEVEL", "value": "INFO"},
            ],
            "secrets": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:openai-key"
                },
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/agent",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}

print(json.dumps(task_definition, indent=2))

# AWS Lambda for serverless deployment
import json
import boto3
from agno.agent import Agent

def lambda_handler(event, context):
    """Handle API Gateway requests via Lambda."""
    
    try:
        # Parse request
        body = json.loads(event.get("body", "{}"))
        message = body.get("message", "")
        
        # Initialize agent (cache this in real deployment)
        agent = Agent(name="Lambda Agent")
        
        # Process request
        response = agent.run(message)
        
        return {
            "statusCode": 200,
            "body": json.dumps({"response": response}),
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
```

---

## Scaling Considerations

### 1. Horizontal Scaling Strategy

```python
# Load balancing across multiple agent instances
from typing import List
import asyncio
from httpx import AsyncClient

class AgentLoadBalancer:
    """Distribute requests across multiple agent instances."""
    
    def __init__(self, agent_urls: List[str]):
        self.agent_urls = agent_urls
        self.current_index = 0
    
    async def route_request(self, message: str) -> str:
        """Route request to next available agent."""
        client = AsyncClient()
        
        for attempt in range(len(self.agent_urls)):
            url = self.agent_urls[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.agent_urls)
            
            try:
                response = await client.post(
                    f"{url}/chat",
                    json={"message": message},
                    timeout=30.0
                )
                return response.json()["response"]
            except Exception as e:
                print(f"Agent {url} failed: {e}")
                continue
        
        raise Exception("All agents failed")

# Usage
lb = AgentLoadBalancer([
    "http://agent-1:8000",
    "http://agent-2:8000",
    "http://agent-3:8000",
])

result = asyncio.run(lb.route_request("Hello"))
```

### 2. Caching Layer

```python
# Redis caching for frequently asked questions
from redis import Redis
import json
import hashlib

class CachedAgent:
    """Agent with Redis caching."""
    
    def __init__(self, agent, redis_client: Redis):
        self.agent = agent
        self.redis = redis_client
        self.ttl = 3600  # 1 hour
    
    def _get_cache_key(self, message: str) -> str:
        """Generate cache key from message."""
        return f"agent:{hashlib.md5(message.encode()).hexdigest()}"
    
    def run(self, message: str) -> str:
        """Run agent with caching."""
        cache_key = self._get_cache_key(message)
        
        # Try cache first
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Cache miss - run agent
        response = self.agent.run(message)
        
        # Store in cache
        self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(response)
        )
        
        return response

# Usage
redis_client = Redis.from_url("redis://localhost:6379")
cached_agent = CachedAgent(agent, redis_client)
response = cached_agent.run("What is Python?")  # Cached on second call
```

### 3. Queue-based Processing

```python
# Celery for background agent processing
from celery import Celery
from kombu import Exchange, Queue

app = Celery("agent_tasks")
app.conf.broker_url = "redis://localhost:6379"

# Define queues
app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("high_priority", Exchange("priority"), routing_key="high"),
    Queue("long_running", Exchange("long"), routing_key="long"),
)

@app.task(queue="default")
def process_agent_request(message: str, user_id: str) -> str:
    """Process agent request asynchronously."""
    agent = Agent(name="Celery Agent")
    return agent.run(message)

@app.task(queue="high_priority", priority=9)
def process_urgent_request(message: str) -> str:
    """Process high-priority request."""
    agent = Agent(name="Urgent Agent")
    return agent.run(message)

# Usage
from agent_tasks import process_agent_request

# Fire and forget
task = process_agent_request.delay("Hello", user_id="user123")

# Get result later
result = task.get(timeout=300)
```

---

## Performance Optimization

### 1. Model Routing Based on Complexity

```python
from typing import Optional
import os

class SmartModelRouter:
    """Route requests to appropriate model based on complexity."""
    
    def __init__(self):
        self.simple_model = "gpt-3.5-turbo"
        self.complex_model = "gpt-4o"
        self.fast_model = "gpt-3.5-turbo-16k"
    
    def detect_complexity(self, message: str) -> str:
        """Detect request complexity."""
        complex_keywords = [
            "analyze", "explain", "research", "comprehensive",
            "reasoning", "math", "code", "architecture"
        ]
        
        word_count = len(message.split())
        has_complex_keywords = any(kw in message.lower() for kw in complex_keywords)
        
        if has_complex_keywords or word_count > 100:
            return "complex"
        elif word_count < 50:
            return "simple"
        else:
            return "medium"
    
    def get_model(self, message: str) -> str:
        """Select appropriate model."""
        complexity = self.detect_complexity(message)
        
        if complexity == "simple":
            return self.simple_model
        elif complexity == "complex":
            return self.complex_model
        else:
            return self.fast_model
    
    def estimate_cost(self, message: str) -> float:
        """Estimate API cost."""
        model = self.get_model(message)
        word_count = len(message.split())
        
        # Approximate costs (in cents per 1K tokens)
        costs = {
            "gpt-3.5-turbo": 0.0005,
            "gpt-4o": 0.003,
            "gpt-3.5-turbo-16k": 0.001,
        }
        
        return (word_count / 750) * costs.get(model, 0.001)

# Usage
router = SmartModelRouter()
print(router.get_model("Hello"))  # simple_model
print(router.get_model("Analyze the 2026 AI market trends comprehensively"))  # complex_model
print(router.estimate_cost("Hello"))  # ~$0.00067
```

### 2. Batch Processing

```python
# Process multiple requests efficiently
from typing import List
import asyncio
from datetime import datetime

class BatchProcessor:
    """Process multiple agent requests in batches."""
    
    def __init__(self, agent, batch_size: int = 10):
        self.agent = agent
        self.batch_size = batch_size
        self.batch = []
        self.results = {}
    
    async def add_request(self, request_id: str, message: str):
        """Add request to batch."""
        self.batch.append((request_id, message))
        
        if len(self.batch) >= self.batch_size:
            await self.process_batch()
    
    async def process_batch(self):
        """Process accumulated requests."""
        if not self.batch:
            return
        
        tasks = []
        for request_id, message in self.batch:
            task = asyncio.create_task(self._run_agent(request_id, message))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        self.batch = []
    
    async def _run_agent(self, request_id: str, message: str):
        """Run agent and store result."""
        try:
            result = self.agent.run(message)
            self.results[request_id] = {
                "status": "success",
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.results[request_id] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    async def get_result(self, request_id: str):
        """Get result of a request."""
        # Wait until result is available
        while request_id not in self.results:
            await asyncio.sleep(0.1)
        return self.results[request_id]

# Usage
processor = BatchProcessor(agent, batch_size=5)

# Add multiple requests
for i in range(10):
    asyncio.run(processor.add_request(f"req-{i}", f"Query {i}"))

# Process remaining batch
asyncio.run(processor.process_batch())

# Get results
result = asyncio.run(processor.get_result("req-0"))
```

---

## Monitoring and Logging

### 1. Comprehensive Logging

```python
# Structured logging with Python logging
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Setup production-grade logging."""
    
    # JSON logging for structured logs
    logger = logging.getLogger()
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

# Usage examples
logger.info("Agent started", extra={
    "agent_name": "ResearchAgent",
    "model": "gpt-4",
    "environment": "production",
})

logger.warning("High latency detected", extra={
    "response_time_ms": 5000,
    "threshold_ms": 3000,
    "user_id": "user123",
})

logger.error("Agent tool execution failed", extra={
    "tool_name": "search_web",
    "error": "Network timeout",
    "retry_count": 3,
})
```

### 2. Prometheus Metrics

```python
# Expose agent metrics for Prometheus
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_count = Counter(
    "agent_requests_total",
    "Total agent requests",
    ["model", "status"],
)

request_duration = Histogram(
    "agent_request_duration_seconds",
    "Agent request duration",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
)

active_requests = Gauge(
    "agent_active_requests",
    "Currently active agent requests",
    ["model"],
)

tokens_used = Counter(
    "agent_tokens_used_total",
    "Total tokens used",
    ["model"],
)

errors_count = Counter(
    "agent_errors_total",
    "Total agent errors",
    ["model", "error_type"],
)

# Usage
import functools

def track_metrics(model_name: str):
    """Decorator to track agent metrics."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            active_requests.labels(model=model_name).inc()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                request_count.labels(model=model_name, status="success").inc()
                return result
            except Exception as e:
                request_count.labels(model=model_name, status="error").inc()
                errors_count.labels(model=model_name, error_type=type(e).__name__).inc()
                raise
            finally:
                duration = time.time() - start_time
                request_duration.labels(model=model_name).observe(duration)
                active_requests.labels(model=model_name).dec()
        
        return wrapper
    return decorator

@track_metrics("gpt-4")
def run_agent(message: str) -> str:
    agent = Agent(name="Monitored Agent")
    return agent.run(message)
```

### 3. Distributed Tracing

```python
# OpenTelemetry for distributed tracing
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

def setup_tracing():
    """Setup OpenTelemetry tracing."""
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    # Instrument frameworks
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument()

# Usage
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("agent_execution") as span:
    span.set_attribute("agent.name", "ResearchAgent")
    span.set_attribute("user.id", "user123")
    
    # Your agent code here
    response = agent.run("What is AI?")
    
    span.set_attribute("response.length", len(response))
```

---

## Error Handling and Recovery

### 1. Comprehensive Error Handling

```python
from typing import Optional
from enum import Enum
import asyncio

class ErrorCategory(Enum):
    TRANSIENT = "transient"  # Retry-able
    PERMANENT = "permanent"  # Non-retry-able
    CONFIGURATION = "configuration"  # Fix required

class AgentException(Exception):
    """Base agent exception."""
    def __init__(self, message: str, category: ErrorCategory, retry_after: int = 0):
        self.message = message
        self.category = category
        self.retry_after = retry_after
        super().__init__(message)

class ResilientAgent:
    """Agent with comprehensive error handling."""
    
    def __init__(self, agent, max_retries: int = 3):
        self.agent = agent
        self.max_retries = max_retries
    
    async def run_with_retry(self, message: str) -> str:
        """Run agent with exponential backoff retry."""
        
        for attempt in range(self.max_retries):
            try:
                return self.agent.run(message)
            
            except Exception as e:
                error_type = type(e).__name__
                
                # Categorize error
                if "rate limit" in str(e).lower():
                    category = ErrorCategory.TRANSIENT
                    wait_time = 2 ** attempt  # Exponential backoff
                elif "authentication" in str(e).lower():
                    category = ErrorCategory.PERMANENT
                else:
                    category = ErrorCategory.TRANSIENT
                
                if category == ErrorCategory.PERMANENT:
                    raise AgentException(str(e), category)
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                    continue
                
                raise AgentException(
                    f"Failed after {self.max_retries} attempts: {e}",
                    ErrorCategory.TRANSIENT
                )

# Fallback strategy
class FallbackAgent:
    """Agent with fallback model selection."""
    
    def __init__(self, primary_model: str, fallback_model: str):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
    
    def run(self, message: str) -> str:
        """Run with fallback."""
        try:
            agent = Agent(model_name=self.primary_model)
            return agent.run(message)
        except Exception as e:
            print(f"Primary model failed: {e}")
            agent = Agent(model_name=self.fallback_model)
            return agent.run(message)
```

### 2. Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to recover."""
        if self.last_failure_time is None:
            return False
        
        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.timeout)

# Usage
breaker = CircuitBreaker(failure_threshold=3)

def call_agent(message: str) -> str:
    agent = Agent()
    return agent.run(message)

try:
    result = breaker.call(call_agent, "Hello")
except Exception as e:
    print(f"Request failed: {e}")
```

---

## Version Management

### 1. Agent Version Control

```yaml
# agent-versions.yaml
versions:
  v1.0.0:
    model: gpt-3.5-turbo
    tools:
      - search_web
      - calculator
    prompt_version: p1.2
    status: deprecated
    deprecated_date: "2026-01-15"
  
  v1.1.0:
    model: gpt-4
    tools:
      - search_web
      - calculator
      - knowledge_base
    prompt_version: p1.3
    status: stable
    release_date: "2025-06-01"
  
  v2.0.0:
    model: gpt-4o
    tools:
      - search_web
      - calculator
      - knowledge_base
      - code_executor
    prompt_version: p2.0
    status: beta
    release_date: "2026-02-15"
    canary_percentage: 10
```

### 2. Prompt Versioning

```python
# Semantic versioning for prompts
from dataclasses import dataclass
from typing import Optional

@dataclass
class PromptVersion:
    """Track prompt changes."""
    version: str  # e.g., "p2.1.0"
    content: str
    description: str
    author: str
    created_at: str
    breaking_changes: Optional[list] = None
    
    def is_compatible_with(self, agent_version: str) -> bool:
        """Check if prompt is compatible with agent version."""
        agent_major = int(agent_version.split(".")[0])
        prompt_major = int(self.version.split(".")[0])
        return agent_major == prompt_major

# Prompt management
class PromptManager:
    """Manage multiple prompt versions."""
    
    def __init__(self):
        self.prompts = {}
        self.active_version = None
    
    def register_prompt(self, prompt: PromptVersion):
        """Register a new prompt version."""
        self.prompts[prompt.version] = prompt
    
    def activate_version(self, version: str):
        """Switch to specific prompt version."""
        if version not in self.prompts:
            raise ValueError(f"Prompt version {version} not found")
        self.active_version = version
    
    def rollback_version(self):
        """Rollback to previous version."""
        versions = sorted(self.prompts.keys())
        current_idx = versions.index(self.active_version)
        if current_idx > 0:
            self.activate_version(versions[current_idx - 1])
    
    def get_active_prompt(self) -> str:
        """Get currently active prompt."""
        return self.prompts[self.active_version].content
```

---

## Cost Optimization

### 1. Token Cost Tracking

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenCost:
    """Track token usage and cost."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_per_1k_input: float  # cents
    cost_per_1k_output: float  # cents
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost in cents."""
        input_cost = (self.input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (self.output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost

# Model pricing database (2026 rates)
MODEL_PRICING = {
    "gpt-3.5-turbo": {
        "input": 0.05,  # cents per 1K tokens
        "output": 0.15,
    },
    "gpt-4": {
        "input": 3.0,
        "output": 6.0,
    },
    "gpt-4o": {
        "input": 2.5,
        "output": 10.0,
    },
    "claude-3-haiku": {
        "input": 0.25,
        "output": 1.25,
    },
}

class CostOptimizer:
    """Optimize agent costs."""
    
    @staticmethod
    def recommend_model(message_length: int, required_quality: str) -> str:
        """Recommend cost-effective model."""
        if required_quality == "high":
            return "gpt-4o"
        elif message_length < 100:
            return "gpt-3.5-turbo"
        else:
            return "gpt-4"
    
    @staticmethod
    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate operation cost."""
        pricing = MODEL_PRICING.get(model, {})
        input_cost = (input_tokens / 1000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1000) * pricing.get("output", 0)
        return input_cost + output_cost

# Cost tracking middleware
class CostTracker:
    """Track and report costs."""
    
    def __init__(self):
        self.daily_costs = {}
    
    def track_request(self, model: str, tokens: dict, cost: float):
        """Record request cost."""
        today = datetime.now().date().isoformat()
        
        if today not in self.daily_costs:
            self.daily_costs[today] = 0
        
        self.daily_costs[today] += cost
    
    def get_daily_cost(self, date: str) -> float:
        """Get cost for specific date."""
        return self.daily_costs.get(date, 0)
    
    def get_monthly_cost(self, month: str) -> float:
        """Get cost for specific month."""
        total = 0
        for date, cost in self.daily_costs.items():
            if date.startswith(month):
                total += cost
        return total
```

### 2. Request Optimization

```python
class RequestOptimizer:
    """Optimize requests to reduce costs."""
    
    @staticmethod
    def truncate_context(context: str, max_length: int = 2000) -> str:
        """Limit context to reduce tokens."""
        if len(context) > max_length:
            return context[:max_length] + "..."
        return context
    
    @staticmethod
    def cache_similar_queries(query: str, cache: dict) -> Optional[str]:
        """Avoid duplicate API calls."""
        import difflib
        
        for cached_query, result in cache.items():
            similarity = difflib.SequenceMatcher(None, query, cached_query).ratio()
            if similarity > 0.9:  # 90% similar
                return result
        
        return None
    
    @staticmethod
    def batch_requests(requests: list, batch_size: int = 5):
        """Process requests in batches to reduce overhead."""
        for i in range(0, len(requests), batch_size):
            yield requests[i:i + batch_size]
```

---

## Security Considerations

### 1. Input Validation and Sanitization

```python
from pydantic import BaseModel, validator, Field
import re

class SecureAgentRequest(BaseModel):
    """Validate and sanitize user input."""
    
    message: str = Field(..., max_length=5000)
    user_id: str = Field(..., min_length=1, max_length=100)
    session_id: Optional[str] = None
    
    @validator("message")
    def validate_message(cls, v):
        """Prevent prompt injection."""
        # Block suspicious patterns
        injection_patterns = [
            r"system:",
            r"admin:",
            r"sql",
            r"<script",
            r"<?php",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially malicious input detected")
        
        return v.strip()
    
    @validator("user_id")
    def validate_user_id(cls, v):
        """Validate user ID format."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Invalid user ID format")
        return v

# Input guardrails
class InputGuardrails:
    """Protect against common attacks."""
    
    @staticmethod
    def remove_pii(text: str) -> str:
        """Remove Personally Identifiable Information."""
        import re
        
        # Email
        text = re.sub(r"\S+@\S+", "[EMAIL_REDACTED]", text)
        
        # Phone number
        text = re.sub(r"\d{3}[-.]?\d{3}[-.]?\d{4}", "[PHONE_REDACTED]", text)
        
        # Credit card
        text = re.sub(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "[CARD_REDACTED]", text)
        
        # SSN
        text = re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN_REDACTED]", text)
        
        return text
    
    @staticmethod
    def detect_prompt_injection(message: str) -> bool:
        """Detect prompt injection attempts."""
        suspicious_patterns = [
            "ignore previous",
            "system override",
            "you are now",
            "pretend you are",
            "act as if",
        ]
        
        message_lower = message.lower()
        return any(pattern in message_lower for pattern in suspicious_patterns)
```

### 2. Output Filtering

```python
class OutputGuardrails:
    """Validate agent output for safety."""
    
    @staticmethod
    def filter_hallucinations(response: str, source_docs: list) -> tuple[str, bool]:
        """Remove unsupported claims."""
        import json
        
        hallucination_score = 0
        filtered_response = response
        
        # Check if response is supported by sources
        for doc in source_docs:
            if any(word in response.lower() for word in doc.lower().split()):
                hallucination_score -= 1
        
        # Mark if potential hallucination
        is_hallucinated = hallucination_score > 0
        
        return filtered_response, is_hallucinated
    
    @staticmethod
    def enforce_schema(response: str, schema: dict) -> bool:
        """Validate output matches expected schema."""
        import json
        
        try:
            response_obj = json.loads(response)
            # Validate required fields
            for field in schema.get("required", []):
                if field not in response_obj:
                    return False
            return True
        except json.JSONDecodeError:
            return False
```

### 3. Access Control

```python
from enum import Enum
from typing import List

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"

class RoleBasedAccessControl:
    """Control what agents can access."""
    
    def __init__(self):
        self.roles = {
            "viewer": [Permission.READ],
            "editor": [Permission.READ, Permission.WRITE],
            "admin": [Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.DELETE],
        }
    
    def check_permission(self, user_role: str, required_permission: Permission) -> bool:
        """Check if user has required permission."""
        permissions = self.roles.get(user_role, [])
        return required_permission in permissions
    
    def get_allowed_tools(self, user_role: str) -> List[str]:
        """Get tools available to user."""
        if user_role == "viewer":
            return ["search", "read_knowledge_base"]
        elif user_role == "editor":
            return ["search", "read_knowledge_base", "write_knowledge_base"]
        else:  # admin
            return ["search", "read_knowledge_base", "write_knowledge_base", "execute_code"]
```

---

## Infrastructure Recommendations

### 1. Recommended Architecture (2026)

```
┌─────────────────────────────────────────────────────────────┐
│                       Users / API Clients                    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Load Balancer (AWS ALB)                    │
│              (Auto-scale based on traffic)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼──┐       ┌────▼──┐      ┌────▼──┐
    │Agent  │       │Agent  │      │Agent  │
    │Pod 1  │       │Pod 2  │      │Pod N  │  (Kubernetes)
    └────┬──┘       └────┬──┘      └────┬──┘
         │               │               │
         └───────────────┼───────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───▼────┐        ┌─────▼──────┐      ┌─────▼──────┐
│ Redis  │        │ PostgreSQL │      │ Vector DB  │
│Cache   │        │(State)     │      │(Knowledge) │
└────────┘        └────────────┘      └────────────┘
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───▼────────┐    ┌──────▼──────┐    ┌──────▼───────┐
│  S3/GCS    │    │ Prometheus  │    │  ELK Stack  │
│(Artifacts) │    │(Metrics)    │    │(Logging)    │
└────────────┘    └─────────────┘    └─────────────┘
```

### 2. Computing Requirements

```python
# Agent resource requirements (2026 benchmarks)
RESOURCE_SPECS = {
    "agent": {
        "cpu": {
            "request": "500m",
            "limit": "2000m",
        },
        "memory": {
            "request": "512Mi",
            "limit": "2Gi",
        },
        "gpu": None,  # Optional for inference
    },
    "cache": {
        "memory": "2Gi",
    },
    "database": {
        "storage": "20Gi",
        "iops": 1000,
    },
}

# Estimated costs (AWS pricing, 2026)
MONTHLY_COST_ESTIMATE = {
    "compute": {
        "3_replicas": 150,  # 3 pods continuously
        "scale_to_10": 500,  # Peak scaling
    },
    "storage": {
        "database": 100,
        "vectors": 50,
        "artifacts": 75,
    },
    "api_calls": {
        "gpt4o_100k_requests": 2000,
    },
    "monitoring": 100,
    "total_monthly": 3000,  # ~$3k/month for small-medium workload
}
```

### 3. High Availability Setup

```yaml
# Multi-region deployment
regions:
  us-east-1:
    primary: true
    replicas: 3
    database:
      type: "rds-aurora"
      multi_az: true
    cache:
      type: "elasticache-redis"
      multi_az: true
  
  us-west-2:
    primary: false
    replicas: 2
    database:
      read_replica: true
      replication_lag: "< 1 second"

disaster_recovery:
  rto: 300  # 5 minutes
  rpo: 60   # 1 minute
  backup_frequency: "hourly"
  cross_region_failover: true
```

---

## Summary

### Key Deployment Principles

1. **Prepare thoroughly** - Use comprehensive checklists before production
2. **Containerize everything** - Docker ensures consistency across environments
3. **Scale horizontally** - Use load balancers and Kubernetes for reliability
4. **Optimize aggressively** - Model routing and caching reduce costs significantly
5. **Monitor comprehensively** - Logging, metrics, and tracing catch issues early
6. **Handle errors gracefully** - Retry logic and fallbacks prevent cascading failures
7. **Manage versions carefully** - Separate prompts, models, and configurations
8. **Optimize costs continuously** - Token tracking and smart routing reduce expenses
9. **Secure rigorously** - Input/output guardrails prevent attacks
10. **Plan for scale** - Design for 10x growth from day one

### 2026 Best Practices

- AgentExecutor is deprecated (EOL Dec 2026) - migrate to LangGraph for complex workflows
- Semantic evaluation required for LLM outputs - not just unit tests
- State persistence essential - use PostgreSQL checkpointers, not in-memory
- Human-in-the-loop first-class feature - pause, inspect, resume workflows
- Progressive deployments mandatory - canary releases for model/prompt changes
- Unified CI/CD essential - treat prompts, configs, and code equally

---

## References

1. Harness AI Deployment Guide 2026: https://harness.io/blog/ai-deployment-in-production-orchestrate-llms-rag-agents
2. AI Agent Deployment Strategies: https://zylos.ai/research/2026-03-05-ai-agent-deployment-strategies-containerization-scaling
3. LangGraph Documentation: https://docs.langchain.com/oss/python/langgraph/overview
4. Kubernetes Best Practices: https://kubernetes.io/docs/concepts/configuration/overview/
5. OpenTelemetry Documentation: https://opentelemetry.io/docs/
6. AWS ECS Task Definition: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html
