# Deployment Pipeline README

Complete guide for deployment pipeline used in LLM-Whisperer.

## Quick Start

### 1. Package Model

```bash
python scripts/deploy.py --model ./training_outputs/lora --version 1.0.0
```

### 2. Deploy to HuggingFace Hub

```bash
python scripts/deploy.py --model ./training_outputs/lora --version 1.0.0 --push-to-hub
```

### 3. Monitor Deployment

```bash
python scripts/monitor.py --model mistral-7b-lora
```

## Architecture

### Core Modules

#### 1. **Packaging** (`src/orchestrator.py`)

Model packaging and preparation:
- Model file organization
- Metadata embedding
- Configuration bundling
- Artifact versioning

#### 2. **Versioning**

Semantic versioning system:
- Major.Minor.Patch (e.g., 1.2.3)
- Version history tracking
- Compatibility checking
- Changelog management

#### 3. **Publishing**

Model registry management:
- HuggingFace Hub integration
- Custom registry support
- Authentication handling
- Model card generation

#### 4. **Rollback**

Deployment recovery:
- Version history tracking
- Safe rollback procedures
- Health checks
- Validation gates

#### 5. **Monitoring**

Post-deployment tracking:
- Performance metrics
- Error tracking
- Latency monitoring
- Usage statistics

## Deployment Process

```
Trained Model
    ↓
Package
    ↓
Version (v1.0.0)
    ↓
Publish
    ↓
Monitor
```

## Configuration

Example deployment configuration:

```yaml
deployment:
  model_path: ./training_outputs/lora
  model_name: mistral-7b-lora
  model_version: 1.0.0
  
  push_to_hub: true
  hub_repo_id: username/mistral-7b-lora
  
  quantization: false
  optimization_level: 1
  
  major: 1
  minor: 0
  patch: 0
```

## Usage Examples

### Example 1: Local Packaging

```python
from pipelines.deployment.src.orchestrator import (
    DeploymentOrchestrator, DeploymentConfig
)

config = DeploymentConfig(
    model_path="./training_outputs/lora",
    model_name="mistral-7b-lora",
    model_version="1.0.0"
)

orchestrator = DeploymentOrchestrator(config)
package_path = orchestrator.package_model()
print(f"Model packaged at: {package_path}")
```

### Example 2: Publishing to HuggingFace

```python
config = DeploymentConfig(
    model_path="./training_outputs/lora",
    model_name="mistral-7b-lora",
    model_version="1.0.0",
    push_to_hub=True,
    hub_repo_id="myuser/mistral-lora"
)

orchestrator = DeploymentOrchestrator(config)
orchestrator.package_model()
result = orchestrator.publish_model()
print(f"Published to: {result['hub_url']}")
```

## Versioning Strategy

### Semantic Versioning
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes

Examples:
- 1.0.0 - Initial release
- 1.1.0 - Added LoRA support
- 1.1.1 - Fixed tokenization bug
- 2.0.0 - New model architecture

## Integration with Training Pipeline

Deploy after successful training:

```bash
# Train model
python pipelines/training/scripts/train.py --config training_config.yaml

# Evaluate model
python pipelines/evaluation/scripts/evaluate.py --model ./training_outputs/lora

# Deploy if evaluation passes
python pipelines/deployment/scripts/deploy.py \
  --model ./training_outputs/lora \
  --version 1.0.0 \
  --push-to-hub
```

## Deployment Checklist

Before deploying:
- [ ] Model trained and checkpointed
- [ ] Evaluation metrics acceptable
- [ ] No regressions detected
- [ ] Model card prepared
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Version number incremented

## Common Issues

### Issue: Model too large for Hub
**Solution**:
- Use Git LFS for large files
- Split model into parts
- Use quantization

### Issue: Publishing fails
**Solution**:
- Check HuggingFace token
- Verify repository permissions
- Check internet connection

### Issue: Version conflict
**Solution**:
- Increment version number
- Check existing tags
- Use unique version identifiers

## License

See LICENSE file in repository root.
