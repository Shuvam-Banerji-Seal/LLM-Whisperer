# Hugging Face Tokens, Auth, and Offline Access - Agentic Skill Prompt

Use this prompt for secure Hugging Face authentication, gated model access, and reproducible online and offline model workflows.

## 1. Mission

Operate Hugging Face credentials and downloads with least privilege, strong reproducibility, and zero token leakage.

Priority order:

1. Credential safety and access correctness.
2. Reproducible model retrieval with pinned revisions.
3. CI and production usability.

## 2. Token Policy

- Prefer fine-grained tokens for service workloads.
- Use read-scoped access for inference-only services.
- Reserve write-scoped tokens only for push or publish workflows.
- Do not use broad write tokens in production inference.

Practical token mapping:

- Local experimentation: fine-grained read token.
- CI inference jobs: dedicated fine-grained read token per environment.
- Model publishing pipeline: separate write token with minimal scope.

## 3. Authentication Workflows

### 3.1 CLI login

```bash
hf auth login
hf auth whoami
```

### 3.2 Python login path (when CLI is not practical)

```python
from huggingface_hub import login

login(token=None)
```

### 3.3 Environment-variable auth

```bash
export HF_TOKEN="<injected-by-secret-manager>"
```

Note: `HF_TOKEN` overrides locally stored tokens for the process environment.

## 4. Gated Model Access Protocol

1. Request access from the model page with the account used in runtime.
2. Confirm approval for that exact account and organization context.
3. Validate with `hf auth whoami` and a test download.

Maintainer-side pending request query pattern:

```bash
curl -H "authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/models/<repo_id>/user-access-request/pending
```

## 5. Cache, Paths, and Offline Controls

Recommended environment setup:

```bash
export HF_HOME=/opt/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_ASSETS_CACHE=$HF_HOME/assets
export HF_TOKEN_PATH=$HF_HOME/token
export HF_HUB_OFFLINE=1
```

Use snapshot pinning for deterministic artifacts:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a",
    local_dir="./artifacts/llama2",
)
```

Offline load pattern:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./artifacts/llama2"
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
```

## 6. CI and Secrets Hygiene

- Inject tokens via CI secret store only.
- Never pass raw tokens on command line flags that can leak to logs or shell history.
- Use distinct tokens for dev, staging, and prod.
- Rotate tokens on schedule and immediately after any suspected leak.

Example GitHub Actions pattern:

```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN_FINE_GRAINED }}
steps:
  - run: python scripts/infer.py
```

## 7. Security Checklist

- No hardcoded token strings in code, notebooks, configs, or Dockerfiles.
- Restrict token scope to the smallest model and action set.
- Keep token storage path permissions strict.
- Redact secrets in logs and incident artifacts.
- Revoke and rotate on incident, then audit access.

## 8. Operational Failure Modes and Fixes

1. 401 or 403 on private model: wrong token scope or missing access approval.
2. Works locally but not in container: token not injected at runtime.
3. Gated model approved in browser but CI fails: CI token belongs to different account.
4. Offline load fails: artifacts not fully downloaded or wrong cache path.
5. Unexpected token in process: environment variable overrides local login state.

## 9. References (Fetched 2026-04-06)

1. https://huggingface.co/docs/hub/security-tokens - Official token roles and least-privilege guidance.
2. https://huggingface.co/docs/hub/models-gated - Gated model request and approval workflows.
3. https://huggingface.co/docs/huggingface_hub/en/guides/cli - CLI authentication and account commands.
4. https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication - Authentication quickstart and behavior.
5. https://huggingface.co/docs/huggingface_hub/en/package_reference/authentication - Programmatic auth APIs.
6. https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables - Cache and token environment variables.
7. https://huggingface.co/docs/huggingface_hub/en/guides/download - Download and snapshot workflows.
8. https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download#huggingface_hub.snapshot_download - Snapshot API reference.
9. https://huggingface.co/docs/transformers/main/en/installation#offline-mode - Transformers offline mode guidance.
10. https://huggingface.co/docs/hub/model-cards - Model card metadata and model governance context.
11. https://huggingface.co/docs/hub/repositories-licenses - License and repository policy context.
12. https://huggingface.co/docs/transformers/main/en/main_classes/model - Loader options tied to authenticated downloads.
