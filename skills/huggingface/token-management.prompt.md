# Master Agentic Skill: Token Management and Chat Template Correctness

## 1. Mission
Guarantee tokenizer, chat template, padding, and stop token correctness across training and inference so that model behavior remains stable and predictable.

## 2. Principles
- Prioritize reproducibility over one-off wins.
- Log every configuration that can alter behavior.
- Validate quality and latency together; never optimize one blindly.
- Keep rollback paths documented and tested.
- Treat safety and governance checks as first-class production requirements.

## 3. Source Index (Docs and Blogs)
1. https://huggingface.co/docs/transformers/chat_templating
2. https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils
3. https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer
4. https://huggingface.co/docs/transformers/conversations
5. https://huggingface.co/docs/trl/dataset_formats
6. https://huggingface.co/docs/transformers/main/en/tasks/language_modeling
7. https://huggingface.co/docs/transformers/main/en/model_doc/mixtral
8. https://huggingface.co/docs/transformers/main/en/model_doc/llama

## 4. Fast Documentation Fetch Commands
Use these commands when someone reports issues and you need to verify behavior against upstream docs quickly.

```bash
mkdir -p /tmp/skill_refs
curl -L "https://huggingface.co/docs/transformers/chat_templating" -o /tmp/skill_refs/huggingface.co_docs_transformers_chat_templating.html
curl -L "https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_internal_tokenization_utils.html
curl -L "https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_main_classes_tokenizer.html
curl -L "https://huggingface.co/docs/transformers/conversations" -o /tmp/skill_refs/huggingface.co_docs_transformers_conversations.html
curl -L "https://huggingface.co/docs/trl/dataset_formats" -o /tmp/skill_refs/huggingface.co_docs_trl_dataset_formats.html
curl -L "https://huggingface.co/docs/transformers/main/en/tasks/language_modeling" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_tasks_language_modeling.html
curl -L "https://huggingface.co/docs/transformers/main/en/model_doc/mixtral" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_model_doc_mixtral.html
curl -L "https://huggingface.co/docs/transformers/main/en/model_doc/llama" -o /tmp/skill_refs/huggingface.co_docs_transformers_main_en_model_doc_llama.html
ls -lh /tmp/skill_refs
```

## 5. Operational Policies
Use this section as the mandatory baseline policy set for token management.

### 5.1 Metrics that must always be tracked
- token_count_per_request
- template_render_error_rate
- stop_token_match_rate
- response_truncation_rate
- generation_prefix_correctness
- train_eval_tokenization_divergence
- invalid_special_token_rate

### 5.2 Guardrails
- Never mix manual prompt formatting and chat templates in same endpoint.
- Enforce tokenizer revision pinning for reproducible deployments.
- Fail CI on template rendering exceptions.
- Keep explicit tests for add_generation_prompt and continue_final_message behavior.
- Validate stop token IDs for every supported model family.
- Require token length budget checks before rollout.

## 6. Codebook
Each recipe is production-oriented and intentionally explicit.

### Recipe 01: Safe chat template rendering for inference
Use this when serving chat models to avoid malformed prompt structures.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", use_fast=True)
messages = [
    {"role": "system", "content": "You are concise and factual."},
    {"role": "user", "content": "Summarize retrieval-augmented generation in 2 lines."},
]

rendered = tok.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
print(rendered.shape)
```

Notes:
- Prefer tokenize=True to avoid accidental special token duplication.
- Keep add_generation_prompt=True for inference in most chat models.

### Recipe 02: Training-time chat formatting
Use this for dataset preprocessing where generation prompt should not be appended.

```python
from datasets import Dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
ds = Dataset.from_dict({
    "chat": [
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}],
        [{"role": "user", "content": "Name a prime"}, {"role": "assistant", "content": "13"}],
    ]
})

def fmt(row):
    return {
        "text": tok.apply_chat_template(
            row["chat"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

ds = ds.map(fmt)
print(ds[0]["text"])
```

Notes:
- For training, set add_generation_prompt=False.
- Keep formatting deterministic across train and eval.

### Recipe 03: Prefill with continue_final_message
Use this to continue an unfinished assistant message such as JSON prefills.

```python
chat = [
    {"role": "user", "content": "Return JSON with keys name and age"},
    {"role": "assistant", "content": '{"name": "'}
]

packed = tok.apply_chat_template(
    chat,
    tokenize=True,
    return_dict=True,
    continue_final_message=True,
)
```

Notes:
- Do not combine continue_final_message with add_generation_prompt.
- Use this only when intentionally prefilling assistant output.

### Recipe 04: Padding and stop token policy
Use this to enforce consistent batching behavior for decoder-only models.

```python
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model_stop_ids = [tok.eos_token_id]
print("pad", tok.pad_token_id, "eos", tok.eos_token_id, "stops", model_stop_ids)
```

Notes:
- Left padding is generally required for batched decoder-only generation.
- Model families may require additional stop IDs.

### Recipe 05: Add special tokens and resize embeddings safely
Use this after adding domain-specific control tokens.

```python
special = {"additional_special_tokens": ["<json_start>", "<json_end>"]}
n_added = tok.add_special_tokens(special)
print("added", n_added)
model.resize_token_embeddings(len(tok))
```

Notes:
- Never add tokens without resizing embeddings.
- Re-evaluate perplexity and downstream metrics after vocabulary changes.

## 7. Failure and Recovery Matrix
This matrix is intentionally exhaustive. Follow one case at a time and log every change.

### Case 001: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 002: responses truncate too early
- Signal: responses truncate too early
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 003: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 004: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 005: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 006: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 007: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 008: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 009: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 010: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 011: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 012: responses truncate too early
- Signal: responses truncate too early
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 013: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 014: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 015: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 016: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 017: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 018: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 019: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 020: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 021: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 022: responses truncate too early
- Signal: responses truncate too early
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 023: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 024: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 025: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 026: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 027: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 028: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 029: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 030: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 031: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 032: responses truncate too early
- Signal: responses truncate too early
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 033: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 034: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 035: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 036: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 037: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 038: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 039: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 040: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 041: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 042: responses truncate too early
- Signal: responses truncate too early
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 043: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 044: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 045: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 046: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 047: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 048: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 049: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 050: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 051: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 052: responses truncate too early
- Signal: responses truncate too early
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 053: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 054: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 055: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 056: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 057: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 058: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 059: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 060: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 061: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 062: responses truncate too early
- Signal: responses truncate too early
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 063: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 064: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 065: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 066: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 067: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 068: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 069: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 070: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 071: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 072: responses truncate too early
- Signal: responses truncate too early
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 073: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 074: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 075: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 076: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 077: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 078: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 079: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 080: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 081: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 082: responses truncate too early
- Signal: responses truncate too early
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 083: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 084: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 085: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 086: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 087: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 088: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 089: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 090: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 091: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 092: responses truncate too early
- Signal: responses truncate too early
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 093: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 094: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 095: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 096: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 097: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 098: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 099: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 100: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 101: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 102: responses truncate too early
- Signal: responses truncate too early
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 103: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 104: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 105: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 106: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 107: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 108: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 109: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 110: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 111: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 112: responses truncate too early
- Signal: responses truncate too early
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 113: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 114: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 115: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 116: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 117: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 118: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 119: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 120: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 121: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 122: responses truncate too early
- Signal: responses truncate too early
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 123: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 124: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 125: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 126: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 127: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 128: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 129: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 130: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 131: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 132: responses truncate too early
- Signal: responses truncate too early
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 133: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 134: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 135: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 136: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 137: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 138: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 139: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 140: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 141: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 142: responses truncate too early
- Signal: responses truncate too early
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 143: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 144: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 145: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 146: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 147: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 148: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 149: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 150: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 151: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 152: responses truncate too early
- Signal: responses truncate too early
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 153: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 154: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 155: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 156: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 157: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 158: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 159: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 160: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 161: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 162: responses truncate too early
- Signal: responses truncate too early
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 163: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 164: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 165: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 166: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 167: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 168: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 169: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 170: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 171: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 172: responses truncate too early
- Signal: responses truncate too early
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 173: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 174: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 175: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 176: tool-calling format fails validation
- Signal: tool-calling format fails validation
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 177: batch generation produces inconsistent lengths
- Signal: batch generation produces inconsistent lengths
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 178: model outputs malformed JSON in structured tasks
- Signal: model outputs malformed JSON in structured tasks
- Likely cause: continue_final_message misused with generation prompt
- Immediate action: switch to apply_chat_template(tokenize=True) for safety
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 179: stop sequence not respected
- Signal: stop sequence not respected
- Likely cause: special tokens duplicated during manual tokenization
- Immediate action: set explicit padding side and pad token policy
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 180: language mixing appears unexpectedly
- Signal: language mixing appears unexpectedly
- Likely cause: padding side incompatible with decode path
- Immediate action: align stop token IDs with tokenizer config
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

### Case 181: assistant continues user text instead of replying
- Signal: assistant continues user text instead of replying
- Likely cause: stop token IDs incomplete for model family
- Immediate action: lock tokenizer revision and verify in CI
- Verification metric: template render success rate remains 100%

```bash
python - <<'PY'
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
print('pad', t.pad_token, 'eos', t.eos_token, 'side', t.padding_side)
PY
```

### Case 182: responses truncate too early
- Signal: responses truncate too early
- Likely cause: template changed without retraining or migration checks
- Immediate action: run train-vs-inference formatting diff on sample set
- Verification metric: stop token match rate stays within expected threshold

```bash
python render_template_diff.py --tokenizer HuggingFaceH4/zephyr-7b-beta --samples data/chat_samples.jsonl
```

### Case 183: duplicate BOS/EOS tokens appear
- Signal: duplicate BOS/EOS tokens appear
- Likely cause: tokenizer revision drift between environments
- Immediate action: remove manual special token insertion where template already handles it
- Verification metric: response truncation rate remains low and stable

```bash
python validate_stop_tokens.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Case 184: template rendering throws runtime errors
- Signal: template rendering throws runtime errors
- Likely cause: missing embedding resize after adding tokens
- Immediate action: add strict tests for JSON and tool-calling output format
- Verification metric: train-eval tokenization divergence remains zero

```bash
python run_generation_regression.py --suite tests/golden_chat_outputs.json
```

### Case 185: train and inference outputs diverge
- Signal: train and inference outputs diverge
- Likely cause: add_generation_prompt misconfigured for inference
- Immediate action: render templates to plain text and inspect exact control tokens
- Verification metric: structured output validity rate remains within target

```bash
python tokenize_length_stats.py --input data/chat_samples.jsonl --out reports/token_lengths.json
```

## 8. Validation Drills
Complete every drill before promoting a change to production.

- [ ] Drill 001: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 002: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 003: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 004: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 005: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 006: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 007: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 008: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 009: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 010: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 011: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 012: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 013: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 014: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 015: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 016: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 017: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 018: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 019: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 020: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 021: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 022: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 023: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 024: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 025: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 026: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 027: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 028: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 029: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 030: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 031: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 032: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 033: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 034: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 035: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 036: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 037: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 038: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 039: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 040: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 041: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 042: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 043: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 044: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 045: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 046: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 047: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 048: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 049: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 050: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 051: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 052: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 053: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 054: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 055: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 056: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 057: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 058: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 059: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 060: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 061: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 062: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 063: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 064: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 065: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 066: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 067: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 068: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 069: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 070: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 071: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 072: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 073: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 074: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 075: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 076: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 077: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 078: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 079: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 080: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 081: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 082: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 083: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 084: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 085: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 086: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 087: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 088: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 089: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 090: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 091: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 092: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 093: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 094: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 095: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 096: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 097: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 098: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 099: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 100: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 101: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 102: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 103: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 104: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 105: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 106: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 107: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 108: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 109: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 110: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 111: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 112: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 113: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 114: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 115: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 116: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 117: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 118: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 119: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 120: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 121: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 122: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 123: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 124: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 125: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 126: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 127: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 128: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 129: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 130: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 131: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 132: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 133: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 134: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 135: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 136: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 137: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 138: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 139: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 140: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 141: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 142: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 143: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 144: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 145: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 146: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 147: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 148: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 149: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 150: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 151: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 152: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 153: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 154: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 155: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 156: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 157: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 158: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 159: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 160: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 161: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 162: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 163: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 164: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 165: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 166: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 167: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 168: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 169: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 170: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 171: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 172: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 173: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 174: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 175: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 176: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 177: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 178: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 179: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 180: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 181: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 182: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 183: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 184: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 185: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 186: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 187: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 188: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 189: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 190: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 191: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 192: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 193: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 194: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 195: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 196: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 197: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 198: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 199: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 200: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 201: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 202: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 203: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 204: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 205: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 206: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 207: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 208: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 209: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 210: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 211: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 212: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 213: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 214: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 215: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 216: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 217: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 218: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 219: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 220: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 221: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 222: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 223: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 224: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 225: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 226: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 227: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 228: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 229: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 230: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 231: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 232: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 233: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 234: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 235: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 236: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 237: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 238: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 239: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 240: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 241: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 242: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 243: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 244: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 245: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 246: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 247: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 248: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 249: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 250: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 251: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 252: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 253: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 254: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 255: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 256: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 257: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 258: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 259: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 260: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 261: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 262: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 263: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 264: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 265: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 266: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 267: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 268: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 269: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 270: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 271: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 272: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 273: Re-run regression pack after any tokenizer or template update.
- [ ] Drill 274: Render 100 sample chats and manually inspect control-token layout.
- [ ] Drill 275: Run add_generation_prompt=True/False comparison suite for inference endpoints.
- [ ] Drill 276: Run continue_final_message prefill tests for JSON tasks.
- [ ] Drill 277: Validate tokenizer revision consistency between local, CI, and production.
- [ ] Drill 278: Test stop token behavior on multilingual and long-context prompts.
- [ ] Drill 279: Run structured output validation for JSON and tool-calling messages.
- [ ] Drill 280: Re-run regression pack after any tokenizer or template update.

