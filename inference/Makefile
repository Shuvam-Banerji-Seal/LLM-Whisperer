PYTHON ?= python
VENV ?= .venv

.PHONY: venv install bench-summary lint-scripts

venv:
	$(PYTHON) -m venv $(VENV)

install:
	. $(VENV)/bin/activate && pip install -r requirements.txt

bench-summary:
	$(PYTHON) benchmarking/scripts/summarize_results.py \
		--input benchmarking/results/latest.ndjson \
		--output benchmarking/results/latest_summary.json

lint-scripts:
	$(PYTHON) -m py_compile \
		engines/vllm/scripts/smoke_openai_chat.py \
		engines/tensorrt/scripts/smoke_openai_chat.py \
		engines/triton/scripts/smoke_infer.py \
		engines/onnxruntime/scripts/run_onnx_inference.py \
		engines/onnxruntime/scripts/optimize_transformer.py \
		engines/llama_cpp/scripts/smoke_openai_chat.py \
		engines/custom_dll/scripts/validate_plugin.py \
		benchmarking/scripts/run_openai_benchmark.py \
		benchmarking/scripts/summarize_results.py \
		benchmarking/scripts/compare_runs.py \
		serving/scripts/healthcheck.py \
		serving/scripts/mock_openai_server.py \
		serving/scripts/load_probe.py \
		quantization/scripts/quantize_bnb.py \
		quantization/scripts/quantize_gptq.py \
		quantization/scripts/quantize_awq.py
