"""LoRA configuration templates."""

from fine_tuning.lora import LoRAConfig


def gpt2_lora_config(**kwargs) -> LoRAConfig:
    \"\"\"LoRA config for GPT-2.\"\"\"
    config = LoRAConfig(
        model_name=\"gpt2\",
        output_dir=\"./gpt2_lora\",
        r=4,
        lora_alpha=8,\n        target_modules=[\"c_attn\"],\n        **kwargs,\n    )\n    return config\n\n\ndef mistral_lora_config(**kwargs) -> LoRAConfig:\n    \"\"\"LoRA config for Mistral-7B.\"\"\"\n    config = LoRAConfig(\n        model_name=\"mistralai/Mistral-7B\",\n        output_dir=\"./mistral_lora\",\n        r=16,\n        lora_alpha=32,\n        target_modules=[\"q_proj\", \"v_proj\"],\n        **kwargs,\n    )\n    return config\n\n\ndef llama_lora_config(**kwargs) -> LoRAConfig:\n    \"\"\"LoRA config for Llama-7B.\"\"\"\n    config = LoRAConfig(\n        model_name=\"meta-llama/Llama-2-7b\",\n        output_dir=\"./llama_lora\",\n        r=16,\n        lora_alpha=32,\n        target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],\n        **kwargs,\n    )\n    return config\n"