#!/usr/bin/env python3
"""
Merge LoRA adapters with base model.

This script merges trained LoRA adapters back into the base model
for easier deployment and inference. Supports various adapter types
including LoRA, QLoRA, and other PEFT adapters.

Usage:
    python merge_adapters.py --base-model meta-llama/Llama-2-7b --adapter ./adapter --output-dir ./merged
    python merge_adapters.py --base-model gpt2 --adapter ./lora_adapter --output-dir ./merged --safe-merge
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for adapter merging."""
    base_model: str
    adapter: str
    output_dir: str
    safe_merge: bool = True
    use_auth_token: Optional[str] = None
    trust_remote_code: bool = True
    replace_if_exists: bool = False


def validate_paths(config: MergeConfig) -> bool:
    """Validate input paths.

    Args:
        config: Merge configuration

    Returns:
        True if valid, False otherwise
    """
    if not config.base_model:
        logger.error("Base model path is required")
        return False

    if not config.adapter:
        logger.error("Adapter path is required")
        return False

    adapter_path = Path(config.adapter)
    if not adapter_path.exists():
        logger.error(f"Adapter path does not exist: {config.adapter}")
        return False

    return True


def get_adapter_info(adapter_path: str) -> Dict[str, Any]:
    """Get information about the adapter.

    Args:
        adapter_path: Path to adapter

    Returns:
        Adapter information
    """
    adapter_path_obj = Path(adapter_path)

    config_file = adapter_path_obj / "adapter_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
            return {
                "type": config.get("peft_type", "unknown"),
                "base_model_name_or_path": config.get("base_model_name_or_path", "unknown"),
                "revision": config.get("revision", None),
            }

    return {"type": "unknown"}


def load_base_model_and_tokenizer(config: MergeConfig) -> tuple:
    """Load base model and tokenizer.

    Args:
        config: Merge configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {config.base_model}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers")
        raise

    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        use_auth_token=config.use_auth_token,
        trust_remote_code=config.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        use_auth_token=config.use_auth_token,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    return model, tokenizer


def load_adapter(adapter_path: str) -> Any:
    """Load adapter using PEFT.

    Args:
        adapter_path: Path to adapter

    Returns:
        Loaded adapter
    """
    logger.info(f"Loading adapter from: {adapter_path}")

    try:
        from peft import PeftModel, PeftConfig
    except ImportError:
        logger.error("peft not installed. Install with: pip install peft")
        raise

    adapter_config = PeftConfig.from_pretrained(adapter_path)

    return adapter_config


def merge_lora_adapter(
    model: Any,
    adapter_path: str,
    config: MergeConfig,
) -> Dict[str, Any]:
    """Merge LoRA adapter into base model.

    Args:
        model: Base model
        adapter_path: Path to adapter
        config: Merge configuration

    Returns:
        Merge metadata
    """
    logger.info("Merging LoRA adapter")

    try:
        from peft import PeftModel
    except ImportError:
        logger.error("peft not installed")
        raise

    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        raise

    peft_model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )

    logger.info("Merging adapter weights with base model")

    merged_model = peft_model.merge_and_unload()

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info(f"Saving merged model to: {config.output_dir}")

    merged_model.save_pretrained(config.output_dir)

    return {
        "adapter_type": "lora",
        "adapter_path": adapter_path,
        "base_model": config.base_model,
        "output_dir": config.output_dir,
        "merged": True,
    }


def merge_qlora_adapter(
    model: Any,
    adapter_path: str,
    config: MergeConfig,
) -> Dict[str, Any]:
    """Merge QLoRA adapter into base model.

    Args:
        model: Base model
        adapter_path: Path to adapter
        config: Merge configuration

    Returns:
        Merge metadata
    """
    logger.info("Merging QLoRA adapter")

    try:
        from peft import PeftModel
    except ImportError:
        logger.error("peft not installed")
        raise

    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.error("Required packages not installed")
        raise

    peft_model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )

    logger.info("Merging QLoRA adapter weights")

    merged_model = peft_model.merge_and_unload()

    os.makedirs(config.output_dir, exist_ok=True)

    merged_model.save_pretrained(config.output_dir)

    return {
        "adapter_type": "qlora",
        "adapter_path": adapter_path,
        "base_model": config.base_model,
        "output_dir": config.output_dir,
        "merged": True,
    }


def safe_merge(
    model: Any,
    adapter_path: str,
    config: MergeConfig,
) -> Dict[str, Any]:
    """Safely merge adapter with validation.

    Args:
        model: Base model
        adapter_path: Path to adapter
        config: Merge configuration

    Returns:
        Merge metadata
    """
    logger.info("Performing safe merge")

    adapter_info = get_adapter_info(adapter_path)
    adapter_type = adapter_info.get("type", "LORA").upper()

    if "QLORA" in adapter_type or "QLORA" in adapter_type:
        result = merge_qlora_adapter(model, adapter_path, config)
    else:
        result = merge_lora_adapter(model, adapter_path, config)

    result["safe_merge"] = True

    return result


def validate_merged_model(output_dir: str) -> bool:
    """Validate merged model files.

    Args:
        output_dir: Output directory

    Returns:
        True if valid, False otherwise
    """
    path = Path(output_dir)

    required_files = ["model.safetensors", "tokenizer_config.json"]
    optional_files = ["config.json"]

    has_model = any((path / f).exists() for f in ["model.safetensors", "pytorch_model.bin", "model.bin"])
    has_tokenizer = (path / "tokenizer_config.json").exists()

    return has_model and has_tokenizer


def main() -> int:
    """Main entry point for adapter merging."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic merge
    python merge_adapters.py --base-model meta-llama/Llama-2-7b --adapter ./adapter --output-dir ./merged

    # Safe merge with verification
    python merge_adapters.py --base-model gpt2 --adapter ./lora_adapter --output-dir ./merged --safe-merge

    # Merge with auth token
    python merge_adapters.py --base-model meta-llama/Llama-2-7b --adapter ./adapter --output-dir ./merged --use-auth-token TOKEN
        """
    )

    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model ID or local path"
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--safe-merge",
        action="store_true",
        default=True,
        help="Perform safe merge with validation"
    )
    parser.add_argument(
        "--replace-if-exists",
        action="store_true",
        help="Replace output if it already exists"
    )
    parser.add_argument(
        "--use-auth-token",
        help="HuggingFace auth token for private models"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code"
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    if output_path.exists() and not args.replace_if_exists:
        logger.error(f"Output directory already exists: {args.output_dir}")
        logger.info("Use --replace-if-exists to overwrite")
        return 1

    config = MergeConfig(
        base_model=args.base_model,
        adapter=args.adapter,
        output_dir=args.output_dir,
        safe_merge=args.safe_merge,
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
        replace_if_exists=args.replace_if_exists,
    )

    if not validate_paths(config):
        return 1

    try:
        adapter_info = get_adapter_info(args.adapter)
        logger.info(f"Adapter info: {adapter_info}")

        model, tokenizer = load_base_model_and_tokenizer(config)

        _ = load_adapter(args.adapter)

        if args.safe_merge:
            result = safe_merge(model, args.adapter, config)
        else:
            result = merge_lora_adapter(model, args.adapter, config)

        tokenizer.save_pretrained(args.output_dir)

        if validate_merged_model(args.output_dir):
            logger.info("Merged model validation passed")
        else:
            logger.warning("Merged model validation failed")

        print(json.dumps(result, indent=2))
        return 0

    except KeyboardInterrupt:
        logger.info("Merge cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())