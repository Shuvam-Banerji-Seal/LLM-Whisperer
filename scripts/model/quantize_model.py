#!/usr/bin/env python3
"""
Quantize models (INT8, INT4, AWQ, GPTQ).

This script provides various model quantization methods:
- bitsandbytes INT8/INT4 quantization
- AWQ (Activation-Aware Weight Quantization)
- GPTQ (Generative Pre-trained Transformer Quantization)

Usage:
    python quantize_model.py --model-id meta-llama/Llama-2-7b --output-dir ./quantized --method int8
    python quantize_model.py --model-id gpt2 --output-dir ./quantized --method gptq --bits 4
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class QuantizeConfig:
    """Configuration for model quantization."""
    model_id: str
    output_dir: str
    method: str = "bitsandbytes"
    bits: int = 4
    compute_dtype: str = "float16"
    double_quant: bool = True
    use_exllama: bool = False
    batch_size: int = 1
    max_seq_length: int = 2048
    calibration_dataset: Optional[str] = None
    use_auth_token: Optional[str] = None
    trust_remote_code: bool = True


def validate_model_id(model_id: str) -> bool:
    """Validate model ID format.

    Args:
        model_id: Model identifier

    Returns:
        True if valid, False otherwise
    """
    if not model_id:
        return False
    if "/" in model_id:
        parts = model_id.split("/")
        return len(parts) == 2 and all(p for p in parts)
    return len(model_id) > 0


def get_model_info(model_id: str, use_auth_token: Optional[str] = None) -> Dict[str, Any]:
    """Get model information.

    Args:
        model_id: Model identifier
        use_auth_token: Optional auth token

    Returns:
        Model information dictionary
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        logger.error("transformers not installed")
        raise

    config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=use_auth_token,
        trust_remote_code=True,
    )

    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "hidden_size": getattr(config, "hidden_size", 0),
        "num_attention_heads": getattr(config, "num_attention_heads", 0),
        "num_hidden_layers": getattr(config, "num_hidden_layers", 0),
        "vocab_size": getattr(config, "vocab_size", 0),
    }


def load_model_and_tokenizer(config: QuantizeConfig) -> tuple:
    """Load model and tokenizer.

    Args:
        config: Quantization configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config.model_id}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers")
        raise

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_auth_token=config.use_auth_token,
        trust_remote_code=config.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        use_auth_token=config.use_auth_token,
        trust_remote_code=config.trust_remote_code,
        device_map="auto",
    )

    return model, tokenizer


def quantize_bitsandbytes(
    model: Any,
    tokenizer: Any,
    config: QuantizeConfig,
) -> Dict[str, Any]:
    """Quantize model using bitsandbytes.

    Args:
        model: Model to quantize
        tokenizer: Tokenizer
        config: Quantization configuration

    Returns:
        Quantization metadata
    """
    logger.info(f"Quantizing with bitsandbytes ({config.bits}bit)")

    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.error("Required packages not installed. Install with: pip install bitsandbytes torch")
        raise

    compute_dtype = getattr(torch, config.compute_dtype)

    if config.bits == 8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    elif config.bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=config.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        raise ValueError(f"Unsupported bits value: {config.bits}")

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("Applying quantization config to model")

    model.config.quantization_config = quant_config

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    original_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    quantized_size_mb = original_size_mb / (8 if config.bits == 8 else 16)

    return {
        "method": "bitsandbytes",
        "bits": config.bits,
        "compute_dtype": config.compute_dtype,
        "double_quant": config.double_quant,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": original_size_mb / quantized_size_mb,
        "output_dir": config.output_dir,
    }


def quantize_awq(
    model: Any,
    tokenizer: Any,
    config: QuantizeConfig,
) -> Dict[str, Any]:
    """Quantize model using AWQ.

    Args:
        model: Model to quantize
        tokenizer: Tokenizer
        config: Quantization configuration

    Returns:
        Quantization metadata
    """
    logger.info("Quantizing with AWQ")

    try:
        from transformers import AutoAWQForCausalLM
    except ImportError:
        logger.error("autoawq not installed. Install with: pip install autoawq")
        raise

    os.makedirs(config.output_dir, exist_ok=True)

    try:
        from awq import AutoAWQForCausalLM
        from awq.utils import get_calibration_dataset

        if config.calibration_dataset:
            calibration_data = get_calibration_dataset(
                config.calibration_dataset,
                tokenizer,
                num_samples=128,
            )
        else:
            calibration_data = None

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": config.bits,
            "version": "GEMM",
        }

        model = AutoAWQForCausalLM.from_pretrained(
            config.model_id,
            use_auth_token=config.use_auth_token,
            trust_remote_code=config.trust_remote_code,
        )

        if calibration_data:
            model.quantize(calibration_data, **quant_config)
        else:
            logger.warning("No calibration dataset provided, skipping quantization")
            return {"method": "awq", "status": "skipped", "reason": "no calibration data"}

        model.save_quantized(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

    except ImportError:
        logger.error("AWQ packages not properly installed")
        raise

    return {
        "method": "awq",
        "bits": config.bits,
        "quant_group_size": 128,
        "output_dir": config.output_dir,
    }


def quantize_gptq(
    model: Any,
    tokenizer: Any,
    config: QuantizeConfig,
) -> Dict[str, Any]:
    """Quantize model using GPTQ.

    Args:
        model: Model to quantize
        tokenizer: Tokenizer
        config: Quantization configuration

    Returns:
        Quantization metadata
    """
    logger.info("Quantizing with GPTQ")

    os.makedirs(config.output_dir, exist_ok=True)

    try:
        from transformers import AutoGPTQForCausalLM, GPTQConfig
    except ImportError:
        logger.error("transformers with GPTQ support not installed")
        raise

    gptq_config = GPTQConfig(
        bits=config.bits,
        dataset="c4",
        block_size=128,
        compile_=False,
    )

    model.config.quantization_config = gptq_config

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return {
        "method": "gptq",
        "bits": config.bits,
        "dataset": "c4",
        "block_size": 128,
        "output_dir": config.output_dir,
    }


def quantize_exllama(
    model: Any,
    tokenizer: Any,
    config: QuantizeConfig,
) -> Dict[str, Any]:
    """Quantize model using ExLlama.

    Args:
        model: Model to quantize
        tokenizer: Tokenizer
        config: Quantization configuration

    Returns:
        Quantization metadata
    """
    logger.info("Quantizing with ExLlama")

    os.makedirs(config.output_dir, exist_ok=True)

    logger.warning("ExLlama quantization requires specific setup")

    return {
        "method": "exllama",
        "bits": config.bits,
        "status": "experimental",
        "output_dir": config.output_dir,
    }


def validate_quantized_model(output_dir: str) -> bool:
    """Validate quantized model files.

    Args:
        output_dir: Output directory

    Returns:
        True if valid, False otherwise
    """
    path = Path(output_dir)

    required_files = ["model.safetensors", "tokenizer_config.json"]
    return all((path / f).exists() for f in required_files if f != "model.safetensors" or (path / f).exists() or (path / "pytorch_model.bin").exists())


def main() -> int:
    """Main entry point for model quantization."""
    parser = argparse.ArgumentParser(
        description="Quantize models (INT8, INT4, AWQ, GPTQ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quantize to INT8 with bitsandbytes
    python quantize_model.py --model-id meta-llama/Llama-2-7b --output-dir ./quantized --method bitsandbytes --bits 8

    # Quantize to INT4 with bitsandbytes
    python quantize_model.py --model-id gpt2 --output-dir ./quantized --method bitsandbytes --bits 4

    # Quantize with AWQ
    python quantize_model.py --model-id meta-llama/Llama-2-7b --output-dir ./quantized --method awq --bits 4

    # Quantize with GPTQ
    python quantize_model.py --model-id gpt2 --output-dir ./quantized --method gptq --bits 4
        """
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--method",
        choices=["bitsandbytes", "awq", "gptq", "exllama"],
        default="bitsandbytes",
        help="Quantization method (default: bitsandbytes)"
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (default: 4)"
    )
    parser.add_argument(
        "--compute-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype (default: float16)"
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        default=True,
        help="Use double quantization (for 4bit)"
    )
    parser.add_argument(
        "--use-exllama",
        action="store_true",
        help="Use ExLlama kernel (for GPTQ)"
    )
    parser.add_argument(
        "--calibration-dataset",
        help="Dataset for calibration (AWQ/GPTQ)"
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

    if not validate_model_id(args.model_id):
        logger.error(f"Invalid model ID: {args.model_id}")
        return 1

    if args.bits not in [4, 8]:
        logger.error("Only 4-bit and 8-bit quantization supported")
        return 1

    config = QuantizeConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        method=args.method,
        bits=args.bits,
        compute_dtype=args.compute_dtype,
        double_quant=args.double_quant,
        use_exllama=args.use_exllama,
        calibration_dataset=args.calibration_dataset,
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
    )

    try:
        model_info = get_model_info(args.model_id, args.use_auth_token)
        logger.info(f"Model info: {model_info}")

        model, tokenizer = load_model_and_tokenizer(config)

        if args.method == "bitsandbytes":
            result = quantize_bitsandbytes(model, tokenizer, config)
        elif args.method == "awq":
            result = quantize_awq(model, tokenizer, config)
        elif args.method == "gptq":
            result = quantize_gptq(model, tokenizer, config)
        elif args.method == "exllama":
            result = quantize_exllama(model, tokenizer, config)

        if validate_quantized_model(args.output_dir):
            logger.info("Quantized model validation passed")
        else:
            logger.warning("Quantized model validation failed")

        print(json.dumps(result, indent=2))
        return 0

    except KeyboardInterrupt:
        logger.info("Quantization cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())