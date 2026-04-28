#!/usr/bin/env python3
"""
Export models to various formats (ONNX, TorchScript).

This script handles model export for deployment:
- ONNX export for cross-platform inference
- TorchScript export for PyTorch serving
- Exported models are optimized for inference

Usage:
    python export_model.py --model-id meta-llama/Llama-2-7b --output-dir ./exported --format onnx
    python export_model.py --model-id gpt2 --output-dir ./exported --format torchscript
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    model_id: str
    output_dir: str
    format: str = "onnx"
    opset_version: int = 14
    optimize: bool = True
    quantize: bool = False
    batch_size: int = 1
    max_seq_length: int = 2048
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


def load_model_for_export(config: ExportConfig) -> tuple:
    """Load model and tokenizer for export.

    Args:
        config: Export configuration

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
    )

    model.eval()

    return model, tokenizer


def export_to_onnx(
    model: Any,
    tokenizer: Any,
    config: ExportConfig,
) -> Dict[str, Any]:
    """Export model to ONNX format.

    Args:
        model: Model to export
        tokenizer: Tokenizer
        config: Export configuration

    Returns:
        Export metadata
    """
    logger.info("Exporting to ONNX format")

    try:
        import torch
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.info("Install with: pip install torch optimum")
        raise

    os.makedirs(config.output_dir, exist_ok=True)

    if hasattr(model, "config"):
        hidden_size = getattr(model.config, "hidden_size", 768)
        num_attention_heads = getattr(model.config, "num_attention_heads", 12)
        num_layers = getattr(model.config, "num_hidden_layers", 12)
        vocab_size = getattr(model.config, "vocab_size", 50257)

        logger.info(f"Model config: hidden_size={hidden_size}, heads={num_attention_heads}, layers={num_layers}")

    sample_text = "Hello, world!"
    inputs = tokenizer(sample_text, return_tensors="pt")

    onnx_path = Path(config.output_dir) / "model.onnx"

    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from optimum.onnxruntime.configuration import AutoOptimizationConfig

        ort_model = ORTModelForCausalLM.from_pretrained(
            config.model_id,
            export=True,
            opset=config.opset_version,
        )

        opt_config = AutoOptimizationConfig(
            optimization_level=2 if config.optimize else 0,
        )

        optimized_model = ort_model.optimize(opt_config)

        optimized_model.save_pretrained(config.output_dir)

        logger.info(f"ONNX model saved to: {onnx_path}")

    except Exception as e:
        logger.warning(f"Optimum export failed, using basic ONNX export: {e}")

        model_path = Path(config.output_dir) / "model.onnx"
        dummy_input = tokenizer(sample_text, return_tensors="pt")

        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input.get("attention_mask")),
            str(model_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=config.opset_version,
        )

        logger.info(f"ONNX model exported to: {model_path}")

    return {
        "format": "onnx",
        "opset_version": config.opset_version,
        "optimized": config.optimize,
        "path": str(onnx_path),
    }


def export_to_torchscript(
    model: Any,
    tokenizer: Any,
    config: ExportConfig,
) -> Dict[str, Any]:
    """Export model to TorchScript format.

    Args:
        model: Model to export
        tokenizer: Tokenizer
        config: Export configuration

    Returns:
        Export metadata
    """
    logger.info("Exporting to TorchScript format")

    try:
        import torch
    except ImportError:
        logger.error("torch not installed")
        raise

    os.makedirs(config.output_dir, exist_ok=True)

    model.eval()

    example_input = tokenizer(
        "Hello, world!",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_seq_length,
    )

    traced_model_path = Path(config.output_dir) / "model.pt"

    try:
        traced_model = torch.jit.trace(
            model,
            example_kwarg_inputs=example_input,
        )

        traced_model.save(str(traced_model_path))
        logger.info(f"TorchScript model saved to: {traced_model_path}")

    except Exception as e:
        logger.warning(f"Tracing failed, using script: {e}")

        scripted_model = torch.jit.script(model)
        scripted_model.save(str(traced_model_path))
        logger.info(f"TorchScript model saved to: {traced_model_path}")

    tokenizer_path = Path(config.output_dir) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)

    return {
        "format": "torchscript",
        "path": str(traced_model_path),
        "tokenizer_path": str(tokenizer_path),
    }


def export_to_safetensors(
    model: Any,
    tokenizer: Any,
    config: ExportConfig,
) -> Dict[str, Any]:
    """Export model to SafeTensors format.

    Args:
        model: Model to export
        tokenizer: Tokenizer
        config: Export configuration

    Returns:
        Export metadata
    """
    logger.info("Exporting to SafeTensors format")

    try:
        from safetensors.torch import save_file
    except ImportError:
        logger.error("safetensors not installed. Install with: pip install safetensors")
        raise

    os.makedirs(config.output_dir, exist_ok=True)

    state_dict = model.state_dict()

    safetensors_path = Path(config.output_dir) / "model.safetensors"
    save_file(state_dict, safetensors_path)

    tokenizer_path = Path(config.output_dir) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)

    num_params = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    logger.info(f"SafeTensors model saved to: {safetensors_path}")
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Model size: {param_size_mb:.2f} MB")

    return {
        "format": "safetensors",
        "path": str(safetensors_path),
        "tokenizer_path": str(tokenizer_path),
        "num_parameters": num_params,
        "size_mb": param_size_mb,
    }


def validate_exported_model(output_dir: str, format: str) -> bool:
    """Validate exported model files.

    Args:
        output_dir: Output directory
        format: Export format

    Returns:
        True if valid, False otherwise
    """
    path = Path(output_dir)

    if format == "onnx":
        return (path / "model.onnx").exists()
    elif format == "torchscript":
        return (path / "model.pt").exists() and (path / "tokenizer").exists()
    elif format == "safetensors":
        return (path / "model.safetensors").exists() and (path / "tokenizer").exists()

    return False


def main() -> int:
    """Main entry point for model export."""
    parser = argparse.ArgumentParser(
        description="Export models to various formats (ONNX, TorchScript)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export to ONNX
    python export_model.py --model-id meta-llama/Llama-2-7b --output-dir ./exported --format onnx

    # Export to TorchScript
    python export_model.py --model-id gpt2 --output-dir ./exported --format torchscript

    # Export to SafeTensors
    python export_model.py --model-id bert-base-uncased --output-dir ./exported --format safetensors

    # Export with optimization
    python export_model.py --model-id gpt2 --output-dir ./exported --format onnx --optimize
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
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript", "safetensors"],
        default="onnx",
        help="Export format (default: onnx)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize exported model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export (default: 1)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
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

    config = ExportConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        format=args.format,
        opset_version=args.opset_version,
        optimize=args.optimize,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
    )

    try:
        model, tokenizer = load_model_for_export(config)

        if args.format == "onnx":
            result = export_to_onnx(model, tokenizer, config)
        elif args.format == "torchscript":
            result = export_to_torchscript(model, tokenizer, config)
        elif args.format == "safetensors":
            result = export_to_safetensors(model, tokenizer, config)

        if validate_exported_model(args.output_dir, args.format):
            logger.info("Model validation passed")
        else:
            logger.warning("Model validation failed")

        print(json.dumps(result, indent=2))
        return 0

    except KeyboardInterrupt:
        logger.info("Export cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())