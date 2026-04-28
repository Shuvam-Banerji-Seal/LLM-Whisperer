#!/usr/bin/env python3
"""
Download and prepare datasets from HuggingFace.

This script handles downloading datasets from HuggingFace Hub,
preprocessing them, and saving in various formats suitable for
fine-tuning LLM models.

Usage:
    python download_dataset.py --dataset-name meta-llama/Llama-2-7b-hf --output-dir ./data
    python download_dataset.py --dataset-name openai/gsm8k --split train --output-dir ./data
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
class DatasetConfig:
    """Configuration for dataset download."""
    name: str
    output_dir: str
    split: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = True
    format: str = "parquet"
    max_size_mb: Optional[int] = None


def validate_dataset_name(name: str) -> bool:
    """Validate dataset name format.

    Args:
        name: Dataset name in format 'owner/dataset'

    Returns:
        True if valid, False otherwise
    """
    if not name:
        return False
    parts = name.split("/")
    if len(parts) != 2:
        return False
    return len(parts[0]) > 0 and len(parts[1]) > 0


def get_dataset_info(dataset_name: str, trust_remote_code: bool = True) -> Dict[str, Any]:
    """Get information about a dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset
        trust_remote_code: Whether to trust remote code

    Returns:
        Dictionary with dataset information
    """
    try:
        from huggingface_hub import dataset_info
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise

    info = dataset_info(dataset_name, trust_remote_code=trust_remote_code)
    return {
        "id": info.id,
        "tags": info.tags,
        "splits": getattr(info, "splits", []),
        "card_data": getattr(info, "card_data", {}),
    }


def download_dataset(config: DatasetConfig) -> Dict[str, Any]:
    """Download a dataset from HuggingFace.

    Args:
        config: Dataset configuration

    Returns:
        Dictionary with download results
    """
    logger.info(f"Downloading dataset: {config.name}")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        raise

    try:
        dataset = load_dataset(
            config.name,
            split=config.split,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    os.makedirs(config.output_dir, exist_ok=True)

    if config.format == "parquet":
        output_path = os.path.join(config.output_dir, config.name.replace("/", "_"))
        dataset.to_parquet(output_path)
        logger.info(f"Saved dataset to {output_path}")
    elif config.format == "json":
        output_path = os.path.join(config.output_dir, f"{config.name.replace('/', '_')}.json")
        dataset.to_json(output_path)
        logger.info(f"Saved dataset to {output_path}")
    elif config.format == "csv":
        output_path = os.path.join(config.output_dir, f"{config.name.replace('/', '_')}.csv")
        dataset.to_csv(output_path)
        logger.info(f"Saved dataset to {output_path}")

    metadata = {
        "dataset_name": config.name,
        "split": config.split,
        "num_examples": len(dataset),
        "format": config.format,
        "output_path": output_path,
        "features": dataset.features,
    }

    metadata_path = os.path.join(config.output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


def prepare_instruction_dataset(
    dataset: Any,
    input_field: str,
    output_field: str,
    system_field: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Prepare instruction-following dataset format.

    Args:
        dataset: Input dataset
        input_field: Field name for input text
        output_field: Field name for output text
        system_field: Optional field name for system prompt

    Returns:
        List of formatted examples
    """
    formatted = []

    for example in dataset:
        formatted_example = {
            "instruction": example.get(input_field, ""),
            "output": example.get(output_field, ""),
        }

        if system_field and system_field in example:
            formatted_example["system"] = example[system_field]

        formatted.append(formatted_example)

    return formatted


def split_dataset(
    dataset: Any,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Split dataset into train and validation sets.

    Args:
        dataset: Input dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed

    Returns:
        Dictionary with train and validation splits
    """
    if not 0 < train_ratio + val_ratio <= 1:
        raise ValueError("train_ratio + val_ratio must be between 0 and 1")

    try:
        from datasets import concatenate_datasets
    except ImportError:
        logger.error("datasets library not installed")
        raise

    dataset = dataset.shuffle(seed=seed)

    split_idx = int(len(dataset) * train_ratio)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))

    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


def validate_downloaded_data(output_dir: str) -> Dict[str, Any]:
    """Validate downloaded dataset files.

    Args:
        output_dir: Directory containing downloaded data

    Returns:
        Validation results
    """
    path = Path(output_dir)
    if not path.exists():
        return {"valid": False, "error": "Directory does not exist"}

    files = list(path.glob("*"))
    if not files:
        return {"valid": False, "error": "No files found"}

    metadata_path = path / "dataset_metadata.json"
    if not metadata_path.exists():
        return {"valid": False, "error": "Metadata file not found"}

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        return {"valid": False, "error": f"Failed to read metadata: {e}"}

    return {
        "valid": True,
        "num_files": len(files),
        "metadata": metadata,
    }


def main() -> int:
    """Main entry point for dataset download."""
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download full dataset
    python download_dataset.py --dataset-name meta-llama/Llama-2-7b-hf --output-dir ./data

    # Download specific split
    python download_dataset.py --dataset-name openai/gsm8k --split train --output-dir ./data

    # Download as JSON format
    python download_dataset.py --dataset-name yelp_review_full --format json --output-dir ./data

    # Download with size limit
    python download_dataset.py --dataset-name allenai/c4 --max-size-mb 1000 --output-dir ./data
        """
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help="HuggingFace dataset name (format: owner/dataset)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--split",
        help="Dataset split to download (e.g., train, test, validation)"
    )
    parser.add_argument(
        "--revision",
        help="Dataset revision to download"
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json", "csv"],
        default="parquet",
        help="Output format (default: parquet)"
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        help="Maximum dataset size in MB to download"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading dataset"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show dataset info without downloading"
    )

    args = parser.parse_args()

    if not validate_dataset_name(args.dataset_name):
        logger.error(f"Invalid dataset name format: {args.dataset_name}")
        return 1

    try:
        if args.info_only:
            info = get_dataset_info(args.dataset_name, args.trust_remote_code)
            print(json.dumps(info, indent=2))
            return 0

        config = DatasetConfig(
            name=args.dataset_name,
            output_dir=args.output_dir,
            split=args.split,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            format=args.format,
            max_size_mb=args.max_size_mb,
        )

        result = download_dataset(config)
        logger.info(f"Successfully downloaded dataset: {result['num_examples']} examples")

        validation = validate_downloaded_data(args.output_dir)
        if validation["valid"]:
            logger.info("Validation passed")
        else:
            logger.warning(f"Validation failed: {validation['error']}")

        print(json.dumps(result, indent=2, default=str))
        return 0

    except KeyboardInterrupt:
        logger.info("Download cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())