#!/usr/bin/env python3
"""
Data augmentation for training.

This script provides various data augmentation techniques
for text data used in LLM training, including:
- Back-translation
- Synonym replacement
- Random insertion/deletion/swap
- Contextual augmentation

Usage:
    python augment_data.py --input-file ./data/train.parquet --output-dir ./data/augmented
    python augment_data.py --input-file ./data/train.parquet --method back-translation --lang en
"""

import argparse
import json
import logging
import os
import random
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
class AugmentationConfig:
    """Configuration for data augmentation."""
    input_path: str
    output_dir: str
    method: str = "synonym"
    augmentation_factor: float = 1.0
    seed: int = 42
    input_column: str = "input"
    output_column: str = "output"
    preserve_columns: List[str] = field(default_factory=list)


class TextAugmenter:
    """Text data augmentation."""

    def __init__(self, config: AugmentationConfig):
        """Initialize augmenter with configuration.

        Args:
            config: Augmentation configuration
        """
        self.config = config
        random.seed(config.seed)

    def load_data(self) -> Any:
        """Load data from input path.

        Returns:
            Loaded dataset
        """
        path = Path(self.config.input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.config.input_path}")

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed. Install with: pip install pandas")
            raise

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms.

        Args:
            text: Input text
            n: Number of words to replace

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < n:
            return text

        synonyms_map = {
            "good": ["great", "excellent", "wonderful"],
            "bad": ["terrible", "awful", "poor"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "compact"],
            "important": ["significant", "crucial", "essential"],
            "think": ["believe", "consider", "suppose"],
            "like": ["enjoy", "prefer", "appreciate"],
            "want": ["desire", "need", "wish"],
            "make": ["create", "produce", "generate"],
            "take": ["grab", "acquire", "obtain"],
        }

        indices = random.sample(range(len(words)), min(n, len(words)))
        for idx in indices:
            word = words[idx].lower()
            if word in synonyms_map:
                words[idx] = random.choice(synonyms_map[word])

        return " ".join(words)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert n words into text.

        Args:
            text: Input text
            n: Number of words to insert

        Returns:
            Augmented text
        """
        words = text.split()
        filler_words = ["the", "a", "an", "really", "very", "quite", "somewhat"]

        for _ in range(n):
            if not words:
                break
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(filler_words))

        return " ".join(words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p.

        Args:
            text: Input text
            p: Probability of deleting each word

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) == 1:
            return text

        remaining = [w for w in words if random.random() > p]
        if not remaining:
            return words[0]

        return " ".join(remaining)

    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap n pairs of words.

        Args:
            text: Input text
            n: Number of swaps

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            if len(words) < 2:
                break
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def back_translation(self, text: str, target_lang: str = "fr") -> str:
        """Perform back-translation augmentation.

        Args:
            text: Input text
            target_lang: Target language for translation

        Returns:
            Augmented text
        """
        try:
            from transformers import pipeline
        except ImportError:
            logger.warning("transformers not installed, skipping back-translation")
            return text

        try:
            translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_lang}")
            translated = translator(text, max_length=512)[0]["translation_text"]

            reverse_translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{target_lang}-en")
            back_translated = reverse_translator(translated, max_length=512)[0]["translation_text"]

            return back_translated
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text

    def contextual_augmentation(self, text: str) -> str:
        """Use a language model for contextual augmentation.

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        try:
            from transformers import pipeline
        except ImportError:
            logger.warning("transformers not installed, skipping contextual augmentation")
            return text

        try:
            fill_mask = pipeline("fill-mask", model="bert-base-uncased")
            words = text.split()

            for i, word in enumerate(words):
                if random.random() < 0.1 and word not in ["<pad>", "[PAD]"]:
                    masked_text = text.replace(word, "[MASK]", 1)
                    predictions = fill_mask(masked_text)
                    if predictions:
                        words[i] = predictions[0]["token_str"]

            return " ".join(words)
        except Exception as e:
            logger.warning(f"Contextual augmentation failed: {e}")
            return text

    def augment_text(self, text: str) -> str:
        """Apply configured augmentation method.

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        method = self.config.method

        if method == "synonym":
            return self.synonym_replacement(text, n=random.randint(1, 3))
        elif method == "random-insert":
            return self.random_insertion(text, n=random.randint(1, 2))
        elif method == "random-delete":
            return self.random_deletion(text, p=0.1)
        elif method == "random-swap":
            return self.random_swap(text, n=random.randint(1, 2))
        elif method == "back-translation":
            return self.back_translation(text)
        elif method == "contextual":
            return self.contextual_augmentation(text)
        else:
            logger.warning(f"Unknown augmentation method: {method}, returning original")
            return text

    def augment_dataset(self) -> Any:
        """Augment the entire dataset.

        Returns:
            Augmented dataset
        """
        logger.info(f"Loading data from: {self.config.input_path}")
        data = self.load_data()
        logger.info(f"Loaded {len(data)} examples")

        augmented_rows = []

        target_size = int(len(data) * self.config.augmentation_factor)

        while len(augmented_rows) < target_size:
            for idx, row in data.iterrows():
                if len(augmented_rows) >= target_size:
                    break

                new_row = row.to_dict()

                if self.config.input_column in new_row:
                    new_row[self.config.input_column] = self.augment_text(str(new_row[self.config.input_column]))

                if self.config.output_column in new_row:
                    new_row[self.config.output_column] = self.augment_text(str(new_row[self.config.output_column]))

                augmented_rows.append(new_row)

                if len(augmented_rows) % 100 == 0:
                    logger.info(f"Augmented {len(augmented_rows)} examples")

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed")
            raise

        augmented_data = pd.DataFrame(augmented_rows)

        os.makedirs(self.config.output_dir, exist_ok=True)

        output_path = Path(self.config.output_dir) / f"augmented_{Path(self.config.input_path).name}"
        if output_path.suffix == ".parquet":
            augmented_data.to_parquet(output_path)
        elif output_path.suffix == ".json":
            augmented_data.to_json(output_path)
        else:
            augmented_data.to_csv(output_path.with_suffix(".csv"))

        logger.info(f"Saved augmented data to: {output_path}")

        metadata = {
            "original_size": len(data),
            "augmented_size": len(augmented_data),
            "method": self.config.method,
            "augmentation_factor": self.config.augmentation_factor,
            "seed": self.config.seed,
        }

        metadata_path = Path(self.config.output_dir) / "augmentation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return augmented_data


def main() -> int:
    """Main entry point for data augmentation."""
    parser = argparse.ArgumentParser(
        description="Data augmentation for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic augmentation with synonym replacement
    python augment_data.py --input-file ./data/train.parquet --output-dir ./data/augmented

    # Back-translation augmentation
    python augment_data.py --input-file ./data/train.parquet --output-dir ./data/augmented --method back-translation

    # Augment with 2x the original size
    python augment_data.py --input-file ./data/train.parquet --output-dir ./data/augmented --augmentation-factor 2.0

    # Random swap augmentation
    python augment_data.py --input-file ./data/train.parquet --output-dir ./data/augmented --method random-swap
        """
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Input data file (parquet, json, or csv)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for augmented data"
    )
    parser.add_argument(
        "--method",
        choices=[
            "synonym",
            "random-insert",
            "random-delete",
            "random-swap",
            "back-translation",
            "contextual",
        ],
        default="synonym",
        help="Augmentation method (default: synonym)"
    )
    parser.add_argument(
        "--augmentation-factor",
        type=float,
        default=1.0,
        help="Factor to multiply dataset size by (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--input-column",
        default="input",
        help="Input column name (default: input)"
    )
    parser.add_argument(
        "--output-column",
        default="output",
        help="Output column name (default: output)"
    )
    parser.add_argument(
        "--back-translation-lang",
        default="fr",
        help="Target language for back-translation (default: fr)"
    )

    args = parser.parse_args()

    if args.augmentation_factor < 1.0:
        logger.warning("augmentation-factor must be >= 1.0, using 1.0")
        args.augmentation_factor = 1.0

    config = AugmentationConfig(
        input_path=args.input_file,
        output_dir=args.output_dir,
        method=args.method,
        augmentation_factor=args.augmentation_factor,
        seed=args.seed,
        input_column=args.input_column,
        output_column=args.output_column,
    )

    try:
        augmenter = TextAugmenter(config)
        result = augmenter.augment_dataset()

        metadata = {
            "method": config.method,
            "original_size": int(len(result) / config.augmentation_factor),
            "augmented_size": len(result),
            "augmentation_factor": config.augmentation_factor,
        }

        print(json.dumps(metadata, indent=2))
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Augmentation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())