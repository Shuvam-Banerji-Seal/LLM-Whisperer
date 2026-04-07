"""Data preprocessing module for tokenization and format conversion."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    tokenizer_name: str = "gpt2"  # Can be "gpt2", "bert", or path
    max_seq_length: int = 512
    truncation_strategy: str = (
        "longest_first"  # "longest_first", "only_first", "only_second"
    )
    padding_strategy: str = "max_length"  # "max_length", "longest"
    format_type: str = "standard"  # "standard", "instruction_tuning", "conversation"
    instruction_template: Optional[str] = None
    response_template: Optional[str] = None


class DataPreprocessing:
    """Data preprocessing for model input preparation."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize preprocessing.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required for preprocessing")

        logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(
        self,
        data: pd.DataFrame,
        text_columns: List[str],
        label_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Preprocess data for model input.

        Args:
            data: Input DataFrame
            text_columns: Columns containing text to process
            label_column: Optional column with labels

        Returns:
            Preprocessed DataFrame with input_ids, attention_mask, etc.
        """
        df = data.copy()

        if self.config.format_type == "instruction_tuning":
            df = self._prepare_instruction_tuning(df, text_columns)
        elif self.config.format_type == "conversation":
            df = self._prepare_conversation(df, text_columns)
        else:
            df = self._prepare_standard(df, text_columns)

        logger.info(f"Preprocessed {len(df)} examples")

        return df

    def _prepare_standard(
        self, df: pd.DataFrame, text_columns: List[str]
    ) -> pd.DataFrame:
        """Prepare standard format (text only)."""
        # Combine text columns
        df["text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

        # Tokenize
        encodings = self.tokenizer(
            df["text"].tolist(),
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=self.config.padding_strategy,
            return_tensors=None,
        )

        # Add tokenized features
        df["input_ids"] = encodings["input_ids"]
        df["attention_mask"] = encodings["attention_mask"]
        if "token_type_ids" in encodings:
            df["token_type_ids"] = encodings["token_type_ids"]

        return df

    def _prepare_instruction_tuning(
        self, df: pd.DataFrame, text_columns: List[str]
    ) -> pd.DataFrame:
        """Prepare instruction tuning format."""
        if len(text_columns) < 2:
            logger.warning(
                "Instruction tuning requires at least 2 text columns (instruction, response)"
            )
            return self._prepare_standard(df, text_columns)

        instruction_col = text_columns[0]
        response_col = text_columns[1]

        instruction_template = (
            self.config.instruction_template or "### Instruction:\n{}\n\n"
        )
        response_template = self.config.response_template or "### Response:\n{}"

        texts = []
        for _, row in df.iterrows():
            instruction = row.get(instruction_col, "")
            response = row.get(response_col, "")

            text = instruction_template.format(instruction) + response_template.format(
                response
            )
            texts.append(text)

        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=self.config.padding_strategy,
            return_tensors=None,
        )

        df["input_ids"] = encodings["input_ids"]
        df["attention_mask"] = encodings["attention_mask"]

        return df

    def _prepare_conversation(
        self, df: pd.DataFrame, text_columns: List[str]
    ) -> pd.DataFrame:
        """Prepare conversation format (multi-turn dialogue)."""
        # For conversation format, expect alternating user/assistant messages
        texts = []

        for _, row in df.iterrows():
            dialogue_parts = []

            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    dialogue_parts.append(str(row[col]))

            text = "\n".join(dialogue_parts)
            texts.append(text)

        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=self.config.padding_strategy,
            return_tensors=None,
        )

        df["input_ids"] = encodings["input_ids"]
        df["attention_mask"] = encodings["attention_mask"]

        return df

    def decode(self, input_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            input_ids: List of token IDs

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get tokenizer information."""
        return {
            "name": self.config.tokenizer_name,
            "vocab_size": self.tokenizer.vocab_size,
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "bos_token": self.tokenizer.bos_token,
            "max_seq_length": self.config.max_seq_length,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = PreprocessingConfig(
        tokenizer_name="gpt2", max_seq_length=512, format_type="standard"
    )

    preprocessor = DataPreprocessing(config)

    df = pd.DataFrame(
        {
            "text": [
                "Hello world",
                "This is a test",
            ]
        }
    )

    processed = preprocessor.preprocess(df, text_columns=["text"])
    print(preprocessor.get_tokenizer_info())
    print(processed.head())
