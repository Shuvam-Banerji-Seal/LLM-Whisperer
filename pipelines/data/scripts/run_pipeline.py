"""End-to-end data pipeline orchestration script."""

import logging
import yaml
import argparse
import sys
from pathlib import Path

from src.ingestion import DataIngestion, IngestionConfig
from src.cleaning import DataCleaning, CleaningConfig
from src.preprocessing import DataPreprocessing, PreprocessingConfig
from src.splitting import DataSplitting, SplittingConfig
from src.validation import DataValidation, ValidationConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(config: dict):
    """Run complete data pipeline."""
    logger.info("=" * 80)
    logger.info("Starting Data Pipeline")
    logger.info("=" * 80)

    # Step 1: Ingestion
    logger.info("\n[STEP 1] Data Ingestion")
    logger.info("-" * 40)

    ingestion_config = IngestionConfig(**config["ingestion"])
    ingestion = DataIngestion()
    data = ingestion.load(ingestion_config)
    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"Statistics: {ingestion.get_statistics()}")

    # Step 2: Cleaning
    logger.info("\n[STEP 2] Data Cleaning")
    logger.info("-" * 40)

    cleaning_config = CleaningConfig(**config["cleaning"])
    cleaner = DataCleaning(cleaning_config)

    # Identify text columns
    text_columns = []
    for col in data.columns:
        if data[col].dtype == "object":
            text_columns.append(col)

    data = cleaner.clean(data, text_columns=text_columns if text_columns else None)
    logger.info(f"Data shape after cleaning: {data.shape}")
    logger.info(f"Cleaning stats: {cleaner.get_statistics()}")

    # Step 3: Preprocessing
    logger.info("\n[STEP 3] Data Preprocessing")
    logger.info("-" * 40)

    preprocessing_config = PreprocessingConfig(**config["preprocessing"])
    preprocessor = DataPreprocessing(preprocessing_config)
    logger.info(f"Tokenizer info: {preprocessor.get_tokenizer_info()}")

    data = preprocessor.preprocess(
        data, text_columns=text_columns if text_columns else [data.columns[0]]
    )
    logger.info(f"Data shape after preprocessing: {data.shape}")

    # Step 4: Splitting
    logger.info("\n[STEP 4] Data Splitting")
    logger.info("-" * 40)

    splitting_config = SplittingConfig(**config["splitting"])
    splitter = DataSplitting(splitting_config)
    splits = splitter.split(data)
    logger.info(f"Split info: {splitter.get_split_info()}")

    # Step 5: Validation
    logger.info("\n[STEP 5] Data Validation")
    logger.info("-" * 40)

    validation_config = ValidationConfig(**config["validation"])
    validator = DataValidation(validation_config)

    for split_name, split_data in splits.items():
        if isinstance(split_data, __import__("pandas").DataFrame):
            logger.info(f"\nValidating {split_name} split...")
            is_valid = validator.validate(split_data)
            logger.info(f"{split_name} split valid: {is_valid}")

    # Save results
    logger.info("\n[SAVE] Saving Results")
    logger.info("-" * 40)

    output_dir = config["output"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if config["output"]["save_splits"]:
        splitter.save_splits(output_dir)

    if config["output"]["save_report"]:
        report_path = f"{output_dir}/validation_report.json"
        validator.save_report(report_path)

    logger.info("\n" + "=" * 80)
    logger.info("Data Pipeline Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)
