"""Data validation module for quality checks and statistics."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for data validation."""

    required_columns: List[str] = field(default_factory=list)
    min_samples: int = 1
    max_duplicates_percentage: float = 10.0
    max_missing_percentage: float = 10.0
    check_data_types: bool = True
    expected_dtypes: Optional[Dict[str, str]] = None


class DataValidation:
    """Data validation and quality assurance."""

    def __init__(self, config: ValidationConfig):
        """Initialize validation.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.validation_results = {}
        self.is_valid = True

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data against configuration.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        logger.info("Starting data validation...")

        self.is_valid = True
        self.validation_results = {
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Run validation checks
        self._check_shape(data)
        self._check_required_columns(data)
        self._check_minimum_samples(data)
        self._check_duplicates(data)
        self._check_missing_values(data)
        self._check_data_types(data)
        self._compute_statistics(data)

        # Log results
        self._log_validation_results()

        return self.is_valid

    def _check_shape(self, data: pd.DataFrame):
        """Check DataFrame shape."""
        shape = {
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "columns": list(data.columns),
        }
        self.validation_results["checks"]["shape"] = shape
        logger.info(
            f"Data shape: {shape['num_rows']} rows × {shape['num_columns']} columns"
        )

    def _check_required_columns(self, data: pd.DataFrame):
        """Check that required columns exist."""
        missing = []
        for col in self.config.required_columns:
            if col not in data.columns:
                missing.append(col)

        if missing:
            self.is_valid = False
            msg = f"Missing required columns: {missing}"
            self.validation_results["errors"].append(msg)
            logger.error(msg)
        else:
            self.validation_results["checks"]["required_columns"] = "OK"
            logger.info("All required columns present")

    def _check_minimum_samples(self, data: pd.DataFrame):
        """Check minimum number of samples."""
        num_samples = len(data)
        if num_samples < self.config.min_samples:
            self.is_valid = False
            msg = f"Insufficient samples: {num_samples} < {self.config.min_samples}"
            self.validation_results["errors"].append(msg)
            logger.error(msg)
        else:
            self.validation_results["checks"]["minimum_samples"] = "OK"
            logger.info(f"Sample count OK: {num_samples} samples")

    def _check_duplicates(self, data: pd.DataFrame):
        """Check for duplicate rows."""
        num_duplicates = data.duplicated().sum()
        percentage = (num_duplicates / len(data) * 100) if len(data) > 0 else 0

        check_result = {
            "num_duplicates": int(num_duplicates),
            "percentage": float(percentage),
        }
        self.validation_results["checks"]["duplicates"] = check_result

        if percentage > self.config.max_duplicates_percentage:
            msg = f"High duplicate percentage: {percentage:.2f}% > {self.config.max_duplicates_percentage}%"
            self.validation_results["warnings"].append(msg)
            logger.warning(msg)
        else:
            logger.info(f"Duplicate check OK: {percentage:.2f}%")

    def _check_missing_values(self, data: pd.DataFrame):
        """Check for missing values."""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data) * 100).to_dict()

        check_result = {
            "missing_counts": missing_counts.to_dict(),
            "missing_percentages": missing_percentages,
        }
        self.validation_results["checks"]["missing_values"] = check_result

        # Check if any column exceeds threshold
        for col, pct in missing_percentages.items():
            if pct > self.config.max_missing_percentage:
                msg = f"High missing percentage in '{col}': {pct:.2f}%"
                self.validation_results["warnings"].append(msg)
                logger.warning(msg)

        if not any(
            pct > self.config.max_missing_percentage
            for pct in missing_percentages.values()
        ):
            logger.info("Missing values check OK")

    def _check_data_types(self, data: pd.DataFrame):
        """Check data types."""
        current_dtypes = data.dtypes.to_dict()
        self.validation_results["checks"]["data_types"] = {
            str(k): str(v) for k, v in current_dtypes.items()
        }

        if self.config.check_data_types and self.config.expected_dtypes:
            for col, expected_dtype in self.config.expected_dtypes.items():
                if col in data.columns:
                    actual_dtype = str(data[col].dtype)
                    if actual_dtype != expected_dtype:
                        msg = f"Type mismatch in '{col}': {actual_dtype} != {expected_dtype}"
                        self.validation_results["warnings"].append(msg)
                        logger.warning(msg)

    def _compute_statistics(self, data: pd.DataFrame):
        """Compute dataset statistics."""
        stats = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "memory_usage_mb": float(data.memory_usage(deep=True).sum() / 1024**2),
            "numeric_columns": int(data.select_dtypes(include=[np.number]).shape[1]),
            "categorical_columns": int(data.select_dtypes(include=["object"]).shape[1]),
        }

        self.validation_results["checks"]["statistics"] = stats
        logger.info(f"Dataset statistics: {stats}")

    def _log_validation_results(self):
        """Log validation results."""
        if self.validation_results["errors"]:
            logger.error(
                f"Validation FAILED with {len(self.validation_results['errors'])} errors"
            )
            for error in self.validation_results["errors"]:
                logger.error(f"  - {error}")

        if self.validation_results["warnings"]:
            logger.warning(
                f"Validation passed with {len(self.validation_results['warnings'])} warnings"
            )
            for warning in self.validation_results["warnings"]:
                logger.warning(f"  - {warning}")

        if self.is_valid and not self.validation_results["warnings"]:
            logger.info("Validation PASSED")

    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report.

        Returns:
            Dictionary with validation report
        """
        report = {
            "is_valid": self.is_valid,
            "num_checks": len(self.validation_results.get("checks", {})),
            "num_errors": len(self.validation_results.get("errors", [])),
            "num_warnings": len(self.validation_results.get("warnings", [])),
            "details": self.validation_results,
        }

        return report

    def save_report(self, output_path: str):
        """Save validation report to file.

        Args:
            output_path: Path to save report (JSON or text)
        """
        import json

        report = self.generate_report()

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = ValidationConfig(
        required_columns=["text", "label"],
        min_samples=10,
        max_missing_percentage=5.0,
    )

    validator = DataValidation(config)

    # Create sample data
    df = pd.DataFrame(
        {
            "text": ["Sample"] * 50,
            "label": np.random.choice([0, 1], 50),
        }
    )

    is_valid = validator.validate(df)
    report = validator.generate_report()
    print(f"Valid: {is_valid}")
    print(f"Report: {report}")
