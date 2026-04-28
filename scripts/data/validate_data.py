#!/usr/bin/env python3
"""
Validate dataset quality and format.

This script validates downloaded datasets for:
- Schema compliance
- Data quality issues
- Missing values
- Data type consistency
- Custom validation rules

Usage:
    python validate_data.py --input-dir ./data --schema schema.json
    python validate_data.py --input-file ./data/dataset.parquet --check-quality
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    input_path: str
    check_quality: bool = True
    check_schema: bool = True
    check_missing: bool = True
    check_duplicates: bool = True
    min_length: int = 1
    max_length: Optional[int] = None
    allowed_schemas: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validation check."""
    check_name: str
    passed: bool
    num_issues: int = 0
    details: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class DataValidator:
    """Validator for dataset quality and format."""

    def __init__(self, config: ValidationConfig):
        """Initialize validator with configuration.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.results: List[ValidationResult] = []

    def load_data(self) -> Any:
        """Load data from input path.

        Returns:
            Loaded dataset
        """
        path = Path(self.config.input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.config.input_path}")

        if path.suffix == ".parquet":
            try:
                import pandas as pd
                return pd.read_parquet(path)
            except ImportError:
                logger.error("pandas not installed. Install with: pip install pandas")
                raise
        elif path.suffix == ".json":
            try:
                import pandas as pd
                return pd.read_json(path)
            except ImportError:
                logger.error("pandas not installed")
                raise
        elif path.suffix == ".csv":
            try:
                import pandas as pd
                return pd.read_csv(path)
            except ImportError:
                logger.error("pandas not installed")
                raise
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def validate_schema(self, data: Any) -> ValidationResult:
        """Validate data schema.

        Args:
            data: Input dataset

        Returns:
            Validation result
        """
        result = ValidationResult(check_name="schema", passed=True)

        try:
            import pandas as pd
        except ImportError:
            result.passed = False
            result.details.append("pandas not available for schema validation")
            return result

        if not isinstance(data, pd.DataFrame):
            result.passed = False
            result.details.append("Data is not a DataFrame")
            return result

        columns = list(data.columns)

        if not columns:
            result.passed = False
            result.details.append("Dataset has no columns")
            result.suggestions.append("Ensure data was loaded correctly")
            return result

        result.details.append(f"Found {len(columns)} columns: {columns}")

        expected_columns = ["input", "output"]
        missing = [col for col in expected_columns if col not in columns]
        if missing:
            result.passed = False
            result.details.append(f"Missing expected columns: {missing}")
            result.suggestions.append(f"Add columns: {missing}")

        return result

    def validate_quality(self, data: Any) -> ValidationResult:
        """Validate data quality.

        Args:
            data: Input dataset

        Returns:
            Validation result
        """
        result = ValidationResult(check_name="quality", passed=True)

        try:
            import pandas as pd
        except ImportError:
            result.passed = False
            result.details.append("pandas not available for quality validation")
            return result

        if not isinstance(data, pd.DataFrame):
            result.passed = False
            result.details.append("Data is not a DataFrame")
            return result

        text_columns = data.select_dtypes(include=["object"]).columns

        for col in text_columns:
            empty_count = data[col].apply(lambda x: isinstance(x, str) and len(x.strip()) == 0).sum()
            if empty_count > 0:
                result.num_issues += empty_count
                result.details.append(f"Column '{col}': {empty_count} empty strings found")

            if self.config.min_length:
                short_count = data[col].apply(
                    lambda x: isinstance(x, str) and len(x) < self.config.min_length
                ).sum()
                if short_count > 0:
                    result.num_issues += short_count
                    result.details.append(
                        f"Column '{col}': {short_count} values shorter than {self.config.min_length}"
                    )

            if self.config.max_length:
                long_count = data[col].apply(
                    lambda x: isinstance(x, str) and len(x) > self.config.max_length
                ).sum()
                if long_count > 0:
                    result.num_issues += long_count
                    result.details.append(
                        f"Column '{col}': {long_count} values longer than {self.config.max_length}"
                    )

        if result.num_issues > 0:
            result.passed = False
            result.suggestions.append("Clean data before training")

        return result

    def validate_missing(self, data: Any) -> ValidationResult:
        """Check for missing values.

        Args:
            data: Input dataset

        Returns:
            Validation result
        """
        result = ValidationResult(check_name="missing", passed=True)

        try:
            import pandas as pd
        except ImportError:
            result.passed = False
            result.details.append("pandas not available for missing value check")
            return result

        if not isinstance(data, pd.DataFrame):
            result.passed = False
            return result

        for col in data.columns:
            null_count = data[col].isnull().sum()
            if null_count > 0:
                result.num_issues += null_count
                percentage = 100 * null_count / len(data)
                result.details.append(
                    f"Column '{col}': {null_count} null values ({percentage:.2f}%)"
                )

        if result.num_issues > 0:
            result.passed = False
            result.suggestions.append("Handle missing values or filter rows")

        return result

    def validate_duplicates(self, data: Any) -> ValidationResult:
        """Check for duplicate entries.

        Args:
            data: Input dataset

        Returns:
            Validation result
        """
        result = ValidationResult(check_name="duplicates", passed=True)

        try:
            import pandas as pd
        except ImportError:
            result.passed = False
            result.details.append("pandas not available for duplicate check")
            return result

        if not isinstance(data, pd.DataFrame):
            result.passed = False
            return result

        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            result.num_issues = duplicate_count
            percentage = 100 * duplicate_count / len(data)
            result.passed = False
            result.details.append(
                f"Found {duplicate_count} duplicate rows ({percentage:.2f}%)"
            )
            result.suggestions.append("Remove duplicates before training")

        return result

    def run_validation(self) -> Dict[str, Any]:
        """Run all validation checks.

        Returns:
            Dictionary with all validation results
        """
        logger.info(f"Starting validation for: {self.config.input_path}")

        data = self.load_data()
        logger.info(f"Loaded {len(data)} rows")

        checks = []

        if self.config.check_schema:
            checks.append(self.validate_schema(data))

        if self.config.check_quality:
            checks.append(self.validate_quality(data))

        if self.config.check_missing:
            checks.append(self.validate_missing(data))

        if self.config.check_duplicates:
            checks.append(self.validate_duplicates(data))

        self.results = checks

        summary = {
            "total_checks": len(checks),
            "passed": sum(1 for r in checks if r.passed),
            "failed": sum(1 for r in checks if not r.passed),
            "total_issues": sum(r.num_issues for r in checks),
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "num_issues": r.num_issues,
                    "details": r.details,
                    "suggestions": r.suggestions,
                }
                for r in checks
            ],
        }

        return summary

    def print_summary(self) -> None:
        """Print validation summary to console."""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 60)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n[{status}] {result.check_name.upper()}")
            print(f"  Issues found: {result.num_issues}")

            if result.details:
                print("  Details:")
                for detail in result.details[:5]:
                    print(f"    - {detail}")
                if len(result.details) > 5:
                    print(f"    ... and {len(result.details) - 5} more")

            if result.suggestions:
                print("  Suggestions:")
                for suggestion in result.suggestions:
                    print(f"    - {suggestion}")

        print("\n" + "=" * 60)


def main() -> int:
    """Main entry point for data validation."""
    parser = argparse.ArgumentParser(
        description="Validate dataset quality and format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python validate_data.py --input-file ./data/dataset.parquet

    # Full validation with checks
    python validate_data.py --input-dir ./data --check-quality --check-schema --check-missing

    # Validation with custom limits
    python validate_data.py --input-file ./data/dataset.parquet --min-length 10 --max-length 4096
        """
    )

    parser.add_argument(
        "--input-file",
        help="Input data file (parquet, json, or csv)"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing data files"
    )
    parser.add_argument(
        "--check-quality",
        action="store_true",
        help="Check data quality (length, empty values)"
    )
    parser.add_argument(
        "--check-schema",
        action="store_true",
        help="Validate schema compliance"
    )
    parser.add_argument(
        "--check-missing",
        action="store_true",
        help="Check for missing values"
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Check for duplicate entries"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum text length (default: 1)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum text length"
    )
    parser.add_argument(
        "--output-json",
        help="Output file for validation results (JSON)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output final summary"
    )

    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        parser.error("Either --input-file or --input-dir is required")

    input_path = args.input_file or args.input_dir

    if args.input_file and args.input_dir:
        logger.warning("Both --input-file and --input-dir specified, using --input-file")

    config = ValidationConfig(
        input_path=input_path,
        check_quality=args.check_quality,
        check_schema=args.check_schema,
        check_missing=args.check_missing,
        check_duplicates=args.check_duplicates,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    if not any([args.check_quality, args.check_schema, args.check_missing, args.check_duplicates]):
        config.check_quality = True
        config.check_schema = True
        config.check_missing = True
        config.check_duplicates = True

    try:
        validator = DataValidator(config)
        results = validator.run_validation()

        if not args.quiet:
            validator.print_summary()

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")

        all_passed = all(r.passed for r in validator.results)
        if all_passed:
            logger.info("All validation checks passed")
            return 0
        else:
            logger.warning("Some validation checks failed")
            return 1

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())