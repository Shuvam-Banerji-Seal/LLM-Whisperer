"""Evaluation pipeline dashboarding module."""

import logging

logger = logging.getLogger(__name__)


def create_dashboard(metrics, output_dir="./dashboards"):
    """Create evaluation dashboard.

    Args:
        metrics: Evaluation metrics
        output_dir: Output directory for dashboard

    Returns:
        Path to created dashboard
    """
    logger.info(f"Creating dashboard in {output_dir}")

    return f"{output_dir}/index.html"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
