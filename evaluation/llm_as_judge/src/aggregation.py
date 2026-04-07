"""
Judge Aggregation and Calibration

Utilities for aggregating judgments and measuring inter-rater agreement.
"""

import statistics
from typing import Dict, List
from collections import Counter
import math


class JudgmentAggregator:
    """Aggregate judgments from multiple judges."""

    @staticmethod
    def mean_score(scores: List[int]) -> float:
        """Compute mean judgment score."""
        return statistics.mean(scores) if scores else 0.0

    @staticmethod
    def median_score(scores: List[int]) -> int:
        """Compute median judgment score."""
        return int(statistics.median(scores)) if scores else 0

    @staticmethod
    def std_dev(scores: List[int]) -> float:
        """Compute standard deviation of scores."""
        return statistics.stdev(scores) if len(scores) > 1 else 0.0

    @staticmethod
    def fleiss_kappa(judgments: Dict[int, List[int]]) -> float:
        """
        Compute Fleiss' kappa for inter-rater agreement.

        Args:
            judgments: Dict mapping content_id -> list of scores (1-5)

        Returns:
            Kappa value (-1 to 1): >0.81 excellent, >0.61 substantial, >0.41 moderate, >0.21 fair
        """
        if not judgments:
            return 0.0

        # Flatten all scores
        all_scores = []
        for scores in judgments.values():
            all_scores.extend(scores)

        if not all_scores:
            return 0.0

        N = len(judgments)  # Number of items
        m = len(all_scores) // N if N > 0 else 1  # Number of judges

        # Count occurrences of each category (1-5)
        k = 5  # Number of categories

        # Compute observed agreement
        p_o = 0.0
        for scores in judgments.values():
            counts = Counter(scores)
            for count in counts.values():
                if m > 1:
                    p_o += count * (count - 1) / (m * (m - 1))

        p_o = p_o / N if N > 0 else 0

        # Compute chance agreement
        p_e = 0.0
        category_counts = Counter(all_scores)
        for count in category_counts.values():
            p_e += (count / len(all_scores)) ** 2

        # Compute kappa
        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0

        return (p_o - p_e) / (1.0 - p_e)

    @staticmethod
    def cohens_kappa(judgments1: List[int], judgments2: List[int]) -> float:
        """
        Compute Cohen's kappa for two judges.
        """
        if len(judgments1) != len(judgments2):
            return 0.0

        n = len(judgments1)
        if n == 0:
            return 0.0

        # Observed agreement
        p_o = sum(1 for j1, j2 in zip(judgments1, judgments2) if j1 == j2) / n

        # Expected agreement
        categories = set(list(judgments1) + list(judgments2))
        p_e = 0.0
        for cat in categories:
            p1 = sum(1 for j in judgments1 if j == cat) / n
            p2 = sum(1 for j in judgments2 if j == cat) / n
            p_e += p1 * p2

        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0

        return (p_o - p_e) / (1.0 - p_e)

    @staticmethod
    def outlier_detection(scores: List[int], threshold: float = 2.0) -> List[bool]:
        """
        Detect outlier judgments using z-score.

        Args:
            scores: List of judgment scores
            threshold: Z-score threshold (default 2.0)

        Returns:
            List of boolean indicating outliers
        """
        if len(scores) < 2:
            return [False] * len(scores)

        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores)

        if stdev == 0:
            return [False] * len(scores)

        return [abs((score - mean) / stdev) > threshold for score in scores]


class JudgeCalibration:
    """Calibrate judge models for consistent scoring."""

    @staticmethod
    def compute_bias(judge_scores: List[int], reference_scores: List[int]) -> float:
        """
        Compute mean bias (systematic over/under-scoring).
        """
        if len(judge_scores) != len(reference_scores):
            return 0.0

        diffs = [j - r for j, r in zip(judge_scores, reference_scores)]
        return statistics.mean(diffs) if diffs else 0.0

    @staticmethod
    def compute_variance(judge_scores: List[int], reference_scores: List[int]) -> float:
        """
        Compute variance (score spread/consistency).
        """
        if len(judge_scores) != len(reference_scores) or len(judge_scores) < 2:
            return 0.0

        diffs = [j - r for j, r in zip(judge_scores, reference_scores)]
        return statistics.variance(diffs)

    @staticmethod
    def pearson_correlation(scores1: List[int], scores2: List[int]) -> float:
        """
        Compute Pearson correlation between two score lists.
        """
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0

        mean1 = statistics.mean(scores1)
        mean2 = statistics.mean(scores2)

        numerator = sum((s1 - mean1) * (s2 - mean2) for s1, s2 in zip(scores1, scores2))

        denom1 = sum((s - mean1) ** 2 for s in scores1)
        denom2 = sum((s - mean2) ** 2 for s in scores2)

        denominator = math.sqrt(denom1 * denom2)

        if denominator == 0:
            return 0.0

        return numerator / denominator
