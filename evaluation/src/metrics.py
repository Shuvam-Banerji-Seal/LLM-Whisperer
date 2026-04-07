"""
Common metric computation utilities for evaluation framework.

Provides functions for computing standard metrics used across evaluation categories.
"""

from typing import List, Dict, Any, Tuple
import statistics
from collections import Counter


class ClassificationMetrics:
    """Metrics for classification/binary evaluation tasks."""

    @staticmethod
    def accuracy(predictions: List[Any], ground_truth: List[Any]) -> float:
        """
        Compute accuracy: (correct predictions) / (total predictions)
        """
        if not predictions or len(predictions) != len(ground_truth):
            return 0.0
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)

    @staticmethod
    def precision(
        predictions: List[Any], ground_truth: List[Any], positive_label: Any = 1
    ) -> float:
        """
        Precision = TP / (TP + FP)
        """
        tp = sum(
            1
            for p, g in zip(predictions, ground_truth)
            if p == positive_label and g == positive_label
        )
        fp = sum(
            1
            for p, g in zip(predictions, ground_truth)
            if p == positive_label and g != positive_label
        )
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(
        predictions: List[Any], ground_truth: List[Any], positive_label: Any = 1
    ) -> float:
        """
        Recall = TP / (TP + FN)
        """
        tp = sum(
            1
            for p, g in zip(predictions, ground_truth)
            if p == positive_label and g == positive_label
        )
        fn = sum(
            1
            for p, g in zip(predictions, ground_truth)
            if p != positive_label and g == positive_label
        )
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(
        predictions: List[Any], ground_truth: List[Any], positive_label: Any = 1
    ) -> float:
        """
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        precision = ClassificationMetrics.precision(
            predictions, ground_truth, positive_label
        )
        recall = ClassificationMetrics.recall(predictions, ground_truth, positive_label)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def confusion_matrix(
        predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, int]:
        """
        Compute confusion matrix for binary classification.
        """
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 0)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 1)
        return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


class RankingMetrics:
    """Metrics for ranking/retrieval evaluation tasks."""

    @staticmethod
    def mean_reciprocal_rank(
        rankings: List[List[Any]], relevant_items: List[List[Any]]
    ) -> float:
        """
        Mean Reciprocal Rank: Average of 1/rank for first relevant item.
        """
        if not rankings:
            return 0.0

        rr_scores = []
        for ranking, relevant in zip(rankings, relevant_items):
            for idx, item in enumerate(ranking, 1):
                if item in relevant:
                    rr_scores.append(1.0 / idx)
                    break
            else:
                rr_scores.append(0.0)

        return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

    @staticmethod
    def ndcg_at_k(
        rankings: List[List[Any]], relevance_scores: List[Dict[Any, int]], k: int = 10
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at K.
        """
        if not rankings:
            return 0.0

        ndcg_scores = []
        for ranking, rel_dict in zip(rankings, relevance_scores):
            # Compute DCG
            dcg = sum(
                rel_dict.get(item, 0) / (2 ** (idx + 1))
                for idx, item in enumerate(ranking[:k])
            )

            # Compute ideal DCG
            ideal_rel = sorted(rel_dict.values(), reverse=True)[:k]
            idcg = sum(rel / (2 ** (idx + 1)) for idx, rel in enumerate(ideal_rel))

            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


class TextMetrics:
    """Metrics for text-based evaluation (similarity, length, etc)."""

    @staticmethod
    def string_match_rate(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Exact string match rate.
        """
        if not predictions:
            return 0.0
        matches = sum(
            1 for p, g in zip(predictions, ground_truth) if p.strip() == g.strip()
        )
        return matches / len(predictions)

    @staticmethod
    def token_overlap(text1: str, text2: str) -> float:
        """
        Token-level overlap (Jaccard similarity).
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def average_length(texts: List[str]) -> float:
        """Compute average text length in tokens."""
        if not texts:
            return 0.0
        total_tokens = sum(len(text.split()) for text in texts)
        return total_tokens / len(texts)


class StatisticalTests:
    """Statistical testing utilities."""

    @staticmethod
    def mean_confidence_interval(
        values: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for mean using t-distribution.
        """
        if len(values) < 2:
            return (0.0, 0.0)

        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        n = len(values)

        # Approximate t-value for 95% CI (use 1.96 for normal approximation)
        from math import sqrt

        margin = 1.96 * (stdev / sqrt(n))

        return (mean - margin, mean + margin)

    @staticmethod
    def kolmogorov_smirnov_statistic(
        sample1: List[float], sample2: List[float]
    ) -> float:
        """
        Compute KS test statistic between two samples.
        D = max|F1(x) - F2(x)|
        """
        combined = sorted(set(sample1 + sample2))

        max_diff = 0.0
        for x in combined:
            f1 = sum(1 for val in sample1 if val <= x) / len(sample1) if sample1 else 0
            f2 = sum(1 for val in sample2 if val <= x) / len(sample2) if sample2 else 0
            max_diff = max(max_diff, abs(f1 - f2))

        return max_diff
