"""Metrics computation and formatting."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    training_time: float = 0.0
    inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for pandas/export."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
        }
    
    def __str__(self) -> str:
        return (
            f"Accuracy:  {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1-Score:  {self.f1:.4f}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_time: float = 0.0,
    inference_time: float = 0.0,
) -> ClassificationMetrics:
    """Compute classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        training_time: Time taken for training (seconds).
        inference_time: Time taken for inference (seconds).
        
    Returns:
        ClassificationMetrics object.
    """
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        training_time=training_time,
        inference_time=inference_time,
    )


def format_metrics_table(
    results: Dict[str, ClassificationMetrics],
) -> str:
    """Format metrics as a markdown table.
    
    Args:
        results: Dictionary mapping approach names to metrics.
        
    Returns:
        Markdown table string.
    """
    header = "| Approach | Accuracy | Precision | Recall | F1-Score | Train Time (s) |"
    separator = "|----------|----------|-----------|--------|----------|----------------|"
    
    rows = [header, separator]
    for name, metrics in results.items():
        row = (
            f"| {name} | {metrics.accuracy:.4f} | {metrics.precision:.4f} | "
            f"{metrics.recall:.4f} | {metrics.f1:.4f} | {metrics.training_time:.1f} |"
        )
        rows.append(row)
    
    return "\n".join(rows)


def print_comparison(results: Dict[str, ClassificationMetrics]) -> None:
    """Print comparison of approaches."""
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    print(format_metrics_table(results))
    print("=" * 70)
    
    # Find best approach
    best_f1 = max(results.items(), key=lambda x: x[1].f1)
    print(f"\n🏆 Best F1-Score: {best_f1[0]} ({best_f1[1].f1:.4f})")
    
    best_acc = max(results.items(), key=lambda x: x[1].accuracy)
    print(f"🏆 Best Accuracy: {best_acc[0]} ({best_acc[1].accuracy:.4f})")
