"""Window-level binary seizure-detection metrics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_binary_arrays(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Validate binary targets and probabilities."""
    labels = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    scores = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(labels) == 0:
        raise ValueError(
            "No observations supplied."
        )

    if len(labels) != len(scores):
        raise ValueError(
            "y_true and probabilities "
            "have different lengths."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    if not np.all(
        (scores >= 0)
        & (scores <= 1)
    ):
        raise ValueError(
            "Probabilities must be in [0, 1]."
        )

    unique_labels = set(
        np.unique(labels)
    )

    if unique_labels != {
        0,
        1,
    }:
        raise ValueError(
            "Both binary classes 0 and 1 "
            "must be present."
        )

    return labels, scores


def calculate_binary_metrics(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate threshold-free and threshold-based metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be in [0, 1]."
        )

    labels, scores = (
        validate_binary_arrays(
            y_true=y_true,
            probabilities=probabilities,
        )
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[
                0,
                1,
            ],
        ).ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else float("nan")
    )

    metrics: dict[
        str,
        float | int,
    ] = {
        "average_precision": float(
            average_precision_score(
                labels,
                scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "matthews_corrcoef": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(
            threshold
        ),
        "positive_prevalence": float(
            labels.mean()
        ),
        "sample_count": int(
            len(labels)
        ),
    }

    for name, value in (
        metrics.items()
    ):
        if isinstance(
            value,
            float,
        ) and name not in {
            "sensitivity",
            "specificity",
        }:
            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"Metric {name} is non-finite."
                )

    return metrics