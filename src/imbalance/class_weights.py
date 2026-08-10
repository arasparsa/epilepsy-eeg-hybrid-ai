"""Class-imbalance utilities for binary seizure detection."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def count_binary_classes(
    windows: pd.DataFrame,
) -> dict[str, int]:
    """Count negative and positive training windows."""
    if "binary_label" not in windows.columns:
        raise ValueError(
            "binary_label column is required."
        )

    labels = windows[
        "binary_label"
    ].dropna().astype(int)

    negative_count = int(
        (labels == 0).sum()
    )

    positive_count = int(
        (labels == 1).sum()
    )

    if negative_count <= 0:
        raise ValueError(
            "No negative training windows."
        )

    if positive_count <= 0:
        raise ValueError(
            "No positive training windows."
        )

    return {
        "negative_count": negative_count,
        "positive_count": positive_count,
    }


def calculate_pos_weight(
    windows: pd.DataFrame,
    *,
    method: str = (
        "negative_divided_by_positive"
    ),
    maximum_weight: float | None = None,
) -> float:
    """Calculate positive-class loss weight."""
    counts = count_binary_classes(
        windows
    )

    negative_count = counts[
        "negative_count"
    ]

    positive_count = counts[
        "positive_count"
    ]

    raw_ratio = (
        negative_count
        / positive_count
    )

    if method == (
        "negative_divided_by_positive"
    ):
        weight = raw_ratio

    elif method == (
        "sqrt_negative_divided_by_positive"
    ):
        weight = math.sqrt(
            raw_ratio
        )

    elif method == "none":
        weight = 1.0

    else:
        raise ValueError(
            f"Unknown class-weight method: "
            f"{method}"
        )

    if maximum_weight is not None:
        if maximum_weight <= 0:
            raise ValueError(
                "maximum_weight must be positive."
            )

        weight = min(
            weight,
            float(maximum_weight),
        )

    return float(weight)


def calculate_sample_weights(
    windows: pd.DataFrame,
) -> np.ndarray:
    """Inverse-frequency weights for WeightedRandomSampler."""
    counts = count_binary_classes(
        windows
    )

    class_weights = {
        0: 1.0
        / counts["negative_count"],
        1: 1.0
        / counts["positive_count"],
    }

    labels = windows[
        "binary_label"
    ].astype(int).to_numpy()

    return np.asarray(
        [
            class_weights[
                int(label)
            ]
            for label in labels
        ],
        dtype=np.float64,
    )