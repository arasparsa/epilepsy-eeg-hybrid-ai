"""Validation-only decision-threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
)


def select_threshold_max_f1(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_threshold: float = 0.01,
    maximum_threshold: float = 0.99,
    number_of_thresholds: int = 199,
    tie_break: str = (
        "closest_to_0_5"
    ),
) -> dict[str, float | int]:
    """Select threshold using validation F1 only."""
    labels = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    scores = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(labels) != len(
        scores
    ):
        raise ValueError(
            "Label and probability "
            "lengths differ."
        )

    if set(
        np.unique(labels)
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Both classes are required "
            "for threshold selection."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "Non-finite probabilities."
        )

    if not (
        0
        <= minimum_threshold
        < maximum_threshold
        <= 1
    ):
        raise ValueError(
            "Invalid threshold interval."
        )

    if number_of_thresholds < 2:
        raise ValueError(
            "At least two thresholds "
            "are required."
        )

    thresholds = np.linspace(
        minimum_threshold,
        maximum_threshold,
        number_of_thresholds,
        dtype=float,
    )

    results: list[
        tuple[
            float,
            float,
        ]
    ] = []

    for threshold in thresholds:
        predictions = (
            scores >= threshold
        ).astype(int)

        score = float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        )

        results.append(
            (
                float(
                    threshold
                ),
                score,
            )
        )

    best_f1 = max(
        result[1]
        for result in results
    )

    tied_thresholds = [
        threshold
        for threshold, score
        in results
        if np.isclose(
            score,
            best_f1,
            rtol=0,
            atol=1e-12,
        )
    ]

    if tie_break == (
        "closest_to_0_5"
    ):
        best_threshold = min(
            tied_thresholds,
            key=lambda threshold: (
                abs(
                    threshold - 0.5
                ),
                threshold,
            ),
        )

    elif tie_break == (
        "lowest"
    ):
        best_threshold = min(
            tied_thresholds
        )

    elif tie_break == (
        "highest"
    ):
        best_threshold = max(
            tied_thresholds
        )

    else:
        raise ValueError(
            f"Unknown tie_break: "
            f"{tie_break}"
        )

    return {
        "threshold": float(
            best_threshold
        ),
        "validation_f1": float(
            best_f1
        ),
        "candidate_threshold_count": int(
            number_of_thresholds
        ),
        "tie_count": int(
            len(tied_thresholds)
        ),
    }