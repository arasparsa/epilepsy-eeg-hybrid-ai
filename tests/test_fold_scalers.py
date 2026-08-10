"""Tests for train-only fold scalers."""

import numpy as np
import pandas as pd

from src.normalization.fold_scalers import (
    build_fold_scaler,
)


def test_scaler_uses_only_selected_subjects() -> None:
    rows = []

    for subject_id, mean_value in [
        ("train_a", 0.0),
        ("train_b", 2.0),
        ("test_c", 1000.0),
    ]:
        for channel_index in range(2):
            rows.append(
                {
                    "subject_id": subject_id,
                    "channel_index": (
                        channel_index
                    ),
                    "channel_name": (
                        f"channel_{channel_index}"
                    ),
                    "sample_count": 100,
                    "mean_uv": mean_value,
                    "m2_uv2": 100.0,
                }
            )

    statistics = pd.DataFrame(rows)

    scaler = build_fold_scaler(
        subject_statistics=statistics,
        training_subjects={
            "train_a",
            "train_b",
        },
        minimum_std_uv=1e-6,
    )

    assert np.allclose(
        scaler["mean_uv"],
        1.0,
    )

    assert not np.any(
        np.isclose(
            scaler["mean_uv"],
            1000.0,
        )
    )