"""Tests for training-only class weights."""

import numpy as np
import pandas as pd

from src.imbalance.class_weights import (
    calculate_pos_weight,
    calculate_sample_weights,
    count_binary_classes,
)


def test_pos_weight_ratio() -> None:
    windows = pd.DataFrame(
        {
            "binary_label": (
                [0] * 90
                + [1] * 10
            )
        }
    )

    weight = calculate_pos_weight(
        windows
    )

    assert weight == 9.0


def test_inverse_frequency_sample_weights() -> None:
    windows = pd.DataFrame(
        {
            "binary_label": [
                0,
                0,
                0,
                1,
            ]
        }
    )

    weights = (
        calculate_sample_weights(
            windows
        )
    )

    assert np.allclose(
        weights[:3],
        1 / 3,
    )

    assert weights[3] == 1.0