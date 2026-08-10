"""Combine subject statistics into train-only fold scalers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.normalization.streaming_stats import (
    RunningChannelStatistics,
)


def build_fold_scaler(
    *,
    subject_statistics: pd.DataFrame,
    training_subjects: set[str],
    minimum_std_uv: float,
) -> pd.DataFrame:
    """Combine statistics of training subjects only."""
    if not training_subjects:
        raise ValueError(
            "No training subjects supplied."
        )

    selected = subject_statistics.loc[
        subject_statistics[
            "subject_id"
        ].isin(training_subjects)
    ].copy()

    observed_subjects = set(
        selected["subject_id"]
    )

    if (
        observed_subjects
        != training_subjects
    ):
        missing = (
            training_subjects
            - observed_subjects
        )

        raise ValueError(
            "Missing subject statistics: "
            f"{sorted(missing)}"
        )

    channel_names = (
        selected[
            [
                "channel_index",
                "channel_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "channel_index"
        )
    )

    channel_count = len(
        channel_names
    )

    combined = (
        RunningChannelStatistics.create(
            channel_count
        )
    )

    for subject_id, group in (
        selected.groupby(
            "subject_id",
            sort=True,
        )
    ):
        del subject_id

        ordered = group.sort_values(
            "channel_index"
        )

        if len(ordered) != channel_count:
            raise ValueError(
                "Incomplete subject channel statistics."
            )

        combined.combine(
            count=ordered[
                "sample_count"
            ].to_numpy(
                dtype=np.int64
            ),
            mean=ordered[
                "mean_uv"
            ].to_numpy(
                dtype=np.float64
            ),
            m2=ordered[
                "m2_uv2"
            ].to_numpy(
                dtype=np.float64
            ),
        )

    values = combined.to_dict()

    std = values["std"]

    invalid_std = (
        ~np.isfinite(std)
        | (std < minimum_std_uv)
    )

    safe_std = std.copy()

    safe_std[invalid_std] = 1.0

    scaler = channel_names.copy()

    scaler["sample_count"] = (
        values["count"]
    )

    scaler["mean_uv"] = (
        values["mean"]
    )

    scaler["variance_uv2"] = (
        values["variance"]
    )

    scaler["std_uv"] = std

    scaler["scale_uv"] = safe_std

    scaler[
        "std_replaced_with_one"
    ] = invalid_std

    return scaler.reset_index(
        drop=True
    )


def save_scaler_npz(
    *,
    scaler: pd.DataFrame,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    """Save a fold scaler in a compact NumPy artifact."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        channel_names=scaler[
            "channel_name"
        ].astype(str).to_numpy(),
        channel_mean_uv=scaler[
            "mean_uv"
        ].to_numpy(
            dtype=np.float64
        ),
        channel_std_uv=scaler[
            "scale_uv"
        ].to_numpy(
            dtype=np.float64
        ),
        sample_count=scaler[
            "sample_count"
        ].to_numpy(
            dtype=np.int64
        ),
        metadata=np.asarray(
            [metadata],
            dtype=object,
        ),
    )


def load_scaler_npz(
    path: Path,
) -> dict[str, object]:
    """Load scaler arrays and metadata."""
    with np.load(
        path,
        allow_pickle=True,
    ) as artifact:
        return {
            "channel_names": (
                artifact[
                    "channel_names"
                ]
            ),
            "channel_mean_uv": (
                artifact[
                    "channel_mean_uv"
                ]
            ),
            "channel_std_uv": (
                artifact[
                    "channel_std_uv"
                ]
            ),
            "sample_count": (
                artifact[
                    "sample_count"
                ]
            ),
            "metadata": (
                artifact[
                    "metadata"
                ][0]
            ),
        }