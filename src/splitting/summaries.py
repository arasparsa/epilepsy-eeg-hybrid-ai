"""Summaries for patient-independent fold assignments."""

from __future__ import annotations

import pandas as pd


def build_outer_fold_summary(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each outer test fold."""
    return (
        windows.groupby(
            "outer_test_fold"
        )
        .agg(
            subject_count=(
                "subject_id",
                "nunique",
            ),
            case_count=(
                "case_id",
                "nunique",
            ),
            recording_count=(
                "recording_id",
                "nunique",
            ),
            total_window_count=(
                "window_id",
                "count",
            ),
            non_ictal_window_count=(
                "label_name",
                lambda values: int(
                    (values == "non_ictal").sum()
                ),
            ),
            boundary_window_count=(
                "label_name",
                lambda values: int(
                    (values == "boundary").sum()
                ),
            ),
            ictal_window_count=(
                "label_name",
                lambda values: int(
                    (values == "ictal").sum()
                ),
            ),
            clean_non_ictal_count=(
                "clean_non_ictal",
                "sum",
            ),
            recording_hours=(
                "window_duration_seconds",
                lambda values: (
                    float(values.sum()) / 3600
                ),
            ),
        )
        .reset_index()
    )


def add_outer_fold_ratios(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary class ratios."""
    output = summary.copy()

    output["binary_window_count"] = (
        output["non_ictal_window_count"]
        + output["ictal_window_count"]
    )

    output["ictal_window_fraction"] = (
        output["ictal_window_count"]
        / output["binary_window_count"]
    )

    return output