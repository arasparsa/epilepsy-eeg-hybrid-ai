"""Select leakage-safe windows for nested CV roles."""

from __future__ import annotations

import pandas as pd


def get_subjects_for_nested_fold(
    *,
    inner_assignments: pd.DataFrame,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, set[str]]:
    """Return train, validation and test subject sets."""
    current = inner_assignments.loc[
        (
            inner_assignments[
                "outer_fold"
            ]
            == outer_fold
        )
        & (
            inner_assignments[
                "inner_fold"
            ]
            == inner_fold
        )
    ]

    if current.empty:
        raise ValueError(
            "No nested assignments found for "
            f"outer={outer_fold}, "
            f"inner={inner_fold}."
        )

    subject_sets = {
        role: set(
            current.loc[
                current["role"] == role,
                "subject_id",
            ].astype(str)
        )
        for role in [
            "train",
            "validation",
            "test",
        ]
    }

    if (
        subject_sets["train"]
        & subject_sets["validation"]
    ):
        raise ValueError(
            "Train/validation subject overlap."
        )

    if (
        subject_sets["train"]
        & subject_sets["test"]
    ):
        raise ValueError(
            "Train/test subject overlap."
        )

    if (
        subject_sets["validation"]
        & subject_sets["test"]
    ):
        raise ValueError(
            "Validation/test subject overlap."
        )

    return subject_sets


def select_binary_windows(
    *,
    windows: pd.DataFrame,
    subject_ids: set[str],
) -> pd.DataFrame:
    """Select binary windows for specified subjects."""
    selected = windows.loc[
        windows["subject_id"].isin(
            subject_ids
        )
        & windows[
            "binary_label"
        ].notna()
    ].copy()

    selected["binary_label"] = (
        selected["binary_label"]
        .astype(int)
    )

    return selected.reset_index(
        drop=True
    )


def get_nested_window_tables(
    *,
    windows: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, pd.DataFrame]:
    """Build train/validation/test window tables."""
    subjects = (
        get_subjects_for_nested_fold(
            inner_assignments=(
                inner_assignments
            ),
            outer_fold=outer_fold,
            inner_fold=inner_fold,
        )
    )

    return {
        role: select_binary_windows(
            windows=windows,
            subject_ids=subject_ids,
        )
        for role, subject_ids
        in subjects.items()
    }


def get_outer_development_tables(
    *,
    windows: pd.DataFrame,
    outer_subject_folds: pd.DataFrame,
    outer_fold: int,
) -> dict[str, pd.DataFrame]:
    """Return full outer-development and outer-test windows."""
    test_subjects = set(
        outer_subject_folds.loc[
            outer_subject_folds[
                "outer_test_fold"
            ]
            == outer_fold,
            "subject_id",
        ].astype(str)
    )

    development_subjects = set(
        outer_subject_folds.loc[
            outer_subject_folds[
                "outer_test_fold"
            ]
            != outer_fold,
            "subject_id",
        ].astype(str)
    )

    if (
        development_subjects
        & test_subjects
    ):
        raise RuntimeError(
            "Outer development/test overlap."
        )

    return {
        "development": (
            select_binary_windows(
                windows=windows,
                subject_ids=(
                    development_subjects
                ),
            )
        ),
        "test": select_binary_windows(
            windows=windows,
            subject_ids=test_subjects,
        ),
    }