"""Tests for patient-independent nested splitting."""

import pandas as pd

from src.splitting.nested_cv import (
    generate_inner_subject_folds,
    generate_outer_subject_folds,
)


def create_synthetic_windows() -> pd.DataFrame:
    rows: list[dict] = []

    for subject_index in range(12):
        subject_id = (
            f"subject_{subject_index:02d}"
        )

        for window_index in range(20):
            rows.append(
                {
                    "window_id": (
                        f"{subject_id}_{window_index}"
                    ),
                    "case_id": (
                        f"case_{subject_index:02d}"
                    ),
                    "recording_id": (
                        f"rec_{subject_index:02d}"
                    ),
                    "subject_id": subject_id,
                    "binary_label": (
                        1
                        if window_index < (
                            2 + subject_index % 3
                        )
                        else 0
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_outer_subjects_are_unique() -> None:
    windows = create_synthetic_windows()

    outer = generate_outer_subject_folds(
        binary_windows=windows,
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    assert outer["subject_id"].is_unique
    assert outer["outer_test_fold"].nunique() == 3


def test_inner_roles_do_not_overlap() -> None:
    windows = create_synthetic_windows()

    outer = generate_outer_subject_folds(
        binary_windows=windows,
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    inner = generate_inner_subject_folds(
        binary_windows=windows,
        outer_subject_folds=outer,
        outer_n_splits=3,
        inner_n_splits=3,
        shuffle=True,
        base_random_state=100,
        random_state_increment=1,
    )

    for (
        outer_fold,
        inner_fold,
    ), group in inner.groupby(
        ["outer_fold", "inner_fold"]
    ):
        train = set(
            group.loc[
                group["role"] == "train",
                "subject_id",
            ]
        )

        validation = set(
            group.loc[
                group["role"]
                == "validation",
                "subject_id",
            ]
        )

        test = set(
            group.loc[
                group["role"] == "test",
                "subject_id",
            ]
        )

        assert not (train & validation)
        assert not (train & test)
        assert not (validation & test)