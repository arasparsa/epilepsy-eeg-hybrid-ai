"""Patient-independent nested cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
)


def prepare_binary_stratification_table(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Keep windows with valid binary labels."""
    required_columns = {
        "window_id",
        "subject_id",
        "case_id",
        "recording_id",
        "binary_label",
    }

    missing = required_columns - set(
        windows.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    binary = windows.loc[
        windows["binary_label"].notna()
    ].copy()

    binary["binary_label"] = (
        binary["binary_label"].astype(int)
    )

    if set(binary["binary_label"].unique()) != {
        0,
        1,
    }:
        raise ValueError(
            "Binary stratification requires labels 0 and 1."
        )

    return binary


def generate_outer_subject_folds(
    binary_windows: pd.DataFrame,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> pd.DataFrame:
    """Assign every subject to exactly one outer test fold."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=(
            random_state if shuffle else None
        ),
    )

    y = binary_windows[
        "binary_label"
    ].to_numpy()

    groups = binary_windows[
        "subject_id"
    ].astype(str).to_numpy()

    # X is not used by the splitter.
    X = np.zeros(
        shape=(len(binary_windows), 1),
        dtype=np.uint8,
    )

    subject_to_fold: dict[str, int] = {}

    for fold_index, (
        development_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            X=X,
            y=y,
            groups=groups,
        )
    ):
        test_subjects = set(
            groups[test_indices]
        )

        development_subjects = set(
            groups[development_indices]
        )

        overlap = (
            test_subjects
            & development_subjects
        )

        if overlap:
            raise RuntimeError(
                "Subject overlap within outer split: "
                f"{sorted(overlap)}"
            )

        for subject_id in test_subjects:
            if subject_id in subject_to_fold:
                raise RuntimeError(
                    f"Subject {subject_id} assigned "
                    "to multiple outer folds."
                )

            subject_to_fold[subject_id] = fold_index

    all_subjects = set(groups)

    if set(subject_to_fold) != all_subjects:
        missing_subjects = (
            all_subjects
            - set(subject_to_fold)
        )

        raise RuntimeError(
            "Subjects missing outer assignment: "
            f"{sorted(missing_subjects)}"
        )

    return pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "outer_test_fold": fold,
            }
            for subject_id, fold
            in sorted(
                subject_to_fold.items()
            )
        ]
    )


def generate_inner_subject_folds(
    binary_windows: pd.DataFrame,
    outer_subject_folds: pd.DataFrame,
    outer_n_splits: int,
    inner_n_splits: int,
    shuffle: bool,
    base_random_state: int,
    random_state_increment: int,
) -> pd.DataFrame:
    """Generate inner train/validation assignments."""
    assignment_rows: list[dict] = []

    outer_fold_lookup = (
        outer_subject_folds.set_index(
            "subject_id"
        )["outer_test_fold"]
        .to_dict()
    )

    for outer_fold in range(
        outer_n_splits
    ):
        outer_test_subjects = {
            subject_id
            for subject_id, fold
            in outer_fold_lookup.items()
            if fold == outer_fold
        }

        outer_development = (
            binary_windows.loc[
                ~binary_windows[
                    "subject_id"
                ].isin(outer_test_subjects)
            ].copy()
        )

        random_state = (
            base_random_state
            + outer_fold
            * random_state_increment
        )

        splitter = StratifiedGroupKFold(
            n_splits=inner_n_splits,
            shuffle=shuffle,
            random_state=(
                random_state
                if shuffle
                else None
            ),
        )

        y = outer_development[
            "binary_label"
        ].to_numpy()

        groups = outer_development[
            "subject_id"
        ].astype(str).to_numpy()

        X = np.zeros(
            shape=(
                len(outer_development),
                1,
            ),
            dtype=np.uint8,
        )

        observed_validation_subjects: set[
            str
        ] = set()

        for inner_fold, (
            train_indices,
            validation_indices,
        ) in enumerate(
            splitter.split(
                X=X,
                y=y,
                groups=groups,
            )
        ):
            train_subjects = set(
                groups[train_indices]
            )

            validation_subjects = set(
                groups[validation_indices]
            )

            if (
                train_subjects
                & validation_subjects
            ):
                raise RuntimeError(
                    "Subject overlap in inner fold."
                )

            observed_validation_subjects.update(
                validation_subjects
            )

            for subject_id in sorted(
                train_subjects
            ):
                assignment_rows.append(
                    {
                        "outer_fold": (
                            outer_fold
                        ),
                        "inner_fold": (
                            inner_fold
                        ),
                        "subject_id": (
                            subject_id
                        ),
                        "role": "train",
                        "inner_random_state": (
                            random_state
                        ),
                    }
                )

            for subject_id in sorted(
                validation_subjects
            ):
                assignment_rows.append(
                    {
                        "outer_fold": (
                            outer_fold
                        ),
                        "inner_fold": (
                            inner_fold
                        ),
                        "subject_id": (
                            subject_id
                        ),
                        "role": "validation",
                        "inner_random_state": (
                            random_state
                        ),
                    }
                )

            for subject_id in sorted(
                outer_test_subjects
            ):
                assignment_rows.append(
                    {
                        "outer_fold": (
                            outer_fold
                        ),
                        "inner_fold": (
                            inner_fold
                        ),
                        "subject_id": (
                            subject_id
                        ),
                        "role": "test",
                        "inner_random_state": (
                            random_state
                        ),
                    }
                )

        expected_development_subjects = (
            set(
                outer_development[
                    "subject_id"
                ].unique()
            )
        )

        if (
            observed_validation_subjects
            != expected_development_subjects
        ):
            raise RuntimeError(
                "Not every development subject was "
                "used once as inner validation."
            )

    assignments = pd.DataFrame(
        assignment_rows
    )

    key_columns = [
        "outer_fold",
        "inner_fold",
        "subject_id",
    ]

    if assignments[
        key_columns
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate nested subject assignments."
        )

    return assignments