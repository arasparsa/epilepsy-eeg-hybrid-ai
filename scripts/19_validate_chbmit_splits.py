"""Validate frozen patient-independent CHB-MIT splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "chbmit_splitting.yaml"
)


def main() -> int:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    output_dir = (
        PROJECT_ROOT
        / config["outputs"][
            "metadata_directory"
        ]
    )

    mapping = pd.read_csv(
        output_dir
        / "chbmit_resolved_subject_mapping.csv"
    )

    outer = pd.read_csv(
        output_dir
        / "chbmit_outer_subject_folds.csv"
    )

    inner = pd.read_csv(
        output_dir
        / "chbmit_inner_subject_folds.csv"
    )

    windows = pd.read_csv(
        output_dir
        / "chbmit_window_outer_folds.csv"
    )

    outer_summary = pd.read_csv(
        output_dir
        / "chbmit_outer_fold_summary.csv"
    )

    errors: list[str] = []
    warnings: list[str] = []
    validation_rows: list[dict] = []

    outer_n_splits = int(
        config["outer_cv"]["n_splits"]
    )

    inner_n_splits = int(
        config["inner_cv"]["n_splits"]
    )

    # 1. Every subject has one outer assignment.
    if outer[
        "subject_id"
    ].duplicated().any():
        errors.append(
            "Subjects assigned to multiple outer folds."
        )

    mapped_subjects = set(
        mapping["subject_id"]
    )

    assigned_subjects = set(
        outer["subject_id"]
    )

    if mapped_subjects != assigned_subjects:
        errors.append(
            "Resolved and assigned subject sets differ."
        )

    # 2. Valid outer fold IDs.
    expected_outer_folds = set(
        range(outer_n_splits)
    )

    observed_outer_folds = set(
        outer["outer_test_fold"]
    )

    if (
        observed_outer_folds
        != expected_outer_folds
    ):
        errors.append(
            "Outer fold identifiers are incomplete."
        )

    # 3. Every case maps to one fold.
    case_folds = (
        mapping.merge(
            outer,
            on="subject_id",
            validate="many_to_one",
        )
        .groupby("case_id")[
            "outer_test_fold"
        ]
        .nunique()
    )

    if not case_folds.eq(1).all():
        errors.append(
            "A case occurs in multiple outer folds."
        )

    # 4. chb01/chb21 relationship.
    known_pair = mapping.loc[
        mapping["case_id"].isin(
            ["chb01", "chb21"]
        )
    ]

    if len(known_pair) == 2:
        if (
            known_pair[
                "subject_id"
            ].nunique()
            != 1
        ):
            errors.append(
                "chb01 and chb21 do not share "
                "the same subject ID."
            )
    else:
        warnings.append(
            "The complete chb01/chb21 pair was "
            "not observed in the retained dataset."
        )

    # 5. Every window has one outer fold.
    if windows[
        "window_id"
    ].duplicated().any():
        errors.append(
            "Duplicate window IDs in split manifest."
        )

    if windows[
        "outer_test_fold"
    ].isna().any():
        errors.append(
            "Windows without outer fold."
        )

    window_subject_folds = (
        windows.groupby("subject_id")[
            "outer_test_fold"
        ].nunique()
    )

    if not window_subject_folds.eq(1).all():
        errors.append(
            "Windows from a subject occur in "
            "multiple outer folds."
        )

    recording_folds = (
        windows.groupby("recording_id")[
            "outer_test_fold"
        ].nunique()
    )

    if not recording_folds.eq(1).all():
        errors.append(
            "A recording occurs in multiple folds."
        )

    # 6. Outer test class presence.
    minimum_outer_subjects = int(
        config["validation"][
            "minimum_subjects_per_outer_test_fold"
        ]
    )

    for row in outer_summary.itertuples(
        index=False
    ):
        fold_errors: list[str] = []

        if (
            row.subject_count
            < minimum_outer_subjects
        ):
            fold_errors.append(
                "too_few_subjects"
            )

        if (
            config["validation"][
                "require_each_outer_test_fold_has_ictal_windows"
            ]
            and row.ictal_window_count <= 0
        ):
            fold_errors.append(
                "no_ictal_windows"
            )

        if (
            config["validation"][
                "require_each_outer_test_fold_has_non_ictal_windows"
            ]
            and row.non_ictal_window_count <= 0
        ):
            fold_errors.append(
                "no_non_ictal_windows"
            )

        validation_rows.append(
            {
                "validation_scope": (
                    "outer_fold"
                ),
                "outer_fold": (
                    row.outer_test_fold
                ),
                "inner_fold": None,
                "role": "test",
                "subject_count": (
                    row.subject_count
                ),
                "non_ictal_window_count": (
                    row.non_ictal_window_count
                ),
                "ictal_window_count": (
                    row.ictal_window_count
                ),
                "status": (
                    "valid"
                    if not fold_errors
                    else "error"
                ),
                "issues": "|".join(
                    fold_errors
                ),
            }
        )

        if fold_errors:
            errors.append(
                f"Outer fold "
                f"{row.outer_test_fold}: "
                f"{fold_errors}"
            )

    # 7. Inner split integrity.
    minimum_inner_subjects = int(
        config["validation"][
            "minimum_subjects_per_inner_validation_fold"
        ]
    )

    for outer_fold in range(
        outer_n_splits
    ):
        expected_test_subjects = set(
            outer.loc[
                outer["outer_test_fold"]
                == outer_fold,
                "subject_id",
            ]
        )

        development_subjects = set(
            outer.loc[
                outer["outer_test_fold"]
                != outer_fold,
                "subject_id",
            ]
        )

        validation_appearances = {
            subject_id: 0
            for subject_id
            in development_subjects
        }

        for inner_fold in range(
            inner_n_splits
        ):
            current = inner.loc[
                (
                    inner["outer_fold"]
                    == outer_fold
                )
                & (
                    inner["inner_fold"]
                    == inner_fold
                )
            ]

            train_subjects = set(
                current.loc[
                    current["role"]
                    == "train",
                    "subject_id",
                ]
            )

            validation_subjects = set(
                current.loc[
                    current["role"]
                    == "validation",
                    "subject_id",
                ]
            )

            test_subjects = set(
                current.loc[
                    current["role"]
                    == "test",
                    "subject_id",
                ]
            )

            if (
                train_subjects
                & validation_subjects
            ):
                errors.append(
                    f"Train/validation overlap: "
                    f"outer={outer_fold}, "
                    f"inner={inner_fold}"
                )

            if (
                train_subjects
                & test_subjects
            ):
                errors.append(
                    f"Train/test overlap: "
                    f"outer={outer_fold}, "
                    f"inner={inner_fold}"
                )

            if (
                validation_subjects
                & test_subjects
            ):
                errors.append(
                    f"Validation/test overlap: "
                    f"outer={outer_fold}, "
                    f"inner={inner_fold}"
                )

            if (
                test_subjects
                != expected_test_subjects
            ):
                errors.append(
                    f"Incorrect test subjects: "
                    f"outer={outer_fold}, "
                    f"inner={inner_fold}"
                )

            if (
                train_subjects
                | validation_subjects
                != development_subjects
            ):
                errors.append(
                    f"Incomplete development partition: "
                    f"outer={outer_fold}, "
                    f"inner={inner_fold}"
                )

            for subject_id in (
                validation_subjects
            ):
                validation_appearances[
                    subject_id
                ] += 1

            role_windows = windows.loc[
                windows["subject_id"].isin(
                    validation_subjects
                )
            ]

            ictal_count = int(
                (
                    role_windows["label_name"]
                    == "ictal"
                ).sum()
            )

            non_ictal_count = int(
                (
                    role_windows["label_name"]
                    == "non_ictal"
                ).sum()
            )

            fold_issues: list[str] = []

            if (
                len(validation_subjects)
                < minimum_inner_subjects
            ):
                fold_issues.append(
                    "too_few_validation_subjects"
                )

            if (
                config["validation"][
                    "require_each_inner_validation_fold_has_ictal_windows"
                ]
                and ictal_count <= 0
            ):
                fold_issues.append(
                    "no_validation_ictal_windows"
                )

            if (
                config["validation"][
                    "require_each_inner_validation_fold_has_non_ictal_windows"
                ]
                and non_ictal_count <= 0
            ):
                fold_issues.append(
                    "no_validation_non_ictal_windows"
                )

            validation_rows.append(
                {
                    "validation_scope": (
                        "inner_fold"
                    ),
                    "outer_fold": (
                        outer_fold
                    ),
                    "inner_fold": (
                        inner_fold
                    ),
                    "role": "validation",
                    "subject_count": (
                        len(
                            validation_subjects
                        )
                    ),
                    "non_ictal_window_count": (
                        non_ictal_count
                    ),
                    "ictal_window_count": (
                        ictal_count
                    ),
                    "status": (
                        "valid"
                        if not fold_issues
                        else "error"
                    ),
                    "issues": "|".join(
                        fold_issues
                    ),
                }
            )

            if fold_issues:
                errors.append(
                    f"Inner validation fold issue: "
                    f"outer={outer_fold}, "
                    f"inner={inner_fold}, "
                    f"{fold_issues}"
                )

        invalid_appearances = {
            subject_id: count
            for subject_id, count
            in validation_appearances.items()
            if count != 1
        }

        if invalid_appearances:
            errors.append(
                "Development subjects must appear "
                "exactly once as inner validation: "
                f"outer={outer_fold}, "
                f"{invalid_appearances}"
            )

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        output_dir
        / "chbmit_split_validation.csv",
        index=False,
    )

    result = {
        "errors": len(errors),
        "warnings": len(warnings),
        "outer_folds": outer_n_splits,
        "inner_folds_per_outer": (
            inner_n_splits
        ),
        "subject_count": int(
            outer["subject_id"].nunique()
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    for error in errors:
        print("ERROR:", error)

    for warning in warnings:
        print("WARNING:", warning)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())