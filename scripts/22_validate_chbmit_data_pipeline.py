"""Validate train-only normalization and EEG DataLoaders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fold_selection import (
    get_nested_window_tables,
    get_subjects_for_nested_fold,
)
from src.data.window_dataset import (
    EEGWindowDataset,
)
from src.normalization.fold_scalers import (
    load_scaler_npz,
)
from src.splitting.subject_mapping import (
    apply_subject_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "chbmit_data_pipeline.yaml"
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

    scaler_dir = (
        PROJECT_ROOT
        / config["outputs"][
            "scaler_directory"
        ]
    )

    windows = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "window_manifest"
        ]
    )

    mapping = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "subject_mapping"
        ]
    )

    windows = apply_subject_mapping(
        windows,
        mapping,
    )

    outer = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "outer_subject_folds"
        ]
    )

    inner = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "inner_subject_folds"
        ]
    )

    subject_statistics = pd.read_csv(
        output_dir
        / "chbmit_subject_channel_statistics.csv"
    )

    class_weights = pd.read_csv(
        output_dir
        / "chbmit_fold_class_weights.csv"
    )

    errors: list[str] = []
    warnings: list[str] = []
    validation_rows: list[dict] = []

    expected_channel_count = int(
        config["signal"][
            "expected_channel_count"
        ]
    )

    expected_sample_count = int(
        config["signal"][
            "expected_sample_count"
        ]
    )

    # Subject statistics integrity.
    channel_counts = (
        subject_statistics.groupby(
            "subject_id"
        )["channel_name"].nunique()
    )

    if not channel_counts.eq(
        expected_channel_count
    ).all():
        errors.append(
            "Incomplete subject-channel statistics."
        )

    if (
        subject_statistics[
            "sample_count"
        ]
        <= 0
    ).any():
        errors.append(
            "Non-positive statistics sample count."
        )

    if not np.isfinite(
        subject_statistics[
            [
                "mean_uv",
                "m2_uv2",
                "variance_uv2",
                "std_uv",
            ]
        ].to_numpy()
    ).all():
        errors.append(
            "Non-finite subject statistics."
        )

    outer_ids = sorted(
        outer["outer_test_fold"]
        .unique()
        .astype(int)
    )

    inner_ids = sorted(
        inner["inner_fold"]
        .unique()
        .astype(int)
    )

    for outer_fold in outer_ids:
        for inner_fold in inner_ids:
            subject_sets = (
                get_subjects_for_nested_fold(
                    inner_assignments=inner,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
            )

            fold_tables = (
                get_nested_window_tables(
                    windows=windows,
                    inner_assignments=inner,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
            )

            scaler_path = (
                scaler_dir
                / f"outer_{outer_fold:02d}"
                / (
                    f"inner_{inner_fold:02d}"
                    "_scaler.npz"
                )
            )

            fold_issues: list[str] = []

            if not scaler_path.exists():
                fold_issues.append(
                    "scaler_missing"
                )

                continue

            scaler = load_scaler_npz(
                scaler_path
            )

            metadata = scaler[
                "metadata"
            ]

            scaler_training_subjects = set(
                metadata[
                    "training_subjects"
                ]
            )

            if (
                scaler_training_subjects
                != subject_sets["train"]
            ):
                fold_issues.append(
                    "scaler_training_subject_mismatch"
                )

            if (
                scaler_training_subjects
                & subject_sets[
                    "validation"
                ]
            ):
                fold_issues.append(
                    "validation_subject_in_scaler"
                )

            if (
                scaler_training_subjects
                & subject_sets["test"]
            ):
                fold_issues.append(
                    "test_subject_in_scaler"
                )

            mean = np.asarray(
                scaler[
                    "channel_mean_uv"
                ]
            )

            std = np.asarray(
                scaler[
                    "channel_std_uv"
                ]
            )

            if mean.shape != (
                expected_channel_count,
            ):
                fold_issues.append(
                    "invalid_scaler_mean_shape"
                )

            if std.shape != (
                expected_channel_count,
            ):
                fold_issues.append(
                    "invalid_scaler_std_shape"
                )

            if (
                ~np.isfinite(mean)
            ).any():
                fold_issues.append(
                    "nonfinite_scaler_mean"
                )

            if (
                ~np.isfinite(std)
            ).any():
                fold_issues.append(
                    "nonfinite_scaler_std"
                )

            if (std <= 0).any():
                fold_issues.append(
                    "nonpositive_scaler_std"
                )

            weight_row = (
                class_weights.loc[
                    (
                        class_weights[
                            "scope"
                        ]
                        == "inner_training"
                    )
                    & (
                        class_weights[
                            "outer_fold"
                        ]
                        == outer_fold
                    )
                    & (
                        class_weights[
                            "inner_fold"
                        ]
                        == inner_fold
                    )
                ]
            )

            if len(weight_row) != 1:
                fold_issues.append(
                    "missing_or_duplicate_class_weight"
                )

            else:
                train_labels = (
                    fold_tables["train"][
                        "binary_label"
                    ].astype(int)
                )

                expected_negative = int(
                    (
                        train_labels == 0
                    ).sum()
                )

                expected_positive = int(
                    (
                        train_labels == 1
                    ).sum()
                )

                observed = (
                    weight_row.iloc[0]
                )

                if (
                    int(
                        observed[
                            "negative_count"
                        ]
                    )
                    != expected_negative
                ):
                    fold_issues.append(
                        "negative_count_mismatch"
                    )

                if (
                    int(
                        observed[
                            "positive_count"
                        ]
                    )
                    != expected_positive
                ):
                    fold_issues.append(
                        "positive_count_mismatch"
                    )

            # Load deterministic samples from every role.
            for role, table in (
                fold_tables.items()
            ):
                if table.empty:
                    fold_issues.append(
                        f"{role}_table_empty"
                    )
                    continue

                sample_table = (
                    table.sort_values(
                        "window_id"
                    )
                    .head(2)
                    .copy()
                )

                dataset = EEGWindowDataset(
                    windows=sample_table,
                    project_root=(
                        PROJECT_ROOT
                    ),
                    channel_mean_uv=mean,
                    channel_std_uv=std,
                    expected_channel_count=(
                        expected_channel_count
                    ),
                    expected_sample_count=(
                        expected_sample_count
                    ),
                    max_open_fif_files=2,
                    preload_fif=False,
                    return_metadata=True,
                    verify_finite_values=True,
                )

                for dataset_index in range(
                    len(dataset)
                ):
                    signal, label, metadata_row = (
                        dataset[
                            dataset_index
                        ]
                    )

                    if tuple(
                        signal.shape
                    ) != (
                        expected_channel_count,
                        expected_sample_count,
                    ):
                        fold_issues.append(
                            f"{role}_shape_invalid"
                        )

                    if not np.isfinite(
                        signal.numpy()
                    ).all():
                        fold_issues.append(
                            f"{role}_nonfinite"
                        )

                    if float(
                        label.item()
                    ) not in {
                        0.0,
                        1.0,
                    }:
                        fold_issues.append(
                            f"{role}_label_invalid"
                        )

                    if not metadata_row[
                        "window_id"
                    ]:
                        fold_issues.append(
                            f"{role}_metadata_invalid"
                        )

            validation_rows.append(
                {
                    "outer_fold": (
                        outer_fold
                    ),
                    "inner_fold": (
                        inner_fold
                    ),
                    "train_subject_count": (
                        len(
                            subject_sets["train"]
                        )
                    ),
                    "validation_subject_count": (
                        len(
                            subject_sets[
                                "validation"
                            ]
                        )
                    ),
                    "test_subject_count": (
                        len(
                            subject_sets["test"]
                        )
                    ),
                    "status": (
                        "valid"
                        if not fold_issues
                        else "error"
                    ),
                    "issues": "|".join(
                        sorted(
                            set(
                                fold_issues
                            )
                        )
                    ),
                }
            )

            if fold_issues:
                errors.append(
                    f"outer={outer_fold}, "
                    f"inner={inner_fold}: "
                    f"{sorted(set(fold_issues))}"
                )

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        output_dir
        / "chbmit_data_pipeline_validation.csv",
        index=False,
    )

    result = {
        "errors": len(errors),
        "warnings": len(warnings),
        "validated_nested_folds": int(
            len(validation)
        ),
        "valid_nested_folds": int(
            validation[
                "status"
            ].eq("valid").sum()
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