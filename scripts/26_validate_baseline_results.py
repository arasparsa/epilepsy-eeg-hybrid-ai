"""Validate the complete CHB-MIT baseline experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.baseline_utils import (
    load_outer_scaler,
    load_yaml,
    round_half_up,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "chbmit_baseline.yaml"
)


def main() -> int:
    config = load_yaml(
        CONFIG_PATH
    )

    output_dir = (
        PROJECT_ROOT
        / config["outputs"][
            "metadata_directory"
        ]
    )

    inner_runs = pd.read_csv(
        output_dir
        / "chbmit_inner_training_runs.csv"
    )

    inner_metrics = pd.read_csv(
        output_dir
        / "chbmit_inner_metrics.csv"
    )

    inner_thresholds = (
        pd.read_csv(
            output_dir
            / "chbmit_inner_thresholds.csv"
        )
    )

    selected = pd.read_csv(
        output_dir
        / "chbmit_selected_baseline_config.csv"
    )

    outer_metrics = pd.read_csv(
        output_dir
        / "chbmit_outer_metrics.csv"
    )

    predictions = pd.read_csv(
        output_dir
        / "chbmit_outer_predictions.csv"
    )

    outer_folds = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "outer_subject_folds"
        ]
    )

    inner_assignments = (
        pd.read_csv(
            PROJECT_ROOT
            / config["inputs"][
                "inner_subject_folds"
            ]
        )
    )

    errors: list[str] = []
    warnings: list[str] = []
    validation_rows: list[dict] = []

    expected_outer_count = int(
        outer_folds[
            "outer_test_fold"
        ].nunique()
    )

    expected_inner_per_outer = int(
        inner_assignments[
            "inner_fold"
        ].nunique()
    )

    expected_inner_runs = (
        expected_outer_count
        * expected_inner_per_outer
    )

    # --------------------------------------------------
    # Inner run counts
    # --------------------------------------------------

    if len(inner_runs) != (
        expected_inner_runs
    ):
        errors.append(
            "Incorrect number of inner runs: "
            f"{len(inner_runs)} vs "
            f"{expected_inner_runs}."
        )

    if len(inner_metrics) != (
        expected_inner_runs
    ):
        errors.append(
            "Incorrect number of inner metric rows."
        )

    if len(inner_thresholds) != (
        expected_inner_runs
    ):
        errors.append(
            "Incorrect number of inner threshold rows."
        )

    key = [
        "outer_fold",
        "inner_fold",
    ]

    for name, table in [
        (
            "inner_runs",
            inner_runs,
        ),
        (
            "inner_metrics",
            inner_metrics,
        ),
        (
            "inner_thresholds",
            inner_thresholds,
        ),
    ]:
        if table[
            key
        ].duplicated().any():
            errors.append(
                f"Duplicate keys in {name}."
            )

    # --------------------------------------------------
    # Block-reading and I/O policy checks
    # --------------------------------------------------
    
    if not inner_runs[
        "batched_block_reading"
    ].astype(bool).all():
    
        errors.append(
            "Some final training runs did not use "
            "batched block reading."
        )
    
    
    if inner_runs[
        "training_preload_fif"
    ].astype(bool).any():
    
        errors.append(
            "preload_fif=True was unexpectedly used "
            "for final training runs."
        )
    
    
    if not inner_runs[
        "validation_block_reading"
    ].astype(bool).all():
    
        errors.append(
            "Some validation runs did not use "
            "recording-block reading."
        )
    
    
    if inner_runs[
        "validation_preload_fif"
    ].astype(bool).any():
    
        errors.append(
            "Validation preload_fif should be False "
            "for baseline_v1."
        )
    
    
    if not inner_runs[
        "validation_order"
    ].eq(
        "recording_contiguous"
    ).all():
    
        errors.append(
            "Validation order was not "
            "recording-contiguous."
        )
    # --------------------------------------------------
    # Threshold leakage
    # --------------------------------------------------

    if not inner_thresholds[
        "threshold_source"
    ].eq(
        "inner_validation"
    ).all():
        errors.append(
            "A threshold was not derived "
            "from inner validation."
        )

    if not inner_thresholds[
        "threshold"
    ].between(
        0,
        1,
    ).all():
        errors.append(
            "Invalid inner threshold."
        )

    # --------------------------------------------------
    # Recalculate selected epoch and threshold
    # --------------------------------------------------

    for selected_row in (
        selected.itertuples(
            index=False
        )
    ):
        outer_fold = int(
            selected_row.outer_fold
        )

        fold_runs = (
            inner_runs.loc[
                inner_runs[
                    "outer_fold"
                ]
                == outer_fold
            ]
        )

        fold_thresholds = (
            inner_thresholds.loc[
                inner_thresholds[
                    "outer_fold"
                ]
                == outer_fold
            ]
        )

        expected_epoch = max(
            1,
            round_half_up(
                float(
                    np.median(
                        fold_runs[
                            "best_epoch"
                        ]
                    )
                )
            ),
        )

        expected_threshold = float(
            np.median(
                fold_thresholds[
                    "threshold"
                ]
            )
        )

        if (
            int(
                selected_row.selected_epoch
            )
            != expected_epoch
        ):
            errors.append(
                f"Outer {outer_fold}: "
                "selected epoch does not equal "
                "median inner epoch."
            )

        if not np.isclose(
            float(
                selected_row.selected_threshold
            ),
            expected_threshold,
            rtol=0,
            atol=1e-12,
        ):
            errors.append(
                f"Outer {outer_fold}: "
                "selected threshold does not equal "
                "median inner threshold."
            )

        if (
            selected_row.threshold_source
            != (
                "median_inner_validation_threshold"
            )
        ):
            errors.append(
                f"Outer {outer_fold}: "
                "invalid threshold source."
            )

    # --------------------------------------------------
    # CNN outer metrics
    # --------------------------------------------------

    cnn_metrics = outer_metrics.loc[
        outer_metrics[
            "model_name"
        ]
        == config["model"]["name"]
    ].copy()

    if len(cnn_metrics) != (
        expected_outer_count
    ):
        errors.append(
            "Expected exactly one CNN metric "
            "row per outer fold."
        )

    if cnn_metrics[
        "outer_fold"
    ].duplicated().any():
        errors.append(
            "Duplicate CNN outer fold results."
        )

    # --------------------------------------------------
    # Check finite metrics
    # --------------------------------------------------

    metric_columns = [
        "average_precision",
        "roc_auc",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "matthews_corrcoef",
    ]

    for column in (
        metric_columns
    ):
        if column not in (
            outer_metrics.columns
        ):
            errors.append(
                f"Missing metric column: "
                f"{column}"
            )

            continue

        values = outer_metrics[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            errors.append(
                f"Non-finite values in "
                f"{column}."
            )

    # --------------------------------------------------
    # Prediction integrity
    # --------------------------------------------------

    if not predictions[
        "window_id"
    ].is_unique:
        errors.append(
            "Outer prediction window IDs "
            "are duplicated."
        )

    if not predictions[
        "probability"
    ].between(
        0,
        1,
    ).all():
        errors.append(
            "Predicted probabilities outside [0,1]."
        )

    if not predictions[
        "true_label"
    ].isin(
        [
            0,
            1,
        ]
    ).all():
        errors.append(
            "Invalid true labels."
        )

    if not predictions[
        "predicted_label"
    ].isin(
        [
            0,
            1,
        ]
    ).all():
        errors.append(
            "Invalid predicted labels."
        )

    recalculated_predictions = (
        predictions[
            "probability"
        ]
        >= predictions[
            "threshold"
        ]
    ).astype(int)

    if not np.array_equal(
        recalculated_predictions.to_numpy(),
        predictions[
            "predicted_label"
        ].astype(int).to_numpy(),
    ):
        errors.append(
            "Predicted labels do not match "
            "stored thresholds."
        )

    if not predictions[
        "threshold_source"
    ].eq(
        "median_inner_validation_threshold"
    ).all():
        errors.append(
            "Outer prediction threshold "
            "source is invalid."
        )

    # --------------------------------------------------
    # Threshold must equal frozen selected value
    # --------------------------------------------------

    merged_thresholds = (
        predictions.merge(
            selected[
                [
                    "outer_fold",
                    "selected_threshold",
                ]
            ],
            on="outer_fold",
            how="left",
            validate="many_to_one",
        )
    )

    threshold_match = np.isclose(
        merged_thresholds[
            "threshold"
        ].to_numpy(
            dtype=float
        ),
        merged_thresholds[
            "selected_threshold"
        ].to_numpy(
            dtype=float
        ),
        rtol=0,
        atol=1e-12,
    )

    if not threshold_match.all():
        errors.append(
            "Outer predictions used a threshold "
            "different from the frozen inner-CV value."
        )

    # --------------------------------------------------
    # Patient-independent test-fold integrity
    # --------------------------------------------------

    expected_subject_fold = (
        outer_folds.set_index(
            "subject_id"
        )[
            "outer_test_fold"
        ].astype(int).to_dict()
    )

    for subject_id, group in (
        predictions.groupby(
            "subject_id"
        )
    ):
        if subject_id not in (
            expected_subject_fold
        ):
            errors.append(
                "Prediction for unknown subject: "
                f"{subject_id}"
            )

            continue

        observed_folds = set(
            group[
                "outer_fold"
            ].astype(int)
        )

        expected_fold = int(
            expected_subject_fold[
                subject_id
            ]
        )

        if observed_folds != {
            expected_fold
        }:
            errors.append(
                f"Subject {subject_id} "
                "appears in wrong outer fold."
            )

    # --------------------------------------------------
    # Outer scaler leakage
    # --------------------------------------------------

    for outer_fold in range(
        expected_outer_count
    ):
        scaler, _ = (
            load_outer_scaler(
                project_root=(
                    PROJECT_ROOT
                ),
                config=config,
                outer_fold=(
                    outer_fold
                ),
            )
        )

        scaler_subjects = set(
            scaler[
                "metadata"
            ][
                "training_subjects"
            ]
        )

        test_subjects = set(
            outer_folds.loc[
                outer_folds[
                    "outer_test_fold"
                ]
                == outer_fold,
                "subject_id",
            ].astype(str)
        )

        overlap = (
            scaler_subjects
            & test_subjects
        )

        if overlap:
            errors.append(
                f"Outer {outer_fold}: "
                "test subject contributed "
                f"to scaler: {sorted(overlap)}"
            )

    # --------------------------------------------------
    # Check checkpoints
    # --------------------------------------------------

    for run in (
        inner_runs.itertuples(
            index=False
        )
    ):
        checkpoint_path = (
            PROJECT_ROOT
            / run.checkpoint_path
        )

        if not checkpoint_path.exists():
            errors.append(
                "Missing inner checkpoint: "
                f"{checkpoint_path}"
            )

    for metric_row in (
        cnn_metrics.itertuples(
            index=False
        )
    ):
        checkpoint_path = (
            PROJECT_ROOT
            / metric_row.checkpoint_path
        )

        if not checkpoint_path.exists():
            errors.append(
                "Missing outer checkpoint: "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------
    # Validation records
    # --------------------------------------------------

    for outer_fold in range(
        expected_outer_count
    ):
        outer_predictions = (
            predictions.loc[
                predictions[
                    "outer_fold"
                ]
                == outer_fold
            ]
        )

        validation_rows.append(
            {
                "outer_fold": (
                    outer_fold
                ),
                "prediction_count": (
                    len(
                        outer_predictions
                    )
                ),
                "subject_count": (
                    outer_predictions[
                        "subject_id"
                    ].nunique()
                ),
                "recording_count": (
                    outer_predictions[
                        "recording_id"
                    ].nunique()
                ),
                "positive_count": int(
                    (
                        outer_predictions[
                            "true_label"
                        ]
                        == 1
                    ).sum()
                ),
                "negative_count": int(
                    (
                        outer_predictions[
                            "true_label"
                        ]
                        == 0
                    ).sum()
                ),
                "status": "valid",
            }
        )

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        output_dir
        / "chbmit_baseline_validation.csv",
        index=False,
    )

    summary = {
        "baseline_version": (
            config[
                "project"
            ]["baseline_version"]
        ),
        "inner_runs_expected": (
            expected_inner_runs
        ),
        "inner_runs_observed": int(
            len(
                inner_runs
            )
        ),
        "outer_runs_expected": (
            expected_outer_count
        ),
        "outer_cnn_runs_observed": int(
            len(
                cnn_metrics
            )
        ),
        "outer_prediction_count": int(
            len(
                predictions
            )
        ),
        "outer_test_subject_count": int(
            predictions[
                "subject_id"
            ].nunique()
        ),
        "errors": int(
            len(errors)
        ),
        "warnings": int(
            len(warnings)
        ),
    }

    with (
        output_dir
        / "chbmit_baseline_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nBaseline validation\n"
    )

    print(
        f"Inner runs: "
        f"{len(inner_runs)} / "
        f"{expected_inner_runs}"
    )

    print(
        f"Outer CNN runs: "
        f"{len(cnn_metrics)} / "
        f"{expected_outer_count}"
    )

    print(
        f"Prediction windows: "
        f"{len(predictions)}"
    )

    print(
        f"Errors: {len(errors)}"
    )

    print(
        f"Warnings: {len(warnings)}"
    )

    for error in errors:
        print(
            "ERROR:",
            error,
        )

    for warning in warnings:
        print(
            "WARNING:",
            warning,
        )

    return (
        1
        if errors
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )