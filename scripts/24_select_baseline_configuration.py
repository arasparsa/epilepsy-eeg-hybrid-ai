"""Freeze outer-fold epoch count and threshold from inner CV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.baseline_utils import (
    load_yaml,
    round_half_up,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_baseline.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    config = load_yaml(
        args.config
    )

    output_dir = (
        PROJECT_ROOT
        / config["outputs"][
            "metadata_directory"
        ]
    )

    runs = pd.read_csv(
        output_dir
        / "chbmit_inner_training_runs.csv"
    )

    metrics = pd.read_csv(
        output_dir
        / "chbmit_inner_metrics.csv"
    )

    thresholds = pd.read_csv(
        output_dir
        / "chbmit_inner_thresholds.csv"
    )

    inner_assignments = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "inner_subject_folds"
        ]
    )

    expected_inner_folds = int(
        inner_assignments[
            "inner_fold"
        ].nunique()
    )

    selected_rows: list[dict] = []

    for outer_fold in sorted(
        runs[
            "outer_fold"
        ].unique().astype(int)
    ):
        outer_runs = runs.loc[
            runs[
                "outer_fold"
            ]
            == outer_fold
        ].copy()

        outer_metrics = metrics.loc[
            metrics[
                "outer_fold"
            ]
            == outer_fold
        ].copy()

        outer_thresholds = (
            thresholds.loc[
                thresholds[
                    "outer_fold"
                ]
                == outer_fold
            ].copy()
        )

        if (
            len(outer_runs)
            != expected_inner_folds
        ):
            raise ValueError(
                f"Outer {outer_fold}: "
                f"expected {expected_inner_folds} "
                f"inner runs, found "
                f"{len(outer_runs)}."
            )

        if (
            len(outer_metrics)
            != expected_inner_folds
        ):
            raise ValueError(
                "Incomplete inner metrics."
            )

        if (
            len(outer_thresholds)
            != expected_inner_folds
        ):
            raise ValueError(
                "Incomplete inner thresholds."
            )

        if (
            outer_runs[
                "inner_fold"
            ].duplicated().any()
        ):
            raise ValueError(
                "Duplicate inner runs."
            )

        median_epoch = float(
            np.median(
                outer_runs[
                    "best_epoch"
                ]
            )
        )

        selected_epoch = max(
            1,
            round_half_up(
                median_epoch
            ),
        )

        selected_threshold = float(
            np.median(
                outer_thresholds[
                    "threshold"
                ]
            )
        )

        selected_rows.append(
            {
                "outer_fold": (
                    outer_fold
                ),
                "model_name": (
                    config[
                        "model"
                    ]["name"]
                ),
                "selected_epoch": (
                    selected_epoch
                ),
                "raw_median_best_epoch": (
                    median_epoch
                ),
                "epoch_source": (
                    "median_best_inner_epoch"
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "threshold_source": (
                    "median_inner_validation_threshold"
                ),
                "mean_inner_average_precision": float(
                    outer_metrics[
                        "average_precision"
                    ].mean()
                ),
                "std_inner_average_precision": float(
                    outer_metrics[
                        "average_precision"
                    ].std(
                        ddof=1
                    )
                ),
                "mean_inner_roc_auc": float(
                    outer_metrics[
                        "roc_auc"
                    ].mean()
                ),
                "std_inner_roc_auc": float(
                    outer_metrics[
                        "roc_auc"
                    ].std(
                        ddof=1
                    )
                ),
                "inner_fold_count": (
                    expected_inner_folds
                ),
                "configuration_frozen": True,
                "timestamp_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }
        )

    selected = pd.DataFrame(
        selected_rows
    ).sort_values(
        "outer_fold"
    )

    output_path = (
        output_dir
        / "chbmit_selected_baseline_config.csv"
    )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            "Selected configuration "
            "already exists. Use --overwrite."
        )

    selected.to_csv(
        output_path,
        index=False,
    )

    print(
        selected.to_string(
            index=False
        )
    )

    print(
        "\nBaseline configuration frozen."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )