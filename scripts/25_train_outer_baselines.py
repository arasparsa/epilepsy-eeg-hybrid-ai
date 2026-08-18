"""Train frozen baseline on outer development and evaluate outer test."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.fold_selection import (
    get_outer_development_tables,
)
from src.evaluation.predictions import (
    predict_loader,
)
from src.evaluation.window_metrics import (
    calculate_binary_metrics,
)
from src.experiments.baseline_utils import (
    build_criterion,
    build_loader,
    build_model_from_config,
    build_optimizer,
    get_experiment_seed,
    get_outer_pos_weight,
    load_outer_scaler,
    load_yaml,
    prepare_master_windows,
    resolve_device,
    seed_experiment,
    upsert_csv,
)
from src.preprocessing.provenance import (
    get_git_commit_hash,
)
from src.training.trainer import (
    train_fixed_epochs,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_baseline.yaml"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--outer-fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    config = load_yaml(
        args.config
    )

    data_pipeline_config = (
        load_yaml(
            PROJECT_ROOT
            / config["inputs"][
                "data_pipeline_config"
            ]
        )
    )

    output_dir = (
        PROJECT_ROOT
        / config["outputs"][
            "metadata_directory"
        ]
    )

    model_root = (
        PROJECT_ROOT
        / config["outputs"][
            "model_directory"
        ]
    )

    report_root = (
        PROJECT_ROOT
        / config["outputs"][
            "report_directory"
        ]
        / "training_curves"
    )

    selected_config = (
        pd.read_csv(
            output_dir
            / "chbmit_selected_baseline_config.csv"
        )
    )

    windows = (
        prepare_master_windows(
            project_root=PROJECT_ROOT,
            config=config,
        )
    )

    outer_subject_folds = (
        pd.read_csv(
            PROJECT_ROOT
            / config["inputs"][
                "outer_subject_folds"
            ]
        )
    )

    class_weights = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "class_weights"
        ]
    )

    outer_fold_ids = sorted(
        selected_config[
            "outer_fold"
        ].unique().astype(int)
    )

    if args.outer_fold is not None:
        if args.outer_fold not in (
            outer_fold_ids
        ):
            raise ValueError(
                "Requested outer fold "
                "does not exist."
            )

        outer_fold_ids = [
            args.outer_fold
        ]

    device = resolve_device(
        config["runtime"][
            "device"
        ]
    )

    print(
        f"Device: {device}"
    )

    git_commit = (
        get_git_commit_hash()
    )

    metric_rows: list[dict] = []
    prediction_rows: list[
        pd.DataFrame
    ] = []

    for outer_fold in (
        outer_fold_ids
    ):
        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FINAL OUTER FOLD "
            f"{outer_fold}"
        )

        print(
            "=" * 70
        )

        selected_row = (
            selected_config.loc[
                selected_config[
                    "outer_fold"
                ]
                == outer_fold
            ]
        )

        if len(
            selected_row
        ) != 1:
            raise ValueError(
                "Expected exactly one frozen "
                "outer configuration."
            )

        selected_row = (
            selected_row.iloc[0]
        )

        if (
            selected_row[
                "threshold_source"
            ]
            != (
                "median_inner_validation_threshold"
            )
        ):
            raise RuntimeError(
                "Outer threshold source "
                "is not valid."
            )

        selected_epochs = int(
            selected_row[
                "selected_epoch"
            ]
        )

        selected_threshold = (
            float(
                selected_row[
                    "selected_threshold"
                ]
            )
        )

        seed = (
            get_experiment_seed(
                config=config,
                outer_fold=(
                    outer_fold
                ),
                inner_fold=None,
            )
        )

        seed_experiment(
            seed=seed,
            data_pipeline_config=(
                data_pipeline_config
            ),
        )

        tables = (
            get_outer_development_tables(
                windows=windows,
                outer_subject_folds=(
                    outer_subject_folds
                ),
                outer_fold=(
                    outer_fold
                ),
            )
        )

        development_windows = (
            tables[
                "development"
            ]
        )

        test_windows = (
            tables[
                "test"
            ]
        )

        development_subjects = set(
            development_windows[
                "subject_id"
            ].astype(str).unique()
        )

        test_subjects = set(
            test_windows[
                "subject_id"
            ].astype(str).unique()
        )

        if (
            development_subjects
            & test_subjects
        ):
            raise RuntimeError(
                "Outer subject leakage detected."
            )

        scaler, scaler_path = (
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

        # Strong leakage audit:
        scaler_metadata = (
            scaler[
                "metadata"
            ]
        )

        scaler_subjects = set(
            scaler_metadata[
                "training_subjects"
            ]
        )

        if (
            scaler_subjects
            != development_subjects
        ):
            raise RuntimeError(
                "Outer scaler subject mismatch."
            )

        if (
            scaler_subjects
            & test_subjects
        ):
            raise RuntimeError(
                "Outer-test subject contributed "
                "to scaler."
            )

        pos_weight = (
            get_outer_pos_weight(
                class_weights=(
                    class_weights
                ),
                outer_fold=(
                    outer_fold
                ),
            )
        )

        development_loader = (
            build_loader(
                windows=(
                    development_windows
                ),
                role="train",
                scaler=scaler,
                seed=seed,
                project_root=(
                    PROJECT_ROOT
                ),
                baseline_config=(
                    config
                ),
                data_pipeline_config=(
                    data_pipeline_config
                ),
            )
        )

        # The test loader is now allowed because
        # epoch count and threshold are already frozen.
        test_loader = (
            build_loader(
                windows=(
                    test_windows
                ),
                role="test",
                scaler=scaler,
                seed=seed + 20_000,
                project_root=(
                    PROJECT_ROOT
                ),
                baseline_config=(
                    config
                ),
                data_pipeline_config=(
                    data_pipeline_config
                ),
            )
        )

        model = (
            build_model_from_config(
                config
            ).to(
                device
            )
        )

        optimizer = (
            build_optimizer(
                model=model,
                config=config,
            )
        )

        criterion = (
            build_criterion(
                pos_weight=(
                    pos_weight
                ),
                device=device,
                config=config,
            )
        )

        fold_directory = (
            model_root
            / (
                f"outer_"
                f"{outer_fold:02d}"
            )
        )

        fold_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            fold_directory
            / "final_outer_model.pt"
        )

        if (
            checkpoint_path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                "Outer checkpoint exists. "
                "Use --overwrite."
            )

        (
            model,
            training_history,
        ) = train_fixed_epochs(
            model=model,
            train_loader=(
                development_loader
            ),
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=(
                selected_epochs
            ),
            checkpoint_path=(
                checkpoint_path
            ),
            outer_fold=(
                outer_fold
            ),
            seed=seed,
            model_name=config[
                "model"
            ]["name"],
            gradient_clip_max_norm=(
                float(
                    config[
                        "training"
                    ][
                        "gradient_clipping"
                    ]["max_norm"]
                )
                if config[
                    "training"
                ][
                    "gradient_clipping"
                ]["enabled"]
                else None
            ),
            mixed_precision=bool(
                config[
                    "training"
                ][
                    "mixed_precision"
                ]["enabled"]
            ),
        )

        history_path = (
            report_root
            / (
                f"outer_"
                f"{outer_fold:02d}"
                "_final_history.csv"
            )
        )

        training_history.to_csv(
            history_path,
            index=False,
        )

        test_predictions = (
            predict_loader(
                model=model,
                loader=(
                    test_loader
                ),
                device=device,
            )
        )

        test_predictions[
            "predicted_label"
        ] = (
            test_predictions[
                "probability"
            ]
            >= selected_threshold
        ).astype(int)

        test_predictions[
            "outer_fold"
        ] = outer_fold

        test_predictions[
            "threshold"
        ] = selected_threshold

        test_predictions[
            "threshold_source"
        ] = (
            "median_inner_validation_threshold"
        )

        # Add segmentation timing metadata needed later
        # for event-level reconstruction.
        timing_columns = [
            "window_id",
            "start_sample",
            "stop_sample_exclusive",
            "start_seconds",
            "end_seconds",
            "label_name",
            "seizure_overlap_seconds",
            "seizure_overlap_fraction",
            "near_seizure",
            "clean_non_ictal",
            "output_fif_path",
        ]

        available_timing_columns = [
            column
            for column
            in timing_columns
            if column
            in test_windows.columns
        ]

        test_predictions = (
            test_predictions.merge(
                test_windows[
                    available_timing_columns
                ],
                on="window_id",
                how="left",
                validate="one_to_one",
            )
        )

        cnn_metrics = (
            calculate_binary_metrics(
                y_true=(
                    test_predictions[
                        "true_label"
                    ].to_numpy()
                ),
                probabilities=(
                    test_predictions[
                        "probability"
                    ].to_numpy()
                ),
                threshold=(
                    selected_threshold
                ),
            )
        )

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        metric_rows.append(
            {
                "outer_fold": (
                    outer_fold
                ),
                "model_name": (
                    config[
                        "model"
                    ]["name"]
                ),
                "evaluation_level": (
                    "window"
                ),
                "selected_epoch": (
                    selected_epochs
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "epoch_source": (
                    "median_best_inner_epoch"
                ),
                "threshold_source": (
                    "median_inner_validation_threshold"
                ),
                "pos_weight": (
                    pos_weight
                ),
                "development_subject_count": (
                    len(
                        development_subjects
                    )
                ),
                "test_subject_count": (
                    len(
                        test_subjects
                    )
                ),
                "development_window_count": (
                    len(
                        development_windows
                    )
                ),
                "test_window_count": (
                    len(
                        test_windows
                    )
                ),
                "scaler_path": str(
                    scaler_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "checkpoint_path": str(
                    checkpoint_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "git_commit": (
                    git_commit
                ),
                "timestamp_utc": (
                    timestamp
                ),
                **cnn_metrics,
            }
        )

        # -------------------------------------------------
        # Constant prevalence baseline
        # -------------------------------------------------

        if config[
            "constant_baseline"
        ]["enabled"]:
            development_prevalence = float(
                development_windows[
                    "binary_label"
                ].astype(int).mean()
            )

            constant_probabilities = (
                np.full(
                    shape=len(
                        test_predictions
                    ),
                    fill_value=(
                        development_prevalence
                    ),
                    dtype=float,
                )
            )

            constant_threshold = float(
                config[
                    "constant_baseline"
                ][
                    "decision_threshold"
                ]
            )

            constant_metrics = (
                calculate_binary_metrics(
                    y_true=(
                        test_predictions[
                            "true_label"
                        ].to_numpy()
                    ),
                    probabilities=(
                        constant_probabilities
                    ),
                    threshold=(
                        constant_threshold
                    ),
                )
            )

            metric_rows.append(
                {
                    "outer_fold": (
                        outer_fold
                    ),
                    "model_name": (
                        "constant_prevalence"
                    ),
                    "evaluation_level": (
                        "window"
                    ),
                    "selected_epoch": 0,
                    "selected_threshold": (
                        constant_threshold
                    ),
                    "epoch_source": (
                        "not_applicable"
                    ),
                    "threshold_source": (
                        "fixed_0_5"
                    ),
                    "pos_weight": (
                        np.nan
                    ),
                    "development_subject_count": (
                        len(
                            development_subjects
                        )
                    ),
                    "test_subject_count": (
                        len(
                            test_subjects
                        )
                    ),
                    "development_window_count": (
                        len(
                            development_windows
                        )
                    ),
                    "test_window_count": (
                        len(
                            test_windows
                        )
                    ),
                    "scaler_path": (
                        "not_applicable"
                    ),
                    "checkpoint_path": (
                        "not_applicable"
                    ),
                    "git_commit": (
                        git_commit
                    ),
                    "timestamp_utc": (
                        timestamp
                    ),
                    **constant_metrics,
                }
            )

            test_predictions[
                "constant_probability"
            ] = (
                development_prevalence
            )

        prediction_rows.append(
            test_predictions
        )

        del (
            model,
            optimizer,
            criterion,
            development_loader,
            test_loader,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(
        metric_rows
    )

    predictions_df = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    upsert_csv(
        path=(
            output_dir
            / "chbmit_outer_metrics.csv"
        ),
        new_rows=(
            metrics_df
        ),
        key_columns=[
            "outer_fold",
            "model_name",
        ],
    )

    # Predictions must be replaced by fold, not
    # merely deduplicated blindly.
    predictions_path = (
        output_dir
        / "chbmit_outer_predictions.csv"
    )

    if predictions_path.exists():
        old_predictions = (
            pd.read_csv(
                predictions_path
            )
        )

        replaced_folds = set(
            predictions_df[
                "outer_fold"
            ].astype(int)
        )

        old_predictions = (
            old_predictions.loc[
                ~old_predictions[
                    "outer_fold"
                ].astype(int).isin(
                    replaced_folds
                )
            ]
        )

        predictions_df = pd.concat(
            [
                old_predictions,
                predictions_df,
            ],
            ignore_index=True,
        )

    if not predictions_df[
        "window_id"
    ].is_unique:
        raise RuntimeError(
            "Outer prediction window IDs "
            "are not unique."
        )

    predictions_df = (
        predictions_df.sort_values(
            [
                "outer_fold",
                "subject_id",
                "recording_id",
                "window_id",
            ]
        )
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print(
        "\nOuter evaluation complete."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )