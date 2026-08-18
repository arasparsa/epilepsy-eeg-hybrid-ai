"""Train SimpleEEGCNN on all inner patient-independent folds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.data.fold_selection import (
    get_nested_window_tables,
    get_subjects_for_nested_fold,
)
from src.evaluation.predictions import (
    predict_loader,
)
from src.evaluation.thresholding import (
    select_threshold_max_f1,
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
    get_inner_pos_weight,
    load_inner_scaler,
    load_yaml,
    prepare_master_windows,
    resolve_device,
    seed_experiment,
    upsert_csv,
)
from src.models.simple_cnn import (
    count_trainable_parameters,
)
from src.preprocessing.provenance import (
    get_git_commit_hash,
)
from src.training.trainer import (
    train_with_early_stopping,
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
        "--inner-fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--maximum-epochs",
        type=int,
        default=None,
        help=(
            "Optional smoke-test override. "
            "Do not use this for final experiments "
            "unless documented."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    
#########
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-validation-windows",
        type=int,
        default=None,
    )
#########


    return parser.parse_args()

#########
def build_smoke_subset(
    table: pd.DataFrame,
    maximum_windows: int,
    random_state: int,
) -> pd.DataFrame:
    """Create a deterministic binary smoke-test subset."""
    if maximum_windows <= 0:
        raise ValueError(
            "maximum_windows must be positive."
        )

    if len(table) <= maximum_windows:
        return table.copy().reset_index(
            drop=True
        )

    positives = table.loc[
        table["binary_label"] == 1
    ]

    negatives = table.loc[
        table["binary_label"] == 0
    ]

    if positives.empty or negatives.empty:
        raise ValueError(
            "Smoke subset requires both classes."
        )

    positive_target = min(
        len(positives),
        maximum_windows // 2,
    )

    negative_target = (
        maximum_windows
        - positive_target
    )

    negative_target = min(
        negative_target,
        len(negatives),
    )

    if (
        positive_target
        + negative_target
        < maximum_windows
    ):
        additional_positive = min(
            len(positives)
            - positive_target,
            maximum_windows
            - positive_target
            - negative_target,
        )

        positive_target += (
            additional_positive
        )

    selected_positive = (
        positives.sample(
            n=positive_target,
            replace=False,
            random_state=random_state,
        )
    )

    selected_negative = (
        negatives.sample(
            n=negative_target,
            replace=False,
            random_state=(
                random_state + 1
            ),
        )
    )

    return (
        pd.concat(
            [
                selected_positive,
                selected_negative,
            ],
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=(
                random_state + 2
            ),
        )
        .reset_index(
            drop=True
        )
    )
#########



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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows = (
        prepare_master_windows(
            project_root=PROJECT_ROOT,
            config=config,
        )
    )

    inner_assignments = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "inner_subject_folds"
        ]
    )

    class_weights = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "class_weights"
        ]
    )

    outer_fold_ids = sorted(
        inner_assignments[
            "outer_fold"
        ].unique().astype(int)
    )

    inner_fold_ids = sorted(
        inner_assignments[
            "inner_fold"
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

    if args.inner_fold is not None:
        if args.inner_fold not in (
            inner_fold_ids
        ):
            raise ValueError(
                "Requested inner fold "
                "does not exist."
            )

        inner_fold_ids = [
            args.inner_fold
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

    maximum_epochs = (
        int(
            args.maximum_epochs
        )
        if args.maximum_epochs
        is not None
        else int(
            config[
                "training"
            ]["maximum_epochs"]
        )
    )

    if args.maximum_epochs is not None:
        print(
            "WARNING: maximum epoch "
            "override is active."
        )

    run_rows: list[dict] = []
    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []

    for outer_fold in (
        outer_fold_ids
    ):
        for inner_fold in (
            inner_fold_ids
        ):
            print(
                "\n"
                + "=" * 70
            )

            print(
                f"OUTER {outer_fold} "
                f"| INNER {inner_fold}"
            )

            print(
                "=" * 70
            )

            seed = (
                get_experiment_seed(
                    config=config,
                    outer_fold=(
                        outer_fold
                    ),
                    inner_fold=(
                        inner_fold
                    ),
                )
            )

            seed_experiment(
                seed=seed,
                data_pipeline_config=(
                    data_pipeline_config
                ),
            )

            subject_sets = (
                get_subjects_for_nested_fold(
                    inner_assignments=(
                        inner_assignments
                    ),
                    outer_fold=(
                        outer_fold
                    ),
                    inner_fold=(
                        inner_fold
                    ),
                )
            )

            # Notice:
            # only train and validation are used below.
            # No outer-test DataLoader is constructed.
            fold_tables = (
                get_nested_window_tables(
                    windows=windows,
                    inner_assignments=(
                        inner_assignments
                    ),
                    outer_fold=(
                        outer_fold
                    ),
                    inner_fold=(
                        inner_fold
                    ),
                )
            )

            train_windows = (
                fold_tables[
                    "train"
                ]
            )

            validation_windows = (
                fold_tables[
                    "validation"
                ]
            )
############### 
            #if args.max_train_windows is not None:
             #   train_windows = (
              #      train_windows
               #     .sort_values("window_id")
                #    .head(
                 #       args.max_train_windows
                  #  )
                   # .reset_index(drop=True)
                #)
            #if (
             #   args.max_validation_windows
              #  is not None
            #):
             #   validation_windows = (
              #      validation_windows
               #     .sort_values("window_id")
                #    .head(
                 #       args.max_validation_windows
                  #  )
                   # .reset_index(drop=True)
                #)    

	#######
            if args.max_train_windows is not None:
                train_windows = (
                    build_smoke_subset(
                        table=train_windows,
                        maximum_windows=(
                            args.max_train_windows
                        ),
                        random_state=seed,
                    )
                )

            if (
                args.max_validation_windows
                is not None
            ):
                validation_windows = (
                    build_smoke_subset(
                        table=validation_windows,
                        maximum_windows=(
                            args.max_validation_windows
                        ),
                        random_state=(
                            seed + 10_000
                        ),
                    )
                )

###############

            scaler, scaler_path = (
                load_inner_scaler(
                    project_root=(
                        PROJECT_ROOT
                    ),
                    config=config,
                    outer_fold=(
                        outer_fold
                    ),
                    inner_fold=(
                        inner_fold
                    ),
                )
            )

            pos_weight = (
                get_inner_pos_weight(
                    class_weights=(
                        class_weights
                    ),
                    outer_fold=(
                        outer_fold
                    ),
                    inner_fold=(
                        inner_fold
                    ),
                )
            )

            train_loader = (
                build_loader(
                    windows=(
                        train_windows
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
            ###########
            print(
                "\nTraining dataset"
            )
            
            print(
                "  Full binary windows:",
                f"{len(train_windows):,}",
            )
            
            print(
                "  Positive windows:",
                f"{int((train_windows['binary_label'] == 1).sum()):,}",
            )
            
            print(
                "  Negative windows:",
                f"{int((train_windows['binary_label'] == 0).sum()):,}",
            )
            
            
            batch_sampler = getattr(
                train_loader,
                "batch_sampler",
                None,
            )
            
            if (
                batch_sampler is not None
                and hasattr(
                    batch_sampler,
                    "epoch_summary",
                )
            ):
                sampling_summary = (
                    batch_sampler.epoch_summary()
                )
            
                print(
                    "\nDynamic epoch sampling"
                )
            
                print(
                    "  Positive / epoch:",
                    f"{sampling_summary['positive_windows']:,}",
                )
            
                print(
                    "  Negative / epoch:",
                    f"{sampling_summary['negative_windows']:,}",
                )
            
                print(
                    "  Total / epoch:",
                    f"{sampling_summary['total_windows']:,}",
                )
            
                print(
                    "  Neg:Pos ratio:",
                    sampling_summary[
                        "negative_to_positive_ratio"
                    ],
                )
        
            ###########


            validation_loader = (
                build_loader(
                    windows=(
                        validation_windows
                    ),
                    role=(
                        "validation"
                    ),
                    scaler=scaler,
                    seed=seed + 10_000,
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

            run_directory = (
                model_root
                / (
                    f"outer_"
                    f"{outer_fold:02d}"
                )
                / (
                    f"inner_"
                    f"{inner_fold:02d}"
                )
            )

            run_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint_path = (
                run_directory
                / "best_model.pt"
            )

            if (
                checkpoint_path.exists()
                and not args.overwrite
            ):
                raise FileExistsError(
                    "Checkpoint already exists: "
                    f"{checkpoint_path}. "
                    "Use --overwrite."
                )

            (
                model,
                history,
                training_result,
            ) = (
                train_with_early_stopping(
                    model=model,
                    train_loader=(
                        train_loader
                    ),
                    validation_loader=(
                        validation_loader
                    ),
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    maximum_epochs=(
                        maximum_epochs
                    ),
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    outer_fold=(
                        outer_fold
                    ),
                    inner_fold=(
                        inner_fold
                    ),
                    seed=seed,
                    model_name=config[
                        "model"
                    ]["name"],
                    patience=int(
                        config[
                            "training"
                        ][
                            "early_stopping"
                        ]["patience"]
                    ),
                    minimum_delta=float(
                        config[
                            "training"
                        ][
                            "early_stopping"
                        ][
                            "minimum_delta"
                        ]
                    ),
                    gradient_clip_max_norm=(
                        float(
                            config[
                                "training"
                            ][
                                "gradient_clipping"
                            ][
                                "max_norm"
                            ]
                        )
                        if config[
                            "training"
                        ][
                            "gradient_clipping"
                        ][
                            "enabled"
                        ]
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
            )

            history_path = (
                report_root
                / (
                    f"outer_"
                    f"{outer_fold:02d}"
                    "_inner_"
                    f"{inner_fold:02d}"
                    "_history.csv"
                )
            )

            history.to_csv(
                history_path,
                index=False,
            )

            validation_predictions = (
                predict_loader(
                    model=model,
                    loader=(
                        validation_loader
                    ),
                    device=device,
                )
            )

            threshold_config = (
                config[
                    "evaluation"
                ][
                    "threshold_selection"
                ]
            )

            threshold_result = (
                select_threshold_max_f1(
                    y_true=(
                        validation_predictions[
                            "true_label"
                        ].to_numpy()
                    ),
                    probabilities=(
                        validation_predictions[
                            "probability"
                        ].to_numpy()
                    ),
                    minimum_threshold=float(
                        threshold_config[
                            "minimum_threshold"
                        ]
                    ),
                    maximum_threshold=float(
                        threshold_config[
                            "maximum_threshold"
                        ]
                    ),
                    number_of_thresholds=int(
                        threshold_config[
                            "number_of_thresholds"
                        ]
                    ),
                    tie_break=str(
                        threshold_config[
                            "tie_break"
                        ]
                    ),
                )
            )

            selected_threshold = (
                float(
                    threshold_result[
                        "threshold"
                    ]
                )
            )

            metrics = (
                calculate_binary_metrics(
                    y_true=(
                        validation_predictions[
                            "true_label"
                        ].to_numpy()
                    ),
                    probabilities=(
                        validation_predictions[
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

            run_rows.append(
                {
                    "outer_fold": (
                        outer_fold
                    ),
                    "inner_fold": (
                        inner_fold
                    ),
                    "model_name": (
                        config[
                            "model"
                        ]["name"]
                    ),
                    "seed": seed,
                    "device": str(
                        device
                    ),
                    "train_subject_count": (
                        len(
                            subject_sets[
                                "train"
                            ]
                        )
                    ),
                    "validation_subject_count": (
                        len(
                            subject_sets[
                                "validation"
                            ]
                        )
                    ),
                    # IDs may be recorded for audit,
                    # even though the test data were not loaded.
                    "test_subject_count": (
                        len(
                            subject_sets[
                                "test"
                            ]
                        )
                    ),
                    "train_window_count": (
                        len(
                            train_windows
                        )
                    ),
                    "validation_window_count": (
                        len(
                            validation_windows
                        )
                    ),
                    "train_positive_count": int(
                        (
                            train_windows[
                                "binary_label"
                            ]
                            == 1
                        ).sum()
                    ),
                    "train_negative_count": int(
                        (
                            train_windows[
                                "binary_label"
                            ]
                            == 0
                        ).sum()
                    ),
                    "pos_weight": (
                        pos_weight
                    ),
                    "best_epoch": int(
                        training_result[
                            "best_epoch"
                        ]
                    ),
                    "epochs_completed": int(
                        training_result[
                            "epochs_completed"
                        ]
                    ),
                    "best_validation_average_precision": float(
                        training_result[
                            "best_validation_average_precision"
                        ]
                    ),
                    "parameter_count": (
                        count_trainable_parameters(
                            model
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
                    "history_path": str(
                        history_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "git_commit": (
                        git_commit
                    ),
                    "timestamp_utc": (
                        timestamp
                    ),

                    ##################
                    "batched_block_reading": bool(
                        config[
                            "data_access"
                        ][
                            "training"
                        ].get(
                            "batched_block_reading",
                            True,
                        )
                    ),
                    
                    "training_preload_fif": bool(
                        config[
                            "data_access"
                        ][
                            "training"
                        ].get(
                            "preload_fif",
                            False,
                        )
                    ),
                    
                    "validation_block_reading": bool(
                        config[
                            "data_access"
                        ][
                            "validation"
                        ].get(
                            "batched_block_reading",
                            True,
                        )
                    ),
                    
                    "validation_preload_fif": bool(
                        config[
                            "data_access"
                        ][
                            "validation"
                        ].get(
                            "preload_fif",
                            False,
                        )
                    ),
                    
                    "validation_order": (
                        "recording_contiguous"
                    ),
                    ##################
                }
            )

            metric_rows.append(
                {
                    "outer_fold": (
                        outer_fold
                    ),
                    "inner_fold": (
                        inner_fold
                    ),
                    "model_name": (
                        config[
                            "model"
                        ]["name"]
                    ),
                    **metrics,
                }
            )

            threshold_rows.append(
                {
                    "outer_fold": (
                        outer_fold
                    ),
                    "inner_fold": (
                        inner_fold
                    ),
                    "threshold": (
                        selected_threshold
                    ),
                    "validation_f1": (
                        threshold_result[
                            "validation_f1"
                        ]
                    ),
                    "threshold_source": (
                        "inner_validation"
                    ),
                    "selection_method": (
                        "validation_max_f1"
                    ),
                    "tie_count": (
                        threshold_result[
                            "tie_count"
                        ]
                    ),
                }
            )

            # Explicitly free GPU memory.
            del (
                model,
                optimizer,
                criterion,
                train_loader,
                validation_loader,
            )

            if (
                torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

    upsert_csv(
        path=(
            output_dir
            / "chbmit_inner_training_runs.csv"
        ),
        new_rows=pd.DataFrame(
            run_rows
        ),
        key_columns=[
            "outer_fold",
            "inner_fold",
        ],
    )

    upsert_csv(
        path=(
            output_dir
            / "chbmit_inner_metrics.csv"
        ),
        new_rows=pd.DataFrame(
            metric_rows
        ),
        key_columns=[
            "outer_fold",
            "inner_fold",
        ],
    )

    upsert_csv(
        path=(
            output_dir
            / "chbmit_inner_thresholds.csv"
        ),
        new_rows=pd.DataFrame(
            threshold_rows
        ),
        key_columns=[
            "outer_fold",
            "inner_fold",
        ],
    )

    print(
        "\nInner baseline training complete."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )