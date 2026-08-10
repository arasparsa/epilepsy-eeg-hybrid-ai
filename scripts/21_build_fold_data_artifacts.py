"""Build train-only scalers and class-weight artifacts for nested CV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.data.fold_selection import (
    get_nested_window_tables,
    get_outer_development_tables,
    get_subjects_for_nested_fold,
)
from src.imbalance.class_weights import (
    calculate_pos_weight,
    count_binary_classes,
)
from src.normalization.fold_scalers import (
    build_fold_scaler,
    save_scaler_npz,
)
from src.preprocessing.provenance import (
    get_git_commit_hash,
)
from src.splitting.subject_mapping import (
    apply_subject_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_data_pipeline.yaml"
)


def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


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
    config = load_yaml(args.config)

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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    minimum_std = float(
        config["normalization"][
            "minimum_std_uv"
        ]
    )

    imbalance_config = (
        config["class_imbalance"][
            "loss_weighting"
        ]
    )

    maximum_pos_weight = (
        imbalance_config[
            "maximum_pos_weight"
        ]
    )

    if maximum_pos_weight is not None:
        maximum_pos_weight = float(
            maximum_pos_weight
        )

    scaler_rows: list[dict] = []
    class_weight_rows: list[dict] = []
    dataset_summary_rows: list[dict] = []

    git_commit = get_git_commit_hash()

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    outer_fold_ids = sorted(
        outer["outer_test_fold"]
        .unique()
        .astype(int)
    )

    inner_fold_ids = sorted(
        inner["inner_fold"]
        .unique()
        .astype(int)
    )

    for outer_fold in outer_fold_ids:
        outer_folder = (
            scaler_dir
            / f"outer_{outer_fold:02d}"
        )

        outer_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        for inner_fold in inner_fold_ids:
            subject_sets = (
                get_subjects_for_nested_fold(
                    inner_assignments=inner,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
            )

            window_tables = (
                get_nested_window_tables(
                    windows=windows,
                    inner_assignments=inner,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
            )

            scaler = build_fold_scaler(
                subject_statistics=(
                    subject_statistics
                ),
                training_subjects=(
                    subject_sets["train"]
                ),
                minimum_std_uv=(
                    minimum_std
                ),
            )

            scaler_path = (
                outer_folder
                / (
                    f"inner_{inner_fold:02d}"
                    "_scaler.npz"
                )
            )

            if (
                scaler_path.exists()
                and not args.overwrite
            ):
                raise FileExistsError(
                    f"Scaler exists: "
                    f"{scaler_path}"
                )

            metadata = {
                "scope": "inner_training",
                "outer_fold": int(
                    outer_fold
                ),
                "inner_fold": int(
                    inner_fold
                ),
                "training_subjects": sorted(
                    subject_sets["train"]
                ),
                "validation_subjects": sorted(
                    subject_sets[
                        "validation"
                    ]
                ),
                "test_subjects": sorted(
                    subject_sets["test"]
                ),
                "git_commit": git_commit,
                "timestamp_utc": (
                    timestamp
                ),
            }

            save_scaler_npz(
                scaler=scaler,
                output_path=scaler_path,
                metadata=metadata,
            )

            train_windows = (
                window_tables["train"]
            )

            counts = (
                count_binary_classes(
                    train_windows
                )
            )

            pos_weight = (
                calculate_pos_weight(
                    train_windows,
                    method=(
                        "negative_divided_by_positive"
                    ),
                    maximum_weight=(
                        maximum_pos_weight
                    ),
                )
            )

            for row in scaler.itertuples(
                index=False
            ):
                scaler_rows.append(
                    {
                        "scope": (
                            "inner_training"
                        ),
                        "outer_fold": (
                            outer_fold
                        ),
                        "inner_fold": (
                            inner_fold
                        ),
                        "channel_index": (
                            row.channel_index
                        ),
                        "channel_name": (
                            row.channel_name
                        ),
                        "sample_count": (
                            row.sample_count
                        ),
                        "mean_uv": (
                            row.mean_uv
                        ),
                        "std_uv": (
                            row.std_uv
                        ),
                        "scale_uv": (
                            row.scale_uv
                        ),
                        "scaler_path": str(
                            scaler_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                    }
                )

            class_weight_rows.append(
                {
                    "scope": (
                        "inner_training"
                    ),
                    "outer_fold": (
                        outer_fold
                    ),
                    "inner_fold": (
                        inner_fold
                    ),
                    **counts,
                    "raw_negative_to_positive_ratio": (
                        counts[
                            "negative_count"
                        ]
                        / counts[
                            "positive_count"
                        ]
                    ),
                    "pos_weight": (
                        pos_weight
                    ),
                    "weight_source": (
                        "training_windows_only"
                    ),
                }
            )

            for role, table in (
                window_tables.items()
            ):
                dataset_summary_rows.append(
                    {
                        "scope": (
                            "nested_fold"
                        ),
                        "outer_fold": (
                            outer_fold
                        ),
                        "inner_fold": (
                            inner_fold
                        ),
                        "role": role,
                        "subject_count": (
                            table[
                                "subject_id"
                            ].nunique()
                        ),
                        "window_count": (
                            len(table)
                        ),
                        "negative_count": int(
                            (
                                table[
                                    "binary_label"
                                ]
                                == 0
                            ).sum()
                        ),
                        "positive_count": int(
                            (
                                table[
                                    "binary_label"
                                ]
                                == 1
                            ).sum()
                        ),
                    }
                )

        # Final scaler after inner model selection:
        outer_tables = (
            get_outer_development_tables(
                windows=windows,
                outer_subject_folds=outer,
                outer_fold=outer_fold,
            )
        )

        development_subjects = set(
            outer_tables[
                "development"
            ]["subject_id"]
            .astype(str)
            .unique()
        )

        test_subjects = set(
            outer_tables["test"][
                "subject_id"
            ]
            .astype(str)
            .unique()
        )

        final_scaler = build_fold_scaler(
            subject_statistics=(
                subject_statistics
            ),
            training_subjects=(
                development_subjects
            ),
            minimum_std_uv=minimum_std,
        )

        final_scaler_path = (
            outer_folder
            / "outer_development_scaler.npz"
        )

        save_scaler_npz(
            scaler=final_scaler,
            output_path=(
                final_scaler_path
            ),
            metadata={
                "scope": (
                    "outer_development"
                ),
                "outer_fold": (
                    int(outer_fold)
                ),
                "training_subjects": sorted(
                    development_subjects
                ),
                "test_subjects": sorted(
                    test_subjects
                ),
                "git_commit": git_commit,
                "timestamp_utc": (
                    timestamp
                ),
            },
        )

        development_counts = (
            count_binary_classes(
                outer_tables[
                    "development"
                ]
            )
        )

        development_pos_weight = (
            calculate_pos_weight(
                outer_tables[
                    "development"
                ],
                method=(
                    "negative_divided_by_positive"
                ),
                maximum_weight=(
                    maximum_pos_weight
                ),
            )
        )

        class_weight_rows.append(
            {
                "scope": (
                    "outer_development"
                ),
                "outer_fold": (
                    outer_fold
                ),
                "inner_fold": None,
                **development_counts,
                "raw_negative_to_positive_ratio": (
                    development_counts[
                        "negative_count"
                    ]
                    / development_counts[
                        "positive_count"
                    ]
                ),
                "pos_weight": (
                    development_pos_weight
                ),
                "weight_source": (
                    "outer_development_windows_only"
                ),
            }
        )

    pd.DataFrame(
        scaler_rows
    ).to_csv(
        output_dir
        / "chbmit_fold_scalers.csv",
        index=False,
    )

    pd.DataFrame(
        class_weight_rows
    ).to_csv(
        output_dir
        / "chbmit_fold_class_weights.csv",
        index=False,
    )

    pd.DataFrame(
        dataset_summary_rows
    ).to_csv(
        output_dir
        / "chbmit_fold_dataset_summary.csv",
        index=False,
    )

    summary = {
        "outer_fold_count": int(
            len(outer_fold_ids)
        ),
        "inner_fold_count_per_outer": int(
            len(inner_fold_ids)
        ),
        "inner_scaler_count": int(
            len(outer_fold_ids)
            * len(inner_fold_ids)
        ),
        "outer_development_scaler_count": int(
            len(outer_fold_ids)
        ),
        "total_scaler_count": int(
            len(outer_fold_ids)
            * len(inner_fold_ids)
            + len(outer_fold_ids)
        ),
        "normalization_method": (
            config["normalization"][
                "method"
            ]
        ),
        "primary_imbalance_strategy": (
            config[
                "class_imbalance"
            ]["primary_strategy"]
        ),
        "git_commit": git_commit,
        "timestamp_utc": timestamp,
    }

    with (
        output_dir
        / "chbmit_data_pipeline_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())