"""Generate frozen patient-independent nested CV splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
import yaml

from src.preprocessing.provenance import (
    get_git_commit_hash,
)
from src.splitting.nested_cv import (
    generate_inner_subject_folds,
    generate_outer_subject_folds,
    prepare_binary_stratification_table,
)
from src.splitting.subject_mapping import (
    apply_subject_mapping,
)
from src.splitting.summaries import (
    add_outer_fold_ratios,
    build_outer_fold_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_splitting.yaml"
)


def load_yaml(path: Path) -> dict:
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outer_path = (
        output_dir
        / "chbmit_outer_subject_folds.csv"
    )

    if (
        outer_path.exists()
        and not args.overwrite
        and not config["runtime"].get(
            "overwrite",
            False,
        )
    ):
        raise FileExistsError(
            "Split outputs already exist. "
            "Use --overwrite."
        )

    windows = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"]["window_manifest"]
    )

    mapping = pd.read_csv(
        output_dir
        / "chbmit_resolved_subject_mapping.csv"
    )

    windows = apply_subject_mapping(
        windows,
        mapping,
    )

    windows[
        "binary_metric_eligible"
    ] = windows["label_name"].isin(
        config["window_eligibility"][
            "binary_metric_eligible_labels"
        ]
    )

    windows[
        "training_candidate"
    ] = windows["label_name"].isin(
        config["window_eligibility"][
            "training_candidate_labels"
        ]
    )

    windows[
        "clean_training_candidate"
    ] = (
        windows["label_name"].eq("ictal")
        | (
            windows["label_name"].eq(
                "non_ictal"
            )
            & windows[
                "clean_non_ictal"
            ].astype(bool)
        )
    )

    windows[
        "continuous_evaluation_candidate"
    ] = windows["label_name"].isin(
        ["non_ictal", "ictal"]
    )

    binary_windows = (
        prepare_binary_stratification_table(
            windows
        )
    )

    outer_config = config["outer_cv"]

    outer_subject_folds = (
        generate_outer_subject_folds(
            binary_windows=(
                binary_windows
            ),
            n_splits=int(
                outer_config["n_splits"]
            ),
            shuffle=bool(
                outer_config["shuffle"]
            ),
            random_state=int(
                outer_config[
                    "random_state"
                ]
            ),
        )
    )

    inner_config = config["inner_cv"]

    inner_subject_folds = (
        generate_inner_subject_folds(
            binary_windows=(
                binary_windows
            ),
            outer_subject_folds=(
                outer_subject_folds
            ),
            outer_n_splits=int(
                outer_config["n_splits"]
            ),
            inner_n_splits=int(
                inner_config["n_splits"]
            ),
            shuffle=bool(
                inner_config["shuffle"]
            ),
            base_random_state=int(
                inner_config[
                    "base_random_state"
                ]
            ),
            random_state_increment=int(
                inner_config[
                    "random_state_increment_per_outer_fold"
                ]
            ),
        )
    )

    windows = windows.merge(
        outer_subject_folds,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    if windows[
        "outer_test_fold"
    ].isna().any():
        raise RuntimeError(
            "Windows without outer-fold assignment."
        )

    windows[
        "outer_test_fold"
    ] = windows[
        "outer_test_fold"
    ].astype(int)

    outer_summary = (
        build_outer_fold_summary(
            windows
        )
    )

    outer_summary = add_outer_fold_ratios(
        outer_summary
    )

    split_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    git_commit = get_git_commit_hash()

    outer_subject_folds[
        "splitting_version"
    ] = config["project"][
        "splitting_version"
    ]

    outer_subject_folds[
        "outer_random_state"
    ] = int(
        outer_config["random_state"]
    )

    outer_subject_folds[
        "git_commit"
    ] = git_commit

    outer_subject_folds[
        "split_timestamp_utc"
    ] = split_timestamp

    outer_subject_folds.to_csv(
        outer_path,
        index=False,
    )

    inner_subject_folds[
        "splitting_version"
    ] = config["project"][
        "splitting_version"
    ]

    inner_subject_folds[
        "git_commit"
    ] = git_commit

    inner_subject_folds[
        "split_timestamp_utc"
    ] = split_timestamp

    inner_subject_folds.to_csv(
        output_dir
        / "chbmit_inner_subject_folds.csv",
        index=False,
    )

    window_fold_columns = [
        "window_id",
        "case_id",
        "subject_id",
        "recording_id",
        "label_name",
        "label_id",
        "binary_label",
        "clean_non_ictal",
        "near_seizure",
        "binary_metric_eligible",
        "training_candidate",
        "clean_training_candidate",
        "continuous_evaluation_candidate",
        "outer_test_fold",
    ]

    windows[
        window_fold_columns
    ].to_csv(
        output_dir
        / "chbmit_window_outer_folds.csv",
        index=False,
    )

    outer_summary.to_csv(
        output_dir
        / "chbmit_outer_fold_summary.csv",
        index=False,
    )

    # Build inner-fold summaries at subject/window level.
    inner_summary_rows: list[dict] = []

    for assignment_key, assignments in (
        inner_subject_folds.groupby(
            ["outer_fold", "inner_fold"]
        )
    ):
        outer_fold, inner_fold = (
            assignment_key
        )

        for role in [
            "train",
            "validation",
            "test",
        ]:
            role_subjects = set(
                assignments.loc[
                    assignments["role"] == role,
                    "subject_id",
                ]
            )

            role_windows = windows.loc[
                windows["subject_id"].isin(
                    role_subjects
                )
            ]

            inner_summary_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "role": role,
                    "subject_count": (
                        len(role_subjects)
                    ),
                    "case_count": (
                        role_windows[
                            "case_id"
                        ].nunique()
                    ),
                    "recording_count": (
                        role_windows[
                            "recording_id"
                        ].nunique()
                    ),
                    "total_window_count": (
                        len(role_windows)
                    ),
                    "non_ictal_window_count": int(
                        (
                            role_windows[
                                "label_name"
                            ]
                            == "non_ictal"
                        ).sum()
                    ),
                    "boundary_window_count": int(
                        (
                            role_windows[
                                "label_name"
                            ]
                            == "boundary"
                        ).sum()
                    ),
                    "ictal_window_count": int(
                        (
                            role_windows[
                                "label_name"
                            ]
                            == "ictal"
                        ).sum()
                    ),
                }
            )

    inner_summary = pd.DataFrame(
        inner_summary_rows
    )

    inner_summary.to_csv(
        output_dir
        / "chbmit_inner_fold_summary.csv",
        index=False,
    )

    summary = {
        "splitting_version": (
            config["project"][
                "splitting_version"
            ]
        ),
        "independent_subject_count": int(
            windows[
                "subject_id"
            ].nunique()
        ),
        "case_count": int(
            windows["case_id"].nunique()
        ),
        "recording_count": int(
            windows[
                "recording_id"
            ].nunique()
        ),
        "window_count": int(
            len(windows)
        ),
        "outer_fold_count": int(
            outer_config["n_splits"]
        ),
        "inner_fold_count_per_outer": int(
            inner_config["n_splits"]
        ),
        "outer_random_state": int(
            outer_config["random_state"]
        ),
        "inner_base_random_state": int(
            inner_config[
                "base_random_state"
            ]
        ),
        "grouping_unit": "subject_id",
        "scikit_learn_version": (
            sklearn.__version__
        ),
        "git_commit": git_commit,
        "split_timestamp_utc": (
            split_timestamp
        ),
    }

    with (
        output_dir
        / "chbmit_splitting_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(json.dumps(summary, indent=2))
    print("\nOuter-fold summary:")
    print(
        outer_summary.to_string(
            index=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())