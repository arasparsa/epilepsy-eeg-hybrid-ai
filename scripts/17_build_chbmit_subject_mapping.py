"""Build the resolved CHB-MIT case-to-subject mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.splitting.subject_mapping import (
    apply_subject_mapping,
    resolve_subject_mapping,
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

    args = parser.parse_args()
    config = load_yaml(args.config)

    windows_path = (
        PROJECT_ROOT
        / config["inputs"]["window_manifest"]
    )

    mapping_path = (
        PROJECT_ROOT
        / config["inputs"][
            "case_subject_mapping"
        ]
    )

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

    windows = pd.read_csv(
        windows_path,
        usecols=[
            "case_id",
            "recording_id",
            "window_id",
            "label_name",
            "binary_label",
            "window_duration_seconds",
            "has_seizure_in_recording",
        ],
    )

    verified_mapping = pd.read_csv(
        mapping_path
    )

    resolved = resolve_subject_mapping(
        observed_cases=(
            windows["case_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        verified_mapping=verified_mapping,
    )

    resolved.to_csv(
        output_dir
        / "chbmit_resolved_subject_mapping.csv",
        index=False,
    )

    mapped_windows = apply_subject_mapping(
        windows,
        resolved,
    )

    subject_summary = (
        mapped_windows.groupby(
            "subject_id",
            dropna=False,
        )
        .agg(
            case_count=(
                "case_id",
                "nunique",
            ),
            case_ids=(
                "case_id",
                lambda values: "|".join(
                    sorted(set(values))
                ),
            ),
            recording_count=(
                "recording_id",
                "nunique",
            ),
            total_window_count=(
                "window_id",
                "count",
            ),
            non_ictal_window_count=(
                "label_name",
                lambda values: int(
                    (values == "non_ictal").sum()
                ),
            ),
            boundary_window_count=(
                "label_name",
                lambda values: int(
                    (values == "boundary").sum()
                ),
            ),
            ictal_window_count=(
                "label_name",
                lambda values: int(
                    (values == "ictal").sum()
                ),
            ),
            seizure_recording_count=(
                "has_seizure_in_recording",
                lambda values: int(
                    mapped_windows.loc[
                        values.index
                    ].loc[
                        values.astype(bool),
                        "recording_id",
                    ].nunique()
                ),
            ),
        )
        .reset_index()
    )

    subject_summary[
        "binary_window_count"
    ] = (
        subject_summary[
            "non_ictal_window_count"
        ]
        + subject_summary[
            "ictal_window_count"
        ]
    )

    subject_summary[
        "ictal_window_fraction"
    ] = (
        subject_summary[
            "ictal_window_count"
        ]
        / subject_summary[
            "binary_window_count"
        ]
    )

    subject_summary.to_csv(
        output_dir
        / "chbmit_subject_summary.csv",
        index=False,
    )

    summary = {
        "observed_case_count": int(
            resolved["case_id"].nunique()
        ),
        "resolved_subject_count": int(
            resolved["subject_id"].nunique()
        ),
        "multi_case_subject_count": int(
            subject_summary[
                "case_count"
            ].gt(1).sum()
        ),
        "multi_case_subjects": (
            subject_summary.loc[
                subject_summary["case_count"] > 1,
                [
                    "subject_id",
                    "case_ids",
                ],
            ].to_dict(orient="records")
        ),
    }

    with (
        output_dir
        / "chbmit_subject_mapping_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())