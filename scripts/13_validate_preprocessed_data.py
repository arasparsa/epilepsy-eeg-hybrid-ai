"""Validate all CHB-MIT preprocessed FIF recordings."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "chbmit_preprocessing.yaml"
)


def main() -> int:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    harmonization_path = (
        PROJECT_ROOT
        / config["inputs"]["harmonization_config"]
    )

    with harmonization_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        harmonization = yaml.safe_load(file)

    target_channels = harmonization[
        "channel_sets"
    ][config["channels"]["channel_set_name"]][
        "channels"
    ]

    inclusion = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"]["inclusion_manifest"]
    )

    preprocessing_manifest = pd.read_csv(
        PROJECT_ROOT
        / config["outputs"]["metadata_directory"]
        / "chbmit_preprocessing_manifest.csv"
    )

    failures = pd.read_csv(
        PROJECT_ROOT
        / config["outputs"]["metadata_directory"]
        / "chbmit_preprocessing_failures.csv"
    )

    included_ids = set(
        inclusion.loc[
            inclusion["include_primary_analysis"],
            "recording_id",
        ]
    )

    processed_ids = set(
        preprocessing_manifest["recording_id"]
    )

    failed_ids = set(
        failures["recording_id"].dropna()
    )

    errors: list[str] = []
    warnings: list[str] = []
    validation_rows: list[dict] = []

    unexplained_missing = (
        included_ids
        - processed_ids
        - failed_ids
    )

    if unexplained_missing:
        errors.append(
            "Included recordings neither processed nor "
            "reported as failed: "
            f"{sorted(unexplained_missing)[:10]}"
        )

    duplicate_manifest_ids = (
        preprocessing_manifest[
            "recording_id"
        ].duplicated()
    )

    if duplicate_manifest_ids.any():
        errors.append(
            "Duplicate recording IDs in preprocessing manifest."
        )

    expected_sfreq = float(
        config["resampling"][
            "target_sampling_rate_hz"
        ]
    )

    for row in preprocessing_manifest.itertuples(
        index=False
    ):
        output_path = (
            PROJECT_ROOT / row.output_fif_path
        )

        row_errors: list[str] = []

        if not output_path.exists():
            row_errors.append("output_file_missing")
            validation_rows.append(
                {
                    "case_id": row.case_id,
                    "recording_id": row.recording_id,
                    "validation_status": "error",
                    "validation_errors": (
                        "|".join(row_errors)
                    ),
                }
            )
            continue

        try:
            raw = mne.io.read_raw_fif(
                output_path,
                preload=False,
                verbose="ERROR",
            )

            channel_order_valid = (
                raw.ch_names == target_channels
            )

            channel_count_valid = (
                len(raw.ch_names)
                == len(target_channels)
            )

            sampling_rate_valid = np.isclose(
                float(raw.info["sfreq"]),
                expected_sfreq,
            )

            duration_seconds = (
                raw.n_times
                / raw.info["sfreq"]
            )

            duration_valid = np.isclose(
                duration_seconds,
                row.output_duration_seconds,
                atol=1 / expected_sfreq,
            )

            ictal_count = int(
                (
                    raw.annotations.description
                    == config["annotations"][
                        "ictal_description"
                    ]
                ).sum()
            )

            annotation_count_valid = (
                ictal_count
                == int(
                    row[
                        "after_"
                        "expected_ictal_annotation_count"
                    ]
                )
                if isinstance(row, dict)
                else ictal_count
                == int(
                    getattr(
                        row,
                        "after_expected_ictal_annotation_count",
                    )
                )
            )

            finite_check_samples = min(
                raw.n_times,
                int(
                    60
                    * raw.info["sfreq"]
                ),
            )

            data_sample = raw.get_data(
                start=0,
                stop=finite_check_samples,
            )

            finite_sample_valid = bool(
                np.isfinite(data_sample).all()
            )

            checks = {
                "channel_order_valid": (
                    channel_order_valid
                ),
                "channel_count_valid": (
                    channel_count_valid
                ),
                "sampling_rate_valid": (
                    sampling_rate_valid
                ),
                "duration_valid": duration_valid,
                "annotation_count_valid": (
                    annotation_count_valid
                ),
                "finite_sample_valid": (
                    finite_sample_valid
                ),
            }

            failed_checks = [
                name
                for name, passed in checks.items()
                if not passed
            ]

            validation_rows.append(
                {
                    "case_id": row.case_id,
                    "recording_id": (
                        row.recording_id
                    ),
                    "output_fif_path": (
                        row.output_fif_path
                    ),
                    **checks,
                    "validation_status": (
                        "valid"
                        if not failed_checks
                        else "error"
                    ),
                    "validation_errors": "|".join(
                        failed_checks
                    ),
                }
            )

            raw.close()

        except Exception as exc:
            validation_rows.append(
                {
                    "case_id": row.case_id,
                    "recording_id": (
                        row.recording_id
                    ),
                    "output_fif_path": (
                        row.output_fif_path
                    ),
                    "validation_status": "error",
                    "validation_errors": (
                        f"{type(exc).__name__}:{exc}"
                    ),
                }
            )

    validation = pd.DataFrame(
        validation_rows
    )

    invalid_count = int(
        validation["validation_status"]
        .ne("valid")
        .sum()
    )

    if invalid_count:
        errors.append(
            f"{invalid_count} preprocessed files "
            "failed validation."
        )

    if not failures.empty:
        errors.append(
            f"{len(failures)} preprocessing failures exist."
        )

    output_metadata_dir = (
        PROJECT_ROOT
        / config["outputs"]["metadata_directory"]
    )

    validation.to_csv(
        output_metadata_dir
        / "chbmit_preprocessing_validation.csv",
        index=False,
    )

    print("Preprocessing validation")
    print("Errors:", len(errors))
    print("Warnings:", len(warnings))

    for error in errors:
        print("ERROR:", error)

    for warning in warnings:
        print("WARNING:", warning)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())