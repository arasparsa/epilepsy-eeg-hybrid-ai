"""Validate frozen CHB-MIT channel harmonization outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "chbmit_channel_harmonization.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "metadata"
    / "harmonization"
)


def main() -> int:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    coverage = pd.read_csv(
        OUTPUT_DIR
        / "chbmit_channel_set_recording_coverage.csv"
    )

    candidate_summary = pd.read_csv(
        OUTPUT_DIR
        / "chbmit_candidate_channel_sets.csv"
    )

    manifest = pd.read_csv(
        OUTPUT_DIR
        / "chbmit_recording_inclusion_manifest.csv"
    )

    failures = pd.read_csv(
        OUTPUT_DIR
        / "chbmit_harmonization_failures.csv"
    )

    errors: list[str] = []
    warnings: list[str] = []

    primary_set = config["inclusion"][
        "primary_channel_set"
    ]

    configured_sets = set(
        config["channel_sets"].keys()
    )

    observed_sets = set(
        candidate_summary["channel_set_name"]
    )

    if configured_sets != observed_sets:
        errors.append(
            "Configured and evaluated channel sets differ."
        )

    primary_rows = coverage.loc[
        coverage["channel_set_name"] == primary_set
    ]

    included_ids = set(
        manifest.loc[
            manifest["include_primary_analysis"],
            "recording_id",
        ]
    )

    complete_primary_ids = set(
        primary_rows.loc[
            primary_rows["has_complete_channel_set"],
            "recording_id",
        ]
    )

    invalid_included = (
        included_ids - complete_primary_ids
    )

    if invalid_included:
        errors.append(
            "Included recordings without complete primary set: "
            f"{sorted(invalid_included)[:10]}"
        )

    duplicate_keys = manifest[
        ["case_id", "recording_id"]
    ].duplicated()

    if duplicate_keys.any():
        errors.append(
            "Duplicate recording keys in inclusion manifest."
        )

    if manifest[
        "include_primary_analysis"
    ].isna().any():
        errors.append(
            "Missing inclusion decisions detected."
        )

    valid_reasons = manifest.loc[
        manifest["include_primary_analysis"],
        "inclusion_reason",
    ].eq("included")

    if not valid_reasons.all():
        errors.append(
            "Included recordings have exclusion reasons."
        )

    fraction_columns = [
        "recording_retention_fraction",
        "recording_hour_retention_fraction",
        "seizure_retention_fraction",
        "ictal_time_retention_fraction",
    ]

    for column in fraction_columns:
        if not candidate_summary[
            column
        ].between(0, 1).all():
            errors.append(
                f"Invalid fraction in {column}."
            )

    if not failures.empty:
        errors.append(
            f"{len(failures)} canonicalization failures exist."
        )

    primary_summary = candidate_summary.loc[
        candidate_summary["channel_set_name"]
        == primary_set
    ].iloc[0]

    if (
        primary_summary[
            "recording_retention_fraction"
        ] < 0.95
    ):
        warnings.append(
            "Primary montage retains less than 95% "
            "of recordings."
        )

    if (
        primary_summary["seizure_retention_fraction"]
        < 0.95
    ):
        warnings.append(
            "Primary montage retains less than 95% "
            "of annotated seizures."
        )

    print("Harmonization validation")
    print("Errors:", len(errors))
    print("Warnings:", len(warnings))

    for error in errors:
        print("ERROR:", error)

    for warning in warnings:
        print("WARNING:", warning)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())