"""Build a frozen recording inclusion manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_channel_harmonization.yaml"
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_manual_exclusions(
    path: Path,
    excluded_decisions: set[str],
) -> set[str]:
    """Return recordings with confirmed exclusion decisions."""
    if not path.exists():
        return set()

    manual = pd.read_csv(path)

    if manual.empty:
        return set()

    required = {
        "recording_id",
        "review_decision",
    }

    if not required.issubset(manual.columns):
        return set()

    return set(
        manual.loc[
            manual["review_decision"].isin(
                excluded_decisions
            ),
            "recording_id",
        ].dropna()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    config = load_config(args.config)

    inputs = config["inputs"]
    output_dir = (
        PROJECT_ROOT / config["outputs"]["directory"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = pd.read_csv(
        PROJECT_ROOT / inputs["recordings_file"]
    )

    coverage = pd.read_csv(
        output_dir
        / "chbmit_channel_set_recording_coverage.csv"
    )

    qc_recordings = pd.read_csv(
        PROJECT_ROOT / inputs["qc_recordings_file"]
    )

    primary_set = config["inclusion"][
        "primary_channel_set"
    ]

    primary_coverage = coverage.loc[
        coverage["channel_set_name"] == primary_set
    ].copy()

    manifest = recordings.merge(
        primary_coverage[
            [
                "case_id",
                "recording_id",
                "has_complete_channel_set",
                "missing_channel_count",
                "missing_channels",
                "available_target_channel_count",
                "target_channel_count",
            ]
        ],
        on=["case_id", "recording_id"],
        how="left",
        validate="one_to_one",
    )

    manifest = manifest.merge(
        qc_recordings[
            [
                "case_id",
                "recording_id",
                "recording_needs_review",
                "reviewed_channel_count",
                "reviewed_channel_fraction",
            ]
        ],
        on=["case_id", "recording_id"],
        how="left",
        validate="one_to_one",
    )

    excluded_manual_decisions = set(
        config["inclusion"][
            "exclude_on_confirmed_manual_decisions"
        ]
    )

    manually_excluded_recordings = (
        load_manual_exclusions(
            PROJECT_ROOT / inputs["manual_qc_file"],
            excluded_manual_decisions,
        )
    )

    manifest["manual_exclusion"] = (
        manifest["recording_id"].isin(
            manually_excluded_recordings
        )
    )

    allowed_metadata_statuses = set(
        config["inclusion"][
            "require_metadata_status"
        ]
    )

    manifest["metadata_eligible"] = (
        manifest["metadata_status"].isin(
            allowed_metadata_statuses
        )
    )

    manifest["duration_eligible"] = (
        manifest["duration_seconds_edf"] > 0
    )

    manifest["sampling_rate_eligible"] = (
        manifest["sampling_rate_hz"] > 0
    )

    manifest["channel_set_eligible"] = (
        manifest["has_complete_channel_set"]
        .fillna(False)
    )

    # QC review flags remain descriptive, not exclusions.
    manifest["qc_eligible"] = (
        ~manifest["manual_exclusion"]
    )

    manifest["include_primary_analysis"] = (
        manifest["metadata_eligible"]
        & manifest["duration_eligible"]
        & manifest["sampling_rate_eligible"]
        & manifest["channel_set_eligible"]
        & manifest["qc_eligible"]
    )

    def build_reason(row: pd.Series) -> str:
        reasons: list[str] = []

        if not row["metadata_eligible"]:
            reasons.append(
                f"metadata_status:{row['metadata_status']}"
            )

        if not row["duration_eligible"]:
            reasons.append("nonpositive_duration")

        if not row["sampling_rate_eligible"]:
            reasons.append("invalid_sampling_rate")

        if not row["channel_set_eligible"]:
            reasons.append(
                "missing_primary_channels:"
                f"{row['missing_channels']}"
            )

        if row["manual_exclusion"]:
            reasons.append(
                "confirmed_manual_qc_exclusion"
            )

        return (
            "|".join(reasons)
            if reasons
            else "included"
        )

    manifest["inclusion_reason"] = manifest.apply(
        build_reason,
        axis=1,
    )

    manifest["harmonization_policy_version"] = (
        config["project"]["policy_version"]
    )
    manifest["primary_channel_set"] = primary_set

    preferred_columns = [
        "dataset",
        "dataset_version",
        "case_id",
        "subject_id",
        "recording_id",
        "edf_relative_path",
        "has_seizure",
        "n_seizures_parsed",
        "total_ictal_seconds",
        "sampling_rate_hz",
        "duration_seconds_edf",
        "target_channel_count",
        "available_target_channel_count",
        "has_complete_channel_set",
        "missing_channel_count",
        "missing_channels",
        "recording_needs_review",
        "reviewed_channel_count",
        "reviewed_channel_fraction",
        "manual_exclusion",
        "metadata_eligible",
        "duration_eligible",
        "sampling_rate_eligible",
        "channel_set_eligible",
        "qc_eligible",
        "include_primary_analysis",
        "inclusion_reason",
        "primary_channel_set",
        "harmonization_policy_version",
    ]

    manifest = manifest[
        [
            column
            for column in preferred_columns
            if column in manifest.columns
        ]
    ].sort_values(
        ["case_id", "recording_id"]
    )

    manifest.to_csv(
        output_dir
        / "chbmit_recording_inclusion_manifest.csv",
        index=False,
    )

    exclusions = manifest.loc[
        ~manifest["include_primary_analysis"]
    ].copy()

    exclusions.to_csv(
        output_dir / "chbmit_exclusion_reasons.csv",
        index=False,
    )

    included = manifest.loc[
        manifest["include_primary_analysis"]
    ]

    summary = {
        "policy_version": config["project"][
            "policy_version"
        ],
        "primary_channel_set": primary_set,
        "total_recordings": int(len(manifest)),
        "included_recordings": int(len(included)),
        "excluded_recordings": int(len(exclusions)),
        "included_cases": int(
            included["case_id"].nunique()
        ),
        "included_seizure_recordings": int(
            included["has_seizure"].sum()
        ),
        "included_seizures": int(
            included["n_seizures_parsed"].sum()
        ),
        "included_ictal_seconds": float(
            included["total_ictal_seconds"].sum()
        ),
        "included_recording_hours": float(
            included["duration_seconds_edf"].sum()
            / 3600
        ),
        "qc_review_recordings_retained": int(
            included["recording_needs_review"]
            .fillna(False)
            .sum()
        ),
    }

    with (
        output_dir
        / "chbmit_harmonization_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())