"""Build final CHB-MIT recording-level and seizure-level metadata tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INVENTORY_PATH = (
    PROJECT_ROOT / "metadata" / "chbmit_file_inventory.csv"
)

DEFAULT_PARSED_RECORDINGS_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "raw_annotations"
    / "parsed_recordings.csv"
)

DEFAULT_PARSED_SEIZURES_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "raw_annotations"
    / "parsed_seizures.csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metadata"


DATASET_NAME = "CHB-MIT"
DATASET_VERSION = "1.0.0"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final CHB-MIT metadata tables."
    )

    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
    )
    parser.add_argument(
        "--parsed-recordings",
        type=Path,
        default=DEFAULT_PARSED_RECORDINGS_PATH,
    )
    parser.add_argument(
        "--parsed-seizures",
        type=Path,
        default=DEFAULT_PARSED_SEIZURES_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def standardize_inventory(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize selected phase-2 inventory column names."""
    rename_mapping = {}

    if "patient_id" in inventory.columns:
        rename_mapping["patient_id"] = "case_id"

    if "relative_path" in inventory.columns:
        rename_mapping["relative_path"] = "edf_relative_path"

    if "duration_seconds" in inventory.columns:
        rename_mapping["duration_seconds"] = (
            "duration_seconds_edf"
        )

    inventory = inventory.rename(
        columns=rename_mapping
    ).copy()

    required_columns = {
        "case_id",
        "recording_id",
        "edf_relative_path",
        "sampling_rate_hz",
        "n_times",
        "duration_seconds_edf",
        "n_channels",
        "file_size_bytes",
    }

    missing = required_columns.difference(
        inventory.columns
    )

    if missing:
        raise ValueError(
            "Phase-2 inventory is missing required columns: "
            f"{sorted(missing)}"
        )

    inventory["edf_filename"] = (
        inventory["edf_relative_path"]
        .map(lambda value: Path(value).name)
    )

    return inventory


def infer_subject_id(case_id: str) -> str:
    """Return a conservative subject identifier.

    This first metadata version does not silently merge different case IDs.
    A separate manually reviewed case-to-subject mapping may be added later.
    """
    return case_id


def build_recording_table(
    inventory: pd.DataFrame,
    parsed_recordings: pd.DataFrame,
) -> pd.DataFrame:
    """Merge EDF-header inventory with summary-derived metadata."""
    inventory = standardize_inventory(inventory)

    if parsed_recordings[
        ["case_id", "recording_id"]
    ].duplicated().any():
        duplicated = parsed_recordings.loc[
            parsed_recordings[
                ["case_id", "recording_id"]
            ].duplicated(keep=False),
            ["case_id", "recording_id"],
        ]

        raise ValueError(
            "Duplicate recording blocks found in parsed summaries:\n"
            f"{duplicated.to_string(index=False)}"
        )

    recordings = inventory.merge(
        parsed_recordings,
        on=[
            "case_id",
            "recording_id",
            "edf_filename",
        ],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    recordings["dataset"] = DATASET_NAME
    recordings["dataset_version"] = DATASET_VERSION
    recordings["subject_id"] = recordings[
        "case_id"
    ].map(infer_subject_id)

    recordings["edf_found"] = recordings[
        "_merge"
    ].isin(["both", "left_only"])

    recordings["summary_record_found"] = recordings[
        "_merge"
    ].isin(["both", "right_only"])

    recordings["metadata_status"] = recordings[
        "_merge"
    ].map(
        {
            "both": "matched",
            "left_only": "edf_without_summary_record",
            "right_only": "summary_record_without_edf",
        }
    )

    # Only matched records remain
    recordings = recordings[recordings["metadata_status"] == "matched"].copy()

    recordings["has_seizure"] = (
        recordings["n_seizures_parsed"]
        .fillna(0)
        .astype(int)
        .gt(0)
    )

    recordings["n_seizures_reported"] = (
        recordings["n_seizures_reported"]
        .astype("Int64")
    )

    recordings["n_seizures_parsed"] = (
        recordings["n_seizures_parsed"]
        .astype("Int64")
    )

    recordings["total_ictal_seconds"] = (
        recordings["total_ictal_seconds_summary"]
        .fillna(0.0)
    )

    recordings["ictal_fraction"] = np.where(
        recordings["duration_seconds_edf"].gt(0),
        recordings["total_ictal_seconds"]
        / recordings["duration_seconds_edf"],
        np.nan,
    )

    recordings["duration_difference_seconds"] = (
        recordings["duration_seconds_edf"]
        - recordings["duration_seconds_summary_clock"]
    )

    recordings["reported_parsed_seizure_count_match"] = (
        recordings["n_seizures_reported"]
        == recordings["n_seizures_parsed"]
    )

    preferred_columns = [
        "dataset",
        "dataset_version",
        "case_id",
        "subject_id",
        "recording_id",
        "edf_filename",
        "edf_relative_path",
        "summary_filename",
        "summary_relative_path",
        "sampling_rate_hz",
        "n_channels",
        "n_times",
        "duration_seconds_edf",
        "duration_seconds_summary_clock",
        "duration_difference_seconds",
        "file_start_time",
        "file_end_time",
        "crosses_midnight",
        "has_seizure",
        "n_seizures_reported",
        "n_seizures_parsed",
        "reported_parsed_seizure_count_match",
        "total_ictal_seconds",
        "ictal_fraction",
        "file_size_bytes",
        "ordered_channel_signature",
        "unordered_channel_signature",
        "has_exact_duplicate_names",
        "has_possible_suffix_duplicates",
        "edf_found",
        "summary_record_found",
        "metadata_status",
    ]

    existing_preferred = [
        column
        for column in preferred_columns
        if column in recordings.columns
    ]

    remaining_columns = [
        column
        for column in recordings.columns
        if column not in existing_preferred
        and column != "_merge"
        and column != "total_ictal_seconds_summary"
    ]

    return recordings[
        existing_preferred + remaining_columns
    ].sort_values(
        ["case_id", "recording_id"],
    ).reset_index(drop=True)


def build_seizure_table(
    parsed_seizures: pd.DataFrame,
    recordings: pd.DataFrame,
) -> pd.DataFrame:
    """Add EDF metadata and sample indices to seizure intervals."""
    recording_columns = [
        "case_id",
        "recording_id",
        "subject_id",
        "sampling_rate_hz",
        "n_times",
        "duration_seconds_edf",
        "edf_relative_path",
        "metadata_status",
    ]

    seizures = parsed_seizures.merge(
        recordings[recording_columns],
        on=["case_id", "recording_id"],
        how="left",
        validate="many_to_one",
    )

    seizures["dataset"] = DATASET_NAME
    seizures["dataset_version"] = DATASET_VERSION

    seizures["seizure_id"] = (
        seizures["case_id"]
        + "__"
        + seizures["recording_id"]
        + "__sz"
        + seizures["seizure_index"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    # Inclusive onset sample and exclusive offset sample.
    seizures["onset_sample"] = np.floor(
        seizures["onset_seconds"]
        * seizures["sampling_rate_hz"]
    ).astype("Int64")

    seizures["offset_sample_exclusive"] = np.ceil(
        seizures["offset_seconds"]
        * seizures["sampling_rate_hz"]
    ).astype("Int64")

    seizures["interval_within_edf"] = (
        seizures["onset_seconds"].ge(0)
        & seizures["offset_seconds"].gt(
            seizures["onset_seconds"]
        )
        & seizures["offset_seconds"].le(
            seizures["duration_seconds_edf"]
        )
    )

    seizures["sample_interval_within_edf"] = (
        seizures["onset_sample"].ge(0)
        & seizures["offset_sample_exclusive"].gt(
            seizures["onset_sample"]
        )
        & seizures["offset_sample_exclusive"].le(
            seizures["n_times"]
        )
    )

    seizures["validation_status"] = np.select(
        [
            seizures["sampling_rate_hz"].isna(),
            ~seizures["interval_within_edf"],
            ~seizures["sample_interval_within_edf"],
        ],
        [
            "missing_edf_metadata",
            "time_interval_outside_edf",
            "sample_interval_outside_edf",
        ],
        default="valid",
    )

    preferred_columns = [
        "dataset",
        "dataset_version",
        "case_id",
        "subject_id",
        "recording_id",
        "edf_filename",
        "edf_relative_path",
        "seizure_index",
        "seizure_id",
        "onset_seconds",
        "offset_seconds",
        "duration_seconds",
        "onset_sample",
        "offset_sample_exclusive",
        "sampling_rate_hz",
        "n_times",
        "duration_seconds_edf",
        "source_summary_file",
        "annotation_source",
        "interval_within_edf",
        "sample_interval_within_edf",
        "validation_status",
    ]

    return seizures[
        preferred_columns
    ].sort_values(
        [
            "case_id",
            "recording_id",
            "seizure_index",
        ]
    ).reset_index(drop=True)


def build_patient_summary(
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
) -> pd.DataFrame:
    """Build case-level dataset statistics."""
    recording_summary = (
        recordings.groupby(
            ["case_id", "subject_id"],
            dropna=False,
        )
        .agg(
            recording_count=("recording_id", "count"),
            total_recording_seconds=(
                "duration_seconds_edf",
                "sum",
            ),
            seizure_recording_count=(
                "has_seizure",
                "sum",
            ),
            minimum_channel_count=(
                "n_channels",
                "min",
            ),
            maximum_channel_count=(
                "n_channels",
                "max",
            ),
            distinct_channel_signatures=(
                "unordered_channel_signature",
                "nunique",
            ),
        )
        .reset_index()
    )

    if seizures.empty:
        seizure_summary = pd.DataFrame(
            columns=[
                "case_id",
                "seizure_count",
                "total_ictal_seconds",
                "median_seizure_duration_seconds",
                "minimum_seizure_duration_seconds",
                "maximum_seizure_duration_seconds",
            ]
        )
    else:
        seizure_summary = (
            seizures.groupby("case_id")
            .agg(
                seizure_count=("seizure_id", "count"),
                total_ictal_seconds=(
                    "duration_seconds",
                    "sum",
                ),
                median_seizure_duration_seconds=(
                    "duration_seconds",
                    "median",
                ),
                minimum_seizure_duration_seconds=(
                    "duration_seconds",
                    "min",
                ),
                maximum_seizure_duration_seconds=(
                    "duration_seconds",
                    "max",
                ),
            )
            .reset_index()
        )

    summary = recording_summary.merge(
        seizure_summary,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    numeric_fill_columns = [
        "seizure_count",
        "total_ictal_seconds",
    ]

    summary[numeric_fill_columns] = summary[
        numeric_fill_columns
    ].fillna(0)

    summary["total_recording_hours"] = (
        summary["total_recording_seconds"] / 3600
    )

    summary["ictal_fraction"] = np.where(
        summary["total_recording_seconds"] > 0,
        summary["total_ictal_seconds"]
        / summary["total_recording_seconds"],
        np.nan,
    )

    return summary.sort_values(
        "case_id"
    ).reset_index(drop=True)


def main() -> int:
    args = parse_arguments()

    required_paths = [
        args.inventory,
        args.parsed_recordings,
        args.parsed_seizures,
    ]

    missing_paths = [
        path for path in required_paths if not path.exists()
    ]

    if missing_paths:
        print(
            f"ERROR: Missing input files: {missing_paths}",
            file=sys.stderr,
        )
        return 1

    inventory = pd.read_csv(args.inventory)
    parsed_recordings = pd.read_csv(
        args.parsed_recordings
    )
    parsed_seizures = pd.read_csv(
        args.parsed_seizures
    )

    recordings = build_recording_table(
        inventory=inventory,
        parsed_recordings=parsed_recordings,
    )

    seizures = build_seizure_table(
        parsed_seizures=parsed_seizures,
        recordings=recordings,
    )

    patient_summary = build_patient_summary(
        recordings=recordings,
        seizures=seizures,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    recordings.to_csv(
        args.output_dir / "chbmit_recordings.csv",
        index=False,
    )

    seizures.to_csv(
        args.output_dir / "chbmit_seizures.csv",
        index=False,
    )

    patient_summary.to_csv(
        args.output_dir / "chbmit_patient_summary.csv",
        index=False,
    )

    dataset_summary = {
        "dataset": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "case_count": int(
            recordings["case_id"].nunique()
        ),
        "recording_count": int(len(recordings)),
        "matched_recording_count": int(
            recordings["metadata_status"]
            .eq("matched")
            .sum()
        ),
        "seizure_recording_count": int(
            recordings["has_seizure"].sum()
        ),
        "seizure_count": int(len(seizures)),
        "total_recording_hours": float(
            recordings["duration_seconds_edf"].sum()
            / 3600
        ),
        "total_ictal_seconds": float(
            seizures["duration_seconds"].sum()
        ),
        "invalid_seizure_interval_count": int(
            seizures["validation_status"]
            .ne("valid")
            .sum()
        ),
    }

    with (
        args.output_dir / "chbmit_dataset_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            dataset_summary,
            file,
            indent=2,
        )

    print("Metadata build completed.")
    print(json.dumps(dataset_summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())