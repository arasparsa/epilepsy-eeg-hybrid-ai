"""Validate CHB-MIT signal-quality outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QC_DIR = PROJECT_ROOT / "metadata" / "signal_quality"
METADATA_DIR = PROJECT_ROOT / "metadata"


def main() -> int:
    recordings = pd.read_csv(
        METADATA_DIR / "chbmit_recordings.csv"
    )

    chunks = pd.read_csv(
        QC_DIR / "chbmit_qc_chunks.csv"
    )

    channels = pd.read_csv(
        QC_DIR / "chbmit_qc_channels.csv"
    )

    recording_qc = pd.read_csv(
        QC_DIR / "chbmit_qc_recordings.csv"
    )

    failures = pd.read_csv(
        QC_DIR / "chbmit_qc_failures.csv"
    )

    errors: list[str] = []
    warnings: list[str] = []

    matched_recordings = recordings.loc[
        recordings["metadata_status"] == "matched"
    ]

    expected_ids = set(
        matched_recordings["recording_id"]
    )

    processed_ids = set(
        recording_qc["recording_id"]
    )

    failed_ids = set(
        failures["recording_id"].dropna()
    )

    unexplained_missing = (
        expected_ids - processed_ids - failed_ids
    )

    if unexplained_missing:
        errors.append(
            "Recordings neither processed nor reported as failed: "
            f"{sorted(unexplained_missing)[:10]}"
        )

    chunk_key = [
        "recording_id",
        "chunk_id",
        "channel_index",
    ]

    if chunks[chunk_key].duplicated().any():
        errors.append("Duplicate chunk-channel keys detected.")

    channel_key = [
        "recording_id",
        "channel_index",
    ]

    if channels[channel_key].duplicated().any():
        errors.append("Duplicate recording-channel keys detected.")

    if recording_qc["recording_id"].duplicated().any():
        errors.append("Duplicate recording QC rows detected.")

    fraction_columns = [
        "nonfinite_fraction",
        "nan_fraction",
        "zero_fraction",
        "flatline_sample_fraction",
        "seizure_overlap_fraction",
    ]

    for column in fraction_columns:
        invalid = ~chunks[column].dropna().between(0, 1)

        if invalid.any():
            errors.append(
                f"Invalid fraction values in {column}."
            )

    if (
        chunks["chunk_end_seconds"]
        <= chunks["chunk_start_seconds"]
    ).any():
        errors.append("Non-positive chunk duration detected.")

    if (
        chunks["stop_sample_exclusive"]
        <= chunks["start_sample"]
    ).any():
        errors.append("Invalid chunk sample interval detected.")

    if (chunks["sampling_rate_hz"] <= 0).any():
        errors.append("Non-positive sampling rate detected.")

    expected_annotation_labels = {
        "non_ictal",
        "boundary",
        "ictal",
    }

    unexpected_labels = (
        set(chunks["annotation_label"].dropna())
        - expected_annotation_labels
    )

    if unexpected_labels:
        errors.append(
            f"Unexpected annotation labels: "
            f"{sorted(unexpected_labels)}"
        )

    if not failures.empty:
        warnings.append(
            f"{len(failures)} EDF files failed QC processing."
        )

    print("QC validation")
    print("Errors:", len(errors))
    print("Warnings:", len(warnings))

    for error in errors:
        print("ERROR:", error)

    for warning in warnings:
        print("WARNING:", warning)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())