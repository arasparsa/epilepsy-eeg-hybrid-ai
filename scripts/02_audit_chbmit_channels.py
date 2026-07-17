"""Audit channels and EDF headers in the CHB-MIT dataset.

This script reads EDF headers without preloading full EEG signals. It creates
file-level, channel-level, channel-coverage, montage-signature, and problem-file
reports for later channel harmonization decisions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "metadata"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "channel_audit"


# Candidate bipolar channels. This is not yet the final channel set.
# The audit will calculate how many files contain each candidate set.
TARGET_CHANNEL_SETS: dict[str, list[str]] = {
    "core_17": [
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FZ-CZ",
        "CZ-PZ",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "FP2-F8",
        "F8-T8",
        "P8-O2",
    ],
    "extended_18": [
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FZ-CZ",
        "CZ-PZ",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
    ],
}


def normalize_channel_name(channel_name: str) -> str:
    """Normalize superficial EDF channel-label differences.

    The function deliberately avoids aggressive anatomical mapping.
    Original labels are preserved separately in all output tables.
    """
    name = channel_name.strip().upper()

    # Normalize Unicode dash variants.
    name = (
        name.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    # Remove repeated whitespace and spaces around hyphens.
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s*-\s*", "-", name)

    # Remove a leading EEG prefix.
    name = re.sub(r"^EEG\s+", "", name)

    # Remove common terminal reference suffixes only.
    name = re.sub(r"-(REF|LE)$", "", name)

    return name


def remove_duplicate_suffix(channel_name: str) -> str:
    """Remove MNE-style duplicate suffixes for a secondary comparison.

    Examples
    --------
    T8-P8-0 -> T8-P8
    T8-P8-1 -> T8-P8

    Important
    ---------
    This representation is used only to identify possible duplicate channels.
    It must not silently replace the original or normalized channel name.
    """
    return re.sub(r"-\d+$", "", channel_name)


def extract_patient_id(file_path: Path, raw_dir: Path) -> str:
    """Extract a CHB-MIT case identifier from path or filename."""
    candidates = [
        file_path.parent.name.lower(),
        file_path.stem.lower(),
    ]

    for candidate in candidates:
        match = re.search(r"chb\d{2}", candidate)
        if match:
            return match.group(0)

    relative_path = file_path.relative_to(raw_dir)
    raise ValueError(
        f"Could not extract a patient/case ID from {relative_path}"
    )


def safe_measurement_date(raw: mne.io.BaseRaw) -> str | None:
    """Convert EDF measurement date to an ISO string when available."""
    meas_date = raw.info.get("meas_date")

    if meas_date is None:
        return None

    try:
        return meas_date.isoformat()
    except AttributeError:
        return str(meas_date)


def inspect_edf_file(file_path: Path, raw_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one EDF header and return file-level and channel-level rows."""
    raw = mne.io.read_raw_edf(
        input_fname=file_path,
        preload=False,
        infer_types=False,
        verbose="ERROR",
    )

    original_channels = list(raw.ch_names)
    normalized_channels = [
        normalize_channel_name(name)
        for name in original_channels
    ]
    base_channels = [
        remove_duplicate_suffix(name)
        for name in normalized_channels
    ]

    normalized_counts = Counter(normalized_channels)
    base_counts = Counter(base_channels)

    exact_duplicates = sorted(
        name
        for name, count in normalized_counts.items()
        if count > 1
    )

    possible_suffix_duplicates = sorted(
        name
        for name, count in base_counts.items()
        if count > 1
    )

    patient_id = extract_patient_id(file_path, raw_dir)
    relative_path = file_path.relative_to(raw_dir)
    sampling_rate = float(raw.info["sfreq"])
    n_times = int(raw.n_times)

    # n_times / sfreq is usually preferable to raw.times[-1],
    # because it reflects the full sample count.
    duration_seconds = n_times / sampling_rate

    channel_signature = "||".join(sorted(normalized_channels))
    ordered_signature = "||".join(normalized_channels)

    file_row: dict[str, Any] = {
        "dataset": "CHB-MIT",
        "patient_id": patient_id,
        "recording_id": file_path.stem,
        "relative_path": relative_path.as_posix(),
        "file_size_bytes": file_path.stat().st_size,
        "sampling_rate_hz": sampling_rate,
        "n_times": n_times,
        "duration_seconds": duration_seconds,
        "n_channels": len(original_channels),
        "n_unique_normalized_channels": len(set(normalized_channels)),
        "measurement_date": safe_measurement_date(raw),
        "original_channel_order": json.dumps(
            original_channels,
            ensure_ascii=False,
        ),
        "normalized_channel_order": json.dumps(
            normalized_channels,
            ensure_ascii=False,
        ),
        "ordered_channel_signature": ordered_signature,
        "unordered_channel_signature": channel_signature,
        "has_exact_duplicate_names": bool(exact_duplicates),
        "exact_duplicate_names": "|".join(exact_duplicates),
        "has_possible_suffix_duplicates": bool(
            possible_suffix_duplicates
        ),
        "possible_suffix_duplicate_bases": "|".join(
            possible_suffix_duplicates
        ),
    }

    channel_rows: list[dict[str, Any]] = []

    for channel_index, (
        original_name,
        normalized_name,
        base_name,
    ) in enumerate(
        zip(
            original_channels,
            normalized_channels,
            base_channels,
            strict=True,
        )
    ):
        channel_rows.append(
            {
                "dataset": "CHB-MIT",
                "patient_id": patient_id,
                "recording_id": file_path.stem,
                "relative_path": relative_path.as_posix(),
                "channel_index": channel_index,
                "original_channel_name": original_name,
                "normalized_channel_name": normalized_name,
                "base_channel_name": base_name,
                "sampling_rate_hz": sampling_rate,
            }
        )

    return file_row, channel_rows


def build_presence_matrix(
    file_inventory: pd.DataFrame,
    channel_long: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per recording and one binary column per channel."""
    presence = (
        channel_long.assign(present=1)
        .pivot_table(
            index=[
                "patient_id",
                "recording_id",
                "relative_path",
            ],
            columns="normalized_channel_name",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    # Ensure all channel columns contain integers.
    identifier_columns = {
        "patient_id",
        "recording_id",
        "relative_path",
    }

    for column in presence.columns:
        if column not in identifier_columns:
            presence[column] = presence[column].astype(int)

    expected_rows = len(file_inventory)

    if len(presence) != expected_rows:
        raise RuntimeError(
            "Presence matrix row count does not match file inventory: "
            f"{len(presence)} versus {expected_rows}"
        )

    return presence


def build_channel_counts(
    file_inventory: pd.DataFrame,
    channel_long: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize file-level and patient-level channel coverage."""
    total_files = file_inventory["recording_id"].nunique()
    total_patients = file_inventory["patient_id"].nunique()

    file_counts = (
        channel_long[
            [
                "patient_id",
                "recording_id",
                "normalized_channel_name",
            ]
        ]
        .drop_duplicates()
        .groupby("normalized_channel_name")
        .agg(
            file_count=("recording_id", "count"),
            patient_count=("patient_id", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "normalized_channel_name": "channel_name",
            }
        )
    )

    file_counts["total_files"] = total_files
    file_counts["total_patients"] = total_patients
    file_counts["file_coverage_fraction"] = (
        file_counts["file_count"] / total_files
    )
    file_counts["patient_coverage_fraction"] = (
        file_counts["patient_count"] / total_patients
    )

    return file_counts.sort_values(
        by=["file_count", "patient_count", "channel_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_signature_summary(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize distinct normalized channel sets."""
    summary = (
        file_inventory.groupby(
            "unordered_channel_signature",
            dropna=False,
        )
        .agg(
            file_count=("recording_id", "count"),
            patient_count=("patient_id", "nunique"),
            example_patient=("patient_id", "first"),
            example_recording=("recording_id", "first"),
            minimum_channel_count=("n_channels", "min"),
            maximum_channel_count=("n_channels", "max"),
        )
        .reset_index()
        .sort_values(
            ["file_count", "patient_count"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary.insert(
        0,
        "signature_id",
        [f"SIG-{index:03d}" for index in range(1, len(summary) + 1)],
    )

    return summary


def build_target_channel_coverage(
    file_inventory: pd.DataFrame,
    channel_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate coverage of predefined candidate channel sets."""
    grouped_channels = (
        channel_long.groupby(
            ["patient_id", "recording_id", "relative_path"]
        )["normalized_channel_name"]
        .apply(set)
        .reset_index(name="available_channels")
    )

    recording_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for set_name, target_channels in TARGET_CHANNEL_SETS.items():
        target_set = set(target_channels)

        for row in grouped_channels.itertuples(index=False):
            missing = sorted(
                target_set.difference(row.available_channels)
            )
            extra = sorted(
                row.available_channels.difference(target_set)
            )

            recording_rows.append(
                {
                    "channel_set_name": set_name,
                    "patient_id": row.patient_id,
                    "recording_id": row.recording_id,
                    "relative_path": row.relative_path,
                    "target_channel_count": len(target_set),
                    "available_target_channel_count": (
                        len(target_set) - len(missing)
                    ),
                    "has_complete_target_set": len(missing) == 0,
                    "missing_target_channels": "|".join(missing),
                    "extra_channels": "|".join(extra),
                }
            )

    recording_coverage = pd.DataFrame(recording_rows)

    for set_name, group in recording_coverage.groupby(
        "channel_set_name"
    ):
        complete = group[group["has_complete_target_set"]]

        summary_rows.append(
            {
                "channel_set_name": set_name,
                "target_channel_count": int(
                    group["target_channel_count"].iloc[0]
                ),
                "complete_file_count": len(complete),
                "total_file_count": len(group),
                "complete_file_fraction": (
                    len(complete) / len(group)
                    if len(group)
                    else 0.0
                ),
                "covered_patient_count": (
                    complete["patient_id"].nunique()
                ),
                "total_patient_count": (
                    group["patient_id"].nunique()
                ),
            }
        )

    coverage_summary = pd.DataFrame(summary_rows)

    return recording_coverage, coverage_summary


def build_problem_files(
    file_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Collect files requiring manual review."""
    modal_sampling_rate = (
        file_inventory["sampling_rate_hz"]
        .mode()
        .iloc[0]
    )
    modal_channel_count = (
        file_inventory["n_channels"]
        .mode()
        .iloc[0]
    )

    problems = file_inventory.copy()

    problems["unusual_sampling_rate"] = (
        problems["sampling_rate_hz"] != modal_sampling_rate
    )
    problems["unusual_channel_count"] = (
        problems["n_channels"] != modal_channel_count
    )
    problems["duplicate_name_problem"] = (
        problems["has_exact_duplicate_names"]
        | problems["has_possible_suffix_duplicates"]
    )

    problem_mask = (
        problems["unusual_sampling_rate"]
        | problems["unusual_channel_count"]
        | problems["duplicate_name_problem"]
    )

    selected_columns = [
        "patient_id",
        "recording_id",
        "relative_path",
        "sampling_rate_hz",
        "n_channels",
        "n_unique_normalized_channels",
        "unusual_sampling_rate",
        "unusual_channel_count",
        "duplicate_name_problem",
        "exact_duplicate_names",
        "possible_suffix_duplicate_bases",
        "normalized_channel_order",
    ]

    return problems.loc[problem_mask, selected_columns].reset_index(
        drop=True
    )


def validate_outputs(
    file_inventory: pd.DataFrame,
    channel_long: pd.DataFrame,
) -> None:
    """Run critical consistency checks before saving outputs."""
    if file_inventory.empty:
        raise ValueError("File inventory is empty.")

    if channel_long.empty:
        raise ValueError("Channel table is empty.")

    if file_inventory["relative_path"].duplicated().any():
        duplicates = file_inventory.loc[
            file_inventory["relative_path"].duplicated(),
            "relative_path",
        ].tolist()
        raise ValueError(
            f"Duplicate EDF paths found: {duplicates[:5]}"
        )

    observed_channel_rows = len(channel_long)
    expected_channel_rows = int(
        file_inventory["n_channels"].sum()
    )

    if observed_channel_rows != expected_channel_rows:
        raise ValueError(
            "Channel-row count mismatch: "
            f"observed={observed_channel_rows}, "
            f"expected={expected_channel_rows}"
        )

    if (file_inventory["sampling_rate_hz"] <= 0).any():
        raise ValueError("Non-positive sampling rate detected.")

    if (file_inventory["duration_seconds"] <= 0).any():
        raise ValueError("Non-positive recording duration detected.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CHB-MIT EDF channel structures."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Root directory containing CHB-MIT EDF files.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Directory for generated CSV metadata.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for text reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    raw_dir = args.raw_dir.resolve()
    metadata_dir = args.metadata_dir.resolve()
    report_dir = args.report_dir.resolve()

    if not raw_dir.exists():
        print(
            f"ERROR: Raw dataset directory not found: {raw_dir}",
            file=sys.stderr,
        )
        return 1

    edf_files = sorted(raw_dir.rglob("*.edf"))

    if not edf_files:
        print(
            f"ERROR: No EDF files found under: {raw_dir}",
            file=sys.stderr,
        )
        return 1

    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, str]] = []

    print(f"Found {len(edf_files)} EDF files.")

    for index, file_path in enumerate(edf_files, start=1):
        relative_path = file_path.relative_to(raw_dir)
        print(f"[{index:04d}/{len(edf_files):04d}] {relative_path}")

        try:
            file_row, current_channel_rows = inspect_edf_file(
                file_path=file_path,
                raw_dir=raw_dir,
            )
            file_rows.append(file_row)
            channel_rows.extend(current_channel_rows)

        except Exception as exc:
            failed_rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(
                f"WARNING: Failed to inspect {relative_path}: {exc}",
                file=sys.stderr,
            )

    file_inventory = pd.DataFrame(file_rows)
    channel_long = pd.DataFrame(channel_rows)
    failed_files = pd.DataFrame(failed_rows)

    if file_inventory.empty:
        print(
            "ERROR: No EDF files were successfully inspected.",
            file=sys.stderr,
        )
        return 1

    validate_outputs(file_inventory, channel_long)

    channel_counts = build_channel_counts(
        file_inventory=file_inventory,
        channel_long=channel_long,
    )

    channel_presence = build_presence_matrix(
        file_inventory=file_inventory,
        channel_long=channel_long,
    )

    signature_summary = build_signature_summary(file_inventory)

    (
        target_recording_coverage,
        target_coverage_summary,
    ) = build_target_channel_coverage(
        file_inventory=file_inventory,
        channel_long=channel_long,
    )

    problem_files = build_problem_files(file_inventory)

    file_inventory.to_csv(
        metadata_dir / "chbmit_file_inventory.csv",
        index=False,
    )
    channel_long.to_csv(
        metadata_dir / "chbmit_channel_long.csv",
        index=False,
    )
    channel_counts.to_csv(
        metadata_dir / "chbmit_channel_counts.csv",
        index=False,
    )
    channel_presence.to_csv(
        metadata_dir / "chbmit_channel_presence_matrix.csv",
        index=False,
    )
    signature_summary.to_csv(
        metadata_dir / "chbmit_channel_signatures.csv",
        index=False,
    )
    target_recording_coverage.to_csv(
        metadata_dir / "chbmit_target_channel_coverage.csv",
        index=False,
    )
    target_coverage_summary.to_csv(
        metadata_dir / "chbmit_target_channel_summary.csv",
        index=False,
    )
    problem_files.to_csv(
        metadata_dir / "chbmit_problem_files.csv",
        index=False,
    )

    if not failed_files.empty:
        failed_files.to_csv(
            metadata_dir / "chbmit_failed_edf_files.csv",
            index=False,
        )

    print("\nAudit completed.")
    print(f"Successfully inspected files: {len(file_inventory)}")
    print(f"Failed files: {len(failed_files)}")
    print(
        "Patients/cases: "
        f"{file_inventory['patient_id'].nunique()}"
    )
    print(
        "Channel-count range: "
        f"{file_inventory['n_channels'].min()}–"
        f"{file_inventory['n_channels'].max()}"
    )
    print(
        "Sampling rates: "
        f"{sorted(file_inventory['sampling_rate_hz'].unique())}"
    )
    print(
        "Distinct normalized channels: "
        f"{channel_long['normalized_channel_name'].nunique()}"
    )
    print(
        "Distinct channel signatures: "
        f"{len(signature_summary)}"
    )
    print(f"Problem files: {len(problem_files)}")
    print(f"Metadata directory: {metadata_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())