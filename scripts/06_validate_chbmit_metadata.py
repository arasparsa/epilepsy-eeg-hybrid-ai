"""Validate final CHB-MIT recording and seizure metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RECORDINGS = (
    PROJECT_ROOT / "metadata" / "chbmit_recordings.csv"
)

DEFAULT_SEIZURES = (
    PROJECT_ROOT / "metadata" / "chbmit_seizures.csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "metadata"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recordings",
        type=Path,
        default=DEFAULT_RECORDINGS,
    )
    parser.add_argument(
        "--seizures",
        type=Path,
        default=DEFAULT_SEIZURES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--clock-duration-tolerance-seconds",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def add_problem(
    problems: list[dict],
    *,
    severity: str,
    problem_type: str,
    case_id: str | None,
    recording_id: str | None,
    seizure_id: str | None = None,
    details: str,
) -> None:
    problems.append(
        {
            "severity": severity,
            "problem_type": problem_type,
            "case_id": case_id,
            "recording_id": recording_id,
            "seizure_id": seizure_id,
            "details": details,
        }
    )


def main() -> int:
    args = parse_arguments()

    recordings = pd.read_csv(args.recordings)
    seizures = pd.read_csv(args.seizures)

    problems: list[dict] = []

    # 1. Unique EDF paths.
    duplicate_paths = recordings.loc[
        recordings["edf_relative_path"].duplicated(
            keep=False
        )
    ]

    for row in duplicate_paths.itertuples(index=False):
        add_problem(
            problems,
            severity="error",
            problem_type="duplicate_edf_path",
            case_id=row.case_id,
            recording_id=row.recording_id,
            details=str(row.edf_relative_path),
        )

    # 2. Unique case/recording keys.
    duplicate_recordings = recordings.loc[
        recordings[
            ["case_id", "recording_id"]
        ].duplicated(keep=False)
    ]

    for row in duplicate_recordings.itertuples(index=False):
        add_problem(
            problems,
            severity="error",
            problem_type="duplicate_recording_key",
            case_id=row.case_id,
            recording_id=row.recording_id,
            details="Duplicate case_id + recording_id",
        )

    # 3. EDF and summary match.
    unmatched = recordings.loc[
        recordings["metadata_status"] != "matched"
    ]

    for row in unmatched.itertuples(index=False):
        add_problem(
            problems,
            severity="error",
            problem_type="unmatched_edf_summary",
            case_id=row.case_id,
            recording_id=row.recording_id,
            details=str(row.metadata_status),
        )

    # 4. Reported versus parsed seizure count.
    count_mismatches = recordings.loc[
        recordings[
            "reported_parsed_seizure_count_match"
        ] == False  # noqa: E712
    ]

    for row in count_mismatches.itertuples(index=False):
        add_problem(
            problems,
            severity="error",
            problem_type="seizure_count_mismatch",
            case_id=row.case_id,
            recording_id=row.recording_id,
            details=(
                f"reported={row.n_seizures_reported}, "
                f"parsed={row.n_seizures_parsed}"
            ),
        )

    # 5. EDF duration versus summary clock duration.
    duration_valid = recordings[
        "duration_seconds_summary_clock"
    ].notna()

    duration_mismatch = recordings.loc[
        duration_valid
        & (
            recordings[
                "duration_difference_seconds"
            ].abs()
            > args.clock_duration_tolerance_seconds
        )
    ]

    for row in duration_mismatch.itertuples(index=False):
        add_problem(
            problems,
            severity="warning",
            problem_type="recording_duration_mismatch",
            case_id=row.case_id,
            recording_id=row.recording_id,
            details=(
                f"EDF={row.duration_seconds_edf}, "
                f"summary={row.duration_seconds_summary_clock}, "
                f"difference={row.duration_difference_seconds}"
            ),
        )

    # 6. Invalid seizure intervals.
    invalid_intervals = seizures.loc[
        seizures["validation_status"] != "valid"
    ]

    for row in invalid_intervals.itertuples(index=False):
        add_problem(
            problems,
            severity="error",
            problem_type="invalid_seizure_interval",
            case_id=row.case_id,
            recording_id=row.recording_id,
            seizure_id=row.seizure_id,
            details=str(row.validation_status),
        )

    # 7. Duplicate seizure IDs.
    duplicate_seizure_ids = seizures.loc[
        seizures["seizure_id"].duplicated(
            keep=False
        )
    ]

    for row in duplicate_seizure_ids.itertuples(
        index=False
    ):
        add_problem(
            problems,
            severity="error",
            problem_type="duplicate_seizure_id",
            case_id=row.case_id,
            recording_id=row.recording_id,
            seizure_id=row.seizure_id,
            details="Duplicate seizure ID",
        )

    # 8. Overlapping seizure intervals within a recording.
    for (
        case_id,
        recording_id,
    ), group in seizures.groupby(
        ["case_id", "recording_id"]
    ):
        ordered = group.sort_values("onset_seconds")

        previous_offset: float | None = None
        previous_id: str | None = None

        for row in ordered.itertuples(index=False):
            if (
                previous_offset is not None
                and row.onset_seconds < previous_offset
            ):
                add_problem(
                    problems,
                    severity="warning",
                    problem_type=(
                        "overlapping_seizure_intervals"
                    ),
                    case_id=case_id,
                    recording_id=recording_id,
                    seizure_id=row.seizure_id,
                    details=(
                        f"{previous_id} overlaps "
                        f"{row.seizure_id}"
                    ),
                )

            previous_offset = row.offset_seconds
            previous_id = row.seizure_id

    # 9. Recalculate recording seizure counts from seizure table.
    recalculated_counts = (
        seizures.groupby(
            ["case_id", "recording_id"]
        )
        .size()
        .rename("recalculated_seizure_count")
        .reset_index()
    )

    validation = recordings.merge(
        recalculated_counts,
        on=["case_id", "recording_id"],
        how="left",
        validate="one_to_one",
    )

    validation[
        "recalculated_seizure_count"
    ] = validation[
        "recalculated_seizure_count"
    ].fillna(0).astype(int)

    validation[
        "seizure_table_count_matches_recording"
    ] = (
        validation["recalculated_seizure_count"]
        == validation["n_seizures_parsed"].fillna(0)
    )

    for row in validation.loc[
        ~validation[
            "seizure_table_count_matches_recording"
        ]
    ].itertuples(index=False):
        add_problem(
            problems,
            severity="error",
            problem_type=(
                "recording_seizure_table_count_mismatch"
            ),
            case_id=row.case_id,
            recording_id=row.recording_id,
            details=(
                f"recording={row.n_seizures_parsed}, "
                f"seizure_table="
                f"{row.recalculated_seizure_count}"
            ),
        )

    problems_df = pd.DataFrame(problems)

    if problems_df.empty:
        problems_df = pd.DataFrame(
            columns=[
                "severity",
                "problem_type",
                "case_id",
                "recording_id",
                "seizure_id",
                "details",
            ]
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    problems_df.to_csv(
        args.output_dir
        / "chbmit_metadata_problems.csv",
        index=False,
    )

    validation_columns = [
        "case_id",
        "recording_id",
        "metadata_status",
        "n_seizures_reported",
        "n_seizures_parsed",
        "recalculated_seizure_count",
        "reported_parsed_seizure_count_match",
        "seizure_table_count_matches_recording",
        "duration_seconds_edf",
        "duration_seconds_summary_clock",
        "duration_difference_seconds",
    ]

    validation[
        validation_columns
    ].to_csv(
        args.output_dir
        / "chbmit_annotation_validation.csv",
        index=False,
    )

    error_count = int(
        problems_df["severity"].eq("error").sum()
    )

    warning_count = int(
        problems_df["severity"].eq("warning").sum()
    )

    print("Metadata validation completed.")
    print(f"Errors: {error_count}")
    print(f"Warnings: {warning_count}")

    if not problems_df.empty:
        print(
            "\nProblem categories:\n",
            problems_df[
                ["severity", "problem_type"]
            ]
            .value_counts()
            .to_string(),
        )

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())