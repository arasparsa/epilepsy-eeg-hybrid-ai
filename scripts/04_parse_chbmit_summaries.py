"""Parse all CHB-MIT summary files into intermediate CSV tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.metadata.chbmit_summary_parser import (
    SummaryParseError,
    calculate_clock_duration_seconds,
    iter_summary_files,
    parse_summary_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "metadata" / "raw_annotations"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse CHB-MIT case summary files."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not raw_dir.exists():
        print(
            f"ERROR: CHB-MIT directory not found: {raw_dir}",
            file=sys.stderr,
        )
        return 1

    summary_paths = list(iter_summary_files(raw_dir))

    if not summary_paths:
        print(
            f"ERROR: No summary files found under {raw_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    recording_rows: list[dict] = []
    seizure_rows: list[dict] = []
    problem_rows: list[dict] = []

    print(f"Summary files found: {len(summary_paths)}")

    for index, summary_path in enumerate(
        summary_paths,
        start=1,
    ):
        relative_path = summary_path.relative_to(raw_dir)

        print(
            f"[{index:02d}/{len(summary_paths):02d}] "
            f"{relative_path}"
        )

        try:
            parsed = parse_summary_file(
                summary_path=summary_path,
                raw_root=raw_dir,
            )

            summary_rows.append(
                {
                    "case_id": parsed.case_id,
                    "summary_filename": parsed.summary_filename,
                    "summary_relative_path": (
                        parsed.summary_relative_path
                    ),
                    "n_declared_channels": len(
                        parsed.declared_channels
                    ),
                    "declared_channels": json.dumps(
                        parsed.declared_channels,
                        ensure_ascii=False,
                    ),
                    "n_recordings_parsed": len(
                        parsed.recordings
                    ),
                    "parse_status": "success",
                }
            )

            for recording in parsed.recordings:
                (
                    clock_duration,
                    crosses_midnight,
                ) = calculate_clock_duration_seconds(
                    start_time=recording.file_start_time,
                    end_time=recording.file_end_time,
                )

                recording_rows.append(
                    {
                        "case_id": parsed.case_id,
                        "recording_id": Path(
                            recording.edf_filename
                        ).stem,
                        "edf_filename": (
                            recording.edf_filename
                        ),
                        "summary_filename": (
                            recording.summary_filename
                        ),
                        "summary_relative_path": (
                            recording.summary_relative_path
                        ),
                        "file_start_time": (
                            recording.file_start_time
                        ),
                        "file_end_time": (
                            recording.file_end_time
                        ),
                        "duration_seconds_summary_clock": (
                            clock_duration
                        ),
                        "crosses_midnight": (
                            crosses_midnight
                        ),
                        "n_seizures_reported": (
                            recording.n_seizures_reported
                        ),
                        "n_seizures_parsed": (
                            recording.n_seizures_parsed
                        ),
                        "total_ictal_seconds_summary": (
                            recording.total_ictal_seconds
                        ),
                    }
                )

                for seizure in recording.seizures:
                    seizure_rows.append(
                        {
                            "case_id": parsed.case_id,
                            "recording_id": Path(
                                recording.edf_filename
                            ).stem,
                            "edf_filename": (
                                recording.edf_filename
                            ),
                            "seizure_index": (
                                seizure.seizure_index
                            ),
                            "onset_seconds": (
                                seizure.onset_seconds
                            ),
                            "offset_seconds": (
                                seizure.offset_seconds
                            ),
                            "duration_seconds": (
                                seizure.duration_seconds
                            ),
                            "source_summary_file": (
                                recording.summary_relative_path
                            ),
                            "annotation_source": (
                                "CHB-MIT case summary"
                            ),
                        }
                    )

        except Exception as exc:
            problem_rows.append(
                {
                    "summary_relative_path": (
                        relative_path.as_posix()
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            print(
                f"WARNING: {relative_path}: {exc}",
                file=sys.stderr,
            )

    summaries = pd.DataFrame(summary_rows)
    recordings = pd.DataFrame(recording_rows)
    seizures = pd.DataFrame(seizure_rows)
    problems = pd.DataFrame(problem_rows)

    summaries.to_csv(
        output_dir / "parsed_summary_files.csv",
        index=False,
    )
    recordings.to_csv(
        output_dir / "parsed_recordings.csv",
        index=False,
    )
    seizures.to_csv(
        output_dir / "parsed_seizures.csv",
        index=False,
    )

    if problems.empty:
        problems = pd.DataFrame(
            columns=[
                "summary_relative_path",
                "error_type",
                "error_message",
            ]
        )

    problems.to_csv(
        output_dir / "summary_parse_problems.csv",
        index=False,
    )

    print("\nSummary parsing completed.")
    print(f"Successful summaries: {len(summaries)}")
    print(f"Failed summaries: {len(problems)}")
    print(f"Recording blocks: {len(recordings)}")
    print(f"Seizure intervals: {len(seizures)}")
    print(f"Output: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())