"""Build deterministic CHB-MIT segmentation and labeling metadata."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mne
import pandas as pd
import yaml

from src.preprocessing.provenance import (
    get_git_commit_hash,
)
from src.segmentation.labeling import (
    label_window,
)
from src.segmentation.windows import (
    generate_fixed_windows,
    seconds_to_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_segmentation.yaml"
)


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def build_seizure_lookup(
    seizures: pd.DataFrame,
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Map each recording to its seizure intervals."""
    lookup: dict[
        tuple[str, str],
        list[tuple[float, float]],
    ] = {}

    for key, group in seizures.groupby(
        ["case_id", "recording_id"],
        sort=True,
    ):
        ordered = group.sort_values(
            ["onset_seconds", "seizure_index"]
        )

        lookup[key] = list(
            zip(
                ordered[
                    "onset_seconds"
                ].astype(float),
                ordered[
                    "offset_seconds"
                ].astype(float),
                strict=True,
            )
        )

    return lookup


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--recording-id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
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

    window_manifest_path = (
        output_dir
        / "chbmit_window_manifest.csv"
    )

    if (
        window_manifest_path.exists()
        and not args.overwrite
        and not config["runtime"].get(
            "overwrite",
            False,
        )
    ):
        print(
            "ERROR: Segmentation output already exists. "
            "Use --overwrite.",
            file=sys.stderr,
        )
        return 1

    preprocessing_manifest = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "preprocessing_manifest"
        ]
    )

    preprocessing_validation = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "preprocessing_validation"
        ]
    )

    seizures = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"]["seizures_file"]
    )

    valid_statuses = set(
        config["quality"][
            "require_preprocessing_validation_status"
        ]
    )

    valid_preprocessed = (
        preprocessing_validation.loc[
            preprocessing_validation[
                "validation_status"
            ].isin(valid_statuses),
            [
                "case_id",
                "recording_id",
                "validation_status",
            ],
        ]
    )

    selected = preprocessing_manifest.merge(
        valid_preprocessed,
        on=["case_id", "recording_id"],
        how="inner",
        validate="one_to_one",
    )

    if args.recording_id:
        selected = selected.loc[
            selected["recording_id"]
            == args.recording_id
        ]

    if args.limit_files is not None:
        selected = selected.head(
            args.limit_files
        )

    if selected.empty:
        print(
            "ERROR: No validated preprocessed "
            "recordings selected.",
            file=sys.stderr,
        )
        return 1

    seizure_lookup = build_seizure_lookup(
        seizures
    )

    window_duration_seconds = float(
        config["windowing"][
            "window_duration_seconds"
        ]
    )

    stride_seconds = float(
        config["windowing"]["stride_seconds"]
    )

    expected_sfreq = float(
        config["windowing"][
            "expected_sampling_rate_hz"
        ]
    )

    expected_window_samples = (
        seconds_to_samples(
            window_duration_seconds,
            expected_sfreq,
        )
    )

    expected_stride_samples = (
        seconds_to_samples(
            stride_seconds,
            expected_sfreq,
        )
    )

    label_ids = {
        name: int(value)
        for name, value in config[
            "labeling"
        ]["labels"].items()
    }

    window_rows: list[dict] = []
    failure_rows: list[dict] = []
    recording_summary_rows: list[dict] = []

    segmentation_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    git_commit = get_git_commit_hash()

    total_recordings = len(selected)

    for file_index, recording in enumerate(
        selected.itertuples(index=False),
        start=1,
    ):
        print(
            f"[{file_index:04d}/"
            f"{total_recordings:04d}] "
            f"{recording.case_id}/"
            f"{recording.recording_id}"
        )

        fif_path = (
            PROJECT_ROOT
            / recording.output_fif_path
        )

        try:
            raw = mne.io.read_raw_fif(
                fif_path,
                preload=False,
                verbose="ERROR",
            )

            sampling_rate = float(
                raw.info["sfreq"]
            )

            if not abs(
                sampling_rate - expected_sfreq
            ) <= 1e-9:
                raise ValueError(
                    f"Unexpected sampling rate: "
                    f"{sampling_rate}"
                )

            windows = generate_fixed_windows(
                n_times=int(raw.n_times),
                sampling_rate_hz=sampling_rate,
                window_duration_seconds=(
                    window_duration_seconds
                ),
                stride_seconds=stride_seconds,
                drop_incomplete_final_window=bool(
                    config["windowing"][
                        "drop_incomplete_final_window"
                    ]
                ),
            )

            seizure_intervals = (
                seizure_lookup.get(
                    (
                        recording.case_id,
                        recording.recording_id,
                    ),
                    [],
                )
            )

            label_counts = {
                "non_ictal": 0,
                "boundary": 0,
                "ictal": 0,
            }

            for window in windows:
                label = label_window(
                    window_start_seconds=(
                        window.start_seconds
                    ),
                    window_end_seconds=(
                        window.end_seconds
                    ),
                    seizure_intervals=(
                        seizure_intervals
                    ),
                    ictal_overlap_threshold=float(
                        config["labeling"][
                            "ictal_overlap_threshold"
                        ]
                    ),
                    near_seizure_margin_seconds=float(
                        config["labeling"][
                            "near_seizure_margin_seconds"
                        ]
                    ),
                    clean_non_ictal_minimum_distance_seconds=float(
                        config["labeling"][
                            "clean_non_ictal_minimum_distance_seconds"
                        ]
                    ),
                    label_ids=label_ids,
                )

                label_counts[
                    label.label_name
                ] += 1

                window_id = (
                    f"{recording.case_id}"
                    f"__{recording.recording_id}"
                    f"__w{window.window_index:06d}"
                )

                window_rows.append(
                    {
                        "dataset": "CHB-MIT",
                        "dataset_version": (
                            config["project"][
                                "dataset_version"
                            ]
                        ),
                        "case_id": (
                            recording.case_id
                        ),
                        "recording_id": (
                            recording.recording_id
                        ),
                        "window_id": window_id,
                        "window_index": (
                            window.window_index
                        ),
                        "output_fif_path": (
                            recording.output_fif_path
                        ),
                        "sampling_rate_hz": (
                            sampling_rate
                        ),
                        "channel_count": int(
                            recording.output_channel_count
                        ),
                        "channel_order": (
                            recording.output_channel_order
                        ),
                        "start_sample": (
                            window.start_sample
                        ),
                        "stop_sample_exclusive": (
                            window.stop_sample_exclusive
                        ),
                        "sample_count": (
                            window.stop_sample_exclusive
                            - window.start_sample
                        ),
                        "start_seconds": (
                            window.start_seconds
                        ),
                        "end_seconds": (
                            window.end_seconds
                        ),
                        "window_duration_seconds": (
                            window.duration_seconds
                        ),
                        "stride_seconds": (
                            stride_seconds
                        ),
                        "overlap_between_windows_seconds": (
                            window_duration_seconds
                            - stride_seconds
                        ),
                        "label_name": (
                            label.label_name
                        ),
                        "label_id": label.label_id,
                        "binary_label": (
                            label.binary_label
                        ),
                        "seizure_overlap_seconds": (
                            label.seizure_overlap_seconds
                        ),
                        "seizure_overlap_fraction": (
                            label.seizure_overlap_fraction
                        ),
                        "overlaps_seizure": (
                            label.overlaps_seizure
                        ),
                        "distance_to_nearest_seizure_seconds": (
                            label.distance_to_nearest_seizure_seconds
                        ),
                        "near_seizure": (
                            label.near_seizure
                        ),
                        "clean_non_ictal": (
                            label.clean_non_ictal
                        ),
                        "has_seizure_in_recording": (
                            len(seizure_intervals) > 0
                        ),
                        "seizure_count_in_recording": (
                            len(seizure_intervals)
                        ),
                        "preprocessing_version": (
                            config["project"][
                                "preprocessing_version"
                            ]
                        ),
                        "segmentation_version": (
                            config["project"][
                                "segmentation_version"
                            ]
                        ),
                        "git_commit": git_commit,
                        "segmentation_timestamp_utc": (
                            segmentation_timestamp
                        ),
                    }
                )

            recording_summary_rows.append(
                {
                    "case_id": (
                        recording.case_id
                    ),
                    "recording_id": (
                        recording.recording_id
                    ),
                    "output_fif_path": (
                        recording.output_fif_path
                    ),
                    "n_times": int(raw.n_times),
                    "sampling_rate_hz": (
                        sampling_rate
                    ),
                    "recording_duration_seconds": (
                        raw.n_times / sampling_rate
                    ),
                    "window_count": len(windows),
                    "non_ictal_window_count": (
                        label_counts["non_ictal"]
                    ),
                    "boundary_window_count": (
                        label_counts["boundary"]
                    ),
                    "ictal_window_count": (
                        label_counts["ictal"]
                    ),
                    "expected_window_samples": (
                        expected_window_samples
                    ),
                    "expected_stride_samples": (
                        expected_stride_samples
                    ),
                }
            )

            raw.close()

        except Exception as exc:
            failure_rows.append(
                {
                    "case_id": (
                        recording.case_id
                    ),
                    "recording_id": (
                        recording.recording_id
                    ),
                    "output_fif_path": (
                        recording.output_fif_path
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_message": str(exc),
                }
            )

            print(
                f"WARNING: "
                f"{recording.recording_id}: "
                f"{exc}",
                file=sys.stderr,
            )

            if config["runtime"].get(
                "fail_fast",
                False,
            ):
                raise

    window_manifest = pd.DataFrame(
        window_rows
    )

    recording_summary = pd.DataFrame(
        recording_summary_rows
    )

    failures = pd.DataFrame(
        failure_rows
    )

    if window_manifest.empty:
        print(
            "ERROR: No windows generated.",
            file=sys.stderr,
        )
        return 1

    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "case_id",
                "recording_id",
                "output_fif_path",
                "error_type",
                "error_message",
            ]
        )

    window_manifest.to_csv(
        window_manifest_path,
        index=False,
    )

    recording_summary.to_csv(
        output_dir
        / "chbmit_recording_window_summary.csv",
        index=False,
    )

    failures.to_csv(
        output_dir
        / "chbmit_segmentation_failures.csv",
        index=False,
    )

    case_summary = (
        window_manifest.groupby(
            ["case_id"]
        )
        .agg(
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
            clean_non_ictal_window_count=(
                "clean_non_ictal",
                "sum",
            ),
        )
        .reset_index()
    )

    case_summary.to_csv(
        output_dir
        / "chbmit_case_window_summary.csv",
        index=False,
    )

    label_distribution = (
        window_manifest[
            "label_name"
        ].value_counts()
    )

    summary = {
        "segmentation_version": (
            config["project"][
                "segmentation_version"
            ]
        ),
        "selected_recordings": int(
            total_recordings
        ),
        "successfully_segmented_recordings": int(
            window_manifest[
                "recording_id"
            ].nunique()
        ),
        "failed_recordings": int(
            len(failures)
        ),
        "total_windows": int(
            len(window_manifest)
        ),
        "non_ictal_windows": int(
            label_distribution.get(
                "non_ictal",
                0,
            )
        ),
        "boundary_windows": int(
            label_distribution.get(
                "boundary",
                0,
            )
        ),
        "ictal_windows": int(
            label_distribution.get(
                "ictal",
                0,
            )
        ),
        "clean_non_ictal_windows": int(
            window_manifest[
                "clean_non_ictal"
            ].sum()
        ),
        "window_duration_seconds": (
            window_duration_seconds
        ),
        "stride_seconds": stride_seconds,
        "window_samples": (
            expected_window_samples
        ),
        "stride_samples": (
            expected_stride_samples
        ),
        "ictal_overlap_threshold": float(
            config["labeling"][
                "ictal_overlap_threshold"
            ]
        ),
        "git_commit": git_commit,
        "segmentation_timestamp_utc": (
            segmentation_timestamp
        ),
    }

    with (
        output_dir
        / "chbmit_segmentation_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nSegmentation completed.")
    print(json.dumps(summary, indent=2))

    return 0 if failures.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())