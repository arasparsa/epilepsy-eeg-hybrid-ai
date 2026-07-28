"""Run chunked signal-quality control on CHB-MIT EDF recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

from src.quality.aggregation import (
    add_provisional_chunk_flags,
    aggregate_channel_metrics,
    aggregate_recording_metrics,
)
from src.quality.interval_utils import (
    classify_chunk_annotation,
)
from src.quality.metrics import (
    SpectralConfig,
    calculate_channel_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "chbmit_qc.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_seizure_lookup(
    seizures: pd.DataFrame,
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Build recording-keyed seizure interval lookup."""
    lookup: dict[
        tuple[str, str],
        list[tuple[float, float]],
    ] = {}

    for key, group in seizures.groupby(
        ["case_id", "recording_id"]
    ):
        lookup[key] = list(
            zip(
                group["onset_seconds"].astype(float),
                group["offset_seconds"].astype(float),
                strict=True,
            )
        )

    return lookup


def iter_chunks(
    n_times: int,
    sampling_rate_hz: float,
    chunk_duration_seconds: float,
    minimum_chunk_duration_seconds: float,
):
    """Yield fixed-duration sample ranges."""
    chunk_samples = int(
        round(chunk_duration_seconds * sampling_rate_hz)
    )
    minimum_samples = int(
        round(minimum_chunk_duration_seconds * sampling_rate_hz)
    )

    if chunk_samples <= 0:
        raise ValueError("Chunk length must be positive.")

    chunk_index = 0

    for start_sample in range(0, n_times, chunk_samples):
        stop_sample = min(
            n_times,
            start_sample + chunk_samples,
        )

        if stop_sample - start_sample < minimum_samples:
            continue

        yield chunk_index, start_sample, stop_sample
        chunk_index += 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional development limit.",
    )
    parser.add_argument(
        "--recording-id",
        type=str,
        default=None,
        help="Run QC for one recording only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    config = load_config(args.config)

    raw_dir = (
        PROJECT_ROOT / config["dataset"]["raw_dir"]
    ).resolve()

    recordings_path = (
        PROJECT_ROOT
        / config["dataset"]["recordings_file"]
    ).resolve()

    seizures_path = (
        PROJECT_ROOT
        / config["dataset"]["seizures_file"]
    ).resolve()

    output_dir = (
        PROJECT_ROOT / config["output"]["directory"]
    ).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_output = output_dir / "chbmit_qc_chunks.csv"
    channel_output = output_dir / "chbmit_qc_channels.csv"
    recording_output = output_dir / "chbmit_qc_recordings.csv"
    failure_output = output_dir / "chbmit_qc_failures.csv"

    existing_outputs = [
        chunk_output,
        channel_output,
        recording_output,
    ]

    overwrite = (
        args.overwrite
        or config["runtime"].get("overwrite", False)
    )

    if any(path.exists() for path in existing_outputs) and not overwrite:
        print(
            "ERROR: QC outputs already exist. "
            "Use --overwrite to replace them.",
            file=sys.stderr,
        )
        return 1

    recordings = pd.read_csv(recordings_path)
    seizures = pd.read_csv(seizures_path)

    recordings = recordings.loc[
        recordings["metadata_status"] == "matched"
    ].copy()

    if args.recording_id:
        recordings = recordings.loc[
            recordings["recording_id"]
            == args.recording_id
        ]

    if args.limit_files is not None:
        recordings = recordings.head(args.limit_files)

    if recordings.empty:
        print(
            "ERROR: No recordings selected.",
            file=sys.stderr,
        )
        return 1

    seizure_lookup = build_seizure_lookup(seizures)

    spectral_section = config["spectral"]

    spectral_config = SpectralConfig(
        fmin_hz=float(spectral_section["fmin_hz"]),
        fmax_hz=float(spectral_section["fmax_hz"]),
        nperseg_seconds=float(
            spectral_section["nperseg_seconds"]
        ),
        overlap_fraction=float(
            spectral_section["overlap_fraction"]
        ),
        line_frequency_hz=float(
            spectral_section["line_frequency_hz"]
        ),
        line_band_half_width_hz=float(
            spectral_section[
                "line_band_half_width_hz"
            ]
        ),
        line_reference_band_half_width_hz=float(
            spectral_section[
                "line_reference_band_half_width_hz"
            ]
        ),
        high_frequency_start_hz=float(
            spectral_section[
                "high_frequency_start_hz"
            ]
        ),
        high_frequency_end_hz=float(
            spectral_section[
                "high_frequency_end_hz"
            ]
        ),
    )

    chunk_rows: list[dict] = []
    failure_rows: list[dict] = []

    total_files = len(recordings)

    for file_index, recording in enumerate(
        recordings.itertuples(index=False),
        start=1,
    ):
        edf_path = raw_dir / recording.edf_relative_path

        print(
            f"[{file_index:04d}/{total_files:04d}] "
            f"{recording.case_id}/{recording.recording_id}"
        )

        try:
            raw = mne.io.read_raw_edf(
                edf_path,
                preload=False,
                infer_types=False,
                verbose="ERROR",
            )

            sampling_rate_hz = float(raw.info["sfreq"])
            seizure_intervals = seizure_lookup.get(
                (
                    recording.case_id,
                    recording.recording_id,
                ),
                [],
            )

            for (
                chunk_index,
                start_sample,
                stop_sample,
            ) in iter_chunks(
                n_times=raw.n_times,
                sampling_rate_hz=sampling_rate_hz,
                chunk_duration_seconds=float(
                    config["processing"][
                        "chunk_duration_seconds"
                    ]
                ),
                minimum_chunk_duration_seconds=float(
                    config["processing"][
                        "minimum_chunk_duration_seconds"
                    ]
                ),
            ):
                chunk_start_seconds = (
                    start_sample / sampling_rate_hz
                )
                chunk_end_seconds = (
                    stop_sample / sampling_rate_hz
                )

                (
                    annotation_label,
                    seizure_overlap_fraction,
                ) = classify_chunk_annotation(
                    chunk_start=chunk_start_seconds,
                    chunk_end=chunk_end_seconds,
                    seizure_intervals=seizure_intervals,
                    ictal_threshold=float(
                        config["annotation"][
                            "ictal_overlap_threshold"
                        ]
                    ),
                )

                data_volts = raw.get_data(
                    start=start_sample,
                    stop=stop_sample,
                    reject_by_annotation=None,
                )

                data_uv = data_volts * 1e6

                for channel_index, channel_name in enumerate(
                    raw.ch_names
                ):
                    metrics = calculate_channel_metrics(
                        signal_uv=data_uv[channel_index],
                        sampling_rate_hz=sampling_rate_hz,
                        difference_tolerance_uv=float(
                            config["flatline"][
                                "difference_tolerance_uv"
                            ]
                        ),
                        minimum_flat_run_seconds=float(
                            config["flatline"][
                                "minimum_run_seconds"
                            ]
                        ),
                        spectral_config=spectral_config,
                    )

                    chunk_rows.append(
                        {
                            "dataset": "CHB-MIT",
                            "dataset_version": "1.0.0",
                            "case_id": recording.case_id,
                            "subject_id": recording.subject_id,
                            "recording_id": (
                                recording.recording_id
                            ),
                            "edf_relative_path": (
                                recording.edf_relative_path
                            ),
                            "chunk_id": (
                                f"{recording.recording_id}"
                                f"__chunk{chunk_index:05d}"
                            ),
                            "chunk_index": chunk_index,
                            "chunk_start_seconds": (
                                chunk_start_seconds
                            ),
                            "chunk_end_seconds": (
                                chunk_end_seconds
                            ),
                            "chunk_duration_seconds": (
                                chunk_end_seconds
                                - chunk_start_seconds
                            ),
                            "start_sample": start_sample,
                            "stop_sample_exclusive": (
                                stop_sample
                            ),
                            "sampling_rate_hz": (
                                sampling_rate_hz
                            ),
                            "channel_index": channel_index,
                            "channel_name": channel_name,
                            "annotation_label": (
                                annotation_label
                            ),
                            "seizure_overlap_fraction": (
                                seizure_overlap_fraction
                            ),
                            **metrics,
                        }
                    )

            raw.close()

        except Exception as exc:
            failure_rows.append(
                {
                    "case_id": recording.case_id,
                    "recording_id": recording.recording_id,
                    "edf_relative_path": (
                        recording.edf_relative_path
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            print(
                f"WARNING: {recording.recording_id}: {exc}",
                file=sys.stderr,
            )

            if config["runtime"].get("fail_fast", False):
                raise

    chunk_metrics = pd.DataFrame(chunk_rows)

    if chunk_metrics.empty:
        print(
            "ERROR: No chunk metrics were generated.",
            file=sys.stderr,
        )
        return 1

    chunk_metrics = add_provisional_chunk_flags(
        metrics=chunk_metrics,
        thresholds=config["provisional_flags"],
    )

    channel_metrics = aggregate_channel_metrics(
        chunk_metrics=chunk_metrics
    )

    recording_metrics = aggregate_recording_metrics(
        channel_metrics=channel_metrics,
        bad_chunk_fraction_threshold=float(
            config["provisional_flags"][
                "bad_chunk_fraction_review"
            ]
        ),
    )

    chunk_metrics.to_csv(
        chunk_output,
        index=False,
    )

    channel_metrics.to_csv(
        channel_output,
        index=False,
    )

    recording_metrics.to_csv(
        recording_output,
        index=False,
    )

    failures = pd.DataFrame(failure_rows)

    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "case_id",
                "recording_id",
                "edf_relative_path",
                "error_type",
                "error_message",
            ]
        )

    failures.to_csv(
        failure_output,
        index=False,
    )

    run_summary = {
        "selected_recordings": int(total_files),
        "processed_recordings": int(
            chunk_metrics["recording_id"].nunique()
        ),
        "failed_recordings": int(len(failures)),
        "chunk_channel_rows": int(len(chunk_metrics)),
        "recording_channel_rows": int(
            len(channel_metrics)
        ),
        "recording_rows": int(
            len(recording_metrics)
        ),
        "review_chunk_rows": int(
            chunk_metrics["needs_review"].sum()
        ),
        "review_recordings": int(
            recording_metrics[
                "recording_needs_review"
            ].sum()
        ),
        "config_file": str(
            args.config.relative_to(PROJECT_ROOT)
        ),
    }

    with (
        output_dir / "chbmit_qc_run_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2)

    print("\nQC completed.")
    print(json.dumps(run_summary, indent=2))

    return 0 if failures.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())