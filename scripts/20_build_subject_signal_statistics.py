"""Build subject-level channel statistics from continuous FIF data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

from src.normalization.streaming_stats import (
    RunningChannelStatistics,
)
from src.splitting.subject_mapping import (
    apply_subject_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_data_pipeline.yaml"
)


def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()
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

    output_path = (
        output_dir
        / "chbmit_subject_channel_statistics.csv"
    )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            "Subject statistics already exist. "
            "Use --overwrite."
        )

    preprocessing = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "preprocessing_manifest"
        ]
    )

    mapping = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "subject_mapping"
        ]
    )

    preprocessing = apply_subject_mapping(
        preprocessing,
        mapping,
    )

    expected_channel_count = int(
        config["signal"][
            "expected_channel_count"
        ]
    )

    expected_sfreq = float(
        config["signal"][
            "expected_sampling_rate_hz"
        ]
    )

    conversion = float(
        config["signal"][
            "volts_to_microvolts"
        ]
    )

    chunk_duration = float(
        config["normalization"][
            "fit_chunk_duration_seconds"
        ]
    )

    subject_rows: list[dict] = []
    failures: list[dict] = []

    for subject_id, subject_files in (
        preprocessing.groupby(
            "subject_id",
            sort=True,
        )
    ):
        subject_statistics = (
            RunningChannelStatistics.create(
                expected_channel_count
            )
        )

        channel_names: list[str] | None = None
        recording_count = 0

        for recording in (
            subject_files.itertuples(
                index=False
            )
        ):
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

                if (
                    len(raw.ch_names)
                    != expected_channel_count
                ):
                    raise ValueError(
                        "Unexpected channel count."
                    )

                if not np.isclose(
                    raw.info["sfreq"],
                    expected_sfreq,
                ):
                    raise ValueError(
                        "Unexpected sampling rate."
                    )

                if channel_names is None:
                    channel_names = list(
                        raw.ch_names
                    )

                elif (
                    raw.ch_names
                    != channel_names
                ):
                    raise ValueError(
                        "Channel-order mismatch "
                        "within subject."
                    )

                chunk_samples = int(
                    round(
                        chunk_duration
                        * raw.info["sfreq"]
                    )
                )

                for start_sample in range(
                    0,
                    raw.n_times,
                    chunk_samples,
                ):
                    stop_sample = min(
                        raw.n_times,
                        start_sample
                        + chunk_samples,
                    )

                    data_uv = (
                        raw.get_data(
                            start=start_sample,
                            stop=stop_sample,
                        )
                        * conversion
                    )

                    subject_statistics.update(
                        data_uv
                    )

                recording_count += 1
                raw.close()

            except Exception as exc:
                failures.append(
                    {
                        "subject_id": (
                            subject_id
                        ),
                        "case_id": (
                            recording.case_id
                        ),
                        "recording_id": (
                            recording.recording_id
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error_message": str(
                            exc
                        ),
                    }
                )

                if config["runtime"].get(
                    "fail_fast",
                    True,
                ):
                    raise

        if channel_names is None:
            continue

        statistics = (
            subject_statistics.to_dict()
        )

        for channel_index, (
            channel_name
        ) in enumerate(channel_names):
            subject_rows.append(
                {
                    "subject_id": (
                        subject_id
                    ),
                    "channel_index": (
                        channel_index
                    ),
                    "channel_name": (
                        channel_name
                    ),
                    "sample_count": int(
                        statistics[
                            "count"
                        ][channel_index]
                    ),
                    "mean_uv": float(
                        statistics[
                            "mean"
                        ][channel_index]
                    ),
                    "m2_uv2": float(
                        statistics[
                            "m2"
                        ][channel_index]
                    ),
                    "variance_uv2": float(
                        statistics[
                            "variance"
                        ][channel_index]
                    ),
                    "std_uv": float(
                        statistics[
                            "std"
                        ][channel_index]
                    ),
                    "recording_count": (
                        recording_count
                    ),
                }
            )

    subject_table = pd.DataFrame(
        subject_rows
    )

    subject_table.to_csv(
        output_path,
        index=False,
    )

    failure_table = pd.DataFrame(
        failures
    )

    if failure_table.empty:
        failure_table = pd.DataFrame(
            columns=[
                "subject_id",
                "case_id",
                "recording_id",
                "error_type",
                "error_message",
            ]
        )

    failure_table.to_csv(
        output_dir
        / "chbmit_data_pipeline_failures.csv",
        index=False,
    )

    summary = {
        "subject_count": int(
            subject_table[
                "subject_id"
            ].nunique()
        ),
        "channel_count": (
            expected_channel_count
        ),
        "subject_channel_rows": int(
            len(subject_table)
        ),
        "failed_recordings": int(
            len(failure_table)
        ),
    }

    with (
        output_dir
        / "chbmit_subject_statistics_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return (
        0
        if failure_table.empty
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())