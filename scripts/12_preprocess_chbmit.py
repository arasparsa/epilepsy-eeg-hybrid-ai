"""Preprocess all CHB-MIT recordings in the frozen inclusion manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mne
############
"""
_orig_read_raw_edf = mne.io.read_raw_edf
def _patched_read_raw_edf(*args, **kwargs):
    raw = _orig_read_raw_edf(*args, **kwargs)
    raw.set_meas_date(None)
    return raw
mne.io.read_raw_edf = _patched_read_raw_edf
"""
############
import pandas as pd
import yaml

from src.channels.naming import (
    load_validated_aliases,
)
from src.preprocessing.pipeline import (
    preprocess_recording,
)
from src.preprocessing.provenance import (
    calculate_sha256,
    get_git_commit_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_preprocessing.yaml"
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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

    raw_dir = (
        PROJECT_ROOT / config["inputs"]["raw_dir"]
    ).resolve()

    inclusion_manifest_path = (
        PROJECT_ROOT
        / config["inputs"]["inclusion_manifest"]
    ).resolve()

    seizures_path = (
        PROJECT_ROOT
        / config["inputs"]["seizures_file"]
    ).resolve()

    aliases_path = (
        PROJECT_ROOT
        / config["inputs"]["alias_file"]
    ).resolve()

    harmonization_config_path = (
        PROJECT_ROOT
        / config["inputs"]["harmonization_config"]
    ).resolve()

    output_signal_dir = (
        PROJECT_ROOT
        / config["outputs"]["signal_directory"]
    ).resolve()

    output_metadata_dir = (
        PROJECT_ROOT
        / config["outputs"]["metadata_directory"]
    ).resolve()

    output_signal_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        inclusion_manifest_path
    )
    seizures = pd.read_csv(seizures_path)
    alias_table = pd.read_csv(aliases_path)
    harmonization_config = load_yaml(
        harmonization_config_path
    )

    selected_recordings = manifest.loc[
        manifest["include_primary_analysis"]
    ].copy()

    if args.recording_id:
        selected_recordings = selected_recordings.loc[
            selected_recordings["recording_id"]
            == args.recording_id
        ]

    if args.limit_files is not None:
        selected_recordings = (
            selected_recordings.head(
                args.limit_files
            )
        )

    if selected_recordings.empty:
        print(
            "ERROR: No recordings selected.",
            file=sys.stderr,
        )
        return 1

    channel_set_name = config["channels"][
        "channel_set_name"
    ]

    target_channels = harmonization_config[
        "channel_sets"
    ][channel_set_name]["channels"]

    alias_rules = load_validated_aliases(
        alias_table
    )

    overwrite = (
        args.overwrite
        or config["saving"]["overwrite"]
    )

    git_commit = get_git_commit_hash()
    run_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    success_rows: list[dict] = []
    failure_rows: list[dict] = []

    total = len(selected_recordings)

    for index, recording in enumerate(
        selected_recordings.itertuples(
            index=False
        ),
        start=1,
    ):
        print(
            f"[{index:04d}/{total:04d}] "
            f"{recording.case_id}/"
            f"{recording.recording_id}"
        )

        input_path = (
            raw_dir / recording.edf_relative_path
        )

        case_output_dir = (
            output_signal_dir / recording.case_id
        )
        case_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            case_output_dir
            / (
                f"{recording.recording_id}"
                "_preproc-v1_raw.fif"
            )
        )

        if output_path.exists() and not overwrite:
            failure_rows.append(
                {
                    "case_id": recording.case_id,
                    "recording_id": (
                        recording.recording_id
                    ),
                    "input_path": (
                        recording.edf_relative_path
                    ),
                    "output_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "error_type": (
                        "OutputAlreadyExists"
                    ),
                    "error_message": (
                        "Use --overwrite to replace."
                    ),
                }
            )
            continue

        try:
            result = preprocess_recording(
                edf_path=input_path,
                recording_id=(
                    recording.recording_id
                ),
                case_id=recording.case_id,
                target_channels=target_channels,
                alias_rules=alias_rules,
                seizures=seizures,
                config=config,
            )

            result.raw.save(
                output_path,
                fmt=config["saving"]["fmt"],
                split_size=config[
                    "saving"
                ]["split_size"],
                overwrite=overwrite,
                verbose="ERROR",
            )

            result.raw.close()

            output_hash = None

            if config["saving"].get(
                "compute_sha256",
                True,
            ):
                output_hash = calculate_sha256(
                    output_path
                )

            success_rows.append(
                {
                    **result.metadata,
                    "output_fif_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "output_file_size_bytes": (
                        output_path.stat().st_size
                    ),
                    "output_sha256": output_hash,
                    "preprocessing_version": (
                        config["project"][
                            "preprocessing_version"
                        ]
                    ),
                    "harmonization_policy_version": (
                        config["project"][
                            "harmonization_policy_version"
                        ]
                    ),
                    "channel_set_name": (
                        channel_set_name
                    ),
                    "mne_version": mne.__version__,
                    "git_commit": git_commit,
                    "run_timestamp_utc": (
                        run_timestamp
                    ),
                    "processing_status": (
                        "success"
                    ),
                }
            )

        except Exception as exc:
            failure_rows.append(
                {
                    "case_id": recording.case_id,
                    "recording_id": (
                        recording.recording_id
                    ),
                    "input_path": (
                        recording.edf_relative_path
                    ),
                    "output_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            print(
                f"WARNING: "
                f"{recording.recording_id}: {exc}",
                file=sys.stderr,
            )

            if config["runtime"].get(
                "fail_fast",
                False,
            ):
                raise

    preprocessing_manifest = pd.DataFrame(
        success_rows
    )

    failures = pd.DataFrame(
        failure_rows
    )

    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "case_id",
                "recording_id",
                "input_path",
                "output_path",
                "error_type",
                "error_message",
            ]
        )

    preprocessing_manifest.to_csv(
        output_metadata_dir
        / "chbmit_preprocessing_manifest.csv",
        index=False,
    )

    failures.to_csv(
        output_metadata_dir
        / "chbmit_preprocessing_failures.csv",
        index=False,
    )

    summary = {
        "preprocessing_version": (
            config["project"][
                "preprocessing_version"
            ]
        ),
        "selected_recordings": int(total),
        "successfully_processed": int(
            len(preprocessing_manifest)
        ),
        "failed_recordings": int(
            len(failures)
        ),
        "target_channel_count": int(
            len(target_channels)
        ),
        "target_sampling_rate_hz": float(
            config["resampling"][
                "target_sampling_rate_hz"
            ]
        ),
        "filter_band_hz": [
            config["filtering"]["l_freq_hz"],
            config["filtering"]["h_freq_hz"],
        ],
        "notch_enabled": bool(
            config["notch_filter"]["enabled"]
        ),
        "resampling_enabled": bool(
            config["resampling"]["enabled"]
        ),
        "normalization_enabled": bool(
            config["normalization"]["enabled"]
        ),
        "git_commit": git_commit,
        "run_timestamp_utc": run_timestamp,
    }

    with (
        output_metadata_dir
        / "chbmit_preprocessing_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nPreprocessing completed.")
    print(json.dumps(summary, indent=2))

    return 0 if failures.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())