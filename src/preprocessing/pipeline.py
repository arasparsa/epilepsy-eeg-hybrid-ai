"""Reproducible preprocessing pipeline for CHB-MIT recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd

from src.channels.harmonize import (
    harmonize_raw_channels,
)
from src.channels.naming import AliasRule
from src.preprocessing.annotations import (
    append_annotations_without_duplicates,
    build_ictal_annotations,
    validate_ictal_annotations,
)


@dataclass(frozen=True)
class PreprocessingResult:
    """Result metadata for one preprocessed recording."""

    raw: mne.io.BaseRaw
    metadata: dict[str, object]


def preprocess_recording(
    *,
    edf_path: Path,
    recording_id: str,
    case_id: str,
    target_channels: list[str],
    alias_rules: list[AliasRule],
    seizures: pd.DataFrame,
    config: dict,
) -> PreprocessingResult:
    """Load, harmonize, annotate, filter and validate one EDF."""
    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        infer_types=False,
        verbose="ERROR",
    )
   # ----------------------------------------
    raw.set_meas_date(None)
   # ----------------------------------------

    input_sampling_rate = float(raw.info["sfreq"])
    input_n_times = int(raw.n_times)
    input_duration = (
        input_n_times / input_sampling_rate
    )
    input_channel_count = len(raw.ch_names)

    # Phase-5 frozen channel naming, selection and ordering.
    processed = harmonize_raw_channels(
        raw=raw,
        target_channels=target_channels,
        recording_id=recording_id,
        alias_rules=alias_rules,
        copy=True,
    )

    if config["channels"].get(
        "set_all_selected_channels_to_eeg",
        True,
    ):
        channel_type_mapping = {
            channel: "eeg"
            for channel in processed.ch_names
            if processed.get_channel_types(
                picks=[channel]
            )[0] != "eeg"
        }

        if channel_type_mapping:
            processed.set_channel_types(
                channel_type_mapping,
                on_unit_change="ignore",
            )

    if processed.ch_names != target_channels:
        raise RuntimeError(
            f"Channel-order mismatch in {recording_id}"
        )

    # Add seizure annotations before filtering so that the same
    # continuous time coordinate system is retained.
    expected_seizure_count = int(
        (
            seizures["recording_id"]
            == recording_id
        ).sum()
    )

    if config["annotations"][
        "add_ictal_annotations"
    ]:
        ictal_annotations = build_ictal_annotations(
            seizures=seizures,
            recording_id=recording_id,
            description=config["annotations"][
                "ictal_description"
            ],
        )

        processed = append_annotations_without_duplicates(
            raw=processed,
            new_annotations=ictal_annotations,
            description=config["annotations"][
                "ictal_description"
            ],
        )

    annotation_validation_before = (
        validate_ictal_annotations(
            raw=processed,
            expected_seizure_count=expected_seizure_count,
            description=config["annotations"][
                "ictal_description"
            ],
        )
    )

    if not annotation_validation_before[
        "ictal_annotation_count_match"
    ]:
        raise ValueError(
            f"Ictal annotation-count mismatch in {recording_id}"
        )

    if not annotation_validation_before[
        "all_ictal_annotations_inside_raw"
    ]:
        raise ValueError(
            f"Ictal annotation outside recording: "
            f"{recording_id}"
        )

    if config["runtime"].get(
        "preload_after_channel_selection",
        True,
    ):
        processed.load_data()

    filtering = config["filtering"]

    if filtering["enabled"]:
        processed.filter(
            l_freq=float(filtering["l_freq_hz"]),
            h_freq=float(filtering["h_freq_hz"]),
            picks=target_channels,
            method=filtering["method"],
            phase=filtering["phase"],
            fir_window=filtering["fir_window"],
            fir_design=filtering["fir_design"],
            filter_length=filtering[
                "filter_length"
            ],
            l_trans_bandwidth=filtering[
                "l_trans_bandwidth"
            ],
            h_trans_bandwidth=filtering[
                "h_trans_bandwidth"
            ],
            pad=filtering["pad"],
            skip_by_annotation=tuple(
                filtering["skip_by_annotation"]
            ),
            verbose="ERROR",
        )

    notch = config["notch_filter"]

    if notch["enabled"]:
        processed.notch_filter(
            freqs=[
                float(frequency)
                for frequency in notch[
                    "frequencies_hz"
                ]
            ],
            picks=target_channels,
            method=notch["method"],
            phase=notch["phase"],
            fir_design=notch["fir_design"],
            verbose="ERROR",
        )

    resampling = config["resampling"]
    resampling_applied = False

    if resampling["enabled"]:
        target_sampling_rate = float(
            resampling["target_sampling_rate_hz"]
        )

        if (
            float(processed.info["sfreq"])
            != target_sampling_rate
        ):
            processed.resample(
                sfreq=target_sampling_rate,
                method=resampling["method"],
                verbose="ERROR",
            )
            resampling_applied = True

    output_sampling_rate = float(
        processed.info["sfreq"]
    )
    output_n_times = int(processed.n_times)
    output_duration = (
        output_n_times / output_sampling_rate
    )

    annotation_validation_after = (
        validate_ictal_annotations(
            raw=processed,
            expected_seizure_count=expected_seizure_count,
            description=config["annotations"][
                "ictal_description"
            ],
        )
    )

    metadata: dict[str, object] = {
        "case_id": case_id,
        "recording_id": recording_id,
        "input_edf_path": edf_path.as_posix(),
        "input_sampling_rate_hz": (
            input_sampling_rate
        ),
        "output_sampling_rate_hz": (
            output_sampling_rate
        ),
        "input_n_times": input_n_times,
        "output_n_times": output_n_times,
        "input_duration_seconds": input_duration,
        "output_duration_seconds": output_duration,
        "input_channel_count": input_channel_count,
        "output_channel_count": len(
            processed.ch_names
        ),
        "output_channel_order": "|".join(
            processed.ch_names
        ),
        "filter_applied": bool(
            filtering["enabled"]
        ),
        "l_freq_hz": (
            float(filtering["l_freq_hz"])
            if filtering["enabled"]
            else None
        ),
        "h_freq_hz": (
            float(filtering["h_freq_hz"])
            if filtering["enabled"]
            else None
        ),
        "notch_applied": bool(
            notch["enabled"]
        ),
        "resampling_applied": (
            resampling_applied
        ),
        "rereferencing_applied": False,
        "normalization_applied": False,
        **{
            f"before_{key}": value
            for key, value
            in annotation_validation_before.items()
        },
        **{
            f"after_{key}": value
            for key, value
            in annotation_validation_after.items()
        },
    }

    return PreprocessingResult(
        raw=processed,
        metadata=metadata,
    )