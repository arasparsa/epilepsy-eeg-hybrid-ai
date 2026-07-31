"""Create and validate seizure annotations for preprocessed Raw data."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd


def build_ictal_annotations(
    seizures: pd.DataFrame,
    recording_id: str,
    description: str = "ictal",
) -> mne.Annotations:
    """Create MNE annotations for one recording."""
    recording_seizures = (
        seizures.loc[
            seizures["recording_id"] == recording_id
        ]
        .sort_values(
            ["onset_seconds", "seizure_index"]
        )
        .copy()
    )

    if recording_seizures.empty:
        return mne.Annotations(
            onset=[],
            duration=[],
            description=[],
        )

    invalid = recording_seizures.loc[
        (recording_seizures["onset_seconds"] < 0)
        | (
            recording_seizures["offset_seconds"]
            <= recording_seizures["onset_seconds"]
        )
    ]

    if not invalid.empty:
        raise ValueError(
            f"Invalid seizure intervals for {recording_id}"
        )

    return mne.Annotations(
        onset=recording_seizures[
            "onset_seconds"
        ].to_numpy(dtype=float),
        duration=recording_seizures[
            "duration_seconds"
        ].to_numpy(dtype=float),
        description=np.repeat(
            description,
            len(recording_seizures),
        ),
    )


def append_annotations_without_duplicates(
    raw: mne.io.BaseRaw,
    new_annotations: mne.Annotations,
    description: str = "ictal",
) -> mne.io.BaseRaw:
    """Append annotations while rejecting duplicate ictal events."""
    if len(new_annotations) == 0:
        return raw

    existing_keys = {
        (
            round(float(onset), 6),
            round(float(duration), 6),
            str(annotation_description),
        )
        for onset, duration, annotation_description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
            strict=True,
        )
    }

    filtered_onsets: list[float] = []
    filtered_durations: list[float] = []
    filtered_descriptions: list[str] = []

    for onset, duration, annotation_description in zip(
        new_annotations.onset,
        new_annotations.duration,
        new_annotations.description,
        strict=True,
    ):
        key = (
            round(float(onset), 6),
            round(float(duration), 6),
            str(annotation_description),
        )

        if key in existing_keys:
            continue

        filtered_onsets.append(float(onset))
        filtered_durations.append(float(duration))
        filtered_descriptions.append(
            str(annotation_description)
        )

    filtered_annotations = mne.Annotations(
        onset=filtered_onsets,
        duration=filtered_durations,
        description=filtered_descriptions,
    )

    if len(filtered_annotations):
        raw.set_annotations(
            raw.annotations + filtered_annotations
        )

    return raw


def validate_ictal_annotations(
    raw: mne.io.BaseRaw,
    expected_seizure_count: int,
    description: str = "ictal",
    tolerance_seconds: float = 1e-6,
) -> dict[str, object]:
    """Validate ictal annotations against Raw duration."""
    ictal_mask = (
        raw.annotations.description == description
    )

    onsets = raw.annotations.onset[ictal_mask]
    durations = raw.annotations.duration[ictal_mask]
    offsets = onsets + durations

    raw_duration = raw.n_times / raw.info["sfreq"]

    inside_raw = bool(
        np.all(onsets >= -tolerance_seconds)
        and np.all(
            offsets
            <= raw_duration + tolerance_seconds
        )
    )

    return {
        "expected_ictal_annotation_count": (
            int(expected_seizure_count)
        ),
        "observed_ictal_annotation_count": (
            int(ictal_mask.sum())
        ),
        "ictal_annotation_count_match": (
            int(ictal_mask.sum())
            == int(expected_seizure_count)
        ),
        "all_ictal_annotations_inside_raw": inside_raw,
    }