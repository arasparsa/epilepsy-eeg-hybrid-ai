"""Tests for adding seizure annotations."""

import mne
import numpy as np
import pandas as pd

from src.preprocessing.annotations import (
    append_annotations_without_duplicates,
    build_ictal_annotations,
    validate_ictal_annotations,
)


def create_raw() -> mne.io.RawArray:
    info = mne.create_info(
        ch_names=["FP1-F7"],
        sfreq=256.0,
        ch_types=["eeg"],
    )

    data = np.zeros((1, 256 * 60))

    return mne.io.RawArray(
        data,
        info,
        verbose="ERROR",
    )


def test_build_ictal_annotations() -> None:
    seizures = pd.DataFrame(
        {
            "recording_id": ["rec1", "rec1"],
            "seizure_index": [1, 2],
            "onset_seconds": [10.0, 30.0],
            "offset_seconds": [15.0, 35.0],
            "duration_seconds": [5.0, 5.0],
        }
    )

    annotations = build_ictal_annotations(
        seizures=seizures,
        recording_id="rec1",
    )

    assert len(annotations) == 2
    assert annotations.onset.tolist() == [
        10.0,
        30.0,
    ]
    assert annotations.duration.tolist() == [
        5.0,
        5.0,
    ]


def test_duplicate_annotation_is_not_added() -> None:
    raw = create_raw()

    annotation = mne.Annotations(
        onset=[10.0],
        duration=[5.0],
        description=["ictal"],
    )

    raw.set_annotations(annotation)

    append_annotations_without_duplicates(
        raw=raw,
        new_annotations=annotation,
    )

    assert (
        raw.annotations.description
        == "ictal"
    ).sum() == 1


def test_annotation_validation() -> None:
    raw = create_raw()

    raw.set_annotations(
        mne.Annotations(
            onset=[10.0],
            duration=[5.0],
            description=["ictal"],
        )
    )

    result = validate_ictal_annotations(
        raw=raw,
        expected_seizure_count=1,
    )

    assert result[
        "ictal_annotation_count_match"
    ]
    assert result[
        "all_ictal_annotations_inside_raw"
    ]