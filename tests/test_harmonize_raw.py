"""Tests for deterministic Raw channel harmonization."""

import mne
import numpy as np
import pytest

from src.channels.harmonize import (
    harmonize_raw_channels,
)


def create_test_raw(
    channel_names: list[str],
) -> mne.io.RawArray:
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=256.0,
        ch_types="eeg",
    )

    data = np.zeros(
        (len(channel_names), 2560)
    )

    return mne.io.RawArray(
        data,
        info,
        verbose="ERROR",
    )


def test_channels_are_selected_and_reordered() -> None:
    raw = create_test_raw(
        [
            "F3-C3",
            "FP1-F7",
            "EXTRA",
            "F7-T7",
        ]
    )

    target = [
        "FP1-F7",
        "F7-T7",
        "F3-C3",
    ]

    harmonized = harmonize_raw_channels(
        raw=raw,
        target_channels=target,
        recording_id="test",
    )

    assert harmonized.ch_names == target
    assert raw.ch_names != target


def test_missing_target_channel_raises() -> None:
    raw = create_test_raw(
        ["FP1-F7", "F7-T7"]
    )

    with pytest.raises(ValueError):
        harmonize_raw_channels(
            raw=raw,
            target_channels=[
                "FP1-F7",
                "F7-T7",
                "T7-P7",
            ],
            recording_id="test",
        )