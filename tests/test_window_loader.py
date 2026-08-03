"""Tests for loading signal windows from FIF."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.segmentation.loader import (
    load_window_from_manifest_row,
)


def test_load_window_from_manifest(
    tmp_path: Path,
) -> None:
    sampling_rate = 256.0

    info = mne.create_info(
        ch_names=["A", "B"],
        sfreq=sampling_rate,
        ch_types=["eeg", "eeg"],
    )

    data = np.arange(
        2 * 2048,
        dtype=float,
    ).reshape(2, 2048)

    raw = mne.io.RawArray(
        data,
        info,
        verbose="ERROR",
    )

    output_path = (
        tmp_path / "example_raw.fif"
    )

    raw.save(
        output_path,
        overwrite=True,
        verbose="ERROR",
    )

    row = pd.Series(
        {
            "output_fif_path": (
                output_path.name
            ),
            "start_sample": 512,
            "stop_sample_exclusive": 1536,
        }
    )

    loaded, channels, sfreq = (
        load_window_from_manifest_row(
            row=row,
            project_root=tmp_path,
        )
    )

    assert loaded.shape == (2, 1024)
    assert channels == ["A", "B"]
    assert sfreq == 256.0