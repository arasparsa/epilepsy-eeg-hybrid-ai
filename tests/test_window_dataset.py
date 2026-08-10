"""Tests for EEGWindowDataset."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

from src.data.window_dataset import (
    EEGWindowDataset,
)


def test_dataset_reads_and_normalizes_window(
    tmp_path: Path,
) -> None:
    channel_count = 2
    sample_count = 2048
    sampling_rate = 256.0

    data_volts = np.vstack(
        [
            np.full(
                sample_count,
                10e-6,
            ),
            np.full(
                sample_count,
                20e-6,
            ),
        ]
    )

    info = mne.create_info(
        ch_names=["A", "B"],
        sfreq=sampling_rate,
        ch_types=["eeg", "eeg"],
    )

    raw = mne.io.RawArray(
        data_volts,
        info,
        verbose="ERROR",
    )

    fif_path = (
        tmp_path / "test_raw.fif"
    )

    raw.save(
        fif_path,
        overwrite=True,
        verbose="ERROR",
    )

    windows = pd.DataFrame(
        {
            "window_id": ["w1"],
            "output_fif_path": [
                fif_path.name
            ],
            "start_sample": [0],
            "stop_sample_exclusive": [
                1024
            ],
            "binary_label": [1],
            "case_id": ["case"],
            "subject_id": ["subject"],
            "recording_id": ["recording"],
        }
    )

    dataset = EEGWindowDataset(
        windows=windows,
        project_root=tmp_path,
        channel_mean_uv=np.asarray(
            [10.0, 20.0]
        ),
        channel_std_uv=np.asarray(
            [2.0, 4.0]
        ),
        expected_channel_count=2,
        expected_sample_count=1024,
        max_open_fif_files=1,
        return_metadata=True,
    )

    signal, label, metadata = (
        dataset[0]
    )

    assert signal.shape == (
        2,
        1024,
    )

    assert torch.allclose(
        signal,
        torch.zeros_like(signal),
        atol=1e-5,
    )

    assert label.item() == 1.0
    assert metadata["window_id"] == "w1"