"""Read EEG windows from continuous preprocessed FIF files."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd


def load_window_from_manifest_row(
    row: pd.Series,
    project_root: Path,
    preload_raw: bool = False,
) -> tuple[np.ndarray, list[str], float]:
    """Load one EEG window as channels × samples."""
    fif_path = (
        project_root / row["output_fif_path"]
    )

    if not fif_path.exists():
        raise FileNotFoundError(
            f"Preprocessed FIF not found: {fif_path}"
        )

    raw = mne.io.read_raw_fif(
        fif_path,
        preload=preload_raw,
        verbose="ERROR",
    )

    start_sample = int(row["start_sample"])
    stop_sample = int(
        row["stop_sample_exclusive"]
    )

    if start_sample < 0:
        raise ValueError(
            "Window start sample cannot be negative."
        )

    if stop_sample > raw.n_times:
        raise ValueError(
            "Window extends beyond Raw data."
        )

    data = raw.get_data(
        start=start_sample,
        stop=stop_sample,
    )

    channel_names = list(raw.ch_names)
    sampling_rate = float(raw.info["sfreq"])

    raw.close()

    return data, channel_names, sampling_rate