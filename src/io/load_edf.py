"""Utilities for loading EEG recordings stored in EDF format."""

from pathlib import Path

import mne


def load_edf_recording(
    file_path: str | Path,
    preload: bool = False,
) -> mne.io.BaseRaw:
    """
    Load an EEG recording from an EDF file.

    Parameters
    ----------
    file_path:
        Path to the EDF recording.
    preload:
        Whether to load the full signal into memory.

    Returns
    -------
    mne.io.BaseRaw
        Loaded MNE Raw object.

    Raises
    ------
    FileNotFoundError
        If the requested EDF file does not exist.
    ValueError
        If the file does not have an EDF extension.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"EDF file was not found: {path}")

    if path.suffix.lower() != ".edf":
        raise ValueError(f"Expected an EDF file, received: {path.suffix}")

    raw = mne.io.read_raw_edf(
        input_fname=path,
        preload=preload,
        verbose="ERROR",
    )

    return raw