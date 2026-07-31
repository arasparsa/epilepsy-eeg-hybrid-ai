"""Tests for preprocessing principles."""

import mne
import numpy as np


def test_bandpass_preserves_shape() -> None:
    sampling_rate = 256.0
    duration = 20.0

    time = np.arange(
        int(duration * sampling_rate)
    ) / sampling_rate

    signal = (
        np.sin(2 * np.pi * 10 * time)
        + 0.5 * np.sin(2 * np.pi * 60 * time)
    )

    info = mne.create_info(
        ch_names=["FP1-F7"],
        sfreq=sampling_rate,
        ch_types=["eeg"],
    )

    raw = mne.io.RawArray(
        signal[np.newaxis, :],
        info,
        verbose="ERROR",
    )

    original_shape = raw.get_data().shape

    raw.filter(
        l_freq=0.5,
        h_freq=45.0,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )

    assert raw.get_data().shape == original_shape
    assert raw.info["sfreq"] == sampling_rate


def test_filter_reduces_60_hz_component() -> None:
    sampling_rate = 256.0
    duration = 20.0

    time = np.arange(
        int(duration * sampling_rate)
    ) / sampling_rate

    signal_60 = np.sin(
        2 * np.pi * 60 * time
    )

    info = mne.create_info(
        ["FP1-F7"],
        sampling_rate,
        ["eeg"],
    )

    raw = mne.io.RawArray(
        signal_60[np.newaxis, :],
        info,
        verbose="ERROR",
    )

    before_rms = float(
        np.sqrt(
            np.mean(raw.get_data() ** 2)
        )
    )

    raw.filter(
        l_freq=0.5,
        h_freq=45.0,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )

    after_rms = float(
        np.sqrt(
            np.mean(raw.get_data() ** 2)
        )
    )

    assert after_rms < before_rms * 0.2