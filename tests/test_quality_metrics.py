"""Unit tests for signal-quality metrics."""

import numpy as np

from src.quality.metrics import (
    SpectralConfig,
    calculate_channel_metrics,
    calculate_flatline_metrics,
    longest_true_run,
)


def test_longest_true_run() -> None:
    values = np.array(
        [False, True, True, False, True]
    )

    assert longest_true_run(values) == 2


def test_flat_signal_is_detected() -> None:
    sampling_rate = 256.0
    signal = np.zeros(int(5 * sampling_rate))

    metrics = calculate_flatline_metrics(
        signal_uv=signal,
        sampling_rate_hz=sampling_rate,
        difference_tolerance_uv=0.01,
        minimum_run_seconds=1.0,
    )

    assert metrics["flatline_sample_fraction"] > 0.99
    assert metrics["longest_flat_run_seconds"] >= 4.9


def test_noise_signal_is_not_flat() -> None:
    generator = np.random.default_rng(42)
    sampling_rate = 256.0

    signal = generator.normal(
        loc=0,
        scale=20,
        size=int(10 * sampling_rate),
    )

    metrics = calculate_flatline_metrics(
        signal_uv=signal,
        sampling_rate_hz=sampling_rate,
        difference_tolerance_uv=0.01,
        minimum_run_seconds=1.0,
    )

    assert metrics["flatline_sample_fraction"] == 0


def test_nonfinite_fraction() -> None:
    signal = np.array(
        [0.0, 1.0, np.nan, np.inf]
    )

    metrics = calculate_channel_metrics(
        signal_uv=signal,
        sampling_rate_hz=256.0,
        difference_tolerance_uv=0.01,
        minimum_flat_run_seconds=1.0,
        spectral_config=SpectralConfig(),
    )

    assert metrics["nonfinite_fraction"] == 0.5
    assert metrics["nan_fraction"] == 0.25


def test_line_noise_ratio_increases_for_60_hz_signal() -> None:
    sampling_rate = 256.0
    duration = 20.0

    time = np.arange(
        int(duration * sampling_rate)
    ) / sampling_rate

    generator = np.random.default_rng(42)

    baseline = generator.normal(
        0,
        1,
        size=len(time),
    )

    contaminated = (
        baseline
        + 20 * np.sin(2 * np.pi * 60 * time)
    )

    config = SpectralConfig(
        line_frequency_hz=60.0,
        fmax_hz=100.0,
    )

    baseline_metrics = calculate_channel_metrics(
        baseline,
        sampling_rate,
        0.01,
        1.0,
        config,
    )

    contaminated_metrics = calculate_channel_metrics(
        contaminated,
        sampling_rate,
        0.01,
        1.0,
        config,
    )

    assert (
        contaminated_metrics["line_noise_ratio"]
        > baseline_metrics["line_noise_ratio"]
    )