"""Tests for deterministic fixed-window generation."""

from src.segmentation.windows import (
    generate_fixed_windows,
    seconds_to_samples,
)


def test_seconds_to_samples() -> None:
    assert seconds_to_samples(
        4.0,
        256.0,
    ) == 1024

    assert seconds_to_samples(
        2.0,
        256.0,
    ) == 512


def test_four_second_windows_with_half_overlap() -> None:
    windows = generate_fixed_windows(
        n_times=2560,
        sampling_rate_hz=256.0,
        window_duration_seconds=4.0,
        stride_seconds=2.0,
        drop_incomplete_final_window=True,
    )

    assert len(windows) == 4

    assert windows[0].start_sample == 0
    assert (
        windows[0].stop_sample_exclusive
        == 1024
    )

    assert windows[1].start_sample == 512
    assert (
        windows[1].stop_sample_exclusive
        == 1536
    )


def test_incomplete_final_window_is_dropped() -> None:
    windows = generate_fixed_windows(
        n_times=1200,
        sampling_rate_hz=256.0,
        window_duration_seconds=4.0,
        stride_seconds=2.0,
        drop_incomplete_final_window=True,
    )

    assert len(windows) == 1