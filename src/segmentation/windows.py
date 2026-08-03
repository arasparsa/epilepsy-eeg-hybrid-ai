"""Deterministic fixed-length window generation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


@dataclass(frozen=True)
class WindowDefinition:
    """Sample-accurate definition of one EEG window."""

    window_index: int
    start_sample: int
    stop_sample_exclusive: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float


def seconds_to_samples(
    seconds: float,
    sampling_rate_hz: float,
) -> int:
    """Convert seconds to samples deterministically."""
    if seconds <= 0:
        raise ValueError(
            "Seconds must be positive."
        )

    if sampling_rate_hz <= 0:
        raise ValueError(
            "Sampling rate must be positive."
        )

    value = (
        Decimal(str(seconds))
        * Decimal(str(sampling_rate_hz))
    )

    return int(
        value.to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def generate_fixed_windows(
    *,
    n_times: int,
    sampling_rate_hz: float,
    window_duration_seconds: float,
    stride_seconds: float,
    drop_incomplete_final_window: bool = True,
) -> list[WindowDefinition]:
    """Generate sample-aligned windows within one recording."""
    if n_times <= 0:
        raise ValueError(
            "n_times must be positive."
        )

    window_samples = seconds_to_samples(
        window_duration_seconds,
        sampling_rate_hz,
    )

    stride_samples = seconds_to_samples(
        stride_seconds,
        sampling_rate_hz,
    )

    if stride_samples > window_samples:
        raise ValueError(
            "Stride cannot exceed window length "
            "in segmentation_v1."
        )

    windows: list[WindowDefinition] = []
    window_index = 0
    start_sample = 0

    while start_sample < n_times:
        stop_sample = start_sample + window_samples

        if stop_sample > n_times:
            if drop_incomplete_final_window:
                break

            stop_sample = n_times

        sample_count = stop_sample - start_sample

        if sample_count <= 0:
            break

        start_seconds = (
            start_sample / sampling_rate_hz
        )
        end_seconds = (
            stop_sample / sampling_rate_hz
        )

        windows.append(
            WindowDefinition(
                window_index=window_index,
                start_sample=start_sample,
                stop_sample_exclusive=stop_sample,
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds),
                duration_seconds=float(
                    end_seconds - start_seconds
                ),
            )
        )

        window_index += 1
        start_sample += stride_samples

    return windows