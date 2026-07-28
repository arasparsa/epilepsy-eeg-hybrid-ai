"""Interval utilities for signal-quality analysis."""

from __future__ import annotations

from collections.abc import Iterable


def interval_overlap_seconds(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """Return overlap duration between two half-open intervals."""
    if end_a <= start_a:
        raise ValueError("Interval A must have positive duration.")

    if end_b <= start_b:
        raise ValueError("Interval B must have positive duration.")

    return max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )


def maximum_seizure_overlap_fraction(
    chunk_start: float,
    chunk_end: float,
    seizure_intervals: Iterable[tuple[float, float]],
) -> float:
    """Return maximum chunk fraction overlapped by any seizure."""
    chunk_duration = chunk_end - chunk_start

    if chunk_duration <= 0:
        raise ValueError("Chunk duration must be positive.")

    maximum_overlap = 0.0

    for seizure_start, seizure_end in seizure_intervals:
        overlap = interval_overlap_seconds(
            chunk_start,
            chunk_end,
            seizure_start,
            seizure_end,
        )
        maximum_overlap = max(
            maximum_overlap,
            overlap / chunk_duration,
        )

    return maximum_overlap


def classify_chunk_annotation(
    chunk_start: float,
    chunk_end: float,
    seizure_intervals: Iterable[tuple[float, float]],
    ictal_threshold: float = 0.50,
) -> tuple[str, float]:
    """Classify a QC chunk as non-ictal, boundary, or ictal."""
    if not 0 < ictal_threshold <= 1:
        raise ValueError(
            "ictal_threshold must be in the interval (0, 1]."
        )

    overlap_fraction = maximum_seizure_overlap_fraction(
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        seizure_intervals=seizure_intervals,
    )

    if overlap_fraction == 0:
        label = "non_ictal"
    elif overlap_fraction >= ictal_threshold:
        label = "ictal"
    else:
        label = "boundary"

    return label, overlap_fraction