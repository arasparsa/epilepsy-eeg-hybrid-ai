"""Interval operations for EEG seizure-window labeling."""

from __future__ import annotations

from collections.abc import Iterable


Interval = tuple[float, float]


def validate_interval(
    start: float,
    end: float,
) -> None:
    """Validate one half-open interval [start, end)."""
    if start < 0:
        raise ValueError(
            f"Interval start cannot be negative: {start}"
        )

    if end <= start:
        raise ValueError(
            f"Interval must have positive duration: "
            f"{start} -> {end}"
        )


def interval_overlap_seconds(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    """Return overlap between two half-open intervals."""
    validate_interval(first_start, first_end)
    validate_interval(second_start, second_end)

    return max(
        0.0,
        min(first_end, second_end)
        - max(first_start, second_start),
    )


def merge_intervals(
    intervals: Iterable[Interval],
) -> list[Interval]:
    """Merge overlapping or touching intervals."""
    sorted_intervals = sorted(
        (
            (float(start), float(end))
            for start, end in intervals
        ),
        key=lambda item: (item[0], item[1]),
    )

    if not sorted_intervals:
        return []

    for start, end in sorted_intervals:
        validate_interval(start, end)

    merged: list[list[float]] = [
        [
            sorted_intervals[0][0],
            sorted_intervals[0][1],
        ]
    ]

    for start, end in sorted_intervals[1:]:
        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [
        (float(start), float(end))
        for start, end in merged
    ]


def total_interval_overlap_seconds(
    window_start: float,
    window_end: float,
    intervals: Iterable[Interval],
) -> float:
    """Return total non-duplicated overlap with intervals."""
    validate_interval(window_start, window_end)

    merged_intervals = merge_intervals(intervals)

    return float(
        sum(
            interval_overlap_seconds(
                window_start,
                window_end,
                interval_start,
                interval_end,
            )
            for interval_start, interval_end
            in merged_intervals
        )
    )


def distance_between_intervals(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    """Return zero for overlap; otherwise interval gap."""
    validate_interval(first_start, first_end)
    validate_interval(second_start, second_end)

    if interval_overlap_seconds(
        first_start,
        first_end,
        second_start,
        second_end,
    ) > 0:
        return 0.0

    if first_end <= second_start:
        return float(second_start - first_end)

    return float(first_start - second_end)


def distance_to_nearest_interval(
    window_start: float,
    window_end: float,
    intervals: Iterable[Interval],
) -> float | None:
    """Return distance to nearest seizure interval."""
    interval_list = list(intervals)

    if not interval_list:
        return None

    return min(
        distance_between_intervals(
            window_start,
            window_end,
            interval_start,
            interval_end,
        )
        for interval_start, interval_end
        in interval_list
    )