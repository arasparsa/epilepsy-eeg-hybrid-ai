"""Tests for segmentation interval operations."""

from src.segmentation.intervals import (
    distance_to_nearest_interval,
    merge_intervals,
    total_interval_overlap_seconds,
)


def test_merge_overlapping_intervals() -> None:
    merged = merge_intervals(
        [
            (10.0, 20.0),
            (15.0, 25.0),
            (30.0, 40.0),
        ]
    )

    assert merged == [
        (10.0, 25.0),
        (30.0, 40.0),
    ]


def test_total_overlap_is_not_double_counted() -> None:
    overlap = total_interval_overlap_seconds(
        window_start=0.0,
        window_end=20.0,
        intervals=[
            (5.0, 15.0),
            (10.0, 18.0),
        ],
    )

    assert overlap == 13.0


def test_distance_before_seizure() -> None:
    distance = distance_to_nearest_interval(
        window_start=0.0,
        window_end=4.0,
        intervals=[(10.0, 20.0)],
    )

    assert distance == 6.0


def test_distance_after_seizure() -> None:
    distance = distance_to_nearest_interval(
        window_start=30.0,
        window_end=34.0,
        intervals=[(10.0, 20.0)],
    )

    assert distance == 10.0


def test_overlapping_distance_is_zero() -> None:
    distance = distance_to_nearest_interval(
        window_start=18.0,
        window_end=22.0,
        intervals=[(10.0, 20.0)],
    )

    assert distance == 0.0