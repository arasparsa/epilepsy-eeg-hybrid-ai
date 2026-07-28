"""Tests for seizure-QC interval relationships."""

from src.quality.interval_utils import (
    classify_chunk_annotation,
    interval_overlap_seconds,
)


def test_no_overlap() -> None:
    assert interval_overlap_seconds(
        0,
        60,
        100,
        120,
    ) == 0


def test_partial_overlap() -> None:
    assert interval_overlap_seconds(
        0,
        60,
        50,
        70,
    ) == 10


def test_boundary_chunk() -> None:
    label, fraction = classify_chunk_annotation(
        chunk_start=0,
        chunk_end=60,
        seizure_intervals=[(50, 70)],
        ictal_threshold=0.50,
    )

    assert label == "boundary"
    assert fraction == 10 / 60


def test_ictal_chunk() -> None:
    label, fraction = classify_chunk_annotation(
        chunk_start=0,
        chunk_end=60,
        seizure_intervals=[(20, 60)],
        ictal_threshold=0.50,
    )

    assert label == "ictal"
    assert fraction == 40 / 60


def test_non_ictal_chunk() -> None:
    label, fraction = classify_chunk_annotation(
        chunk_start=0,
        chunk_end=60,
        seizure_intervals=[],
        ictal_threshold=0.50,
    )

    assert label == "non_ictal"
    assert fraction == 0