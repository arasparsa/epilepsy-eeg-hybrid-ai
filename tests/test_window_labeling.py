"""Tests for seizure-window labeling."""

from src.segmentation.labeling import (
    label_window,
)


LABEL_IDS = {
    "non_ictal": 0,
    "boundary": 1,
    "ictal": 2,
}


def test_non_ictal_window() -> None:
    result = label_window(
        window_start_seconds=0.0,
        window_end_seconds=4.0,
        seizure_intervals=[
            (10.0, 20.0)
        ],
        ictal_overlap_threshold=0.5,
        near_seizure_margin_seconds=60.0,
        clean_non_ictal_minimum_distance_seconds=60.0,
        label_ids=LABEL_IDS,
    )

    assert result.label_name == "non_ictal"
    assert result.binary_label == 0
    assert result.seizure_overlap_fraction == 0
    assert result.near_seizure
    assert not result.clean_non_ictal


def test_boundary_window() -> None:
    result = label_window(
        window_start_seconds=0.0,
        window_end_seconds=4.0,
        seizure_intervals=[
            (3.0, 10.0)
        ],
        ictal_overlap_threshold=0.5,
        near_seizure_margin_seconds=60.0,
        clean_non_ictal_minimum_distance_seconds=60.0,
        label_ids=LABEL_IDS,
    )

    assert result.label_name == "boundary"
    assert result.binary_label is None
    assert result.seizure_overlap_seconds == 1.0
    assert result.seizure_overlap_fraction == 0.25


def test_exact_threshold_is_ictal() -> None:
    result = label_window(
        window_start_seconds=0.0,
        window_end_seconds=4.0,
        seizure_intervals=[
            (2.0, 10.0)
        ],
        ictal_overlap_threshold=0.5,
        near_seizure_margin_seconds=60.0,
        clean_non_ictal_minimum_distance_seconds=60.0,
        label_ids=LABEL_IDS,
    )

    assert result.label_name == "ictal"
    assert result.binary_label == 1
    assert result.seizure_overlap_fraction == 0.5


def test_full_ictal_window() -> None:
    result = label_window(
        window_start_seconds=10.0,
        window_end_seconds=14.0,
        seizure_intervals=[
            (5.0, 20.0)
        ],
        ictal_overlap_threshold=0.5,
        near_seizure_margin_seconds=60.0,
        clean_non_ictal_minimum_distance_seconds=60.0,
        label_ids=LABEL_IDS,
    )

    assert result.label_name == "ictal"
    assert result.seizure_overlap_fraction == 1.0