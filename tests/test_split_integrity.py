"""Tests for common EEG split-leakage conditions."""

import pandas as pd


def test_all_windows_of_subject_share_fold() -> None:
    windows = pd.DataFrame(
        {
            "subject_id": [
                "a",
                "a",
                "b",
                "b",
            ],
            "outer_test_fold": [
                0,
                0,
                1,
                1,
            ],
        }
    )

    counts = (
        windows.groupby("subject_id")[
            "outer_test_fold"
        ].nunique()
    )

    assert counts.eq(1).all()


def test_overlapping_windows_not_split() -> None:
    windows = pd.DataFrame(
        {
            "recording_id": [
                "rec1",
                "rec1",
                "rec2",
            ],
            "outer_test_fold": [
                0,
                0,
                1,
            ],
        }
    )

    fold_counts = (
        windows.groupby("recording_id")[
            "outer_test_fold"
        ].nunique()
    )

    assert fold_counts.eq(1).all()