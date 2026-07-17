"""Tests for CHB-MIT channel-audit metadata."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "metadata"


def load_table(filename: str) -> pd.DataFrame:
    path = METADATA_DIR / filename
    assert path.exists(), f"Missing metadata file: {path}"
    return pd.read_csv(path)


def test_file_inventory_is_not_empty() -> None:
    inventory = load_table(
        "chbmit_file_inventory.csv"
    )
    assert not inventory.empty


def test_edf_paths_are_unique() -> None:
    inventory = load_table(
        "chbmit_file_inventory.csv"
    )
    assert inventory["relative_path"].is_unique


def test_recording_durations_are_positive() -> None:
    inventory = load_table(
        "chbmit_file_inventory.csv"
    )
    assert (inventory["duration_seconds"] > 0).all()


def test_sampling_rates_are_positive() -> None:
    inventory = load_table(
        "chbmit_file_inventory.csv"
    )
    assert (inventory["sampling_rate_hz"] > 0).all()


def test_channel_counts_match_long_table() -> None:
    inventory = load_table(
        "chbmit_file_inventory.csv"
    )
    channel_long = load_table(
        "chbmit_channel_long.csv"
    )

    expected = int(inventory["n_channels"].sum())
    observed = len(channel_long)

    assert observed == expected


def test_presence_matrix_has_one_row_per_file() -> None:
    inventory = load_table(
        "chbmit_file_inventory.csv"
    )
    presence = load_table(
        "chbmit_channel_presence_matrix.csv"
    )

    assert len(presence) == len(inventory)


def test_coverage_fractions_are_valid() -> None:
    counts = load_table(
        "chbmit_channel_counts.csv"
    )

    assert counts["file_coverage_fraction"].between(
        0,
        1,
    ).all()

    assert counts["patient_coverage_fraction"].between(
        0,
        1,
    ).all()


def test_target_complete_count_is_not_impossible() -> None:
    summary = load_table(
        "chbmit_target_channel_summary.csv"
    )

    assert (
        summary["complete_file_count"]
        <= summary["total_file_count"]
    ).all()

    assert (
        summary["covered_patient_count"]
        <= summary["total_patient_count"]
    ).all()