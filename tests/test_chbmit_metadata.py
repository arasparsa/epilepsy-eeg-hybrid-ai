"""Integrity tests for final CHB-MIT metadata."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "metadata"


def load_csv(filename: str) -> pd.DataFrame:
    path = METADATA_DIR / filename

    assert path.exists(), f"Missing file: {path}"

    return pd.read_csv(path)


def test_recording_table_is_not_empty() -> None:
    recordings = load_csv(
        "chbmit_recordings.csv"
    )

    assert not recordings.empty


def test_recording_keys_are_unique() -> None:
    recordings = load_csv(
        "chbmit_recordings.csv"
    )

    assert not recordings[
        ["case_id", "recording_id"]
    ].duplicated().any()


def test_all_recordings_match_edf_and_summary() -> None:
    recordings = load_csv(
        "chbmit_recordings.csv"
    )

    assert recordings[
        "metadata_status"
    ].eq("matched").all()


def test_reported_and_parsed_counts_match() -> None:
    recordings = load_csv(
        "chbmit_recordings.csv"
    )

    assert recordings[
        "reported_parsed_seizure_count_match"
    ].all()


def test_seizure_ids_are_unique() -> None:
    seizures = load_csv(
        "chbmit_seizures.csv"
    )

    assert seizures["seizure_id"].is_unique


def test_all_seizures_have_positive_duration() -> None:
    seizures = load_csv(
        "chbmit_seizures.csv"
    )

    assert (
        seizures["duration_seconds"] > 0
    ).all()


def test_all_seizures_are_inside_edf() -> None:
    seizures = load_csv(
        "chbmit_seizures.csv"
    )

    assert seizures[
        "interval_within_edf"
    ].all()

    assert seizures[
        "sample_interval_within_edf"
    ].all()


def test_seizure_counts_reconcile() -> None:
    recordings = load_csv(
        "chbmit_recordings.csv"
    )

    seizures = load_csv(
        "chbmit_seizures.csv"
    )

    calculated = (
        seizures.groupby(
            ["case_id", "recording_id"]
        )
        .size()
        .rename("calculated_count")
        .reset_index()
    )

    merged = recordings.merge(
        calculated,
        on=["case_id", "recording_id"],
        how="left",
    )

    merged["calculated_count"] = (
        merged["calculated_count"]
        .fillna(0)
        .astype(int)
    )

    assert (
        merged["calculated_count"]
        == merged["n_seizures_parsed"]
    ).all()


def test_ictal_fraction_is_valid() -> None:
    recordings = load_csv(
        "chbmit_recordings.csv"
    )

    assert recordings["ictal_fraction"].between(
        0,
        1,
    ).all()