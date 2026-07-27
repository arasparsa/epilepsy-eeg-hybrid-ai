"""Unit tests for CHB-MIT summary parsing."""

from pathlib import Path

import pytest

from src.metadata.chbmit_summary_parser import (
    calculate_clock_duration_seconds,
    parse_clock_time,
    parse_summary_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"


def test_parse_indexed_seizure_summary() -> None:
    parsed = parse_summary_file(
        summary_path=(
            FIXTURE_DIR
            / "chb01_summary_excerpt.txt"
        ),
        raw_root=FIXTURE_DIR,
    )

    assert parsed.case_id == "chb01"
    assert len(parsed.recordings) == 2

    first = parsed.recordings[0]

    assert first.edf_filename == "chb01_03.edf"
    assert first.n_seizures_reported == 1
    assert first.n_seizures_parsed == 1

    seizure = first.seizures[0]

    assert seizure.seizure_index == 1
    assert seizure.onset_seconds == 2996
    assert seizure.offset_seconds == 3036
    assert seizure.duration_seconds == 40


def test_parse_unindexed_seizure_summary() -> None:
    parsed = parse_summary_file(
        summary_path=(
            FIXTURE_DIR
            / "chb03_summary_excerpt.txt"
        ),
        raw_root=FIXTURE_DIR,
    )

    recording = parsed.recordings[0]

    assert recording.n_seizures_parsed == 1
    assert recording.seizures[0].seizure_index == 1
    assert recording.seizures[0].onset_seconds == 362
    assert recording.seizures[0].offset_seconds == 414


def test_parse_extended_hour_clock() -> None:
    assert parse_clock_time("26:52:35") == (
        26 * 3600 + 52 * 60 + 35
    )


def test_calculate_standard_duration() -> None:
    duration, crosses_midnight = (
        calculate_clock_duration_seconds(
            "13:00:00",
            "14:00:00",
        )
    )

    assert duration == 3600
    assert crosses_midnight is False


def test_calculate_midnight_crossing() -> None:
    duration, crosses_midnight = (
        calculate_clock_duration_seconds(
            "23:30:00",
            "00:30:00",
        )
    )

    assert duration == 3600
    assert crosses_midnight is True


def test_calculate_extended_hour_duration() -> None:
    duration, crosses_midnight = (
        calculate_clock_duration_seconds(
            "22:52:35",
            "26:52:35",
        )
    )

    assert duration == 4 * 3600
    assert crosses_midnight is False