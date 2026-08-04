"""Tests for CHB-MIT case-to-subject mapping."""

import pandas as pd
import pytest

from src.splitting.subject_mapping import (
    apply_subject_mapping,
    resolve_subject_mapping,
)


def test_chb01_and_chb21_share_subject() -> None:
    observed = [
        "chb01",
        "chb02",
        "chb21",
    ]

    verified = pd.DataFrame(
        {
            "case_id": [
                "chb01",
                "chb21",
            ],
            "subject_id": [
                "subject_shared",
                "subject_shared",
            ],
            "mapping_status": [
                "verified",
                "verified",
            ],
            "mapping_source": [
                "official",
                "official",
            ],
            "notes": [
                "",
                "",
            ],
        }
    )

    resolved = resolve_subject_mapping(
        observed_cases=observed,
        verified_mapping=verified,
    )

    pair = resolved.loc[
        resolved["case_id"].isin(
            ["chb01", "chb21"]
        )
    ]

    assert pair["subject_id"].nunique() == 1


def test_unmapped_case_gets_independent_id() -> None:
    verified = pd.DataFrame(
        columns=[
            "case_id",
            "subject_id",
            "mapping_status",
            "mapping_source",
            "notes",
        ]
    )

    resolved = resolve_subject_mapping(
        observed_cases=["chb05"],
        verified_mapping=verified,
    )

    assert (
        resolved.iloc[0]["subject_id"]
        == "subject_chb05"
    )


def test_duplicate_case_mapping_raises() -> None:
    verified = pd.DataFrame(
        {
            "case_id": [
                "chb01",
                "chb01",
            ],
            "subject_id": [
                "subject_a",
                "subject_b",
            ],
            "mapping_status": [
                "verified",
                "verified",
            ],
            "mapping_source": [
                "source",
                "source",
            ],
            "notes": ["", ""],
        }
    )

    with pytest.raises(ValueError):
        resolve_subject_mapping(
            observed_cases=["chb01"],
            verified_mapping=verified,
        )