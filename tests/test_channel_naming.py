"""Tests for channel naming and aliases."""

import pytest

from src.channels.naming import (
    AliasRule,
    canonicalize_channel_names,
    normalize_channel_name,
)


def test_superficial_normalization() -> None:
    assert (
        normalize_channel_name(
            " EEG FP1 - F7-REF "
        )
        == "FP1-F7"
    )


def test_dash_normalization() -> None:
    assert (
        normalize_channel_name("FP1–F7")
        == "FP1-F7"
    )


def test_unverified_alias_is_not_applied() -> None:
    names, _ = canonicalize_channel_names(
        original_names=["T8-P8-0"],
        alias_rules=[],
        recording_id="chb01_01",
    )

    assert names == ["T8-P8-0"]


def test_verified_global_alias_is_applied() -> None:
    rule = AliasRule(
        original_name="T8-P8-0",
        normalized_name="T8-P8-0",
        canonical_name="T8-P8",
        alias_type="suffix_duplicate",
        validation_status="verified_global",
        applicable_scope="global",
    )

    names, _ = canonicalize_channel_names(
        original_names=["T8-P8-0"],
        alias_rules=[rule],
        recording_id="chb01_01",
    )

    assert names == ["T8-P8"]


def test_alias_cannot_create_duplicate_names() -> None:
    rule = AliasRule(
        original_name="T8-P8-0",
        normalized_name="T8-P8-0",
        canonical_name="T8-P8",
        alias_type="suffix_duplicate",
        validation_status="verified_global",
        applicable_scope="global",
    )

    with pytest.raises(ValueError):
        canonicalize_channel_names(
            original_names=[
                "T8-P8",
                "T8-P8-0",
            ],
            alias_rules=[rule],
            recording_id="example",
        )