"""Channel-name normalization and validated alias handling."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


DASH_TRANSLATION = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


def normalize_channel_name(name: str) -> str:
    """Normalize superficial label differences only."""
    normalized = str(name).strip().upper()
    normalized = normalized.translate(DASH_TRANSLATION)

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"\s*-\s*",
        "-",
        normalized,
    )
    normalized = re.sub(
        r"^EEG\s+",
        "",
        normalized,
    )
    normalized = re.sub(
        r"-(REF|LE)$",
        "",
        normalized,
    )

    return normalized


@dataclass(frozen=True)
class AliasRule:
    """One validated channel alias rule."""

    original_name: str
    normalized_name: str
    canonical_name: str
    alias_type: str
    validation_status: str
    applicable_scope: str
    recording_id: str | None = None


def load_validated_aliases(
    alias_table: pd.DataFrame,
) -> list[AliasRule]:
    """Load only validated alias rules."""
    valid_statuses = {
        "verified_global",
        "verified_recording_specific",
    }

    validated = alias_table.loc[
        alias_table["validation_status"].isin(
            valid_statuses
        )
    ].copy()

    rules: list[AliasRule] = []

    for row in validated.itertuples(index=False):
        recording_id = getattr(
            row,
            "recording_id",
            None,
        )

        if pd.isna(recording_id):
            recording_id = None

        rules.append(
            AliasRule(
                original_name=str(row.original_name),
                normalized_name=str(row.normalized_name),
                canonical_name=str(row.canonical_name),
                alias_type=str(row.alias_type),
                validation_status=str(
                    row.validation_status
                ),
                applicable_scope=str(
                    row.applicable_scope
                ),
                recording_id=recording_id,
            )
        )

    return rules


def canonicalize_channel_names(
    original_names: list[str],
    alias_rules: list[AliasRule],
    recording_id: str,
) -> tuple[list[str], list[dict]]:
    """Apply normalization and explicitly validated aliases."""
    canonical_names: list[str] = []
    audit_rows: list[dict] = []

    for original_name in original_names:
        normalized_name = normalize_channel_name(
            original_name
        )
        canonical_name = normalized_name
        applied_rule: AliasRule | None = None

        for rule in alias_rules:
            normalized_rule_name = normalize_channel_name(
                rule.normalized_name
            )

            if normalized_name != normalized_rule_name:
                continue

            if (
                rule.validation_status
                == "verified_recording_specific"
                and rule.recording_id != recording_id
            ):
                continue

            canonical_name = normalize_channel_name(
                rule.canonical_name
            )
            applied_rule = rule
            break

        canonical_names.append(canonical_name)

        audit_rows.append(
            {
                "recording_id": recording_id,
                "original_channel_name": original_name,
                "normalized_channel_name": normalized_name,
                "canonical_channel_name": canonical_name,
                "alias_applied": applied_rule is not None,
                "alias_validation_status": (
                    applied_rule.validation_status
                    if applied_rule
                    else "not_applied"
                ),
            }
        )

    duplicates = sorted(
        {
            name
            for name in canonical_names
            if canonical_names.count(name) > 1
        }
    )

    if duplicates:
        raise ValueError(
            "Canonicalization created duplicate channel names "
            f"for {recording_id}: {duplicates}"
        )

    return canonical_names, audit_rows