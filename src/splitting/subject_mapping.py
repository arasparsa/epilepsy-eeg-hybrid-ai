"""Resolve CHB-MIT case identifiers to independent subjects."""

from __future__ import annotations

import re

import pandas as pd


REQUIRED_MAPPING_COLUMNS = {
    "case_id",
    "subject_id",
    "mapping_status",
    "mapping_source",
    "notes",
}


def normalize_case_id(case_id: str) -> str:
    """Normalize a CHB-MIT case identifier."""
    normalized = str(case_id).strip().lower()

    if not re.fullmatch(r"chb\d{2}", normalized):
        raise ValueError(
            f"Invalid CHB-MIT case identifier: {case_id!r}"
        )

    return normalized


def default_subject_id(case_id: str) -> str:
    """Generate a deterministic subject ID for one case."""
    normalized = normalize_case_id(case_id)
    return f"subject_{normalized}"


def resolve_subject_mapping(
    observed_cases: list[str],
    verified_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve every observed case to exactly one subject."""
    missing_columns = (
        REQUIRED_MAPPING_COLUMNS
        - set(verified_mapping.columns)
    )

    if missing_columns:
        raise ValueError(
            "Mapping file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    mapping = verified_mapping.copy()

    mapping["case_id"] = mapping[
        "case_id"
    ].map(normalize_case_id)

    if mapping["case_id"].duplicated().any():
        duplicated = mapping.loc[
            mapping["case_id"].duplicated(
                keep=False
            ),
            "case_id",
        ].tolist()

        raise ValueError(
            "Duplicate case rows in verified mapping: "
            f"{duplicated}"
        )

    mapping_lookup = (
        mapping.set_index("case_id")
        .to_dict(orient="index")
    )

    resolved_rows: list[dict] = []

    for case_id in sorted(
        {
            normalize_case_id(case)
            for case in observed_cases
        }
    ):
        if case_id in mapping_lookup:
            source = mapping_lookup[case_id]

            resolved_rows.append(
                {
                    "case_id": case_id,
                    "subject_id": str(
                        source["subject_id"]
                    ),
                    "mapping_status": str(
                        source["mapping_status"]
                    ),
                    "mapping_source": str(
                        source["mapping_source"]
                    ),
                    "notes": str(
                        source.get("notes", "")
                    ),
                }
            )

        else:
            resolved_rows.append(
                {
                    "case_id": case_id,
                    "subject_id": (
                        default_subject_id(case_id)
                    ),
                    "mapping_status": (
                        "case_treated_as_independent"
                    ),
                    "mapping_source": (
                        "CHB-MIT case grouping"
                    ),
                    "notes": (
                        "No documented cross-case identity "
                        "relationship was applied."
                    ),
                }
            )

    resolved = pd.DataFrame(resolved_rows)

    if resolved["case_id"].duplicated().any():
        raise RuntimeError(
            "Resolved mapping contains duplicate cases."
        )

    observed_set = {
        normalize_case_id(case)
        for case in observed_cases
    }

    resolved_set = set(resolved["case_id"])

    if observed_set != resolved_set:
        raise RuntimeError(
            "Resolved mapping does not cover all observed cases."
        )

    return resolved


def apply_subject_mapping(
    table: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Attach resolved subject IDs to a case-level table."""
    if "case_id" not in table.columns:
        raise ValueError(
            "Input table must contain case_id."
        )

    output = table.copy()

    # Remove any placeholder subject_id from earlier phases.
    output = output.drop(
        columns=["subject_id"],
        errors="ignore",
    )

    output["case_id"] = output[
        "case_id"
    ].map(normalize_case_id)

    output = output.merge(
        mapping[
            [
                "case_id",
                "subject_id",
            ]
        ],
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    if output["subject_id"].isna().any():
        missing_cases = sorted(
            output.loc[
                output["subject_id"].isna(),
                "case_id",
            ].unique()
        )

        raise ValueError(
            "Cases without resolved subject IDs: "
            f"{missing_cases}"
        )

    return output