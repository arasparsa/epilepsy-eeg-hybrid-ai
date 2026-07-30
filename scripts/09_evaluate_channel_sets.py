"""Evaluate candidate CHB-MIT channel sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.channels.coverage import (
    build_recording_channel_sets,
    evaluate_channel_sets,
)
from src.channels.naming import (
    canonicalize_channel_names,
    load_validated_aliases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "chbmit_channel_harmonization.yaml"
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_candidate_sets(
    config: dict,
) -> dict[str, list[str]]:
    """Extract channel lists from harmonization config."""
    candidate_sets = {}

    for set_name, set_config in config[
        "channel_sets"
    ].items():
        candidate_sets[set_name] = [
            str(channel).upper()
            for channel in set_config["channels"]
        ]

    return candidate_sets


def standardize_channel_long(
    channel_long: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize phase-2 channel metadata columns."""
    output = channel_long.copy()

    if "patient_id" in output.columns:
        output = output.rename(
            columns={"patient_id": "case_id"}
        )

    required = {
        "case_id",
        "recording_id",
        "original_channel_name",
    }

    missing = required.difference(output.columns)

    if missing:
        raise ValueError(
            "Channel-long table is missing columns: "
            f"{sorted(missing)}"
        )

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    config = load_config(args.config)

    inputs = config["inputs"]
    output_dir = (
        PROJECT_ROOT / config["outputs"]["directory"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = pd.read_csv(
        PROJECT_ROOT / inputs["recordings_file"]
    )
    seizures = pd.read_csv(
        PROJECT_ROOT / inputs["seizures_file"]
    )
    channel_long = pd.read_csv(
        PROJECT_ROOT / inputs["channel_long_file"]
    )
    alias_table = pd.read_csv(
        PROJECT_ROOT / inputs["alias_file"]
    )

    channel_long = standardize_channel_long(
        channel_long
    )

    alias_rules = load_validated_aliases(
        alias_table
    )

    canonical_rows: list[dict] = []
    alias_audit_rows: list[dict] = []
    failure_rows: list[dict] = []

    for (
        case_id,
        recording_id,
    ), group in channel_long.groupby(
        ["case_id", "recording_id"],
        sort=True,
    ):
        ordered = group.sort_values("channel_index")

        try:
            canonical_names, audit_rows = (
                canonicalize_channel_names(
                    original_names=ordered[
                        "original_channel_name"
                    ].astype(str).tolist(),
                    alias_rules=alias_rules,
                    recording_id=recording_id,
                )
            )

            for source_row, canonical_name in zip(
                ordered.itertuples(index=False),
                canonical_names,
                strict=True,
            ):
                canonical_rows.append(
                    {
                        "case_id": case_id,
                        "recording_id": recording_id,
                        "channel_index": (
                            source_row.channel_index
                        ),
                        "original_channel_name": (
                            source_row.original_channel_name
                        ),
                        "canonical_channel_name": (
                            canonical_name
                        ),
                    }
                )

            alias_audit_rows.extend(audit_rows)

        except Exception as exc:
            failure_rows.append(
                {
                    "case_id": case_id,
                    "recording_id": recording_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    canonical_long = pd.DataFrame(canonical_rows)

    if canonical_long.empty:
        print(
            "ERROR: No canonical channel metadata produced.",
            file=sys.stderr,
        )
        return 1

    recording_channel_sets = (
        build_recording_channel_sets(
            canonical_long
        )
    )

    candidate_sets = extract_candidate_sets(config)

    coverage, summary = evaluate_channel_sets(
        recording_channels=recording_channel_sets,
        recordings=recordings,
        seizures=seizures,
        candidate_sets=candidate_sets,
    )

    canonical_long.to_csv(
        output_dir
        / "chbmit_canonical_channel_long.csv",
        index=False,
    )

    coverage.to_csv(
        output_dir
        / "chbmit_channel_set_recording_coverage.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "chbmit_candidate_channel_sets.csv",
        index=False,
    )

    pd.DataFrame(alias_audit_rows).to_csv(
        output_dir
        / "chbmit_channel_alias_validation.csv",
        index=False,
    )

    failures = pd.DataFrame(failure_rows)

    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "case_id",
                "recording_id",
                "error_type",
                "error_message",
            ]
        )

    failures.to_csv(
        output_dir
        / "chbmit_harmonization_failures.csv",
        index=False,
    )

    print("\nCandidate channel-set evaluation:")
    print(summary.to_string(index=False))

    print(
        "\nCanonicalization failures:",
        len(failures),
    )

    return 0 if failures.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())