"""Evaluate channel-set coverage and data retention."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def build_recording_channel_sets(
    channel_long: pd.DataFrame,
) -> pd.DataFrame:
    """Create one canonical channel set per recording."""
    required_columns = {
        "case_id",
        "recording_id",
        "canonical_channel_name",
    }

    missing = required_columns.difference(
        channel_long.columns
    )

    if missing:
        raise ValueError(
            f"Missing channel columns: {sorted(missing)}"
        )

    return (
        channel_long[
            [
                "case_id",
                "recording_id",
                "canonical_channel_name",
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["case_id", "recording_id"]
        )["canonical_channel_name"]
        .apply(set)
        .reset_index(name="available_channels")
    )


def evaluate_channel_sets(
    recording_channels: pd.DataFrame,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
    candidate_sets: Mapping[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate recording and dataset retention per channel set."""
    recording_rows: list[dict] = []

    for set_name, channel_names in candidate_sets.items():
        target = set(channel_names)

        for row in recording_channels.itertuples(
            index=False
        ):
            available = set(row.available_channels)
            missing = sorted(target - available)
            extra = sorted(available - target)

            recording_rows.append(
                {
                    "channel_set_name": set_name,
                    "case_id": row.case_id,
                    "recording_id": row.recording_id,
                    "target_channel_count": len(target),
                    "available_target_channel_count": (
                        len(target) - len(missing)
                    ),
                    "has_complete_channel_set": (
                        len(missing) == 0
                    ),
                    "missing_channel_count": len(missing),
                    "missing_channels": "|".join(missing),
                    "extra_channel_count": len(extra),
                    "extra_channels": "|".join(extra),
                }
            )

    coverage = pd.DataFrame(recording_rows)

    recording_metadata = recordings[
        [
            "case_id",
            "recording_id",
            "duration_seconds_edf",
            "has_seizure",
            "n_seizures_parsed",
            "total_ictal_seconds",
            "metadata_status",
        ]
    ].copy()

    coverage = coverage.merge(
        recording_metadata,
        on=["case_id", "recording_id"],
        how="left",
        validate="many_to_one",
    )

    seizure_counts = (
        seizures.groupby(
            ["case_id", "recording_id"]
        )
        .agg(
            seizure_table_count=(
                "seizure_id",
                "count",
            ),
            seizure_table_ictal_seconds=(
                "duration_seconds",
                "sum",
            ),
        )
        .reset_index()
    )

    coverage = coverage.merge(
        seizure_counts,
        on=["case_id", "recording_id"],
        how="left",
        validate="many_to_one",
    )

    coverage[
        [
            "seizure_table_count",
            "seizure_table_ictal_seconds",
        ]
    ] = coverage[
        [
            "seizure_table_count",
            "seizure_table_ictal_seconds",
        ]
    ].fillna(0)

    summary_rows: list[dict] = []

    for set_name, group in coverage.groupby(
        "channel_set_name"
    ):
        retained = group.loc[
            group["has_complete_channel_set"]
        ]

        total_seizures = group[
            "seizure_table_count"
        ].sum()

        retained_seizures = retained[
            "seizure_table_count"
        ].sum()

        total_ictal_seconds = group[
            "seizure_table_ictal_seconds"
        ].sum()

        retained_ictal_seconds = retained[
            "seizure_table_ictal_seconds"
        ].sum()

        total_duration = group[
            "duration_seconds_edf"
        ].sum()

        retained_duration = retained[
            "duration_seconds_edf"
        ].sum()

        summary_rows.append(
            {
                "channel_set_name": set_name,
                "target_channel_count": int(
                    group[
                        "target_channel_count"
                    ].iloc[0]
                ),
                "total_recordings": len(group),
                "retained_recordings": len(retained),
                "recording_retention_fraction": (
                    len(retained) / len(group)
                    if len(group)
                    else 0
                ),
                "total_cases": (
                    group["case_id"].nunique()
                ),
                "retained_cases_with_any_file": (
                    retained["case_id"].nunique()
                ),
                "total_recording_hours": (
                    total_duration / 3600
                ),
                "retained_recording_hours": (
                    retained_duration / 3600
                ),
                "recording_hour_retention_fraction": (
                    retained_duration / total_duration
                    if total_duration > 0
                    else 0
                ),
                "total_seizure_recordings": int(
                    group["has_seizure"].sum()
                ),
                "retained_seizure_recordings": int(
                    retained["has_seizure"].sum()
                ),
                "total_seizures": int(total_seizures),
                "retained_seizures": int(
                    retained_seizures
                ),
                "seizure_retention_fraction": (
                    retained_seizures / total_seizures
                    if total_seizures > 0
                    else 0
                ),
                "total_ictal_seconds": float(
                    total_ictal_seconds
                ),
                "retained_ictal_seconds": float(
                    retained_ictal_seconds
                ),
                "ictal_time_retention_fraction": (
                    retained_ictal_seconds
                    / total_ictal_seconds
                    if total_ictal_seconds > 0
                    else 0
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return coverage, summary