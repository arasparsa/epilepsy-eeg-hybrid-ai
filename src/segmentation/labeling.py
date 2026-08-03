"""Label fixed-length EEG windows using seizure intervals."""

from __future__ import annotations

from dataclasses import dataclass

from src.segmentation.intervals import (
    Interval,
    distance_to_nearest_interval,
    merge_intervals,
    total_interval_overlap_seconds,
)


@dataclass(frozen=True)
class WindowLabel:
    """Labeling result for one EEG window."""

    label_name: str
    label_id: int
    binary_label: int | None
    seizure_overlap_seconds: float
    seizure_overlap_fraction: float
    overlaps_seizure: bool
    distance_to_nearest_seizure_seconds: float | None
    near_seizure: bool
    clean_non_ictal: bool


def label_window(
    *,
    window_start_seconds: float,
    window_end_seconds: float,
    seizure_intervals: list[Interval],
    ictal_overlap_threshold: float,
    near_seizure_margin_seconds: float,
    clean_non_ictal_minimum_distance_seconds: float,
    label_ids: dict[str, int],
) -> WindowLabel:
    """Assign non-ictal, boundary, or ictal label."""
    window_duration = (
        window_end_seconds - window_start_seconds
    )

    if window_duration <= 0:
        raise ValueError(
            "Window duration must be positive."
        )

    if not 0 < ictal_overlap_threshold <= 1:
        raise ValueError(
            "ictal_overlap_threshold must be in (0, 1]."
        )

    merged_seizures = merge_intervals(
        seizure_intervals
    )

    overlap_seconds = (
        total_interval_overlap_seconds(
            window_start=window_start_seconds,
            window_end=window_end_seconds,
            intervals=merged_seizures,
        )
    )

    overlap_fraction = (
        overlap_seconds / window_duration
    )

    distance = distance_to_nearest_interval(
        window_start=window_start_seconds,
        window_end=window_end_seconds,
        intervals=merged_seizures,
    )

    overlaps_seizure = overlap_seconds > 0

    if overlap_seconds == 0:
        label_name = "non_ictal"
        binary_label: int | None = 0

    elif overlap_fraction >= ictal_overlap_threshold:
        label_name = "ictal"
        binary_label = 1

    else:
        label_name = "boundary"
        binary_label = None

    near_seizure = bool(
        distance is not None
        and distance <= near_seizure_margin_seconds
    )

    clean_non_ictal = bool(
        label_name == "non_ictal"
        and (
            distance is None
            or distance
            >= clean_non_ictal_minimum_distance_seconds
        )
    )

    return WindowLabel(
        label_name=label_name,
        label_id=int(label_ids[label_name]),
        binary_label=binary_label,
        seizure_overlap_seconds=float(
            overlap_seconds
        ),
        seizure_overlap_fraction=float(
            overlap_fraction
        ),
        overlaps_seizure=overlaps_seizure,
        distance_to_nearest_seizure_seconds=(
            float(distance)
            if distance is not None
            else None
        ),
        near_seizure=near_seizure,
        clean_non_ictal=clean_non_ictal,
    )