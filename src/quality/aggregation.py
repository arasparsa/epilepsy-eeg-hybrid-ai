"""Aggregation and provisional QC flags."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_provisional_chunk_flags(
    metrics: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """Add transparent provisional review flags."""
    output = metrics.copy()

    output["flag_nonfinite"] = (
        output["nonfinite_fraction"]
        > thresholds["nonfinite_fraction_error"]
    )

    output["flag_flatline"] = (
        output["flatline_sample_fraction"]
        >= thresholds["flatline_fraction_review"]
    )

    output["flag_zero_heavy"] = (
        output["zero_fraction"]
        >= thresholds["zero_fraction_review"]
    )

    output["flag_near_constant"] = (
        output["std_uv"]
        <= thresholds["constant_channel_std_uv"]
    )

    output["flag_extreme_amplitude"] = (
        output["peak_to_peak_uv"]
        >= thresholds["extreme_peak_to_peak_uv"]
    )

    output["flag_line_noise"] = (
        output["line_noise_ratio"]
        >= thresholds["line_noise_ratio_review"]
    )

    output["flag_high_frequency"] = (
        output["high_frequency_ratio"]
        >= thresholds["high_frequency_ratio_review"]
    )

    flag_columns = [
        "flag_nonfinite",
        "flag_flatline",
        "flag_zero_heavy",
        "flag_near_constant",
        "flag_extreme_amplitude",
        "flag_line_noise",
        "flag_high_frequency",
    ]

    output["flag_count"] = (
        output[flag_columns]
        .fillna(False)
        .sum(axis=1)
        .astype(int)
    )

    output["needs_review"] = output["flag_count"] > 0

    return output


def aggregate_channel_metrics(
    chunk_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate chunk-level metrics to recording-channel level."""
    grouped = chunk_metrics.groupby(
        [
            "case_id",
            "recording_id",
            "channel_name",
            "channel_index",
        ],
        dropna=False,
    )

    channels = grouped.agg(
        chunk_count=("chunk_id", "count"),
        non_ictal_chunk_count=(
            "annotation_label",
            lambda values: int((values == "non_ictal").sum()),
        ),
        ictal_chunk_count=(
            "annotation_label",
            lambda values: int((values == "ictal").sum()),
        ),
        boundary_chunk_count=(
            "annotation_label",
            lambda values: int((values == "boundary").sum()),
        ),
        median_rms_uv=("rms_uv", "median"),
        maximum_rms_uv=("rms_uv", "max"),
        median_std_uv=("std_uv", "median"),
        minimum_std_uv=("std_uv", "min"),
        median_peak_to_peak_uv=(
            "peak_to_peak_uv",
            "median",
        ),
        maximum_peak_to_peak_uv=(
            "peak_to_peak_uv",
            "max",
        ),
        median_robust_range_uv=(
            "robust_range_01_99_uv",
            "median",
        ),
        maximum_flatline_fraction=(
            "flatline_sample_fraction",
            "max",
        ),
        median_flatline_fraction=(
            "flatline_sample_fraction",
            "median",
        ),
        maximum_zero_fraction=(
            "zero_fraction",
            "max",
        ),
        median_line_noise_ratio=(
            "line_noise_ratio",
            "median",
        ),
        maximum_line_noise_ratio=(
            "line_noise_ratio",
            "max",
        ),
        median_high_frequency_ratio=(
            "high_frequency_ratio",
            "median",
        ),
        maximum_high_frequency_ratio=(
            "high_frequency_ratio",
            "max",
        ),
        review_chunk_count=("needs_review", "sum"),
        maximum_flag_count=("flag_count", "max"),
    ).reset_index()

    channels["review_chunk_fraction"] = (
        channels["review_chunk_count"]
        / channels["chunk_count"]
    )

    return channels


def aggregate_recording_metrics(
    channel_metrics: pd.DataFrame,
    bad_chunk_fraction_threshold: float,
) -> pd.DataFrame:
    """Aggregate channel-level QC to recording level."""
    channels = channel_metrics.copy()

    channels["channel_needs_review"] = (
        channels["review_chunk_fraction"]
        >= bad_chunk_fraction_threshold
    )

    grouped = channels.groupby(
        ["case_id", "recording_id"],
        dropna=False,
    )

    recordings = grouped.agg(
        channel_count=("channel_name", "count"),
        reviewed_channel_count=(
            "channel_needs_review",
            "sum",
        ),
        median_channel_rms_uv=(
            "median_rms_uv",
            "median",
        ),
        maximum_channel_rms_uv=(
            "maximum_rms_uv",
            "max",
        ),
        maximum_flatline_fraction=(
            "maximum_flatline_fraction",
            "max",
        ),
        maximum_line_noise_ratio=(
            "maximum_line_noise_ratio",
            "max",
        ),
        maximum_high_frequency_ratio=(
            "maximum_high_frequency_ratio",
            "max",
        ),
    ).reset_index()

    recordings["reviewed_channel_fraction"] = (
        recordings["reviewed_channel_count"]
        / recordings["channel_count"]
    )

    recordings["recording_needs_review"] = (
        recordings["reviewed_channel_count"] > 0
    )

    return recordings