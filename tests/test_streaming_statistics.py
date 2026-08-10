"""Tests for streaming channel statistics."""

import numpy as np

from src.normalization.streaming_stats import (
    RunningChannelStatistics,
)


def test_streaming_matches_numpy() -> None:
    generator = (
        np.random.default_rng(42)
    )

    data = generator.normal(
        size=(3, 1000)
    )

    statistics = (
        RunningChannelStatistics.create(
            3
        )
    )

    statistics.update(
        data[:, :400]
    )

    statistics.update(
        data[:, 400:]
    )

    assert np.allclose(
        statistics.mean,
        data.mean(axis=1),
    )

    assert np.allclose(
        statistics.variance,
        data.var(
            axis=1,
            ddof=0,
        ),
    )


def test_combining_subject_statistics() -> None:
    first = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ]
    )

    second = np.asarray(
        [
            [4.0, 5.0],
            [40.0, 50.0],
        ]
    )

    first_stats = (
        RunningChannelStatistics.create(
            2
        )
    )

    second_stats = (
        RunningChannelStatistics.create(
            2
        )
    )

    combined = (
        RunningChannelStatistics.create(
            2
        )
    )

    first_stats.update(first)
    second_stats.update(second)

    combined.combine(
        count=first_stats.count,
        mean=first_stats.mean,
        m2=first_stats.m2,
    )

    combined.combine(
        count=second_stats.count,
        mean=second_stats.mean,
        m2=second_stats.m2,
    )

    expected = np.concatenate(
        [first, second],
        axis=1,
    )

    assert np.allclose(
        combined.mean,
        expected.mean(axis=1),
    )

    assert np.allclose(
        combined.variance,
        expected.var(axis=1),
    )