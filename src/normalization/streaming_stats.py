"""Numerically stable streaming statistics for EEG channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningChannelStatistics:
    """Per-channel running mean and sum of squared deviations."""

    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def create(
        cls,
        channel_count: int,
    ) -> "RunningChannelStatistics":
        if channel_count <= 0:
            raise ValueError(
                "channel_count must be positive."
            )

        return cls(
            count=np.zeros(
                channel_count,
                dtype=np.int64,
            ),
            mean=np.zeros(
                channel_count,
                dtype=np.float64,
            ),
            m2=np.zeros(
                channel_count,
                dtype=np.float64,
            ),
        )

    def update(
        self,
        data: np.ndarray,
    ) -> None:
        """Update from a channels × samples array."""
        values = np.asarray(
            data,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "Expected channels × samples array."
            )

        if values.shape[0] != len(self.count):
            raise ValueError(
                "Channel count does not match statistics."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Non-finite samples encountered."
            )

        batch_count = np.full(
            values.shape[0],
            values.shape[1],
            dtype=np.int64,
        )

        batch_mean = values.mean(
            axis=1,
            dtype=np.float64,
        )

        centered = (
            values
            - batch_mean[:, np.newaxis]
        )

        batch_m2 = np.sum(
            centered * centered,
            axis=1,
            dtype=np.float64,
        )

        self.combine(
            count=batch_count,
            mean=batch_mean,
            m2=batch_m2,
        )

    def combine(
        self,
        *,
        count: np.ndarray,
        mean: np.ndarray,
        m2: np.ndarray,
    ) -> None:
        """Combine another set of sufficient statistics."""
        other_count = np.asarray(
            count,
            dtype=np.int64,
        )

        other_mean = np.asarray(
            mean,
            dtype=np.float64,
        )

        other_m2 = np.asarray(
            m2,
            dtype=np.float64,
        )

        if not (
            other_count.shape
            == self.count.shape
            == other_mean.shape
            == other_m2.shape
        ):
            raise ValueError(
                "Statistic arrays must share shape."
            )

        total_count = (
            self.count + other_count
        )

        nonzero = other_count > 0

        delta = (
            other_mean - self.mean
        )

        safe_total = np.where(
            total_count > 0,
            total_count,
            1,
        )

        updated_mean = (
            self.mean
            + delta
            * other_count
            / safe_total
        )

        cross_term = (
            delta * delta
            * self.count
            * other_count
            / safe_total
        )

        updated_m2 = (
            self.m2
            + other_m2
            + cross_term
        )

        self.mean = np.where(
            nonzero,
            updated_mean,
            self.mean,
        )

        self.m2 = np.where(
            nonzero,
            updated_m2,
            self.m2,
        )

        self.count = total_count

    @property
    def variance(self) -> np.ndarray:
        """Population variance with ddof=0."""
        return np.divide(
            self.m2,
            self.count,
            out=np.full_like(
                self.m2,
                np.nan,
            ),
            where=self.count > 0,
        )

    @property
    def standard_deviation(
        self,
    ) -> np.ndarray:
        return np.sqrt(self.variance)

    def to_dict(
        self,
    ) -> dict[str, np.ndarray]:
        return {
            "count": self.count.copy(),
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
            "variance": self.variance,
            "std": self.standard_deviation,
        }