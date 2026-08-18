"""Metric-based early stopping."""

from __future__ import annotations

import math


class EarlyStopping:
    """Track validation performance and decide when to stop."""

    def __init__(
        self,
        *,
        mode: str = "max",
        patience: int = 8,
        minimum_delta: float = 0.0,
    ) -> None:
        if mode not in {
            "max",
            "min",
        }:
            raise ValueError(
                "mode must be 'max' or 'min'."
            )

        if patience < 1:
            raise ValueError(
                "patience must be at least 1."
            )

        if minimum_delta < 0:
            raise ValueError(
                "minimum_delta cannot be negative."
            )

        self.mode = mode
        self.patience = int(
            patience
        )
        self.minimum_delta = float(
            minimum_delta
        )

        self.best_value: (
            float | None
        ) = None

        self.best_epoch: (
            int | None
        ) = None

        self.bad_epochs = 0

    def update(
        self,
        *,
        value: float,
        epoch: int,
    ) -> tuple[
        bool,
        bool,
    ]:
        """Return (improved, should_stop)."""
        if not math.isfinite(value):
            raise ValueError(
                "Early-stopping metric "
                "must be finite."
            )

        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0

            return True, False

        if self.mode == "max":
            improved = (
                value
                > self.best_value
                + self.minimum_delta
            )
        else:
            improved = (
                value
                < self.best_value
                - self.minimum_delta
            )

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0

            return True, False

        self.bad_epochs += 1

        should_stop = (
            self.bad_epochs
            >= self.patience
        )

        return (
            False,
            should_stop,
        )