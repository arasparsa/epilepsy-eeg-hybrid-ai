"""Checkpoint handling for baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_average_precision: float,
    outer_fold: int,
    inner_fold: int | None,
    seed: int,
    model_name: str,
) -> None:
    """Save a complete training checkpoint."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "epoch": int(epoch),
        "validation_average_precision": float(
            validation_average_precision
        ),
        "outer_fold": int(
            outer_fold
        ),
        "inner_fold": (
            int(inner_fold)
            if inner_fold is not None
            else None
        ),
        "seed": int(seed),
        "model_name": str(
            model_name
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def save_final_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    outer_fold: int,
    seed: int,
    model_name: str,
) -> None:
    """Save an outer-development trained model."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "epoch": int(epoch),
            "outer_fold": int(
                outer_fold
            ),
            "seed": int(seed),
            "model_name": str(
                model_name
            ),
        },
        path,
    )


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: (
        torch.optim.Optimizer
        | None
    ) = None,
) -> dict[str, Any]:
    """Load a project-generated checkpoint."""
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return checkpoint