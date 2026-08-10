"""Reproducibility utilities for NumPy and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool,
    warn_only: bool,
    cudnn_deterministic: bool,
    cudnn_benchmark: bool,
) -> torch.Generator:
    """Set deterministic random states where supported."""
    if seed < 0:
        raise ValueError(
            "Seed must be non-negative."
        )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = (
        cudnn_deterministic
    )

    torch.backends.cudnn.benchmark = (
        cudnn_benchmark
    )

    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=warn_only,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_worker(
    worker_id: int,
) -> None:
    """Seed NumPy and Python inside DataLoader workers."""
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % 2**32
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)