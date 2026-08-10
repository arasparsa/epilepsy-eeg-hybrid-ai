"""Build leakage-safe PyTorch DataLoaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)

from src.data.window_dataset import (
    EEGWindowDataset,
)
from src.imbalance.class_weights import (
    calculate_sample_weights,
)
from src.utils.reproducibility import (
    seed_worker,
)


def build_eeg_dataloader(
    *,
    windows: pd.DataFrame,
    project_root: Path,
    channel_mean_uv: np.ndarray,
    channel_std_uv: np.ndarray,
    batch_size: int,
    role: str,
    imbalance_strategy: str,
    generator: torch.Generator,
    expected_channel_count: int,
    expected_sample_count: int,
    max_open_fif_files: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: bool,
    return_metadata: bool,
) -> DataLoader:
    """Create one train, validation or test DataLoader."""
    if role not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            f"Invalid loader role: {role}"
        )

    dataset = EEGWindowDataset(
        windows=windows,
        project_root=project_root,
        channel_mean_uv=(
            channel_mean_uv
        ),
        channel_std_uv=(
            channel_std_uv
        ),
        expected_channel_count=(
            expected_channel_count
        ),
        expected_sample_count=(
            expected_sample_count
        ),
        max_open_fif_files=(
            max_open_fif_files
        ),
        preload_fif=False,
        return_metadata=(
            return_metadata
        ),
        verify_finite_values=True,
    )

    sampler = None
    shuffle = False

    if role == "train":
        if (
            imbalance_strategy
            == "weighted_sampler"
        ):
            sample_weights = (
                calculate_sample_weights(
                    windows
                )
            )

            sampler = (
                WeightedRandomSampler(
                    weights=torch.as_tensor(
                        sample_weights,
                        dtype=torch.double,
                    ),
                    num_samples=len(
                        sample_weights
                    ),
                    replacement=True,
                    generator=generator,
                )
            )

            shuffle = False

        elif imbalance_strategy in {
            "loss_weighting",
            "deterministic_negative_sampling",
            "none",
        }:
            shuffle = True

        else:
            raise ValueError(
                "Unknown imbalance strategy: "
                f"{imbalance_strategy}"
            )

    loader_arguments = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "worker_init_fn": (
            seed_worker
        ),
        "generator": generator,
        "persistent_workers": (
            persistent_workers
            if num_workers > 0
            else False
        ),
    }

    if num_workers > 0:
        loader_arguments[
            "prefetch_factor"
        ] = prefetch_factor

    return DataLoader(
        **loader_arguments
    )