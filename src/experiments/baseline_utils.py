"""Shared utilities for baseline experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataloaders import (
    build_eeg_dataloader,
)
from src.data.fold_selection import (
    get_nested_window_tables,
    get_outer_development_tables,
)
from src.models.simple_cnn import (
    SimpleEEGCNN,
)
from src.normalization.fold_scalers import (
    load_scaler_npz,
)
from src.splitting.subject_mapping import (
    apply_subject_mapping,
)
from src.utils.reproducibility import (
    seed_everything,
)


def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(
            file
        )


def save_json(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def resolve_device(
    requested: str,
) -> torch.device:
    """Resolve auto/cpu/cuda device."""
    requested = (
        requested.strip().lower()
    )

    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but "
                "is not available."
            )

        return torch.device(
            "cuda"
        )

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    raise ValueError(
        f"Unsupported device: {requested}"
    )


def build_model_from_config(
    config: dict,
) -> SimpleEEGCNN:
    """Instantiate the frozen SimpleEEGCNN baseline."""
    model_config = config[
        "model"
    ]

    return SimpleEEGCNN(
        input_channels=int(
            model_config[
                "input_channels"
            ]
        ),
        conv_channels=tuple(
            int(value)
            for value
            in model_config[
                "conv_channels"
            ]
        ),
        kernel_sizes=tuple(
            int(value)
            for value
            in model_config[
                "kernel_sizes"
            ]
        ),
        pooling_sizes=tuple(
            int(value)
            for value
            in model_config[
                "pooling_sizes"
            ]
        ),
        dropout=float(
            model_config[
                "dropout"
            ]
        ),
        classifier_hidden_units=int(
            model_config[
                "classifier_hidden_units"
            ]
        ),
        classifier_dropout=float(
            model_config[
                "classifier_dropout"
            ]
        ),
    )


def build_optimizer(
    *,
    model: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    training = config[
        "training"
    ]

    optimizer_name = str(
        training["optimizer"]
    )

    if optimizer_name != "AdamW":
        raise ValueError(
            "baseline_v1 currently supports "
            "AdamW only."
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(
            training[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training[
                "weight_decay"
            ]
        ),
    )


##############################

def build_criterion(
    *,
    pos_weight: float,
    device: torch.device,
    config: dict,
) -> torch.nn.Module:
    """Build BCEWithLogitsLoss.

    Dynamic negative sampling and full pos_weight must not
    be active simultaneously in baseline_v1.
    """

    loss_config = (
        config[
            "training"
        ]["loss"]
    )

    if (
        loss_config["name"]
        != "BCEWithLogitsLoss"
    ):
        raise ValueError(
            "Unsupported baseline loss."
        )

    training_access = (
        config[
            "data_access"
        ]["training"]
    )

    dynamic_sampling = (
        training_access[
            "mode"
        ]
        == (
            "recording_aware_dynamic_negative_sampling"
        )
    )

    use_pos_weight = bool(
        loss_config[
            "use_pos_weight"
        ]
    )

    if (
        dynamic_sampling
        and use_pos_weight
    ):
        raise ValueError(
            "Do not combine full pos_weight with "
            "dynamic negative sampling in baseline_v1."
        )

    if use_pos_weight:
        return (
            torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [
                        float(
                            pos_weight
                        )
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )
        )

    return (
        torch.nn.BCEWithLogitsLoss()
    )

#################################

def get_experiment_seed(
    *,
    config: dict,
    outer_fold: int,
    inner_fold: (
        int | None
    ) = None,
) -> int:
    seed_config = config[
        "reproducibility"
    ]

    seed = (
        int(
            seed_config[
                "base_seed"
            ]
        )
        + outer_fold
        * int(
            seed_config[
                "seed_increment_per_outer_fold"
            ]
        )
    )

    if inner_fold is not None:
        seed += (
            inner_fold
            * int(
                seed_config[
                    "seed_increment_per_inner_fold"
                ]
            )
        )

    return int(seed)


def create_generator(
    seed: int,
) -> torch.Generator:
    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    return generator


def prepare_master_windows(
    *,
    project_root: Path,
    config: dict,
) -> pd.DataFrame:
    """Read window metadata and attach resolved subject IDs."""
    windows = pd.read_csv(
        project_root
        / config["inputs"][
            "window_manifest"
        ]
    )

    mapping = pd.read_csv(
        project_root
        / config["inputs"][
            "subject_mapping"
        ]
    )

    windows = apply_subject_mapping(
        windows,
        mapping,
    )

    return windows


def get_inner_pos_weight(
    *,
    class_weights: pd.DataFrame,
    outer_fold: int,
    inner_fold: int,
) -> float:
    row = class_weights.loc[
        (
            class_weights[
                "scope"
            ]
            == "inner_training"
        )
        & (
            class_weights[
                "outer_fold"
            ]
            == outer_fold
        )
        & (
            class_weights[
                "inner_fold"
            ]
            == inner_fold
        )
    ]

    if len(row) != 1:
        raise ValueError(
            "Expected exactly one inner "
            "class-weight row."
        )

    return float(
        row.iloc[0][
            "pos_weight"
        ]
    )


def get_outer_pos_weight(
    *,
    class_weights: pd.DataFrame,
    outer_fold: int,
) -> float:
    row = class_weights.loc[
        (
            class_weights[
                "scope"
            ]
            == "outer_development"
        )
        & (
            class_weights[
                "outer_fold"
            ]
            == outer_fold
        )
    ]

    if len(row) != 1:
        raise ValueError(
            "Expected exactly one outer-development "
            "class-weight row."
        )

    return float(
        row.iloc[0][
            "pos_weight"
        ]
    )


def load_inner_scaler(
    *,
    project_root: Path,
    config: dict,
    outer_fold: int,
    inner_fold: int,
) -> tuple[
    dict[str, object],
    Path,
]:
    scaler_path = (
        project_root
        / config["inputs"][
            "scaler_directory"
        ]
        / f"outer_{outer_fold:02d}"
        / (
            f"inner_{inner_fold:02d}"
            "_scaler.npz"
        )
    )

    return (
        load_scaler_npz(
            scaler_path
        ),
        scaler_path,
    )


def load_outer_scaler(
    *,
    project_root: Path,
    config: dict,
    outer_fold: int,
) -> tuple[
    dict[str, object],
    Path,
]:
    scaler_path = (
        project_root
        / config["inputs"][
            "scaler_directory"
        ]
        / f"outer_{outer_fold:02d}"
        / "outer_development_scaler.npz"
    )

    return (
        load_scaler_npz(
            scaler_path
        ),
        scaler_path,
    )


#####################################

def build_loader(
    *,
    windows: pd.DataFrame,
    role: str,
    scaler: dict[str, object],
    seed: int,
    project_root: Path,
    baseline_config: dict,
    data_pipeline_config: dict,
):
    """Build recording-aware block-reading DataLoader."""

    loader_config = (
        data_pipeline_config[
            "dataloader"
        ]
    )

    signal_config = (
        data_pipeline_config[
            "signal"
        ]
    )

    access = (
        baseline_config[
            "data_access"
        ]
    )

    if role == "train":

        role_config = (
            access[
                "training"
            ]
        )

        return build_eeg_dataloader(
            windows=windows,
            project_root=project_root,
            channel_mean_uv=np.asarray(
                scaler[
                    "channel_mean_uv"
                ],
                dtype=np.float64,
            ),
            channel_std_uv=np.asarray(
                scaler[
                    "channel_std_uv"
                ],
                dtype=np.float64,
            ),
            batch_size=int(
                role_config[
                    "batch_size"
                ]
            ),
            role="train",
            generator=create_generator(
                seed
            ),
            expected_channel_count=int(
                signal_config[
                    "expected_channel_count"
                ]
            ),
            expected_sample_count=int(
                signal_config[
                    "expected_sample_count"
                ]
            ),
            max_open_fif_files=int(
                role_config.get(
                    "max_open_fif_files",
                    32,
                )
            ),
            preload_fif=bool(
                role_config.get(
                    "preload_fif",
                    False,
                )
            ),
            num_workers=int(
                loader_config[
                    "num_workers_windows"
                ]
            ),
            pin_memory=bool(
                loader_config[
                    "pin_memory"
                ]
            ),
            persistent_workers=bool(
                loader_config[
                    "persistent_workers"
                ]
            ),
            prefetch_factor=int(
                loader_config[
                    "prefetch_factor"
                ]
            ),
            drop_last=bool(
                role_config[
                    "drop_last"
                ]
            ),
            return_metadata=True,
            batched_block_reading=bool(
                role_config.get(
                    "batched_block_reading",
                    True,
                )
            ),
            recording_aware_training=True,
            negative_to_positive_ratio=float(
                role_config[
                    "negative_to_positive_ratio"
                ]
            ),
            local_block_size=int(
                role_config[
                    "local_block_size"
                ]
            ),
            sampler_seed=int(
                seed
            ),
            shuffle_blocks=bool(
                role_config[
                    "shuffle_blocks"
                ]
            ),
            shuffle_within_block=bool(
                role_config[
                    "shuffle_within_block"
                ]
            ),
            negative_sampling_replacement=bool(
                role_config[
                    "negative_sampling_replacement"
                ]
            ),
        )

    if role not in {
        "validation",
        "test",
    }:
        raise ValueError(
            f"Unsupported role: {role}"
        )

    role_config = (
        access[
            role
        ]
    )

    return build_eeg_dataloader(
        windows=windows,
        project_root=project_root,
        channel_mean_uv=np.asarray(
            scaler[
                "channel_mean_uv"
            ],
            dtype=np.float64,
        ),
        channel_std_uv=np.asarray(
            scaler[
                "channel_std_uv"
            ],
            dtype=np.float64,
        ),
        batch_size=int(
            role_config[
                "batch_size"
            ]
        ),
        role=role,
        generator=create_generator(
            seed
        ),
        expected_channel_count=int(
            signal_config[
                "expected_channel_count"
            ]
        ),
        expected_sample_count=int(
            signal_config[
                "expected_sample_count"
            ]
        ),
        max_open_fif_files=int(
            role_config.get(
                "max_open_fif_files",
                4,
            )
        ),
        preload_fif=bool(
            role_config.get(
                "preload_fif",
                False,
            )
        ),
        num_workers=int(
            loader_config[
                "num_workers_windows"
            ]
        ),
        pin_memory=bool(
            loader_config[
                "pin_memory"
            ]
        ),
        persistent_workers=bool(
            loader_config[
                "persistent_workers"
            ]
        ),
        prefetch_factor=int(
            loader_config[
                "prefetch_factor"
            ]
        ),
        drop_last=False,
        return_metadata=True,
        batched_block_reading=bool(
            role_config.get(
                "batched_block_reading",
                True,
            )
        ),
        recording_aware_training=False,
    )

    
#############################################


def seed_experiment(
    *,
    seed: int,
    data_pipeline_config: dict,
) -> torch.Generator:
    reproducibility = (
        data_pipeline_config[
            "reproducibility"
        ]
    )

    return seed_everything(
        seed=seed,
        deterministic_algorithms=bool(
            reproducibility[
                "deterministic_algorithms"
            ]
        ),
        warn_only=bool(
            reproducibility[
                "warn_only_deterministic"
            ]
        ),
        cudnn_deterministic=bool(
            reproducibility[
                "cudnn_deterministic"
            ]
        ),
        cudnn_benchmark=bool(
            reproducibility[
                "cudnn_benchmark"
            ]
        ),
    )


def round_half_up(
    value: float,
) -> int:
    """Avoid Python banker's rounding for selected epoch."""
    return int(
        math.floor(
            float(value) + 0.5
        )
    )


def upsert_csv(
    *,
    path: Path,
    new_rows: pd.DataFrame,
    key_columns: list[str],
) -> None:
    """Insert or replace keyed rows in a CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = pd.read_csv(
            path
        )

        combined = pd.concat(
            [
                existing,
                new_rows,
            ],
            ignore_index=True,
        )

        combined = (
            combined.drop_duplicates(
                subset=key_columns,
                keep="last",
            )
        )

    else:
        combined = new_rows.copy()

    combined = combined.sort_values(
        key_columns
    )

    combined.to_csv(
        path,
        index=False,
    )