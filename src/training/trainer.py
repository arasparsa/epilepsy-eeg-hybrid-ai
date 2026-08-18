"""Training engines for binary EEG classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.evaluation.predictions import (
    predict_loader,
)
from src.evaluation.window_metrics import (
    calculate_binary_metrics,
)
from src.training.checkpointing import (
    load_checkpoint,
    save_final_checkpoint,
    save_training_checkpoint,
)
from src.training.early_stopping import (
    EarlyStopping,
)


######
import time
######

def _create_grad_scaler(
    *,
    device: torch.device,
    mixed_precision: bool,
) -> torch.amp.GradScaler:
    enabled = bool(
        mixed_precision
        and device.type == "cuda"
    )

    return torch.amp.GradScaler(
        device.type,
        enabled=enabled,
    )


def _run_training_epoch(
    *,
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    gradient_clip_max_norm: (
        float | None
    ),
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    amp_enabled = bool(
        mixed_precision
        and device.type == "cuda"
    )

    for (
        signals,
        labels,
        _,
    ) in train_loader:
        signals = signals.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(
                signals
            )

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        if (
            gradient_clip_max_norm
            is not None
        ):
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(
                    gradient_clip_max_norm
                ),
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = int(
            labels.shape[0]
        )

        total_loss += (
            float(
                loss.detach().item()
            )
            * batch_size
        )

        total_samples += (
            batch_size
        )

    if total_samples == 0:
        raise RuntimeError(
            "Training DataLoader "
            "produced no samples."
        )

    return (
        total_loss
        / total_samples
    )


def train_with_early_stopping(
    *,
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    maximum_epochs: int,
    checkpoint_path: Path,
    outer_fold: int,
    inner_fold: int,
    seed: int,
    model_name: str,
    patience: int,
    minimum_delta: float,
    gradient_clip_max_norm: (
        float | None
    ),
    mixed_precision: bool,
) -> tuple[
    torch.nn.Module,
    pd.DataFrame,
    dict[str, float | int],
]:
    """Train using validation Average Precision."""
    if maximum_epochs < 1:
        raise ValueError(
            "maximum_epochs must be positive."
        )

    stopper = EarlyStopping(
        mode="max",
        patience=patience,
        minimum_delta=minimum_delta,
    )

    grad_scaler = (
        _create_grad_scaler(
            device=device,
            mixed_precision=(
                mixed_precision
            ),
        )
    )

    history_rows: list[
        dict
    ] = []

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        batch_sampler = getattr(
            train_loader,
            "batch_sampler",
            None,
        )

        if (
            batch_sampler is not None
            and hasattr(
                batch_sampler,
                "set_epoch",
            )
        ):
            batch_sampler.set_epoch(
                epoch - 1
            )
            
        ###############
        training_start = (
            time.perf_counter()
        )
        
        training_loss = (
            _run_training_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=grad_scaler,
                mixed_precision=(
                    mixed_precision
                ),
                gradient_clip_max_norm=(
                    gradient_clip_max_norm
                ),
            )
        )
        
        training_elapsed = (
            time.perf_counter()
            - training_start
        )
        ###############
        ###############
        validation_start = (
            time.perf_counter()
        )
        ###############
        validation_predictions = (
            predict_loader(
                model=model,
                loader=validation_loader,
                device=device,
            )
        )
        ###########
        validation_elapsed = (
            time.perf_counter()
            - validation_start
        )
        ##########

        validation_metrics = (
            calculate_binary_metrics(
                y_true=(
                    validation_predictions[
                        "true_label"
                    ].to_numpy()
                ),
                probabilities=(
                    validation_predictions[
                        "probability"
                    ].to_numpy()
                ),
                # Only temporary for threshold-dependent
                # monitoring. Model selection uses AP.
                threshold=0.5,
            )
        )

        validation_ap = float(
            validation_metrics[
                "average_precision"
            ]
        )

        (
            improved,
            should_stop,
        ) = stopper.update(
            value=validation_ap,
            epoch=epoch,
        )

        ###############
        history_rows.append(
            {
                "epoch": epoch,
                "training_loss": float(
                    training_loss
                ),
                "validation_average_precision": (
                    validation_ap
                ),
                "validation_roc_auc": float(
                    validation_metrics[
                        "roc_auc"
                    ]
                ),
                "validation_f1_at_0_5": float(
                    validation_metrics[
                        "f1"
                    ]
                ),
                "training_seconds": float(
                    training_elapsed
                ),
                "validation_seconds": float(
                    validation_elapsed
                ),
                "epoch_seconds": float(
                    training_elapsed
                    + validation_elapsed
                ),
                "improved": bool(
                    improved
                ),
            }
        )
        ###############

        if improved:
            save_training_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_average_precision=(
                    validation_ap
                ),
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                seed=seed,
                model_name=model_name,
            )

        print(
            f"Outer {outer_fold} | "
            f"Inner {inner_fold} | "
            f"Epoch {epoch:03d} | "
            f"loss={training_loss:.6f} | "
            f"val_AP={validation_ap:.6f}"
        )

        if should_stop:
            print(
                "Early stopping triggered."
            )
            break

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=device,
    )

    history = pd.DataFrame(
        history_rows
    )

    result = {
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "best_validation_average_precision": float(
            checkpoint[
                "validation_average_precision"
            ]
        ),
        "epochs_completed": int(
            len(history)
        ),
    }

    return (
        model,
        history,
        result,
    )


def train_fixed_epochs(
    *,
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    outer_fold: int,
    seed: int,
    model_name: str,
    gradient_clip_max_norm: (
        float | None
    ),
    mixed_precision: bool,
) -> tuple[
    torch.nn.Module,
    pd.DataFrame,
]:
    """Train on the complete outer-development set.

    No validation or outer-test information is used here.
    """
    if epochs < 1:
        raise ValueError(
            "epochs must be positive."
        )

    grad_scaler = (
        _create_grad_scaler(
            device=device,
            mixed_precision=(
                mixed_precision
            ),
        )
    )

    history_rows = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        batch_sampler = getattr(
            train_loader,
            "batch_sampler",
            None,
        )

        if (
            batch_sampler is not None
            and hasattr(
                batch_sampler,
                "set_epoch",
            )
        ):
            batch_sampler.set_epoch(
                epoch - 1
            )
        ########
        epoch_start = (
            time.perf_counter()
        )
        ########
            
        loss = _run_training_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=grad_scaler,
            mixed_precision=(
                mixed_precision
            ),
            gradient_clip_max_norm=(
                gradient_clip_max_norm
            ),
        )
        ###########
        epoch_elapsed = (
            time.perf_counter()
            - epoch_start
        )
        ###########

        history_rows.append(
            {
                "epoch": epoch,
                "training_loss": float(
                    loss
                ),
                "training_seconds": float(
                    epoch_elapsed
                ),
            }
        )

        ####################
        
        print(
            f"Outer {outer_fold} | "
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"loss={loss:.6f} | "
            f"time={epoch_elapsed / 60:.1f}m",
            flush=True,
        )
        ####################

    save_final_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        outer_fold=outer_fold,
        seed=seed,
        model_name=model_name,
    )

    return (
        model,
        pd.DataFrame(
            history_rows
        ),
    )