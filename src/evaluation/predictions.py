"""Generate deterministic model predictions."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import torch


@torch.inference_mode()
def predict_loader(
    *,
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
) -> pd.DataFrame:
    """Generate prediction probabilities and identifiers."""
    model.eval()

    rows: list[dict] = []

    for (
        signals,
        labels,
        metadata,
    ) in loader:
        signals = signals.to(
            device,
            non_blocking=True,
        )

        logits = model(
            signals
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        labels_cpu = (
            labels.detach()
            .cpu()
            .numpy()
        )

        probabilities_cpu = (
            probabilities.detach()
            .cpu()
            .numpy()
        )

        batch_size = len(
            labels_cpu
        )

        for index in range(
            batch_size
        ):
            rows.append(
                {
                    "window_id": str(
                        metadata[
                            "window_id"
                        ][index]
                    ),
                    "subject_id": str(
                        metadata[
                            "subject_id"
                        ][index]
                    ),
                    "case_id": str(
                        metadata[
                            "case_id"
                        ][index]
                    ),
                    "recording_id": str(
                        metadata[
                            "recording_id"
                        ][index]
                    ),
                    "true_label": int(
                        labels_cpu[
                            index
                        ]
                    ),
                    "probability": float(
                        probabilities_cpu[
                            index
                        ]
                    ),
                }
            )

    predictions = pd.DataFrame(
        rows
    )

    if predictions.empty:
        raise RuntimeError(
            "Prediction loader produced "
            "no observations."
        )

    if not predictions[
        "window_id"
    ].is_unique:
        raise RuntimeError(
            "Duplicate window IDs were produced."
        )

    return predictions