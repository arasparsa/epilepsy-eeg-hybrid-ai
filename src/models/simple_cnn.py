"""Simple 1D CNN baseline for multichannel EEG seizure detection."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pooling_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")

        if pooling_size <= 0:
            raise ValueError("pooling_size must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pooling_size,
                stride=pooling_size,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class SimpleEEGCNN(nn.Module):
    """Baseline temporal CNN.

    Expected input shape:
        [batch, channels, samples]

    Output:
        one logit per EEG window, shape [batch]
    """

    def __init__(
        self,
        *,
        input_channels: int = 17,
        conv_channels: Sequence[int] = (
            32,
            64,
            128,
        ),
        kernel_sizes: Sequence[int] = (
            7,
            5,
            3,
        ),
        pooling_sizes: Sequence[int] = (
            2,
            2,
            2,
        ),
        dropout: float = 0.30,
        classifier_hidden_units: int = 64,
        classifier_dropout: float = 0.40,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if classifier_hidden_units <= 0:
            raise ValueError(
                "classifier_hidden_units must be positive."
            )

        if not (
            len(conv_channels)
            == len(kernel_sizes)
            == len(pooling_sizes)
        ):
            raise ValueError(
                "conv_channels, kernel_sizes and "
                "pooling_sizes must have equal lengths."
            )

        if len(conv_channels) == 0:
            raise ValueError(
                "At least one convolution block is required."
            )

        blocks: list[nn.Module] = []

        current_channels = input_channels

        for (
            output_channels,
            kernel_size,
            pooling_size,
        ) in zip(
            conv_channels,
            kernel_sizes,
            pooling_sizes,
            strict=True,
        ):
            blocks.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=int(
                        output_channels
                    ),
                    kernel_size=int(
                        kernel_size
                    ),
                    pooling_size=int(
                        pooling_size
                    ),
                    dropout=float(
                        dropout
                    ),
                )
            )

            current_channels = int(
                output_channels
            )

        self.features = nn.Sequential(
            *blocks
        )

        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                current_channels,
                classifier_hidden_units,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(
                p=classifier_dropout
            ),
            nn.Linear(
                classifier_hidden_units,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "SimpleEEGCNN expects input "
                "[batch, channels, samples]."
            )

        features = self.features(x)

        pooled = self.global_pool(
            features
        )

        logits = self.classifier(
            pooled
        )

        return logits.squeeze(-1)


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable model parameters."""
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )