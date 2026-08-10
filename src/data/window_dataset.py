"""PyTorch Dataset for CHB-MIT EEG windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.fif_cache import (
    FIFRawCache,
)


class EEGWindowDataset(Dataset):
    """Read sample-aligned windows from continuous FIF files."""

    def __init__(
        self,
        *,
        windows: pd.DataFrame,
        project_root: Path,
        channel_mean_uv: np.ndarray | None,
        channel_std_uv: np.ndarray | None,
        volts_to_microvolts: float = 1e6,
        expected_channel_count: int = 17,
        expected_sample_count: int = 1024,
        max_open_fif_files: int = 4,
        preload_fif: bool = False,
        return_metadata: bool = True,
        verify_finite_values: bool = True,
        clip_range: tuple[
            float,
            float,
        ] | None = None,
    ) -> None:
        self.windows = (
            windows.reset_index(
                drop=True
            ).copy()
        )

        required_columns = {
            "window_id",
            "output_fif_path",
            "start_sample",
            "stop_sample_exclusive",
            "binary_label",
            "case_id",
            "subject_id",
            "recording_id",
        }

        missing = (
            required_columns
            - set(self.windows.columns)
        )

        if missing:
            raise ValueError(
                f"Missing Dataset columns: "
                f"{sorted(missing)}"
            )

        if self.windows[
            "binary_label"
        ].isna().any():
            raise ValueError(
                "Binary Dataset cannot contain "
                "missing labels."
            )

        self.project_root = Path(
            project_root
        ).resolve()

        self.expected_channel_count = int(
            expected_channel_count
        )

        self.expected_sample_count = int(
            expected_sample_count
        )

        self.volts_to_microvolts = float(
            volts_to_microvolts
        )

        self.return_metadata = (
            return_metadata
        )

        self.verify_finite_values = (
            verify_finite_values
        )

        self.clip_range = clip_range

        if (
            channel_mean_uv is None
            or channel_std_uv is None
        ):
            self.channel_mean_uv = None
            self.channel_std_uv = None

        else:
            self.channel_mean_uv = np.asarray(
                channel_mean_uv,
                dtype=np.float32,
            )

            self.channel_std_uv = np.asarray(
                channel_std_uv,
                dtype=np.float32,
            )

            expected_shape = (
                self.expected_channel_count,
            )

            if (
                self.channel_mean_uv.shape
                != expected_shape
            ):
                raise ValueError(
                    "Scaler mean shape mismatch."
                )

            if (
                self.channel_std_uv.shape
                != expected_shape
            ):
                raise ValueError(
                    "Scaler std shape mismatch."
                )

            if np.any(
                self.channel_std_uv <= 0
            ):
                raise ValueError(
                    "Scaler std values must "
                    "be positive."
                )

        self._cache_settings = {
            "project_root": (
                self.project_root
            ),
            "max_open_files": (
                max_open_fif_files
            ),
            "preload": preload_fif,
        }

        self._cache: (
            FIFRawCache | None
        ) = None

    def _get_cache(
        self,
    ) -> FIFRawCache:
        """Create cache lazily in the current worker."""
        if self._cache is None:
            self._cache = FIFRawCache(
                **self._cache_settings
            )

        return self._cache

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            dict[str, Any],
        ]
        | tuple[
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        row = self.windows.iloc[index]

        raw = self._get_cache().get(
            str(row["output_fif_path"])
        )

        start_sample = int(
            row["start_sample"]
        )

        stop_sample = int(
            row[
                "stop_sample_exclusive"
            ]
        )

        data_volts = raw.get_data(
            start=start_sample,
            stop=stop_sample,
        )

        expected_shape = (
            self.expected_channel_count,
            self.expected_sample_count,
        )

        if data_volts.shape != expected_shape:
            raise ValueError(
                f"Unexpected window shape for "
                f"{row['window_id']}: "
                f"{data_volts.shape}"
            )

        data_uv = (
            data_volts
            * self.volts_to_microvolts
        ).astype(
            np.float32,
            copy=False,
        )

        if self.verify_finite_values:
            if not np.isfinite(
                data_uv
            ).all():
                raise ValueError(
                    "Non-finite values in "
                    f"{row['window_id']}"
                )

        if (
            self.channel_mean_uv
            is not None
        ):
            data_uv = (
                data_uv
                - self.channel_mean_uv[
                    :, np.newaxis
                ]
            ) / self.channel_std_uv[
                :, np.newaxis
            ]

        if self.clip_range is not None:
            data_uv = np.clip(
                data_uv,
                self.clip_range[0],
                self.clip_range[1],
            )

        data_uv = np.ascontiguousarray(
            data_uv,
            dtype=np.float32,
        )

        signal_tensor = torch.from_numpy(
            data_uv
        )

        label_tensor = torch.tensor(
            float(row["binary_label"]),
            dtype=torch.float32,
        )

        if not self.return_metadata:
            return (
                signal_tensor,
                label_tensor,
            )

        metadata: dict[str, Any] = {
            "window_id": str(
                row["window_id"]
            ),
            "subject_id": str(
                row["subject_id"]
            ),
            "case_id": str(
                row["case_id"]
            ),
            "recording_id": str(
                row["recording_id"]
            ),
            "start_sample": (
                start_sample
            ),
            "stop_sample_exclusive": (
                stop_sample
            ),
        }

        return (
            signal_tensor,
            label_tensor,
            metadata,
        )