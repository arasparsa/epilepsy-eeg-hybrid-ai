"""Worker-local least-recently-used cache for MNE FIF objects."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import mne


class FIFRawCache:
    """Keep a limited number of FIF files open per worker."""

    def __init__(
        self,
        *,
        project_root: Path,
        max_open_files: int = 4,
        preload: bool = False,
    ) -> None:
        if max_open_files <= 0:
            raise ValueError(
                "max_open_files must be positive."
            )

        self.project_root = (
            Path(project_root).resolve()
        )
        self.max_open_files = max_open_files
        self.preload = preload

        self._cache: OrderedDict[
            str,
            mne.io.BaseRaw,
        ] = OrderedDict()

    def get(
        self,
        relative_path: str,
    ) -> mne.io.BaseRaw:
        """Return a cached Raw object."""
        key = str(relative_path)

        if key in self._cache:
            raw = self._cache.pop(key)
            self._cache[key] = raw
            return raw

        full_path = (
            self.project_root / key
        ).resolve()

        if not full_path.exists():
            raise FileNotFoundError(
                f"FIF file not found: {full_path}"
            )

        raw = mne.io.read_raw_fif(
            full_path,
            preload=self.preload,
            verbose="ERROR",
        )

        self._cache[key] = raw

        while (
            len(self._cache)
            > self.max_open_files
        ):
            _, oldest_raw = (
                self._cache.popitem(
                    last=False
                )
            )

            oldest_raw.close()

        return raw

    def close_all(self) -> None:
        """Close every cached FIF object."""
        for raw in self._cache.values():
            raw.close()

        self._cache.clear()

    def __del__(self) -> None:
        try:
            self.close_all()
        except Exception:
            pass