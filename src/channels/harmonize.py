"""Apply a frozen channel order to MNE Raw objects."""

from __future__ import annotations

from collections.abc import Sequence

import mne

from src.channels.naming import (
    AliasRule,
    canonicalize_channel_names,
)


def harmonize_raw_channels(
    raw: mne.io.BaseRaw,
    target_channels: Sequence[str],
    recording_id: str,
    alias_rules: list[AliasRule] | None = None,
    copy: bool = True,
) -> mne.io.BaseRaw:
    """Rename validated aliases, select and reorder channels.

    This function does not filter, resample, rereference, interpolate,
    or modify the raw EDF file on disk.
    """
    if alias_rules is None:
        alias_rules = []

    harmonized = raw.copy() if copy else raw

    original_names = list(harmonized.ch_names)

    canonical_names, _ = canonicalize_channel_names(
        original_names=original_names,
        alias_rules=alias_rules,
        recording_id=recording_id,
    )

    rename_mapping = {
        original: canonical
        for original, canonical in zip(
            original_names,
            canonical_names,
            strict=True,
        )
        if original != canonical
    }

    if rename_mapping:
        harmonized.rename_channels(
            rename_mapping,
            allow_duplicates=False,
        )

    target = list(target_channels)
    available = set(harmonized.ch_names)
    missing = [
        channel
        for channel in target
        if channel not in available
    ]

    if missing:
        raise ValueError(
            f"{recording_id} is missing target channels: "
            f"{missing}"
        )

    harmonized.pick(target)

    if harmonized.ch_names != target:
        harmonized.reorder_channels(target)

    if harmonized.ch_names != target:
        raise RuntimeError(
            "Final channel order does not match target order."
        )

    return harmonized