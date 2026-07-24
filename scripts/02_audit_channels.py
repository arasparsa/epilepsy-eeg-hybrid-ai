"""Audit channel availability across CHB-MIT EDF recordings."""

from collections import Counter
from pathlib import Path

import mne
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
METADATA_DIR = PROJECT_ROOT / "metadata"


def normalize_channel_name(name: str) -> str:
    """Normalize superficial differences without changing electrode identity."""
    normalized = name.strip().upper()
    normalized = normalized.replace("EEG ", "")
    normalized = normalized.replace("-REF", "")
    normalized = normalized.replace("-LE", "")
    return normalized


def audit_file(file_path: Path) -> dict:
    raw = mne.io.read_raw_edf(
        file_path,
        preload=False,
        verbose="ERROR",
    )

    normalized_channels = [
        normalize_channel_name(channel)
        for channel in raw.ch_names
    ]

    return {
        "patient_id": file_path.parent.name,
        "recording_id": file_path.stem,
        "relative_path": str(file_path.relative_to(RAW_DIR)),
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.times[-1]),
        "n_channels": len(raw.ch_names),
        "original_channels": "|".join(raw.ch_names),
        "normalized_channels": "|".join(normalized_channels),
    }


def main() -> None:
    edf_files = sorted(RAW_DIR.rglob("*.edf"))

    if not edf_files:
        raise FileNotFoundError(f"No EDF files found under {RAW_DIR}")

    rows = []

    for index, file_path in enumerate(edf_files, start=1):
        print(f"[{index}/{len(edf_files)}] {file_path.name}")
        rows.append(audit_file(file_path))

    inventory = pd.DataFrame(rows)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(
        METADATA_DIR / "chbmit_file_inventory.csv",
        index=False,
    )

    channel_counter: Counter[str] = Counter()

    for channel_string in inventory["normalized_channels"]:
        channel_counter.update(channel_string.split("|"))

    channel_counts = (
        pd.DataFrame(
            channel_counter.items(),
            columns=["channel", "file_count"],
        )
        .sort_values("file_count", ascending=False)
        .reset_index(drop=True)
    )

    channel_counts["coverage_fraction"] = (
        channel_counts["file_count"] / len(inventory)
    )

    channel_counts.to_csv(
        METADATA_DIR / "chbmit_channel_counts.csv",
        index=False,
    )


if __name__ == "__main__":
    main()