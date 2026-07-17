"""Compare potentially duplicated EDF channels in flagged CHB-MIT files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit"
DEFAULT_PROBLEM_FILE = (
    PROJECT_ROOT / "metadata" / "chbmit_problem_files.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "metadata"
    / "chbmit_duplicate_channel_comparison.csv"
)


def normalize_name(name: str) -> str:
    normalized = name.strip().upper()
    normalized = normalized.replace("–", "-")
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"^EEG\s+", "", normalized)
    normalized = re.sub(r"-(REF|LE)$", "", normalized)
    return normalized


def base_name(name: str) -> str:
    return re.sub(r"-\d+$", "", name)


def compare_channels(
    raw: mne.io.BaseRaw,
    first_index: int,
    second_index: int,
    maximum_seconds: float,
) -> dict[str, float | bool]:
    sfreq = float(raw.info["sfreq"])
    stop_sample = min(
        raw.n_times,
        int(maximum_seconds * sfreq),
    )

    data = raw.get_data(
        picks=[first_index, second_index],
        start=0,
        stop=stop_sample,
    )

    first = data[0]
    second = data[1]

    difference = first - second

    identical = bool(np.array_equal(first, second))
    allclose = bool(
        np.allclose(
            first,
            second,
            rtol=1e-7,
            atol=1e-12,
        )
    )

    if np.std(first) == 0 or np.std(second) == 0:
        correlation = np.nan
    else:
        correlation = float(
            np.corrcoef(first, second)[0, 1]
        )

    return {
        "samples_compared": len(first),
        "exactly_identical": identical,
        "numerically_allclose": allclose,
        "pearson_correlation": correlation,
        "maximum_absolute_difference": float(
            np.max(np.abs(difference))
        ),
        "root_mean_square_difference": float(
            np.sqrt(np.mean(difference**2))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--problem-file",
        type=Path,
        default=DEFAULT_PROBLEM_FILE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--maximum-seconds",
        type=float,
        default=300.0,
    )
    args = parser.parse_args()

    problems = pd.read_csv(args.problem_file)

    flagged = problems.loc[
        problems["duplicate_name_problem"]
    ].copy()

    rows: list[dict] = []

    for file_row in flagged.itertuples(index=False):
        file_path = args.raw_dir / file_row.relative_path

        raw = mne.io.read_raw_edf(
            file_path,
            preload=False,
            verbose="ERROR",
        )

        normalized = [
            normalize_name(name)
            for name in raw.ch_names
        ]

        grouped: dict[str, list[int]] = {}

        for index, channel_name in enumerate(normalized):
            grouped.setdefault(
                base_name(channel_name),
                [],
            ).append(index)

        for duplicate_base, indices in grouped.items():
            if len(indices) < 2:
                continue

            for first_position in range(len(indices) - 1):
                for second_position in range(
                    first_position + 1,
                    len(indices),
                ):
                    first_index = indices[first_position]
                    second_index = indices[second_position]

                    comparison = compare_channels(
                        raw=raw,
                        first_index=first_index,
                        second_index=second_index,
                        maximum_seconds=args.maximum_seconds,
                    )

                    rows.append(
                        {
                            "patient_id": file_row.patient_id,
                            "recording_id": file_row.recording_id,
                            "relative_path": file_row.relative_path,
                            "duplicate_base": duplicate_base,
                            "first_channel": normalized[first_index],
                            "second_channel": normalized[second_index],
                            **comparison,
                        }
                    )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Comparisons: {len(output)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()