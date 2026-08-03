"""Validate CHB-MIT segmentation and window labels."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

from src.segmentation.loader import (
    load_window_from_manifest_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "chbmit_segmentation.yaml"
)


def main() -> int:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    metadata_dir = (
        PROJECT_ROOT
        / config["outputs"][
            "metadata_directory"
        ]
    )

    windows = pd.read_csv(
        metadata_dir
        / "chbmit_window_manifest.csv"
    )

    recording_summary = pd.read_csv(
        metadata_dir
        / "chbmit_recording_window_summary.csv"
    )

    failures = pd.read_csv(
        metadata_dir
        / "chbmit_segmentation_failures.csv"
    )

    preprocessing_manifest = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "preprocessing_manifest"
        ]
    )

    preprocessing_validation = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"][
            "preprocessing_validation"
        ]
    )

    seizures = pd.read_csv(
        PROJECT_ROOT
        / config["inputs"]["seizures_file"]
    )

    errors: list[str] = []
    warnings: list[str] = []
    validation_rows: list[dict] = []

    expected_sfreq = float(
        config["windowing"][
            "expected_sampling_rate_hz"
        ]
    )

    expected_samples = int(
        round(
            config["windowing"][
                "window_duration_seconds"
            ]
            * expected_sfreq
        )
    )

    expected_stride = int(
        round(
            config["windowing"][
                "stride_seconds"
            ]
            * expected_sfreq
        )
    )

    # 1. Unique window IDs.
    if not windows["window_id"].is_unique:
        errors.append(
            "Window IDs are not unique."
        )

    # 2. Unique recording/window-index keys.
    if windows[
        ["recording_id", "window_index"]
    ].duplicated().any():
        errors.append(
            "Duplicate recording/window-index keys."
        )

    # 3. Valid sample intervals.
    if (
        windows["start_sample"] < 0
    ).any():
        errors.append(
            "Negative start samples detected."
        )

    if (
        windows["stop_sample_exclusive"]
        <= windows["start_sample"]
    ).any():
        errors.append(
            "Invalid sample intervals detected."
        )

    # 4. Fixed sample count.
    if not windows[
        "sample_count"
    ].eq(expected_samples).all():
        errors.append(
            "Not all windows have the expected "
            f"{expected_samples} samples."
        )

    # 5. Fixed sampling rate.
    if not np.allclose(
        windows["sampling_rate_hz"],
        expected_sfreq,
    ):
        errors.append(
            "Unexpected sampling rates detected."
        )

    # 6. Valid label set.
    valid_labels = set(
        config["labeling"]["labels"].keys()
    )

    observed_labels = set(
        windows["label_name"].unique()
    )

    invalid_labels = (
        observed_labels - valid_labels
    )

    if invalid_labels:
        errors.append(
            f"Invalid labels: {invalid_labels}"
        )

    # 7. Fractions.
    if not windows[
        "seizure_overlap_fraction"
    ].between(0, 1).all():
        errors.append(
            "Invalid seizure overlap fractions."
        )

    # 8. Label/fraction consistency.
    threshold = float(
        config["labeling"][
            "ictal_overlap_threshold"
        ]
    )

    invalid_non_ictal = windows.loc[
        (windows["label_name"] == "non_ictal")
        & (
            windows[
                "seizure_overlap_fraction"
            ] != 0
        )
    ]

    if not invalid_non_ictal.empty:
        errors.append(
            "Non-ictal windows with seizure overlap."
        )

    invalid_ictal = windows.loc[
        (windows["label_name"] == "ictal")
        & (
            windows[
                "seizure_overlap_fraction"
            ] < threshold
        )
    ]

    if not invalid_ictal.empty:
        errors.append(
            "Ictal windows below overlap threshold."
        )

    invalid_boundary = windows.loc[
        (windows["label_name"] == "boundary")
        & (
            (
                windows[
                    "seizure_overlap_fraction"
                ] <= 0
            )
            | (
                windows[
                    "seizure_overlap_fraction"
                ] >= threshold
            )
        )
    ]

    if not invalid_boundary.empty:
        errors.append(
            "Boundary windows have invalid overlaps."
        )

    # 9. Binary-label consistency.
    invalid_binary_non_ictal = windows.loc[
        (windows["label_name"] == "non_ictal")
        & (
            windows["binary_label"] != 0
        )
    ]

    invalid_binary_ictal = windows.loc[
        (windows["label_name"] == "ictal")
        & (
            windows["binary_label"] != 1
        )
    ]

    boundary_binary_values = windows.loc[
        windows["label_name"] == "boundary",
        "binary_label",
    ]

    if not invalid_binary_non_ictal.empty:
        errors.append(
            "Invalid non-ictal binary labels."
        )

    if not invalid_binary_ictal.empty:
        errors.append(
            "Invalid ictal binary labels."
        )

    if boundary_binary_values.notna().any():
        errors.append(
            "Boundary windows must have missing "
            "binary labels."
        )

    # 10. Stride consistency within recordings.
    for recording_id, group in windows.groupby(
        "recording_id"
    ):
        ordered = group.sort_values(
            "window_index"
        )

        starts = ordered[
            "start_sample"
        ].to_numpy()

        if len(starts) > 1:
            differences = np.diff(starts)

            if not np.all(
                differences == expected_stride
            ):
                errors.append(
                    f"Inconsistent stride in "
                    f"{recording_id}."
                )

    # 11. Window stays within each FIF.
    merged = windows.merge(
        preprocessing_manifest[
            [
                "recording_id",
                "output_n_times",
            ]
        ],
        on="recording_id",
        how="left",
        validate="many_to_one",
    )

    outside = merged.loc[
        merged["stop_sample_exclusive"]
        > merged["output_n_times"]
    ]

    if not outside.empty:
        errors.append(
            "Windows extend beyond preprocessed FIF."
        )

    # 12. Reconcile summary counts.
    recalculated = (
        windows.groupby("recording_id")
        .agg(
            recalculated_window_count=(
                "window_id",
                "count",
            ),
            recalculated_non_ictal=(
                "label_name",
                lambda x: int(
                    (x == "non_ictal").sum()
                ),
            ),
            recalculated_boundary=(
                "label_name",
                lambda x: int(
                    (x == "boundary").sum()
                ),
            ),
            recalculated_ictal=(
                "label_name",
                lambda x: int(
                    (x == "ictal").sum()
                ),
            ),
        )
        .reset_index()
    )

    summary_check = recording_summary.merge(
        recalculated,
        on="recording_id",
        how="left",
        validate="one_to_one",
    )

    count_checks = {
        "window_count": (
            "recalculated_window_count"
        ),
        "non_ictal_window_count": (
            "recalculated_non_ictal"
        ),
        "boundary_window_count": (
            "recalculated_boundary"
        ),
        "ictal_window_count": (
            "recalculated_ictal"
        ),
    }

    for expected_column, observed_column in (
        count_checks.items()
    ):
        mismatch = (
            summary_check[expected_column]
            != summary_check[observed_column]
        )

        if mismatch.any():
            errors.append(
                f"Summary mismatch for "
                f"{expected_column}."
            )

    # 13. Every seizure should overlap at least one
    # generated window, even if it only creates boundary
    # windows.
    seizure_coverage_rows: list[dict] = []

    for seizure in seizures.itertuples(
        index=False
    ):
        recording_windows = windows.loc[
            windows["recording_id"]
            == seizure.recording_id
        ]

        overlapping = recording_windows.loc[
            (
                recording_windows[
                    "start_seconds"
                ]
                < seizure.offset_seconds
            )
            & (
                recording_windows[
                    "end_seconds"
                ]
                > seizure.onset_seconds
            )
        ]

        seizure_coverage_rows.append(
            {
                "case_id": seizure.case_id,
                "recording_id": (
                    seizure.recording_id
                ),
                "seizure_id": (
                    seizure.seizure_id
                ),
                "seizure_duration_seconds": (
                    seizure.duration_seconds
                ),
                "overlapping_window_count": (
                    len(overlapping)
                ),
                "ictal_window_count": int(
                    (
                        overlapping[
                            "label_name"
                        ]
                        == "ictal"
                    ).sum()
                ),
                "boundary_window_count": int(
                    (
                        overlapping[
                            "label_name"
                        ]
                        == "boundary"
                    ).sum()
                ),
                "has_any_overlapping_window": (
                    len(overlapping) > 0
                ),
                "has_any_ictal_window": bool(
                    (
                        overlapping[
                            "label_name"
                        ]
                        == "ictal"
                    ).any()
                ),
            }
        )

    seizure_coverage = pd.DataFrame(
        seizure_coverage_rows
    )

    no_overlap_seizures = seizure_coverage.loc[
        ~seizure_coverage[
            "has_any_overlapping_window"
        ]
    ]

    if not no_overlap_seizures.empty:
        errors.append(
            f"{len(no_overlap_seizures)} seizures "
            "overlap no generated window."
        )

    no_ictal_windows = seizure_coverage.loc[
        ~seizure_coverage[
            "has_any_ictal_window"
        ]
    ]

    if not no_ictal_windows.empty:
        warnings.append(
            f"{len(no_ictal_windows)} seizures create "
            "no ictal window at the current 50% "
            "overlap threshold."
        )

    seizure_coverage.to_csv(
        metadata_dir
        / "chbmit_seizure_window_coverage.csv",
        index=False,
    )

    # 14. Read a deterministic validation sample.
    sample_size = min(
        50,
        len(windows),
    )

    sampled_windows = windows.sample(
        n=sample_size,
        random_state=42,
    )

    for row_index, row in (
        sampled_windows.iterrows()
    ):
        try:
            data, channels, sfreq = (
                load_window_from_manifest_row(
                    row=row,
                    project_root=PROJECT_ROOT,
                    preload_raw=False,
                )
            )

            expected_shape = (
                int(row["channel_count"]),
                expected_samples,
            )

            shape_valid = (
                data.shape == expected_shape
            )

            finite_valid = bool(
                np.isfinite(data).all()
            )

            channel_count_valid = (
                len(channels)
                == int(row["channel_count"])
            )

            validation_rows.append(
                {
                    "window_id": (
                        row["window_id"]
                    ),
                    "shape_valid": shape_valid,
                    "finite_valid": finite_valid,
                    "channel_count_valid": (
                        channel_count_valid
                    ),
                    "sampling_rate_valid": (
                        np.isclose(
                            sfreq,
                            expected_sfreq,
                        )
                    ),
                    "validation_status": (
                        "valid"
                        if (
                            shape_valid
                            and finite_valid
                            and channel_count_valid
                            and np.isclose(
                                sfreq,
                                expected_sfreq,
                            )
                        )
                        else "error"
                    ),
                }
            )

        except Exception as exc:
            validation_rows.append(
                {
                    "window_id": (
                        row["window_id"]
                    ),
                    "shape_valid": False,
                    "finite_valid": False,
                    "channel_count_valid": False,
                    "sampling_rate_valid": False,
                    "validation_status": (
                        f"error:{type(exc).__name__}:"
                        f"{exc}"
                    ),
                }
            )

    validation = pd.DataFrame(
        validation_rows
    )

    invalid_samples = validation.loc[
        validation["validation_status"]
        != "valid"
    ]

    if not invalid_samples.empty:
        errors.append(
            f"{len(invalid_samples)} sampled windows "
            "failed signal loading validation."
        )

    if not failures.empty:
        errors.append(
            f"{len(failures)} segmentation failures "
            "exist."
        )

    validation.to_csv(
        metadata_dir
        / "chbmit_segmentation_validation.csv",
        index=False,
    )

    print("Segmentation validation")
    print("Errors:", len(errors))
    print("Warnings:", len(warnings))

    for error in errors:
        print("ERROR:", error)

    for warning in warnings:
        print("WARNING:", warning)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())