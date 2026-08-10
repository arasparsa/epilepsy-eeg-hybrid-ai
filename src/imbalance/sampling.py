"""Training-only deterministic negative-window sampling."""

from __future__ import annotations

import math

import pandas as pd


def deterministic_negative_sampling(
    windows: pd.DataFrame,
    *,
    negative_to_positive_ratio: float,
    use_clean_non_ictal_only: bool,
    minimum_negative_windows_per_subject: int,
    random_state: int,
) -> pd.DataFrame:
    """Keep all positives and sample training negatives only."""
    if negative_to_positive_ratio <= 0:
        raise ValueError(
            "negative_to_positive_ratio "
            "must be positive."
        )

    positives = windows.loc[
        windows["binary_label"] == 1
    ].copy()

    negatives = windows.loc[
        windows["binary_label"] == 0
    ].copy()

    if use_clean_non_ictal_only:
        negatives = negatives.loc[
            negatives[
                "clean_non_ictal"
            ].astype(bool)
        ].copy()

    if positives.empty:
        raise ValueError(
            "No positive windows available."
        )

    if negatives.empty:
        raise ValueError(
            "No eligible negative windows."
        )

    target_negative_count = min(
        len(negatives),
        int(
            math.ceil(
                len(positives)
                * negative_to_positive_ratio
            )
        ),
    )

    selected_parts: list[
        pd.DataFrame
    ] = []

    already_selected_ids: set[str] = set()

    # Ensure each training subject contributes negatives.
    for subject_index, (
        subject_id,
        subject_negatives,
    ) in enumerate(
        negatives.groupby(
            "subject_id",
            sort=True,
        )
    ):
        sample_count = min(
            len(subject_negatives),
            minimum_negative_windows_per_subject,
        )

        sampled = (
            subject_negatives.sample(
                n=sample_count,
                replace=False,
                random_state=(
                    random_state
                    + subject_index
                ),
            )
        )

        selected_parts.append(sampled)

        already_selected_ids.update(
            sampled["window_id"]
        )

    selected_negative_count = sum(
        len(part)
        for part in selected_parts
    )

    remaining_target = max(
        0,
        target_negative_count
        - selected_negative_count,
    )

    remaining_pool = negatives.loc[
        ~negatives["window_id"].isin(
            already_selected_ids
        )
    ]

    if (
        remaining_target > 0
        and not remaining_pool.empty
    ):
        additional = remaining_pool.sample(
            n=min(
                remaining_target,
                len(remaining_pool),
            ),
            replace=False,
            random_state=(
                random_state + 100000
            ),
        )

        selected_parts.append(
            additional
        )

    selected_negatives = pd.concat(
        selected_parts,
        ignore_index=True,
    ).drop_duplicates(
        subset=["window_id"]
    )

    # The per-subject minimum can exceed the target.
    if (
        len(selected_negatives)
        > target_negative_count
    ):
        selected_negatives = (
            selected_negatives.sample(
                n=target_negative_count,
                replace=False,
                random_state=(
                    random_state + 200000
                ),
            )
        )

    selected = pd.concat(
        [
            positives,
            selected_negatives,
        ],
        ignore_index=True,
    )

    selected = selected.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    selected[
        "sampling_strategy"
    ] = (
        "deterministic_negative_sampling"
    )

    return selected