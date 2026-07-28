"""Signal-quality metrics for multichannel EEG chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for Welch-based QC spectral metrics."""

    fmin_hz: float = 0.5
    fmax_hz: float = 100.0
    nperseg_seconds: float = 4.0
    overlap_fraction: float = 0.5
    line_frequency_hz: float = 60.0
    line_band_half_width_hz: float = 1.0
    line_reference_band_half_width_hz: float = 5.0
    high_frequency_start_hz: float = 40.0
    high_frequency_end_hz: float = 100.0


def safe_fraction(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Safely calculate a fraction."""
    if denominator <= 0:
        return float("nan")

    return float(numerator / denominator)


def longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive run of True values."""
    boolean_values = np.asarray(values, dtype=bool)

    if boolean_values.size == 0:
        return 0

    padded = np.concatenate(
        [
            np.array([False]),
            boolean_values,
            np.array([False]),
        ]
    )

    changes = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(changes == 1)
    run_ends = np.flatnonzero(changes == -1)

    if len(run_starts) == 0:
        return 0

    return int(np.max(run_ends - run_starts))


def calculate_flatline_metrics(
    signal_uv: np.ndarray,
    sampling_rate_hz: float,
    difference_tolerance_uv: float,
    minimum_run_seconds: float,
) -> dict[str, float]:
    """Calculate flatline-related metrics for one channel."""
    signal = np.asarray(signal_uv, dtype=np.float64)

    finite_mask = np.isfinite(signal)
    finite_signal = signal[finite_mask]

    if finite_signal.size < 2:
        return {
            "flat_difference_fraction": float("nan"),
            "longest_flat_run_seconds": float("nan"),
            "flatline_sample_fraction": float("nan"),
        }

    differences = np.abs(np.diff(finite_signal))
    flat_differences = differences <= difference_tolerance_uv

    longest_run_differences = longest_true_run(flat_differences)

    # A run of N equal differences corresponds approximately to N+1 samples.
    longest_run_samples = (
        longest_run_differences + 1
        if longest_run_differences > 0
        else 0
    )

    minimum_run_samples = max(
        2,
        int(round(minimum_run_seconds * sampling_rate_hz)),
    )

    flatline_sample_count = 0

    if longest_run_samples >= minimum_run_samples:
        # Count all flat-difference runs meeting the duration criterion.
        padded = np.concatenate(
            [
                np.array([False]),
                flat_differences,
                np.array([False]),
            ]
        )
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)

        for start, end in zip(starts, ends, strict=True):
            run_difference_count = end - start
            run_sample_count = run_difference_count + 1

            if run_sample_count >= minimum_run_samples:
                flatline_sample_count += run_sample_count

    return {
        "flat_difference_fraction": safe_fraction(
            int(flat_differences.sum()),
            len(flat_differences),
        ),
        "longest_flat_run_seconds": (
            longest_run_samples / sampling_rate_hz
        ),
        "flatline_sample_fraction": safe_fraction(
            flatline_sample_count,
            len(finite_signal),
        ),
    }


def integrate_band_power(
    frequencies: np.ndarray,
    power_spectral_density: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> float:
    """Integrate PSD within a frequency interval."""
    mask = (
        (frequencies >= lower_hz)
        & (frequencies <= upper_hz)
    )

    if mask.sum() < 2:
        return float("nan")

    return float(
        np.trapezoid(
            power_spectral_density[mask],
            frequencies[mask],
        )
    )


def calculate_spectral_metrics(
    signal_uv: np.ndarray,
    sampling_rate_hz: float,
    config: SpectralConfig,
) -> dict[str, float]:
    """Calculate Welch-based spectral QC metrics."""
    signal = np.asarray(signal_uv, dtype=np.float64)
    signal = signal[np.isfinite(signal)]

    minimum_samples = int(
        max(2, config.nperseg_seconds * sampling_rate_hz)
    )

    if len(signal) < minimum_samples:
        return {
            "total_power_uv2": float("nan"),
            "line_noise_power_uv2": float("nan"),
            "line_reference_power_uv2": float("nan"),
            "line_noise_ratio": float("nan"),
            "high_frequency_power_uv2": float("nan"),
            "high_frequency_ratio": float("nan"),
            "spectral_entropy": float("nan"),
        }

    nyquist_hz = sampling_rate_hz / 2
    effective_fmax = min(config.fmax_hz, nyquist_hz)

    nperseg = min(
        len(signal),
        int(round(config.nperseg_seconds * sampling_rate_hz)),
    )
    noverlap = int(round(nperseg * config.overlap_fraction))
    noverlap = min(noverlap, nperseg - 1)

    frequencies, psd = welch(
        signal,
        fs=sampling_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        average="median",
    )

    analysis_mask = (
        (frequencies >= config.fmin_hz)
        & (frequencies <= effective_fmax)
    )

    frequencies = frequencies[analysis_mask]
    psd = psd[analysis_mask]

    if len(frequencies) < 2:
        return {
            "total_power_uv2": float("nan"),
            "line_noise_power_uv2": float("nan"),
            "line_reference_power_uv2": float("nan"),
            "line_noise_ratio": float("nan"),
            "high_frequency_power_uv2": float("nan"),
            "high_frequency_ratio": float("nan"),
            "spectral_entropy": float("nan"),
        }

    total_power = float(np.trapezoid(psd, frequencies))

    line_lower = (
        config.line_frequency_hz
        - config.line_band_half_width_hz
    )
    line_upper = (
        config.line_frequency_hz
        + config.line_band_half_width_hz
    )

    reference_lower = (
        config.line_frequency_hz
        - config.line_reference_band_half_width_hz
    )
    reference_upper = (
        config.line_frequency_hz
        + config.line_reference_band_half_width_hz
    )

    line_power = integrate_band_power(
        frequencies,
        psd,
        line_lower,
        line_upper,
    )

    reference_power = integrate_band_power(
        frequencies,
        psd,
        reference_lower,
        reference_upper,
    )

    high_frequency_upper = min(
        config.high_frequency_end_hz,
        effective_fmax,
    )

    high_frequency_power = integrate_band_power(
        frequencies,
        psd,
        config.high_frequency_start_hz,
        high_frequency_upper,
    )

    positive_psd = np.clip(psd, a_min=0.0, a_max=None)
    psd_sum = float(positive_psd.sum())

    if psd_sum > 0:
        probabilities = positive_psd / psd_sum
        probabilities = probabilities[probabilities > 0]
        spectral_entropy = float(
            -np.sum(probabilities * np.log(probabilities))
            / np.log(len(positive_psd))
        )
    else:
        spectral_entropy = float("nan")

    return {
        "total_power_uv2": total_power,
        "line_noise_power_uv2": line_power,
        "line_reference_power_uv2": reference_power,
        "line_noise_ratio": (
            line_power / reference_power
            if np.isfinite(reference_power)
            and reference_power > 0
            else float("nan")
        ),
        "high_frequency_power_uv2": high_frequency_power,
        "high_frequency_ratio": (
            high_frequency_power / total_power
            if np.isfinite(total_power)
            and total_power > 0
            else float("nan")
        ),
        "spectral_entropy": spectral_entropy,
    }


def calculate_channel_metrics(
    signal_uv: np.ndarray,
    sampling_rate_hz: float,
    difference_tolerance_uv: float,
    minimum_flat_run_seconds: float,
    spectral_config: SpectralConfig,
) -> dict[str, float]:
    """Calculate QC metrics for one channel and one chunk."""
    signal = np.asarray(signal_uv, dtype=np.float64)
    sample_count = len(signal)

    finite_mask = np.isfinite(signal)
    finite_signal = signal[finite_mask]

    nonfinite_count = int((~finite_mask).sum())
    nan_count = int(np.isnan(signal).sum())
    positive_infinity_count = int(np.isposinf(signal).sum())
    negative_infinity_count = int(np.isneginf(signal).sum())

    if finite_signal.size == 0:
        return {
            "sample_count": sample_count,
            "finite_sample_count": 0,
            "nonfinite_fraction": 1.0,
            "nan_fraction": safe_fraction(nan_count, sample_count),
            "positive_infinity_fraction": safe_fraction(
                positive_infinity_count,
                sample_count,
            ),
            "negative_infinity_fraction": safe_fraction(
                negative_infinity_count,
                sample_count,
            ),
        }

    median = float(np.median(finite_signal))
    mean = float(np.mean(finite_signal))
    standard_deviation = float(np.std(finite_signal))
    variance = float(np.var(finite_signal))
    rms = float(np.sqrt(np.mean(finite_signal**2)))

    q01, q05, q25, q75, q95, q99 = np.quantile(
        finite_signal,
        [0.01, 0.05, 0.25, 0.75, 0.95, 0.99],
    )

    mad = float(
        np.median(
            np.abs(finite_signal - median)
        )
    )

    differences = np.diff(finite_signal)

    metrics: dict[str, float] = {
        "sample_count": sample_count,
        "finite_sample_count": len(finite_signal),
        "nonfinite_fraction": safe_fraction(
            nonfinite_count,
            sample_count,
        ),
        "nan_fraction": safe_fraction(
            nan_count,
            sample_count,
        ),
        "positive_infinity_fraction": safe_fraction(
            positive_infinity_count,
            sample_count,
        ),
        "negative_infinity_fraction": safe_fraction(
            negative_infinity_count,
            sample_count,
        ),
        "zero_fraction": float(
            np.mean(finite_signal == 0)
        ),
        "mean_uv": mean,
        "median_uv": median,
        "std_uv": standard_deviation,
        "variance_uv2": variance,
        "rms_uv": rms,
        "minimum_uv": float(np.min(finite_signal)),
        "maximum_uv": float(np.max(finite_signal)),
        "peak_to_peak_uv": float(np.ptp(finite_signal)),
        "q01_uv": float(q01),
        "q05_uv": float(q05),
        "q25_uv": float(q25),
        "q75_uv": float(q75),
        "q95_uv": float(q95),
        "q99_uv": float(q99),
        "iqr_uv": float(q75 - q25),
        "robust_range_01_99_uv": float(q99 - q01),
        "mad_uv": mad,
        "skewness": float(
            skew(
                finite_signal,
                bias=False,
                nan_policy="omit",
            )
        ),
        "kurtosis_fisher": float(
            kurtosis(
                finite_signal,
                fisher=True,
                bias=False,
                nan_policy="omit",
            )
        ),
        "mean_absolute_first_difference_uv": (
            float(np.mean(np.abs(differences)))
            if differences.size
            else float("nan")
        ),
        "line_length_uv": (
            float(np.sum(np.abs(differences)))
            if differences.size
            else float("nan")
        ),
        "maximum_absolute_first_difference_uv": (
            float(np.max(np.abs(differences)))
            if differences.size
            else float("nan")
        ),
    }

    metrics.update(
        calculate_flatline_metrics(
            signal_uv=finite_signal,
            sampling_rate_hz=sampling_rate_hz,
            difference_tolerance_uv=difference_tolerance_uv,
            minimum_run_seconds=minimum_flat_run_seconds,
        )
    )

    metrics.update(
        calculate_spectral_metrics(
            signal_uv=finite_signal,
            sampling_rate_hz=sampling_rate_hz,
            config=spectral_config,
        )
    )

    return metrics