"""
Station 4: Frequency Domain Signal Processing
----------------------------------------------
Computes FFT-based features from raw bearing vibration signals and merges
them with the time-domain features produced by Station 3.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR        = PROJ_ROOT / "data" / "raw" / "2nd_test" / "2nd_test"
PROCESSED_CSV_PATH  = PROJ_ROOT / "data" / "processed" / "bearing_features_preprocessing.csv"
OUTPUT_CSV_PATH     = PROJ_ROOT / "data" / "processed" / "final_bearing_features_processing.csv"
PLOTS_DIR           = PROJ_ROOT / "docs"

# Ensure output directories exist
(PROJ_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Sampling frequency (Hz)
FS = 20480


# ---------------------------------------------------------------------------
# Feature Extraction: FFT
# ---------------------------------------------------------------------------
def compute_fft_features(signal, fs=FS, top_n=5):
    """
    Compute frequency-domain features from a 1D vibration signal.

    Steps:
      1. Remove DC component (subtract mean).
      2. Compute single-sided FFT magnitude spectrum.
      3. Exclude frequencies below 5 Hz (structural noise).
      4. Detect spectral peaks and select the top N by amplitude.
      5. Compute spectral centroid and spectral energy.

    Parameters
    ----------
    signal : np.ndarray
        1D array of vibration samples.
    fs : int
        Sampling frequency in Hz.
    top_n : int
        Number of dominant peaks to extract.

    Returns
    -------
    features : dict
        Dictionary of extracted frequency-domain features.
    freqs : np.ndarray
        Frequency axis (Hz).
    magnitude : np.ndarray
        Single-sided magnitude spectrum.
    """
    n = len(signal)

    # Remove DC component before FFT
    signal_ac = signal - np.mean(signal)

    # Compute FFT and scale to single-sided real magnitude
    fft_vals = rfft(signal_ac)
    freqs = rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(fft_vals) * (2.0 / n)

    # Exclude low frequencies (< 5 Hz) to avoid structural noise
    valid_mask = freqs >= 5.0
    freqs_valid = freqs[valid_mask]
    mag_valid = magnitude[valid_mask]

    # Detect spectral peaks with minimum distance of ~10 Hz
    min_distance = int(fs * 0.005)
    peak_indices, _ = find_peaks(mag_valid, distance=min_distance)

    # Sort peaks by amplitude (descending) and keep top N
    if len(peak_indices) > 0:
        sorted_idx = peak_indices[np.argsort(mag_valid[peak_indices])[::-1]]
        top_idx = sorted_idx[:top_n]
        top_freqs = freqs_valid[top_idx]
        top_amps  = mag_valid[top_idx]
        # Dominant peak (highest amplitude)
        peak_freq_hz = float(top_freqs[0])
    else:
        top_freqs = np.array([])
        top_amps  = np.array([])
        peak_freq_hz = 0.0

    # Pad to top_n if fewer peaks were found
    pad_len = top_n - len(top_freqs)
    top_freqs = np.concatenate([top_freqs, np.zeros(pad_len)])
    top_amps  = np.concatenate([top_amps,  np.zeros(pad_len)])

    # Spectral centroid: amplitude-weighted mean frequency
    total_mag = np.sum(mag_valid)
    spectral_centroid = float(
        np.sum(freqs_valid * mag_valid) / total_mag if total_mag > 0 else 0.0
    )

    # Spectral energy: sum of squared magnitudes
    spectral_energy = float(np.sum(mag_valid ** 2))

    # Build feature dictionary
    features = {
        "peak_freq_hz":      peak_freq_hz,
        "spectral_centroid": spectral_centroid,
        "spectral_energy":   spectral_energy,
    }
    for i in range(top_n):
        features[f"top_freq_{i + 1}_hz"] = float(top_freqs[i])
        features[f"top_amp_{i + 1}"]     = float(top_amps[i])

    return features, freqs, magnitude


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
def process_all_files(raw_dir):
    """
    Process all raw bearing files in raw_dir and extract FFT features.

    The last 4 files are excluded because they are corrupted/null.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the raw tab-separated signal files.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per file and FFT-based feature columns.
    """
    file_list = sorted(
        [f for f in glob.glob(str(raw_dir / "*")) if os.path.isfile(f)]
    )

    # Exclude the last 4 corrupted/null files
    file_list = file_list[:-4]

    print(f"Processing {len(file_list)} files for frequency-domain features...")

    records = []
    for file_path in file_list:
        filename = os.path.basename(file_path)
        data = np.loadtxt(file_path)
        signal = data[:, 0]  # First column (Bearing 1)

        features, _, _ = compute_fft_features(signal)
        features["filename"] = filename
        records.append(features)

    df_freq = pd.DataFrame(records)
    print(f"Frequency-domain feature extraction complete. Shape: {df_freq.shape}")
    return df_freq


# ---------------------------------------------------------------------------
# Comparison Plot: Normal vs Fault
# ---------------------------------------------------------------------------
def plot_fft_comparison(normal_signal, fault_signal, fs=FS, save_path=None):
    """
    Plot the FFT magnitude spectrum for a Normal and a Fault signal side by side.

    Parameters
    ----------
    normal_signal : np.ndarray
        1D vibration signal from a normal operating condition.
    fault_signal : np.ndarray
        1D vibration signal from a fault/failure condition.
    fs : int
        Sampling frequency in Hz.
    save_path : str or Path, optional
        If provided, the figure is saved to this path at 300 DPI.
    """
    _, freqs_n, mag_n = compute_fft_features(normal_signal, fs=fs)
    _, freqs_f, mag_f = compute_fft_features(fault_signal,  fs=fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)  # type: ignore

    fig.suptitle(
        "FFT Spectrum Comparison: Normal vs Fault (Bearing 1)",
        fontsize=15,
        fontweight="bold",
    )

    # Normal state
    ax1.plot(freqs_n, mag_n, color="#2ecc71", linewidth=0.8)
    ax1.set_title("Normal State", fontsize=12)
    ax1.set_ylabel("Amplitude (g)", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, fs / 2)
    ax1.grid(True)

    # Fault state
    ax2.plot(freqs_f, mag_f, color="#e74c3c", linewidth=0.8)
    ax2.set_title("Fault / Failure State", fontsize=12)
    ax2.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Amplitude (g)", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, fs / 2)
    ax2.grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Main Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Step 1: Load time-domain features from Station 3 ---
    if PROCESSED_CSV_PATH.exists():
        df_time = pd.read_csv(PROCESSED_CSV_PATH)
        print(f"Time-domain dataset loaded. Shape: {df_time.shape}")
    else:
        print(
            f"Warning: Time-domain CSV not found at {PROCESSED_CSV_PATH}. "
            "Proceeding without merging."
        )
        df_time = None

    # --- Step 2: Extract frequency-domain features ---
    df_freq = process_all_files(RAW_DATA_DIR)

    # --- Step 3: Merge time-domain and frequency-domain features ---
    if df_time is not None:
        df_final = pd.merge(df_time, df_freq, on="filename", how="inner")
        print(f"Merged dataset shape: {df_final.shape}")
    else:
        df_final = df_freq

    # --- Step 4: Save final merged dataset ---
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nFinal merged feature table saved to: {OUTPUT_CSV_PATH}")
    print(f"Final table dimensions: {df_final.shape}")
    print(df_final.head())
