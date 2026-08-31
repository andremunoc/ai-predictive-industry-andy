import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from pathlib import Path

# --- Paths and Constants ---
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJ_ROOT / "data" / "raw" / "2nd_test" / "2nd_test"
OUTPUT_CSV = PROJ_ROOT / "data" / "processed" / "bearing_features_preprocessing.csv"


# --- File List Generation ---
file_list = sorted([f for f in glob.glob(str(DATA_DIR / "*")) if os.path.isfile(f)])

# Discard the last 4 files (corrupted/null files at the end)
file_list = file_list[:-4]

print(f"Total number of files found: {len(file_list)}")


def extract_window_features(signal):
    """
    Calculate RMS (Root Mean Square), Kurtosis, and Skewness
    for a given 1D numpy array signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array of vibration signal samples.

    Returns
    -------
    rms_val : float
        Root Mean Square of the signal.
    kurt_val : float
        Kurtosis of the signal.
    skew_val : float
        Skewness of the signal.
    """
    rms_val = np.sqrt(np.mean(signal**2))
    kurt_val = kurtosis(signal)
    skew_val = skew(signal)
    return rms_val, kurt_val, skew_val


# --- Main Processing Loop ---
features_list = []

for idx, file_path in enumerate(file_list):
    filename = os.path.basename(file_path)

    df_raw = pd.read_csv(file_path, sep='\t', header=None)
    df_raw.columns = ['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4']

    b1_signal = df_raw['Bearing1'].values

    rms, kurt, skew_stat = extract_window_features(b1_signal)

    # --- Labeling Logic ---
    if idx < 600:
        label = 'Normal'
    elif idx < 800:
        label = 'Wear'
    else:
        label = 'Failure'

    features_list.append({
        'file_id': idx,
        'filename': filename,
        'b1_rms': rms,
        'b1_kurtosis': kurt,
        'b1_skewness': skew_stat,
        'label': label
    })

# --- Save to CSV ---
df_features = pd.DataFrame(features_list)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_features.to_csv(OUTPUT_CSV, index=False)

# --- Final Prints ---
print("\nPreprocessing completed successfully!")
print(f"Final table dimensions: {df_features.shape}")
print(df_features.head())
