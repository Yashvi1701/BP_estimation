import os
import glob

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter

def extract_nonfiducial_features(signal):

    signal = np.asarray(signal, dtype=np.float64)

    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        return {}

    # -----------------------------
    # Basic statistical features
    # -----------------------------

    mean_val = np.mean(signal)
    median_val = np.median(signal)
    std_val = np.std(signal)
    variance_val = np.var(signal)

    iqr_val = (
        np.percentile(signal, 75)
        - np.percentile(signal, 25)
    )

    skewness_val = skew(signal)
    kurtosis_val = kurtosis(signal)

    # -----------------------------
    # Zero crossing rate
    # -----------------------------

    zero_crossings = np.sum(
        signal[:-1] * signal[1:] < 0
    )

    zero_crossing_rate = (
        zero_crossings / (len(signal) - 1)
    )

    # -----------------------------
    # Shannon entropy
    # -----------------------------

    hist, _ = np.histogram(
        signal,
        bins=50
    )

    probabilities = hist / np.sum(hist)

    probabilities = probabilities[
        probabilities > 0
    ]

    shannon_entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    # -----------------------------
    # Energy
    # -----------------------------

    energy = signal ** 2

    energy_mean = np.mean(energy)
    energy_variance = np.var(energy)
    energy_skewness = skew(energy)
    energy_kurtosis = kurtosis(energy)

    energy_iqr = (
        np.percentile(energy, 75)
        - np.percentile(energy, 25)
    )

    # -----------------------------
    # Kaiser-Teager Energy
    # -----------------------------

    if len(signal) >= 3:

        kte = (
            signal[1:-1] ** 2
            - signal[:-2] * signal[2:]
        )

        kte_mean = np.mean(kte)
        kte_variance = np.var(kte)
        kte_skewness = skew(kte)
        kte_kurtosis = kurtosis(kte)

        kte_iqr = (
            np.percentile(kte, 75)
            - np.percentile(kte, 25)
        )

    else:

        kte_mean = np.nan
        kte_variance = np.nan
        kte_skewness = np.nan
        kte_kurtosis = np.nan
        kte_iqr = np.nan

    return {

        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "variance": variance_val,
        "iqr": iqr_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,

        "zero_crossing_rate": zero_crossing_rate,
        "shannon_entropy": shannon_entropy,

        "energy_mean": energy_mean,
        "energy_variance": energy_variance,
        "energy_skewness": energy_skewness,
        "energy_kurtosis": energy_kurtosis,
        "energy_iqr": energy_iqr,

        "kte_mean": kte_mean,
        "kte_variance": kte_variance,
        "kte_skewness": kte_skewness,
        "kte_kurtosis": kte_kurtosis,
        "kte_iqr": kte_iqr
    }


def extract_all_features(ppg, fs=125):

    ppg = np.asarray(ppg, dtype=np.float64)

    # =========================================
    # 1. Original PPG
    # =========================================

    ppg_features = extract_nonfiducial_features(ppg)

    # =========================================
    # 2. VPG - First derivative
    # =========================================

    vpg = np.gradient(ppg)

    # Smooth VPG
    vpg = savgol_filter(
        vpg,
        window_length=11,
        polyorder=3
    )

    # =========================================
    # 3. APG - Second derivative
    # =========================================

    apg = np.gradient(vpg)

    # Smooth APG
    apg = savgol_filter(
        apg,
        window_length=11,
        polyorder=3
    )

    # =========================================
    # 4. Extract features
    # =========================================

    vpg_features = extract_nonfiducial_features(vpg)

    apg_features = extract_nonfiducial_features(apg)

    # =========================================
    # 5. Combine all 57 features
    # =========================================

    features = {}

    for name, value in ppg_features.items():
        features["ppg_" + name] = value

    for name, value in vpg_features.items():
        features["vpg_" + name] = value

    for name, value in apg_features.items():
        features["apg_" + name] = value

    return features


from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np


def extract_features_single_sample(i, X, y, fs=125):

    # VitalDB X shape = (N, 1000)
    ppg = X[i, :]

    features = extract_all_features(
        ppg,
        fs
    )

    features["SBP"] = y[i, 0]
    features["DBP"] = y[i, 1]

    return features


def extract_features_from_dataset(
    X,
    y,
    fs=125,
    n_jobs=2
):

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=0
    )(
        delayed(extract_features_single_sample)(
            i, X, y, fs
        )
        for i in range(len(X))
    )

    return pd.DataFrame(results)