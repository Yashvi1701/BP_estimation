import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

# from skrebate import RReliefF

from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")


def extract_bp_from_abp(abp_window):

    # Detect systolic peaks
    peaks, _ = find_peaks(
        abp_window,
        distance=50,       # ~0.4 sec at 125 Hz
        prominence=5
    )


    # Need enough beats in the window
    if len(peaks) < 2:
        return None, None


    # SBP = pressure at systolic peaks
    sbp_values = abp_window[peaks]


    dbp_values = []


    # Find minimum pressure between consecutive systolic peaks
    for i in range(len(peaks)-1):

        beat_segment = abp_window[
            peaks[i]:peaks[i+1]
        ]

        if len(beat_segment) > 0:

            dbp_values.append(
                np.min(beat_segment)
            )


    if len(dbp_values) == 0:
        return None, None


    # Use median to reduce effect of noisy beats
    sbp = np.median(sbp_values)
    dbp = np.median(dbp_values)


    # Optional sanity check
    if sbp < 50 or sbp > 250:
        return None, None

    if dbp < 20 or dbp > 150:
        return None, None


    return sbp, dbp

def process_recording(recording, WINDOW_SIZE, STEP_SIZE ):

    data = recording[:]

    ppg = data[:,0]
    abp = data[:,1]

    X = []
    y = []

    for start in range(
        0,
        len(ppg)-WINDOW_SIZE+1,
        STEP_SIZE
    ):

        ppg_window = ppg[start:start+WINDOW_SIZE]

        abp_window = abp[start:start+WINDOW_SIZE]

        sbp, dbp = extract_bp_from_abp(abp_window)

        if sbp is None:
            continue

        X.append(ppg_window)
        y.append([sbp,dbp])

    if len(X)==0:
        return None,None

    return np.array(X),np.array(y)




def process_split(record_list, WINDOW_SIZE, STEP_SIZE):

    X_all = []
    y_all = []

    skipped = 0

    for f, ref in tqdm(record_list):

        recording = f[ref]

        X, y = process_recording(recording, WINDOW_SIZE, STEP_SIZE)

        if X is None:
            skipped += 1
            continue

        X_all.append(X)
        y_all.append(y)

    X_all = np.concatenate(X_all)

    y_all = np.concatenate(y_all)

    print("Skipped recordings:", skipped)

    return X_all, y_all



def load_UCI_dataset(
    WINDOW_SIZE, 
    STEP_SIZE,
    data_path="/data1/yashvi_bhuva/BP_estimation_using_PPG/UCI/data",
    test_size=0.1,
    val_size=1/9,
    random_state=42,
    
):


    # ============================
    # Load recordings
    # ============================

    recordings = []


    for part in range(1,5):

        file_path = f"{data_path}/Part_{part}.mat"

        f = h5py.File(file_path, "r")

        dataset = f[f"Part_{part}"]


        for i in range(dataset.shape[0]):

            recordings.append(
                (f, dataset[i,0])
            )


    print("Total recordings:", len(recordings))


    # ============================
    # Train-test split
    # ============================

    train_records, test_records = train_test_split(
        recordings,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )


    # ============================
    # Train-validation split
    # ============================

    train_records, val_records = train_test_split(
        train_records,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )


    print(
        f"Train recordings: {len(train_records)}"
    )

    print(
        f"Validation recordings: {len(val_records)}"
    )

    print(
        f"Test recordings: {len(test_records)}"
    )


    # ============================
    # Preprocessing
    # ============================

    X_train, y_train = process_split(train_records,WINDOW_SIZE, 
    STEP_SIZE)

    X_val, y_val = process_split(val_records,WINDOW_SIZE, 
    STEP_SIZE)

    X_test, y_test = process_split(test_records,WINDOW_SIZE, 
    STEP_SIZE)

    # ===========================
    # Add channel dimension
    # CNN input:
    # (batch, channels, samples)
    # ============================

    X_train = X_train[:,None,:]

    X_val = X_val[:,None,:]

    X_test = X_test[:,None,:]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


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

    ppg = X[i, 0, :]

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


def select_f_test(X, y, k=15):

    scores, p_values = f_regression(
        X,
        y
    )

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Highest F-score = most relevant
    indices = np.argsort(scores)[::-1][:k]

    selected_features = X.columns[indices].tolist()

    results = pd.DataFrame({
        "feature": X.columns,
        "F_score": scores,
        "p_value": p_values
    })

    results = results.sort_values(
        "F_score",
        ascending=False
    ).reset_index(drop=True)

    return selected_features, results



import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
# from skrebate import RReliefF


# ============================================================
# mRMR FEATURE SELECTION
# ============================================================

def select_mrmr(
    X,
    y,
    k=15,
    sample_size=50000,
    random_state=42
):
    
    X = X.copy()
    y = np.asarray(y, dtype=np.float64)

    # --------------------------------------------------------
    # 1. Sample training data for feature selection
    # --------------------------------------------------------
    
    rng = np.random.RandomState(random_state)

    n_samples = min(sample_size, len(X))

    sample_indices = rng.choice(
        len(X),
        size=n_samples,
        replace=False
    )

    X_sample = X.iloc[sample_indices]
    y_sample = y[sample_indices]

    print(f"mRMR: using {n_samples} samples for feature selection")

    # Convert to numpy
    X_values = X_sample.values.astype(np.float64)
    y_values = y_sample.astype(np.float64)

    n_features = X_values.shape[1]

    # --------------------------------------------------------
    # 2. Relevance: MI(feature, target)
    # --------------------------------------------------------

    relevance = mutual_info_regression(
        X_values,
        y_values,
        random_state=random_state
    )

    # --------------------------------------------------------
    # 3. MI between every pair of features
    # --------------------------------------------------------

    redundancy = np.zeros(
        (n_features, n_features)
    )

    for i in range(n_features):

        for j in range(i + 1, n_features):

            mi_ij = mutual_info_regression(
                X_values[:, [i]],
                X_values[:, j],
                random_state=random_state
            )[0]

            redundancy[i, j] = mi_ij
            redundancy[j, i] = mi_ij

    # --------------------------------------------------------
    # 4. Greedy mRMR selection
    # --------------------------------------------------------

    selected = []

    # First feature = highest relevance
    first_feature = np.argmax(relevance)

    selected.append(first_feature)

    remaining = set(range(n_features))
    remaining.remove(first_feature)

    while len(selected) < k:

        best_feature = None
        best_score = -np.inf

        for candidate in remaining:

            # Average redundancy with
            # already selected features
            avg_redundancy = np.mean([
                redundancy[candidate, s]
                for s in selected
            ])

            # mRMR score
            score = (
                relevance[candidate]
                - avg_redundancy
            )

            if score > best_score:

                best_score = score
                best_feature = candidate

        selected.append(best_feature)
        remaining.remove(best_feature)

    # --------------------------------------------------------
    # 5. Get selected feature names
    # --------------------------------------------------------

    selected_features = [
        X.columns[i]
        for i in selected
    ]

    # --------------------------------------------------------
    # 6. Results table
    # --------------------------------------------------------

    results = pd.DataFrame({
        "feature": X.columns,
        "relevance_MI": relevance
    })

    results = results.sort_values(
        "relevance_MI",
        ascending=False
    ).reset_index(drop=True)

    return selected_features, results


# ============================================================
# RELIEFF FEATURE SELECTION
# ============================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def select_reliefF(
    X,
    y,
    k=15,
    n_neighbors=100,
    sample_size=50000,
    random_state=42
):
    """
    Regression ReliefF (RReliefF) feature selection.

    Parameters
    ----------
    X : pandas DataFrame
        Training features only.
    y : array-like
        Continuous target (SBP or DBP).
    k : int
        Number of features to select.
    n_neighbors : int
        Number of nearest neighbours.
    sample_size : int
        Number of training samples used for feature selection.
    random_state : int
        Random seed.
    """

    X = X.copy()
    y = np.asarray(y, dtype=np.float64)

    rng = np.random.RandomState(random_state)

    # --------------------------------------------------------
    # Sample training data
    # --------------------------------------------------------

    n_samples = min(sample_size, len(X))

    sample_indices = rng.choice(
        len(X),
        size=n_samples,
        replace=False
    )

    X_sample = X.iloc[sample_indices].values.astype(np.float64)
    y_sample = y[sample_indices]

    print(
        f"RReliefF: using {n_samples} samples "
        f"for feature selection"
    )

    # --------------------------------------------------------
    # Standardize features
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # --------------------------------------------------------
    # Find nearest neighbours
    # --------------------------------------------------------

    n_neighbors = min(n_neighbors + 1, n_samples)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    nn.fit(X_scaled)

    distances, indices = nn.kneighbors(X_scaled)

    # Remove self-neighbour
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    # --------------------------------------------------------
    # Calculate feature scores
    # --------------------------------------------------------

    n_features = X_scaled.shape[1]

    scores = np.zeros(n_features)

    # Normalize target differences
    y_range = np.ptp(y_sample)

    if y_range == 0:
        raise ValueError("Target has zero variance.")

    for i in range(n_samples):

        neighbors = indices[i]

        # Target differences
        target_diff = (
            np.abs(y_sample[i] - y_sample[neighbors])
            / y_range
        )

        # Feature differences
        feature_diff = np.abs(
            X_scaled[i] - X_scaled[neighbors]
        )

        # Weight by target difference
        scores += np.mean(
            target_diff[:, None] * feature_diff,
            axis=0
        )

    scores /= n_samples

    # Higher score = more relevant
    indices_sorted = np.argsort(scores)[::-1]

    selected_indices = indices_sorted[:k]

    selected_features = [
        X.columns[i]
        for i in selected_indices
    ]

    results = pd.DataFrame({
        "feature": X.columns,
        "RReliefF_score": scores
    })

    results = results.sort_values(
        "RReliefF_score",
        ascending=False
    ).reset_index(drop=True)

    return selected_features, results


import os
import joblib

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_feature_set(
    name,
    selected_features,
    target,
    df_train,
    df_val,
    df_test,
    save_dir=SAVE_DIR
):
    
    print("=" * 60)
    print(f"Processing: {name}")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Select the 15 features
    # --------------------------------------------------

    X_train = df_train[selected_features].copy()
    X_val   = df_val[selected_features].copy()
    X_test  = df_test[selected_features].copy()

    y_train = df_train[target].copy()
    y_val   = df_val[target].copy()
    y_test  = df_test[target].copy()

    print("Number of selected features:", len(selected_features))

    # --------------------------------------------------
    # 2. Fit StandardScaler ONLY on UCI TRAIN
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # IMPORTANT:
    # Do NOT fit scaler again on validation/test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # 3. Convert standardized arrays back to DataFrames
    # --------------------------------------------------

    train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=selected_features,
        index=df_train.index
    )

    val_scaled_df = pd.DataFrame(
        X_val_scaled,
        columns=selected_features,
        index=df_val.index
    )

    test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=selected_features,
        index=df_test.index
    )

    # Add target
    train_scaled_df[target] = y_train.values
    val_scaled_df[target]   = y_val.values
    test_scaled_df[target]  = y_test.values

    # --------------------------------------------------
    # 4. Save STANDARDIZED CSVs
    # --------------------------------------------------

    train_path = os.path.join(
        save_dir,
        f"{name}_train.csv"
    )

    val_path = os.path.join(
        save_dir,
        f"{name}_val.csv"
    )

    test_path = os.path.join(
        save_dir,
        f"{name}_test.csv"
    )

    train_scaled_df.to_csv(train_path, index=False)
    val_scaled_df.to_csv(val_path, index=False)
    test_scaled_df.to_csv(test_path, index=False)

    # --------------------------------------------------
    # 5. Save feature names + scaler
    # --------------------------------------------------

    package = {
        "feature_names": selected_features,
        "target": target,
        "scaler": scaler
    }

    scaler_path = os.path.join(
        save_dir,
        f"{name}_preprocessing.pkl"
    )

    joblib.dump(
        package,
        scaler_path
    )

    # --------------------------------------------------
    # 6. Save selected feature names as CSV
    # --------------------------------------------------

    feature_info = pd.DataFrame({
        "feature": selected_features
    })

    feature_info.to_csv(
        os.path.join(
            save_dir,
            f"{name}_features.csv"
        ),
        index=False
    )

    # --------------------------------------------------
    # 7. Print information
    # --------------------------------------------------

    print(f"Train shape: {train_scaled_df.shape}")
    print(f"Val shape:   {val_scaled_df.shape}")
    print(f"Test shape:  {test_scaled_df.shape}")

    print(f"\nSaved:")
    print(train_path)
    print(val_path)
    print(test_path)
    print(scaler_path)

    print()