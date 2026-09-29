import os
import glob

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter

def process_vitaldb_dataset(
    valid_df,
    WINDOW_SIZE,
    STEP_SIZE,
    fs=500,
    target_fs=125
):

    X_all = []
    y_all = []

    successful_cases = []
    failed_cases = []

    total_windows = 0

    for case_id in tqdm(
        valid_df["case_id"].tolist(),
        desc="Processing VitalDB",
        unit="case"
    ):

        X, y = process_vitaldb_recording(
            case_id=case_id,
            WINDOW_SIZE=WINDOW_SIZE,
            STEP_SIZE=STEP_SIZE,
            fs=fs,
            target_fs=target_fs
        )

        if X is None:
            failed_cases.append(case_id)
            continue

        X_all.append(X)
        y_all.append(y)

        successful_cases.append(case_id)

        total_windows += len(X)

    # -----------------------------------------
    # Combine
    # -----------------------------------------

    if len(X_all) == 0:

        raise ValueError(
            "No valid windows were generated."
        )

    X_all = np.concatenate(
        X_all,
        axis=0
    )

    y_all = np.concatenate(
        y_all,
        axis=0
    )

    print("\n================================")
    print("VitalDB preprocessing complete")
    print("================================")

    print(
        "Successful cases:",
        len(successful_cases)
    )

    print(
        "Failed cases:",
        len(failed_cases)
    )

    print(
        "Total windows:",
        len(X_all)
    )

    print(
        "X shape:",
        X_all.shape
    )

    print(
        "y shape:",
        y_all.shape
    )

    return (
        X_all,
        y_all,
        successful_cases,
        failed_cases
    )

def extract_bp_from_abp(abp_window, fs=500):

    # -----------------------------------------
    # Basic ART quality check
    # -----------------------------------------

    if not np.isfinite(abp_window).all():
        return None, None

    # Very flat signal
    if np.ptp(abp_window) < 20:
        return None, None

    # -----------------------------------------
    # Find systolic peaks
    # -----------------------------------------

    peaks, _ = find_peaks(
        abp_window,
        distance=int(0.4 * fs),
        prominence=10
    )

    # Need at least 2 beats
    if len(peaks) < 2:
        return None, None

    # -----------------------------------------
    # SBP
    # -----------------------------------------

    sbp_values = abp_window[peaks]

    # -----------------------------------------
    # DBP
    # -----------------------------------------

    dbp_values = []

    for i in range(len(peaks) - 1):

        beat_segment = abp_window[
            peaks[i]:peaks[i + 1]
        ]

        if len(beat_segment) > 0:

            dbp_values.append(
                np.min(beat_segment)
            )

    if len(dbp_values) == 0:
        return None, None

    # -----------------------------------------
    # Median across beats
    # -----------------------------------------

    sbp = np.median(sbp_values)
    dbp = np.median(dbp_values)

    # -----------------------------------------
    # Physiological range
    # -----------------------------------------

    if not np.isfinite(sbp):
        return None, None

    if not np.isfinite(dbp):
        return None, None

    if sbp < 50 or sbp > 250:
        return None, None

    if dbp < 20 or dbp > 150:
        return None, None

    return sbp, dbp

import vitaldb

from scipy.signal import find_peaks, resample_poly

def downsample_signal(signal):

    return resample_poly(
        signal,
        up=1,
        down=4
    )

from scipy.signal import butter, filtfilt

def remove_baseline_wander(ppg, fs=500, cutoff=0.5, order=4):
    """
    Remove low-frequency baseline wander from PPG.

    ppg: 1D numpy array
    fs: sampling frequency
    cutoff: baseline cutoff frequency in Hz
    """

    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype='low'
    )

    baseline = filtfilt(b, a, ppg)

    corrected_ppg = ppg - baseline

    return corrected_ppg


def robust_minmax_normalize(ppg):

    p_low = np.percentile(ppg, 1)
    p_high = np.percentile(ppg, 99)

    if p_high == p_low:
        return None

    ppg = (ppg - p_low) / (p_high - p_low)

    # Clip extreme noise/outliers
    ppg = np.clip(ppg, 0, 1)

    return ppg
def process_vitaldb_recording(
    case_id,
    WINDOW_SIZE,
    STEP_SIZE,
    fs=500,
    target_fs=125
):

    try:

        # =========================================
        # Load case
        # =========================================

        vf = vitaldb.VitalFile(case_id)

        tracks = vf.trks

        # =========================================
        # Check required tracks
        # =========================================

        if "SNUADC/PLETH" not in tracks:
            return None, None

        if "SNUADC/ART" not in tracks:
            return None, None

        ppg_track = tracks["SNUADC/PLETH"]
        art_track = tracks["SNUADC/ART"]

        # =========================================
        # Check sampling rates
        # =========================================

        ppg_fs = float(ppg_track.srate)
        art_fs = float(art_track.srate)

        if ppg_fs != fs:
            return None, None

        if art_fs != fs:
            return None, None

        # =========================================
        # Check records
        # =========================================

        if len(ppg_track.recs) == 0:
            return None, None

        if len(art_track.recs) == 0:
            return None, None

        # =========================================
        # Get first record
        # =========================================

        ppg_rec = ppg_track.recs[0]
        art_rec = art_track.recs[0]

        ppg = np.asarray(
            ppg_rec["val"],
            dtype=np.float32
        )

        art_raw = np.asarray(
            art_rec["val"],
            dtype=np.float32
        )

        art = (
            art_raw * float(art_track.gain)
            + float(art_track.offset)
        )

        # =========================================
        # Align using timestamps
        # =========================================

        ppg_start = float(ppg_rec["dt"])
        art_start = float(art_rec["dt"])

        start_time = max(
            ppg_start,
            art_start
        )

        ppg_start_idx = int(
            round(
                (start_time - ppg_start) * fs
            )
        )

        art_start_idx = int(
            round(
                (start_time - art_start) * fs
            )
        )

        ppg = ppg[ppg_start_idx:]
        art = art[art_start_idx:]

        # =========================================
        # Make same length
        # =========================================

        n = min(
            len(ppg),
            len(art)
        )

        ppg = ppg[:n]
        art = art[:n]
        ppg = remove_baseline_wander(ppg)
        ppg = robust_minmax_normalize(ppg)

        # =========================================
        # Recording too short?
        # =========================================

        if n < WINDOW_SIZE:
            return None, None

        # =========================================
        # Normalize PPG
        # =========================================

        valid_ppg = ppg[
            np.isfinite(ppg)
        ]

        if len(valid_ppg) == 0:
            return None, None

        ppg_mean = np.mean(valid_ppg)
        ppg_std = np.std(valid_ppg)

        if ppg_std == 0 or not np.isfinite(ppg_std):
            return None, None

        

        # =========================================
        # Windowing
        # =========================================

        X = []
        y = []

        for start in range(
            0,
            n - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            # -------------------------------------
            # Extract windows
            # -------------------------------------

            ppg_window = ppg[
                start:start + WINDOW_SIZE
            ]

            art_window = art[
                start:start + WINDOW_SIZE
            ]

            # -------------------------------------
            # NaN / invalid value check
            # -------------------------------------

            if not np.isfinite(ppg_window).all():
                continue

            if not np.isfinite(art_window).all():
                continue

            # -------------------------------------
            # Extract SBP / DBP
            # -------------------------------------

            sbp, dbp = extract_bp_from_abp(
                art_window,
                fs=fs
            )

            # No valid peaks / BP
            if sbp is None or dbp is None:
                continue

            # -------------------------------------
            # SBP / DBP range check
            # -------------------------------------

            if sbp < 50 or sbp > 250:
                continue

            if dbp < 20 or dbp > 150:
                continue

            # -------------------------------------
            # Downsample PPG
            # -------------------------------------

            ppg_window_125 = downsample_signal(
                ppg_window,
            )


            # -------------------------------------
            # Store
            # -------------------------------------

            X.append(ppg_window_125)

            y.append([
                sbp,
                dbp
            ])

        # =========================================
        # No valid windows
        # =========================================

        if len(X) == 0:
            return None, None

        # =========================================
        # Return
        # =========================================

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32)
        )

    except Exception as e:

        print(
            f"Error processing case {case_id}: "
            f"{type(e).__name__}: {e}"
        )

        return None, None

import os
import time
import numpy as np

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed
)

from tqdm import tqdm


# ============================================================
# Worker
# ============================================================

def process_single_case(args):

    case_id, WINDOW_SIZE, STEP_SIZE, fs, target_fs = args

    X, y = process_vitaldb_recording(
        case_id=case_id,
        WINDOW_SIZE=WINDOW_SIZE,
        STEP_SIZE=STEP_SIZE,
        fs=fs,
        target_fs=target_fs
    )

    return case_id, X, y


# ============================================================
# Main dataset processing
# ============================================================

def process_vitaldb_dataset(
    valid_df,
    WINDOW_SIZE,
    STEP_SIZE,
    fs=500,
    target_fs=125,
    num_workers=10,
    output_dir="vitaldb_checkpoints",
    timeout_per_case=120
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    case_ids = valid_df["case_id"].tolist()

    # ========================================================
    # Check already processed cases
    # ========================================================

    completed_cases = set()

    for filename in os.listdir(output_dir):

        if filename.endswith(".npz"):

            try:

                case_id = int(
                    filename.replace(
                        "case_", ""
                    ).replace(
                        ".npz", ""
                    )
                )

                completed_cases.add(case_id)

            except ValueError:
                pass

    remaining_cases = [
        case_id
        for case_id in case_ids
        if case_id not in completed_cases
    ]

    print("\n================================")
    print("VitalDB Processing")
    print("================================")

    print(
        "Total cases:",
        len(case_ids)
    )

    print(
        "Already completed:",
        len(completed_cases)
    )

    print(
        "Remaining:",
        len(remaining_cases)
    )

    if len(remaining_cases) == 0:

        print(
            "\nAll cases already processed!"
        )

    else:

        # ====================================================
        # Prepare arguments
        # ====================================================

        args_list = [
            (
                case_id,
                WINDOW_SIZE,
                STEP_SIZE,
                fs,
                target_fs
            )
            for case_id in remaining_cases
        ]

        # ====================================================
        # Process in batches
        #
        # Important:
        # We don't submit all 3238 cases at once.
        # ====================================================

        batch_size = num_workers * 2

        for batch_start in range(
            0,
            len(args_list),
            batch_size
        ):

            batch = args_list[
                batch_start:
                batch_start + batch_size
            ]

            print(
                f"\nProcessing batch "
                f"{batch_start + 1}-"
                f"{batch_start + len(batch)}"
            )

            executor = ProcessPoolExecutor(
                max_workers=num_workers
            )

            futures = {}

            start_times = {}

            for args in batch:

                future = executor.submit(
                    process_single_case,
                    args
                )

                futures[future] = args[0]

                start_times[future] = time.time()

            finished = set()

            try:

                while len(finished) < len(futures):

                    # ----------------------------------------
                    # Check completed futures
                    # ----------------------------------------

                    for future in list(futures):

                        if future in finished:
                            continue

                        if not future.done():
                            continue

                        case_id = futures[future]

                        finished.add(future)

                        try:

                            returned_case_id, X, y = (
                                future.result()
                            )

                            # --------------------------------
                            # Failed case
                            # --------------------------------

                            if X is None:

                                print(
                                    f"\nCase {case_id}: "
                                    f"no valid windows"
                                )

                                continue

                            # --------------------------------
                            # Save immediately
                            # --------------------------------

                            save_path = os.path.join(
                                output_dir,
                                f"case_{case_id}.npz"
                            )

                            np.savez_compressed(
                                save_path,
                                X=X,
                                y=y,
                                case_id=case_id
                            )

                            print(
                                f"\nSaved case {case_id}: "
                                f"{len(X)} windows"
                            )

                        except Exception as e:

                            print(
                                f"\nError processing "
                                f"case {case_id}: "
                                f"{type(e).__name__}: {e}"
                            )

                    # ----------------------------------------
                    # Check for timeout
                    # ----------------------------------------

                    now = time.time()

                    timed_out = []

                    for future in futures:

                        if future in finished:
                            continue

                        elapsed = (
                            now -
                            start_times[future]
                        )

                        if elapsed > timeout_per_case:

                            timed_out.append(
                                futures[future]
                            )

                    # ----------------------------------------
                    # Kill pool if a worker is stuck
                    # ----------------------------------------

                    if len(timed_out) > 0:

                        print(
                            "\n================================"
                        )

                        print(
                            "TIMEOUT DETECTED"
                        )

                        print(
                            "Stuck cases:",
                            timed_out
                        )

                        print(
                            "Stopping current worker pool..."
                        )

                        print(
                            "Completed cases have already "
                            "been saved."
                        )

                        print(
                            "================================"
                        )

                        # ------------------------------------
                        # Cancel futures that haven't started
                        # ------------------------------------

                        for future in futures:

                            if not future.done():

                                future.cancel()

                        # ------------------------------------
                        # Terminate worker processes
                        # ------------------------------------

                        for process in executor._processes.values():

                            if process.is_alive():

                                process.terminate()

                        executor.shutdown(
                            wait=False,
                            cancel_futures=True
                        )

                        # ------------------------------------
                        # Stop this batch
                        # ------------------------------------

                        break

                    time.sleep(0.5)

            finally:

                # =================================================
                # Normal shutdown
                # =================================================

                if not any(
                    not f.done()
                    for f in futures
                ):

                    executor.shutdown(
                        wait=True
                    )

    # ============================================================
    # Load all successfully processed cases
    # ============================================================

    print(
        "\n================================"
    )

    print(
        "Loading saved cases..."
    )

    print(
        "================================"
    )

    X_all = []
    y_all = []
    successful_cases = []

    for case_id in case_ids:

        save_path = os.path.join(
            output_dir,
            f"case_{case_id}.npz"
        )

        if not os.path.exists(save_path):
            continue

        try:

            data = np.load(
                save_path
            )

            X = data["X"]
            y = data["y"]

            if len(X) == 0:
                continue

            X_all.append(X)
            y_all.append(y)

            successful_cases.append(
                case_id
            )

        except Exception as e:

            print(
                f"Could not load case {case_id}: "
                f"{e}"
            )

    # ============================================================
    # Combine
    # ============================================================

    if len(X_all) == 0:

        raise ValueError(
            "No successfully processed cases found."
        )

    X_all = np.concatenate(
        X_all,
        axis=0
    )

    y_all = np.concatenate(
        y_all,
        axis=0
    )

    # ============================================================
    # Failed cases
    # ============================================================

    successful_set = set(
        successful_cases
    )

    failed_cases = [
        case_id
        for case_id in case_ids
        if case_id not in successful_set
    ]

    # ============================================================
    # Final statistics
    # ============================================================

    print(
        "\n================================"
    )

    print(
        "VitalDB preprocessing complete"
    )

    print(
        "================================"
    )

    print(
        "Successful cases:",
        len(successful_cases)
    )

    print(
        "Failed cases:",
        len(failed_cases)
    )

    print(
        "Total windows:",
        len(X_all)
    )

    print(
        "X shape:",
        X_all.shape
    )

    print(
        "y shape:",
        y_all.shape
    )

    # ============================================================
    # Save final combined dataset
    # ============================================================

    np.savez_compressed(
        "vitaldb_zero_shot_test.npz",
        X=X_all,
        y=y_all,
        case_ids=np.array(
            successful_cases
        )
    )

    print(
        "\nFinal dataset saved to:"
    )

    print(
        "vitaldb_zero_shot_test.npz"
    )

    return (
        X_all,
        y_all,
        successful_cases,
        failed_cases
    )








# ----------------------------------------------------
# For features.ipynb


import os
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed







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



# --------------------------------------------------
# Function
# --------------------------------------------------
import joblib
def create_vitaldb_feature_set(df_vitaldb, UCI_PREPROCESSING_DIR, VITALDB_SAVE_DIR, method, target):

    name = f"{target}_{method}"

    print("=" * 70)
    print(f"Processing: {name}")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load UCI preprocessing package
    # --------------------------------------------------

    pkl_path = os.path.join(
        UCI_PREPROCESSING_DIR,
        f"{name}_preprocessing.pkl"
    )

    print("Loading:", pkl_path)

    package = joblib.load(pkl_path)

    selected_features = package["feature_names"]
    scaler = package["scaler"]

    print("Target:", package["target"])
    print("Number of features:", len(selected_features))
    print("Features:")
    print(selected_features)


    # --------------------------------------------------
    # 2. Check VitalDB features
    # --------------------------------------------------

    missing_features = [
        f for f in selected_features
        if f not in df_vitaldb.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in df_vitaldb: {missing_features}"
        )


    # --------------------------------------------------
    # 3. Select features from VitalDB
    # --------------------------------------------------

    X_vital = df_vitaldb[selected_features].copy()

    y_vital = df_vitaldb[target].copy()


    # --------------------------------------------------
    # 4. Apply UCI TRAIN-FITTED scaler
    # --------------------------------------------------

    X_vital_scaled = scaler.transform(X_vital)


    # --------------------------------------------------
    # 5. Create DataFrame
    # --------------------------------------------------

    vital_scaled_df = pd.DataFrame(
        X_vital_scaled,
        columns=selected_features,
        index=df_vitaldb.index
    )

    # Add target
    vital_scaled_df[target] = y_vital.values

    # Preserve case ID
    if "case_id" in df_vitaldb.columns:
        vital_scaled_df["case_id"] = df_vitaldb["case_id"].values


    # --------------------------------------------------
    # 6. Save CSV
    # --------------------------------------------------

    output_path = os.path.join(
        VITALDB_SAVE_DIR,
        f"{name}_vitaldb.csv"
    )

    vital_scaled_df.to_csv(
        output_path,
        index=False
    )

    print("\nVitalDB shape:", vital_scaled_df.shape)
    print("Saved:", output_path)
    print()

    return vital_scaled_df