import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import (
    find_peaks,
    savgol_filter
)
import h5py

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader


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
    if sbp - dbp < 20:
        return None, None


    return sbp, dbp


from scipy.signal import butter, filtfilt

def remove_baseline_wander(ppg, fs=125, cutoff=0.5, order=4):
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


def get_systolic_peak(ppg_window, fs=125):

    peaks, properties = find_peaks(
        ppg_window,
        height=np.mean(ppg_window),
        distance=int(0.3 * fs),
        prominence=0.1
    )

    if len(peaks) == 0:
        return np.array([], dtype=int)

    return peaks

def calculate_derivatives(ppg, fs=125):

    # -----------------------------------
    # Smooth PPG
    # -----------------------------------

    smooth_ppg = savgol_filter(
        ppg,
        window_length=15,
        polyorder=3
    )

    # -----------------------------------
    # First derivative = VPG
    # -----------------------------------

    vpg = np.gradient(
        smooth_ppg
    ) * fs

    # -----------------------------------
    # Second derivative = APG
    # -----------------------------------

    apg = np.gradient(
        vpg
    ) * fs

    return smooth_ppg, vpg, apg

def get_dicrotic_notch(
    ppg_window,
    sys_idx,
    vpg,
    apg,
    fs=125
):

    if sys_idx is None:
        return None



    # ------------------------------------------
    # Search region after systolic peak
    # ------------------------------------------

    search_start = sys_idx + int(0.08 * fs)
    search_end = sys_idx + int(0.35 * fs)

    search_end = min(
        search_end,
        len(ppg_window) - 1
    )

    if search_start >= search_end:
        return None

    # ------------------------------------------
    # APG extrema
    # ------------------------------------------

    apg_segment = apg[
        search_start:search_end
    ]

    extrema, _ = find_peaks(
        np.abs(apg_segment),
        prominence=0.10 * np.std(apg_segment)
    )

    if len(extrema) == 0:
        return None

    candidates = search_start + extrema

    # ------------------------------------------
    # Keep candidates on falling limb
    # ------------------------------------------

    candidates = [
        idx
        for idx in candidates
        if vpg[idx] < 0
    ]

    if len(candidates) == 0:
        return None

    # ------------------------------------------
    # Find strongest APG candidate
    # ------------------------------------------

    scores = [
        abs(apg[idx])
        for idx in candidates
    ]

    candidate = candidates[
        np.argmax(scores)
    ]

    # ==================================================
    # IMPORTANT:
    # Move FORWARD from APG candidate
    # ==================================================

    refine_start = candidate + int(0.02 * fs)
    refine_end = candidate + int(0.08 * fs)

    refine_end = min(
        refine_end,
        search_end
    )

    if refine_end - refine_start < 2:
        return candidate

    # ------------------------------------------
    # Look at how VPG changes
    # ------------------------------------------

    local_vpg = vpg[
        refine_start:refine_end
    ]

    # We want the point where the falling
    # slope starts becoming less negative.
    #
    # Calculate change in VPG
    # ------------------------------------------

    dvpg = np.gradient(local_vpg)

    # Find strongest positive change
    # in the falling slope

    best_relative_idx = np.argmax(dvpg)

    best_idx = (
        refine_start +
        best_relative_idx
    )

    return best_idx



def get_delay_time(ppg_window, fs=125):
    """
    Detect all systolic peaks and estimate the
    peak-to-dicrotic-notch delay for each beat.

    Returns:
        delays : np.ndarray
            Valid delay for each detected beat.
    """

    # -----------------------------------------
    # Detect ALL systolic peaks
    # -----------------------------------------

    sys_indices = get_systolic_peak(
        ppg_window,
        fs
    )

    if sys_indices is None or len(sys_indices) == 0:
        return np.array([])
    smooth_ppg, vpg, apg = calculate_derivatives(
        ppg_window,
        fs
    )

    delays = []

    # -----------------------------------------
    # Process each systolic peak
    # -----------------------------------------

    for sys_idx in sys_indices:

        notch_idx = get_dicrotic_notch(
            ppg_window,
            sys_idx,
            vpg,
            apg,
            fs
        )

        if notch_idx is None:
            continue

        # -------------------------------------
        # Peak → notch delay
        # -------------------------------------

        delay = (
            notch_idx - sys_idx
        ) / fs

        # -------------------------------------
        # Reject unreasonable delays
        # -------------------------------------

        if delay < 0.08 or delay > 0.35:
            continue

        delays.append(delay)

    return np.array(
        delays,
        dtype=np.float32
    )


def extract_delay_times(
    ppg_windows,
    fs=125
):
    """
    Extract one representative delay
    for each PPG window.

    Multiple beats are detected within
    each window and their median delay
    is used.

    Failed detections are stored as NaN.

    Returns:
        delay_times : np.ndarray of shape (N,)
    """

    delay_times = []

    for w in ppg_windows:

        # -----------------------------------------
        # Get beat-level delays
        # -----------------------------------------

        delays = get_delay_time(
            w,
            fs
        )

        # -----------------------------------------
        # No valid delays
        # -----------------------------------------

        if len(delays) == 0:

            delay_times.append(np.nan)

            continue

        # -----------------------------------------
        # Median delay for this window
        # -----------------------------------------

        window_delay = np.median(
            delays
        )

        delay_times.append(
            window_delay
        )

    return np.array(
        delay_times,
        dtype=np.float32
    )




def process_recording(
    recording,
    window_size,
    step_size,
    fs=125
):

    data = recording[:]

    # -----------------------------------------
    # Extract channels
    # -----------------------------------------

    ppg = data[:, 0]
    abp = data[:, 1]
    ppg = remove_baseline_wander(ppg)
    ppg = robust_minmax_normalize(ppg)

    # -----------------------------------------
    # Recording-level PPG normalization
    # -----------------------------------------

    if ppg is None:
        return None, None, None
 

    X = []
    y = []
    delays = []

    # -----------------------------------------
    # Windowing
    # -----------------------------------------

    for start in range(
        0,
        len(ppg) - window_size + 1,
        step_size
    ):

        ppg_window = ppg[
            start:start + window_size
        ]

        abp_window = abp[
            start:start + window_size
        ]

        # -------------------------------------
        # Extract SBP / DBP
        # -------------------------------------

        sbp, dbp = extract_bp_from_abp(
            abp_window
        )

        if sbp is None:
            continue

        # -------------------------------------
        # Extract beat-level delays
        # -------------------------------------

        beat_delays = get_delay_time(
            ppg_window,
            fs
        )

        # -------------------------------------
        # No valid delay detected
        # -------------------------------------

        if len(beat_delays) == 0:
            continue

        # -------------------------------------
        # One representative delay per window
        # -------------------------------------

        window_delay = np.median(
            beat_delays
        )

        # -------------------------------------
        # Store
        # -------------------------------------

        X.append(ppg_window)

        y.append([
            sbp,
            dbp
        ])

        delays.append(
            window_delay
        )

    # -----------------------------------------
    # No valid windows
    # -----------------------------------------

    if len(X) == 0:
        return None, None, None

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(delays, dtype=np.float32)
    )


def process_split(
    record_list,
    window_size,
    step_size,
    fs=125
):

    X_all = []
    y_all = []
    delay_all = []

    skipped_recordings = 0

    # -----------------------------------------
    # Process each recording
    # -----------------------------------------

    for f, ref in tqdm(
        record_list,
        desc="Processing recordings"
    ):

        recording = f[ref]

        X, y, delays = process_recording(
            recording,
            window_size,
            step_size,
            fs
        )

        if X is None:

            skipped_recordings += 1

            continue

        X_all.append(X)
        y_all.append(y)
        delay_all.append(delays)

    # -----------------------------------------
    # Combine all recordings
    # -----------------------------------------

    X_all = np.concatenate(
        X_all,
        axis=0
    )

    y_all = np.concatenate(
        y_all,
        axis=0
    )

    delay_all = np.concatenate(
        delay_all,
        axis=0
    )

    print(
        "Skipped recordings:",
        skipped_recordings
    )

    return (
        X_all,
        y_all,
        delay_all
    )

def load_UCI_recordings(
    data_path
):

    recordings = []

    files = []

    # -----------------------------------------
    # Load Parts 1-4
    # -----------------------------------------

    for part in range(1, 5):

        file_path = (
            f"{data_path}/Part_{part}.mat"
        )

        f = h5py.File(
            file_path,
            "r"
        )

        dataset = f[
            f"Part_{part}"
        ]

        files.append(f)

        # -------------------------------------
        # Store recording references
        # -------------------------------------

        for i in range(
            dataset.shape[0]
        ):

            recordings.append(
                (
                    f,
                    dataset[i, 0]
                )
            )

    print(
        "Total recordings:",
        len(recordings)
    )

    return recordings, files

def split_recordings(
    recordings,
    test_size=0.20,
    val_size=0.10,
    random_state=42
):

    # -----------------------------------------
    # First: 80% train+val, 20% test
    # -----------------------------------------

    train_val, test_records = train_test_split(
        recordings,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # -----------------------------------------
    # Validation fraction within remaining 80%
    #
    # 0.10 / 0.80 = 0.125
    # -----------------------------------------

    val_fraction = (
        val_size /
        (1 - test_size)
    )

    train_records, val_records = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True
    )

    print(
        "Train recordings:",
        len(train_records)
    )

    print(
        "Validation recordings:",
        len(val_records)
    )

    print(
        "Test recordings:",
        len(test_records)
    )

    return (
        train_records,
        val_records,
        test_records
    )

def prepare_UCI_dataset(
    data_path,
    window_size,
    step_size,
    fs=125,
    test_size=0.20,
    val_size=0.10,
    random_state=42
):

    # ==================================================
    # 1. Load recording references
    # ==================================================

    recordings, files = load_UCI_recordings(
        data_path
    )

    # ==================================================
    # 2. SPLIT RECORDINGS FIRST
    # ==================================================

    (
        train_records,
        val_records,
        test_records
    ) = split_recordings(
        recordings,
        test_size,
        val_size,
        random_state
    )

    # ==================================================
    # 3. Process TRAIN recordings
    # ==================================================

    print("\nProcessing TRAIN recordings...")

    (
        X_train,
        y_train,
        delay_train
    ) = process_split(
        train_records,
        window_size,
        step_size,
        fs
    )

    # ==================================================
    # 4. Process VALIDATION recordings
    # ==================================================

    print("\nProcessing VALIDATION recordings...")

    (
        X_val,
        y_val,
        delay_val
    ) = process_split(
        val_records,
        window_size,
        step_size,
        fs
    )

    # ==================================================
    # 5. Process TEST recordings
    # ==================================================

    print("\nProcessing TEST recordings...")

    (
        X_test,
        y_test,
        delay_test
    ) = process_split(
        test_records,
        window_size,
        step_size,
        fs
    )

    # ==================================================
    # 6. Add CNN channel dimension
    # ==================================================

    X_train = X_train[:, None, :]
    X_val = X_val[:, None, :]
    X_test = X_test[:, None, :]

    # ==================================================
    # 7. Convert to tensors
    # ==================================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    delay_train = torch.tensor(
        delay_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.float32
    )

    delay_val = torch.tensor(
        delay_val,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )

    delay_test = torch.tensor(
        delay_test,
        dtype=torch.float32
    )

    # ==================================================
    # 8. Print shapes
    # ==================================================

    print(
        "\n======================================"
    )

    print(
        "DATASET SHAPES"
    )

    print(
        "======================================"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "delay_train:",
        delay_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "y_val:",
        y_val.shape
    )

    print(
        "delay_val:",
        delay_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    print(
        "delay_test:",
        delay_test.shape
    )

    return (
        X_train,
        y_train,
        delay_train,

        X_val,
        y_val,
        delay_val,

        X_test,
        y_test,
        delay_test
    )



def estimate_RC(
    sbp,
    dbp,
    delay,
    min_delay=0.08,
    max_delay=0.35
):
    """
    Estimate Windkessel RC from training data.

    Physics:
        DBP = SBP * exp(-delay / RC)

    Therefore:
        RC = -delay / ln(DBP / SBP)

    Parameters
    ----------
    sbp : array-like
        SBP values from training windows.

    dbp : array-like
        DBP values from training windows.

    delay : array-like
        Peak-to-notch delay for training windows.

    Returns
    -------
    RC : float
        Estimated Windkessel time constant in seconds.
    """

    sbp = np.asarray(sbp, dtype=np.float64)
    dbp = np.asarray(dbp, dtype=np.float64)
    delay = np.asarray(delay, dtype=np.float64)

    # ------------------------------------------------
    # Valid values only
    # ------------------------------------------------

    valid = (
        np.isfinite(sbp) &
        np.isfinite(dbp) &
        np.isfinite(delay)
    )

    # Physiological constraints
    valid &= sbp > 0
    valid &= dbp > 0
    valid &= dbp < sbp

    # Delay constraints
    valid &= delay >= min_delay
    valid &= delay <= max_delay

    sbp_valid = sbp[valid]
    dbp_valid = dbp[valid]
    delay_valid = delay[valid]

    print("Total windows:", len(sbp))
    print("Valid windows:", len(sbp_valid))
    print(
        "Valid percentage:",
        100 * len(sbp_valid) / len(sbp),
        "%"
    )

    if len(sbp_valid) == 0:
        raise ValueError(
            "No valid samples available for RC estimation."
        )

    # ------------------------------------------------
    # Calculate RC for every valid window
    # ------------------------------------------------

    ratio = dbp_valid / sbp_valid

    RC_values = (
        -delay_valid /
        np.log(ratio)
    )

    # ------------------------------------------------
    # Remove invalid RC values
    # ------------------------------------------------

    RC_values = RC_values[
        np.isfinite(RC_values) &
        (RC_values > 0)
    ]

    if len(RC_values) == 0:
        raise ValueError(
            "No valid RC values obtained."
        )

    # ------------------------------------------------
    # Robust estimate
    # ------------------------------------------------

    RC = np.median(RC_values)

    print("\n================================")
    print("Windkessel RC Estimation")
    print("================================")

    print(
        f"Median RC : {RC:.4f} s"
    )

    print(
        f"Mean RC   : {np.mean(RC_values):.4f} s"
    )

    print(
        f"Std RC    : {np.std(RC_values):.4f} s"
    )

    print(
        f"Min RC    : {np.min(RC_values):.4f} s"
    )

    print(
        f"Max RC    : {np.max(RC_values):.4f} s"
    )

    return RC, RC_values


def grid_search_RC(
    sbp,
    dbp,
    delay,
    rc_min=0.10,
    rc_max=1.00,
    rc_step=0.005
):

    sbp = np.asarray(sbp, dtype=np.float64)
    dbp = np.asarray(dbp, dtype=np.float64)
    delay = np.asarray(delay, dtype=np.float64)

    # -----------------------------------------
    # Valid samples
    # -----------------------------------------

    valid = (
        np.isfinite(sbp) &
        np.isfinite(dbp) &
        np.isfinite(delay) &
        (sbp > 0) &
        (dbp > 0) &
        (dbp < sbp)
    )

    sbp = sbp[valid]
    dbp = dbp[valid]
    delay = delay[valid]

    print("Valid samples:", len(sbp))

    # -----------------------------------------
    # Candidate RC values
    # -----------------------------------------

    rc_values = np.arange(
        rc_min,
        rc_max + rc_step,
        rc_step
    )

    mae_values = []
    rmse_values = []

    # -----------------------------------------
    # Evaluate every RC
    # -----------------------------------------

    for RC in rc_values:

        dbp_physics = (
            sbp *
            np.exp(-delay / RC)
        )

        mae = np.mean(
            np.abs(dbp_physics - dbp)
        )

        rmse = np.sqrt(
            np.mean(
                (dbp_physics - dbp) ** 2
            )
        )

        mae_values.append(mae)
        rmse_values.append(rmse)

    mae_values = np.array(mae_values)
    rmse_values = np.array(rmse_values)

    # -----------------------------------------
    # Best RC
    # -----------------------------------------

    best_mae_idx = np.argmin(mae_values)
    best_rmse_idx = np.argmin(rmse_values)

    best_RC_MAE = rc_values[best_mae_idx]
    best_RC_RMSE = rc_values[best_rmse_idx]

    print("\n======================================")
    print("RC GRID SEARCH RESULTS")
    print("======================================")

    print(
        f"Best RC by MAE  : {best_RC_MAE:.4f} s"
    )

    print(
        f"Best MAE        : {mae_values[best_mae_idx]:.4f} mmHg"
    )

    print()

    print(
        f"Best RC by RMSE : {best_RC_RMSE:.4f} s"
    )

    print(
        f"Best RMSE       : {rmse_values[best_rmse_idx]:.4f} mmHg"
    )

    return (
        rc_values,
        mae_values,
        rmse_values,
        best_RC_MAE,
        best_RC_RMSE
    )

class PPGDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

def create_data_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=256
):

    train_dataset = PPGDataset(
        X_train,
        y_train
    )

    val_dataset = PPGDataset(
        X_val,
        y_val
    )

    test_dataset = PPGDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def print_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"MAE : {mae:.4f}")
    

from matplotlib import pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
def plot_training_loss(history, model_name="ResNet1D"):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title(
        f"{model_name}: Training and Validation Loss"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pinn_training_metrics(
    history,
    model_name="Windkessel PINN"
):

    epochs = range(1, len(history["train_loss"]) + 1)

    # ============================================================
    # 1. TOTAL LOSS
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"{model_name}: Total Loss vs Epoch"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 2. SBP RMSE
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_sbp_rmse"],
        label="Train SBP RMSE"
    )

    plt.plot(
        epochs,
        history["val_sbp_rmse"],
        label="Validation SBP RMSE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("RMSE (mmHg)")

    plt.title(
        f"{model_name}: SBP RMSE vs Epoch"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 3. DBP RMSE
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_dbp_rmse"],
        label="Train DBP RMSE"
    )

    plt.plot(
        epochs,
        history["val_dbp_rmse"],
        label="Validation DBP RMSE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("RMSE (mmHg)")

    plt.title(
        f"{model_name}: DBP RMSE vs Epoch"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 4. PHYSICS LOSS
    # ============================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_phys_loss"],
        label="Train Physics Loss"
    )

    plt.plot(
        epochs,
        history["val_phys_loss"],
        label="Validation Physics Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Physics Loss")

    plt.title(
        f"{model_name}: Physics Loss vs Epoch"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_regression_results(
    y_true,
    y_pred,
    model_name,
    target_name
):

    # Convert to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    # ----------------------------------------
    # 1. TRUE VS PREDICTED
    # ----------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.3
    )

    min_val = min(
        y_true.min(),
        y_pred.min()
    )

    max_val = max(
        y_true.max(),
        y_pred.max()
    )

    # Ideal y = x line
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Ideal (y = x)"
    )

    plt.xlabel(
        f"True {target_name} (mmHg)"
    )

    plt.ylabel(
        f"Predicted {target_name} (mmHg)"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"True vs Predicted"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ----------------------------------------
    # 2. RESIDUAL VS PREDICTED
    # ----------------------------------------

    plt.figure(figsize=(7, 5))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.3
    )

    # Zero-error line
    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        f"Predicted {target_name} (mmHg)"
    )

    plt.ylabel(
        "Residual (mmHg)"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"Residual Plot"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ----------------------------------------
    # 3. RESIDUAL DISTRIBUTION
    # ----------------------------------------

    plt.figure(figsize=(7, 5))

    plt.hist(
        residuals,
        bins=50,
        alpha=0.7
    )

    # Zero-error line
    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Residual (mmHg)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"Residual Distribution"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ----------------------------------------
    # 4. MAE VS TRUE TARGET RANGE
    #    WITH ERROR BARS
    # ----------------------------------------

    if target_name.upper() == "DBP":

        bins = [
            50,
            60,
            70,
            80,
            90,
            100,
            120,
            150
        ]

    else:

        # SBP
        bins = [
            70,
            80,
            100,
            120,
            140,
            160,
            180,
            200
        ]


    mae_values = []
    error_std = []
    sample_counts = []
    range_labels = []


    for i in range(len(bins) - 1):

        lower = bins[i]
        upper = bins[i + 1]

        mask = (
            (y_true >= lower) &
            (y_true < upper)
        )

        count = np.sum(mask)

        # Skip range if there are no samples
        if count == 0:
            continue


        # ----------------------------------------
        # Absolute errors for this BP range
        # ----------------------------------------

        absolute_errors = np.abs(
            y_true[mask] -
            y_pred[mask]
        )


        # Mean Absolute Error
        mae = np.mean(
            absolute_errors
        )


        # Standard deviation of absolute errors
        std = np.std(
            absolute_errors
        )


        mae_values.append(mae)
        error_std.append(std)
        sample_counts.append(count)

        range_labels.append(
            f"{lower}–{upper}"
        )


    # ----------------------------------------
    # Plot MAE + error bars
    # ----------------------------------------

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        range_labels,
        mae_values,
        yerr=error_std,
        capsize=5,
        alpha=0.8
    )


    plt.xlabel(
        f"True {target_name} range (mmHg)"
    )

    plt.ylabel(
        "MAE (mmHg)"
    )

    plt.title(
        f"{model_name} — {target_name}: "
        f"MAE by True Target Range"
    )


    # ----------------------------------------
    # Add MAE ± SD and sample count
    # ----------------------------------------

    for bar, mae, std, count in zip(
        bars,
        mae_values,
        error_std,
        sample_counts
    ):

        plt.text(
            bar.get_x() +
            bar.get_width() / 2,

            bar.get_height()+std+1,

            f"{mae:.2f} ± {std:.2f}\n"
            f"(n={count:,})",

            ha="center",
            va="bottom",
            fontsize=9
        )


    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()




##############################3
### RESNET 
def train_resnet1d(
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=50,
    lr=1e-4,
    weight_decay=1e-3,
    batch_size=None
):

    # ============================================================
    # 1. RESNET MODEL
    # ============================================================

    class BasicBlock1D(nn.Module):

        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()

            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False
            )

            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            )

            self.bn2 = nn.BatchNorm1d(out_channels)

            self.downsample = None

            if stride != 1 or in_channels != out_channels:

                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_channels)
                )

        def forward(self, x):

            identity = x

            # Main branch
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            # Skip connection
            if self.downsample is not None:
                identity = self.downsample(x)

            # Residual addition
            out = out + identity
            out = self.relu(out)

            return out


    class SmallResNet1D(nn.Module):

        def __init__(self):
            super().__init__()

            self.in_channels = 32

            # Initial convolution
            self.conv1 = nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            )

            self.bn1 = nn.BatchNorm1d(32)
            self.relu = nn.ReLU(inplace=True)

            self.maxpool = nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )

            # Residual stages
            self.layer1 = self._make_layer(
                out_channels=32,
                blocks=2,
                stride=1
            )

            self.layer2 = self._make_layer(
                out_channels=64,
                blocks=2,
                stride=2
            )

            self.layer3 = self._make_layer(
                out_channels=128,
                blocks=2,
                stride=2
            )

            self.layer4 = self._make_layer(
                out_channels=256,
                blocks=2,
                stride=2
            )

            # Global average pooling
            self.avgpool = nn.AdaptiveAvgPool1d(1)

            # Dropout
            self.dropout = nn.Dropout(p=0.4)

            # SBP + DBP
            self.fc = nn.Linear(256, 2)


        def _make_layer(
            self,
            out_channels,
            blocks,
            stride
        ):

            layers = []

            layers.append(
                BasicBlock1D(
                    self.in_channels,
                    out_channels,
                    stride
                )
            )

            self.in_channels = out_channels

            for _ in range(1, blocks):

                layers.append(
                    BasicBlock1D(
                        out_channels,
                        out_channels,
                        stride=1
                    )
                )

            return nn.Sequential(*layers)


        def forward(self, x):

            # [B, 1, 1000]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)

            # [B, 256, 1] -> [B, 256]
            x = torch.flatten(x, start_dim=1)

            x = self.dropout(x)

            # [B, 2]
            # [:,0] = SBP
            # [:,1] = DBP
            x = self.fc(x)

            return x


    # ============================================================
    # 2. CREATE MODEL
    # ============================================================

    model = SmallResNet1D().to(device)


    # ============================================================
    # 3. LOSS
    # ============================================================

    criterion = nn.MSELoss()


    # ============================================================
    # 4. OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


    # ============================================================
    # 5. LEARNING RATE SCHEDULER
    # ============================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )


    # ============================================================
    # 6. TRAINING HISTORY
    # ============================================================

    train_losses = []

    train_sbp_mse = []
    train_dbp_mse = []

    train_sbp_rmse = []
    train_dbp_rmse = []

    val_losses = []

    val_sbp_mse = []
    val_dbp_mse = []

    val_sbp_rmse = []
    val_dbp_rmse = []


    # ============================================================
    # 7. BEST MODEL
    # ============================================================

    best_val_loss = float("inf")
    best_model_state = None


    # ============================================================
    # 8. TRAINING LOOP
    # ============================================================

    for epoch in range(epochs):

        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------

        model.train()

        running_train_loss = 0.0

        running_train_sbp_mse = 0.0
        running_train_dbp_mse = 0.0

        train_samples = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)


            # ====================================================
            # TOTAL MSE LOSS
            # ====================================================

            loss = criterion(
                y_pred,
                y_batch
            )


            # ====================================================
            # SEPARATE SBP / DBP MSE
            # ====================================================

            sbp_loss = nn.functional.mse_loss(
                y_pred[:, 0],
                y_batch[:, 0]
            )

            dbp_loss = nn.functional.mse_loss(
                y_pred[:, 1],
                y_batch[:, 1]
            )


            # Backpropagation
            loss.backward()

            optimizer.step()


            # ====================================================
            # ACCUMULATE
            # ====================================================

            batch_size_actual = X_batch.size(0)

            running_train_loss += (
                loss.item() * batch_size_actual
            )

            running_train_sbp_mse += (
                sbp_loss.item() * batch_size_actual
            )

            running_train_dbp_mse += (
                dbp_loss.item() * batch_size_actual
            )

            train_samples += batch_size_actual


        # ========================================================
        # EPOCH TRAIN METRICS
        # ========================================================

        epoch_train_loss = (
            running_train_loss /
            train_samples
        )

        epoch_train_sbp_mse = (
            running_train_sbp_mse /
            train_samples
        )

        epoch_train_dbp_mse = (
            running_train_dbp_mse /
            train_samples
        )

        epoch_train_sbp_rmse = np.sqrt(
            epoch_train_sbp_mse
        )

        epoch_train_dbp_rmse = np.sqrt(
            epoch_train_dbp_mse
        )


        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        model.eval()

        running_val_loss = 0.0

        running_val_sbp_mse = 0.0
        running_val_dbp_mse = 0.0

        val_samples = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)


                # =================================================
                # TOTAL MSE
                # =================================================

                loss = criterion(
                    y_pred,
                    y_batch
                )


                # =================================================
                # SEPARATE SBP / DBP MSE
                # =================================================

                sbp_loss = nn.functional.mse_loss(
                    y_pred[:, 0],
                    y_batch[:, 0]
                )

                dbp_loss = nn.functional.mse_loss(
                    y_pred[:, 1],
                    y_batch[:, 1]
                )


                batch_size_actual = X_batch.size(0)

                running_val_loss += (
                    loss.item() * batch_size_actual
                )

                running_val_sbp_mse += (
                    sbp_loss.item() * batch_size_actual
                )

                running_val_dbp_mse += (
                    dbp_loss.item() * batch_size_actual
                )

                val_samples += batch_size_actual


        # ========================================================
        # EPOCH VALIDATION METRICS
        # ========================================================

        epoch_val_loss = (
            running_val_loss /
            val_samples
        )

        epoch_val_sbp_mse = (
            running_val_sbp_mse /
            val_samples
        )

        epoch_val_dbp_mse = (
            running_val_dbp_mse /
            val_samples
        )

        epoch_val_sbp_rmse = np.sqrt(
            epoch_val_sbp_mse
        )

        epoch_val_dbp_rmse = np.sqrt(
            epoch_val_dbp_mse
        )


        # ========================================================
        # SAVE HISTORY
        # ========================================================

        train_losses.append(
            epoch_train_loss
        )

        train_sbp_mse.append(
            epoch_train_sbp_mse
        )

        train_dbp_mse.append(
            epoch_train_dbp_mse
        )

        train_sbp_rmse.append(
            epoch_train_sbp_rmse
        )

        train_dbp_rmse.append(
            epoch_train_dbp_rmse
        )


        val_losses.append(
            epoch_val_loss
        )

        val_sbp_mse.append(
            epoch_val_sbp_mse
        )

        val_dbp_mse.append(
            epoch_val_dbp_mse
        )

        val_sbp_rmse.append(
            epoch_val_sbp_rmse
        )

        val_dbp_rmse.append(
            epoch_val_dbp_rmse
        )


        # ========================================================
        # LEARNING RATE SCHEDULER
        # ========================================================

        scheduler.step(epoch_val_loss)


        # ========================================================
        # SAVE BEST MODEL
        # ========================================================

        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss

            best_model_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }


        # ========================================================
        # PRINT EPOCH
        # ========================================================

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"LR: {current_lr:.2e} | "
            f"Train MSE: {epoch_train_loss:.4f} | "
            f"Train SBP MSE: {epoch_train_sbp_mse:.4f} | "
            f"Train DBP MSE: {epoch_train_dbp_mse:.4f} | "
            f"Train SBP RMSE: {epoch_train_sbp_rmse:.4f} | "
            f"Train DBP RMSE: {epoch_train_dbp_rmse:.4f} | "
            f"Val MSE: {epoch_val_loss:.4f} | "
            f"Val SBP MSE: {epoch_val_sbp_mse:.4f} | "
            f"Val DBP MSE: {epoch_val_dbp_mse:.4f} | "
            f"Val SBP RMSE: {epoch_val_sbp_rmse:.4f} | "
            f"Val DBP RMSE: {epoch_val_dbp_rmse:.4f}"
        )


    # ============================================================
    # 9. LOAD BEST VALIDATION MODEL
    # ============================================================

    model.load_state_dict(
        best_model_state
    )

    model = model.to(device)

    print("\nBest validation loss:")
    print(f"{best_val_loss:.6f}")


    # ============================================================
    # 10. TEST SET PREDICTIONS
    # ============================================================

    model.eval()

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            y_pred = model(X_batch)

            all_y_pred.append(
                y_pred.cpu().numpy()
            )

            all_y_true.append(
                y_batch.numpy()
            )


    y_true = np.concatenate(
        all_y_true,
        axis=0
    )

    y_pred = np.concatenate(
        all_y_pred,
        axis=0
    )


    # ============================================================
    # 11. SEPARATE SBP / DBP
    # ============================================================

    y_true_sbp = y_true[:, 0]
    y_pred_sbp = y_pred[:, 0]

    y_true_dbp = y_true[:, 1]
    y_pred_dbp = y_pred[:, 1]


    # ============================================================
    # 12. SBP TEST RESULTS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("SBP TEST RESULTS")
    print("=" * 60)

    print_metrics(
        y_true_sbp,
        y_pred_sbp
    )

    plot_regression_results(
        y_true_sbp,
        y_pred_sbp,
        model_name="ResNet1D",
        target_name="SBP"
    )


    # ============================================================
    # 13. DBP TEST RESULTS
    # ============================================================

    print("\n")
    print("=" * 60)
    print("DBP TEST RESULTS")
    print("=" * 60)

    print_metrics(
        y_true_dbp,
        y_pred_dbp
    )

    plot_regression_results(
        y_true_dbp,
        y_pred_dbp,
        model_name="ResNet1D",
        target_name="DBP"
    )


    # ============================================================
    # 14. RETURN EVERYTHING
    # ============================================================

    history = {

        # Overall
        "train_loss": train_losses,
        "val_loss": val_losses,

        # SBP
        "train_sbp_mse": train_sbp_mse,
        "train_sbp_rmse": train_sbp_rmse,

        "val_sbp_mse": val_sbp_mse,
        "val_sbp_rmse": val_sbp_rmse,

        # DBP
        "train_dbp_mse": train_dbp_mse,
        "train_dbp_rmse": train_dbp_rmse,

        "val_dbp_mse": val_dbp_mse,
        "val_dbp_rmse": val_dbp_rmse
    }


    return (
        model,
        history,
        y_true,
        y_pred
    )


#########################################33333333

class PPGPhysicsDataset(Dataset):

    def __init__(self, X, y, delay):

        self.X = X
        self.y = y
        self.delay = delay

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx],
            self.delay[idx]
        )

def run_windkessel_pinn(
    X_train, y_train, delay_train,
    X_val, y_val, delay_val,
    X_test, y_test, delay_test,
    RC,
    device,
    lambda_phys,
    batch_size=256,
    epochs=50,
    patience=10,
    lr=1e-4,
    weight_decay=1e-3
):

    import copy
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils import clip_grad_norm_

    # ============================================================
    # 1. DATASET
    # ============================================================

    class PPGPhysicsDataset(Dataset):

        def __init__(self, X, y, delay):
            self.X = X
            self.y = y
            self.delay = delay

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return (
                self.X[idx],
                self.y[idx],
                self.delay[idx]
            )

    train_dataset = PPGPhysicsDataset(
        X_train, y_train, delay_train
    )

    val_dataset = PPGPhysicsDataset(
        X_val, y_val, delay_val
    )

    test_dataset = PPGPhysicsDataset(
        X_test, y_test, delay_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    # ============================================================
    # 2. MODEL
    # ============================================================

    class BasicBlock1D(nn.Module):

        def __init__(self, in_channels, out_channels, stride=1):

            super().__init__()

            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False
            )

            self.bn1 = nn.BatchNorm1d(out_channels)

            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            )

            self.bn2 = nn.BatchNorm1d(out_channels)

            self.downsample = None

            if stride != 1 or in_channels != out_channels:

                self.downsample = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_channels)
                )

        def forward(self, x):

            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class SmallResNet1D(nn.Module):

        def __init__(self):

            super().__init__()

            self.conv1 = nn.Conv1d(
                1,
                32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            )

            self.bn1 = nn.BatchNorm1d(32)
            self.relu = nn.ReLU(inplace=True)

            self.maxpool = nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )

            self.layer1 = nn.Sequential(
                BasicBlock1D(32, 32),
                BasicBlock1D(32, 32)
            )

            self.layer2 = nn.Sequential(
                BasicBlock1D(32, 64, stride=2),
                BasicBlock1D(64, 64)
            )

            self.layer3 = nn.Sequential(
                BasicBlock1D(64, 128, stride=2),
                BasicBlock1D(128, 128)
            )

            self.layer4 = nn.Sequential(
                BasicBlock1D(128, 256, stride=2),
                BasicBlock1D(256, 256)
            )

            self.avgpool = nn.AdaptiveAvgPool1d(1)

            self.dropout = nn.Dropout(0.4)

            self.fc = nn.Linear(256, 2)

        def forward(self, x):

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)

            x = torch.flatten(x, 1)

            x = self.dropout(x)

            x = self.fc(x)

            return x


    model = SmallResNet1D().to(device)

    # ============================================================
    # 3. OPTIMIZER + SCHEDULER
    # ============================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # ============================================================
    # 4. PHYSICS LOSS
    # ============================================================

    def windkessel_physics_loss(
        predictions,
        delays,
        RC
    ):

        sbp_pred = predictions[:, 0]
        dbp_pred = predictions[:, 1]

        delays = delays.float().view(-1)

        dbp_wk = (
            sbp_pred *
            torch.exp(-delays / RC)
        )

        physics_loss = torch.mean(
            (dbp_wk - dbp_pred) ** 2
        )

        return physics_loss


    # ============================================================
    # 5. PINN LOSS
    # ============================================================

    def pinn_loss(
        predictions,
        targets,
        delays
    ):

        # SBP MSE
        sbp_loss = nn.functional.mse_loss(
            predictions[:, 0],
            targets[:, 0]
        )

        # DBP MSE
        dbp_loss = nn.functional.mse_loss(
            predictions[:, 1],
            targets[:, 1]
        )

        # Windkessel physics loss
        physics_loss = windkessel_physics_loss(
            predictions,
            delays,
            RC
        )

        # Supervised BP loss
        bp_loss = sbp_loss + dbp_loss

        # Total PINN loss
        total_loss = (
            bp_loss +
            lambda_phys * physics_loss
        )

        return (
            total_loss,
            sbp_loss,
            dbp_loss,
            physics_loss
        )


    # ============================================================
    # 6. TRAINING
    # ============================================================

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_sbp_loss": [],
        "train_dbp_loss": [],
        "train_phys_loss": [],

        "val_loss": [],
        "val_sbp_loss": [],
        "val_dbp_loss": [],
        "val_phys_loss": [],

        "train_sbp_rmse": [],
        "train_dbp_rmse": [],

        "val_sbp_rmse": [],
        "val_dbp_rmse": []
    }

    for epoch in range(epochs):

        # ========================================================
        # TRAIN
        # ========================================================

        model.train()

        train_total = 0.0
        train_sbp = 0.0
        train_dbp = 0.0
        train_phys = 0.0

        train_sbp_errors = []
        train_dbp_errors = []

        for X, y, delay in train_loader:

            X = X.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            )

            y = y.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            )

            delay = delay.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            ).view(-1)

            # Forward pass
            predictions = model(X)

            # Calculate losses
            (
                total_loss,
                sbp_loss,
                dbp_loss,
                physics_loss
            ) = pinn_loss(
                predictions,
                y,
                delay
            )

            # Backpropagation
            optimizer.zero_grad()

            total_loss.backward()

            clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # Weighted accumulation
            bs = X.size(0)

            train_total += total_loss.item() * bs
            train_sbp += sbp_loss.item() * bs
            train_dbp += dbp_loss.item() * bs
            train_phys += physics_loss.item() * bs

            # Errors for RMSE
            train_sbp_errors.extend(
                (
                    predictions[:, 0] -
                    y[:, 0]
                )
                .detach()
                .cpu()
                .numpy()
            )

            train_dbp_errors.extend(
                (
                    predictions[:, 1] -
                    y[:, 1]
                )
                .detach()
                .cpu()
                .numpy()
            )

        n_train = len(train_loader.dataset)

        train_total /= n_train
        train_sbp /= n_train
        train_dbp /= n_train
        train_phys /= n_train

        train_sbp_rmse = np.sqrt(
            np.mean(
                np.array(train_sbp_errors) ** 2
            )
        )

        train_dbp_rmse = np.sqrt(
            np.mean(
                np.array(train_dbp_errors) ** 2
            )
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        model.eval()

        val_total = 0.0
        val_sbp = 0.0
        val_dbp = 0.0
        val_phys = 0.0

        val_sbp_errors = []
        val_dbp_errors = []

        with torch.no_grad():

            for X, y, delay in val_loader:

                X = X.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True
                )

                y = y.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True
                )

                delay = delay.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True
                ).view(-1)

                predictions = model(X)

                (
                    total_loss,
                    sbp_loss,
                    dbp_loss,
                    physics_loss
                ) = pinn_loss(
                    predictions,
                    y,
                    delay
                )

                bs = X.size(0)

                val_total += total_loss.item() * bs
                val_sbp += sbp_loss.item() * bs
                val_dbp += dbp_loss.item() * bs
                val_phys += physics_loss.item() * bs

                val_sbp_errors.extend(
                    (
                        predictions[:, 0] -
                        y[:, 0]
                    )
                    .cpu()
                    .numpy()
                )

                val_dbp_errors.extend(
                    (
                        predictions[:, 1] -
                        y[:, 1]
                    )
                    .cpu()
                    .numpy()
                )

        n_val = len(val_loader.dataset)

        val_total /= n_val
        val_sbp /= n_val
        val_dbp /= n_val
        val_phys /= n_val

        val_sbp_rmse = np.sqrt(
            np.mean(
                np.array(val_sbp_errors) ** 2
            )
        )

        val_dbp_rmse = np.sqrt(
            np.mean(
                np.array(val_dbp_errors) ** 2
            )
        )

        # ========================================================
        # LEARNING RATE SCHEDULER
        # ========================================================

        scheduler.step(val_total)

        # ========================================================
        # SAVE HISTORY
        # ========================================================

        history["train_loss"].append(train_total)
        history["train_sbp_loss"].append(train_sbp)
        history["train_dbp_loss"].append(train_dbp)
        history["train_phys_loss"].append(train_phys)

        history["val_loss"].append(val_total)
        history["val_sbp_loss"].append(val_sbp)
        history["val_dbp_loss"].append(val_dbp)
        history["val_phys_loss"].append(val_phys)

        history["train_sbp_rmse"].append(train_sbp_rmse)
        history["train_dbp_rmse"].append(train_dbp_rmse)

        history["val_sbp_rmse"].append(val_sbp_rmse)
        history["val_dbp_rmse"].append(val_dbp_rmse)

        # ========================================================
        # EARLY STOPPING
        # ========================================================

        if val_total < best_val_loss:

            best_val_loss = val_total

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            status = "✓"

        else:

            patience_counter += 1

            status = (
                f"patience "
                f"{patience_counter}/{patience}"
            )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"LR: {current_lr:.2e} | "
            f"Train Total: {train_total:.4f} | "
            f"Train SBP MSE: {train_sbp:.4f} | "
            f"Train DBP MSE: {train_dbp:.4f} | "
            f"Train Phys: {train_phys:.4f} | "
            f"Train RMSE: "
            f"SBP={train_sbp_rmse:.3f}, "
            f"DBP={train_dbp_rmse:.3f} | "
            f"Val Total: {val_total:.4f} | "
            f"Val SBP MSE: {val_sbp:.4f} | "
            f"Val DBP MSE: {val_dbp:.4f} | "
            f"Val Phys: {val_phys:.4f} | "
            f"Val RMSE: "
            f"SBP={val_sbp_rmse:.3f}, "
            f"DBP={val_dbp_rmse:.3f} | "
            f"{status}"
        )

        if patience_counter >= patience:

            print("\nEarly stopping.")

            break

    # ============================================================
    # 7. RESTORE BEST MODEL
    # ============================================================

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )

    # ============================================================
    # 8. TEST
    # ============================================================

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X, y, delay in test_loader:

            X = X.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            )

            y = y.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            )

            predictions = model(X)

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                y.cpu().numpy()
            )

    y_pred = np.concatenate(
        all_predictions,
        axis=0
    )

    y_true = np.concatenate(
        all_targets,
        axis=0
    )

    # ============================================================
    # 9. SBP / DBP
    # ============================================================

    y_true_sbp = y_true[:, 0]
    y_pred_sbp = y_pred[:, 0]

    y_true_dbp = y_true[:, 1]
    y_pred_dbp = y_pred[:, 1]

    # ============================================================
    # 10. METRICS + PLOTS
    # ============================================================

    print("\n" + "=" * 60)
    print("WINDKESSEL PINN — SBP TEST RESULTS")
    print("=" * 60)

    print_metrics(
        y_true_sbp,
        y_pred_sbp
    )

    plot_regression_results(
        y_true_sbp,
        y_pred_sbp,
        model_name="Windkessel PINN",
        target_name="SBP"
    )

    print("\n" + "=" * 60)
    print("WINDKESSEL PINN — DBP TEST RESULTS")
    print("=" * 60)

    print_metrics(
        y_true_dbp,
        y_pred_dbp
    )

    plot_regression_results(
        y_true_dbp,
        y_pred_dbp,
        model_name="Windkessel PINN",
        target_name="DBP"
    )

    # ============================================================
    # 11. RETURN RESULTS
    # ============================================================

    results = {
        "predictions": y_pred,
        "targets": y_true,

        "sbp_true": y_true_sbp,
        "sbp_pred": y_pred_sbp,

        "dbp_true": y_true_dbp,
        "dbp_pred": y_pred_dbp,

        "best_val_loss": best_val_loss,

        "history": history
    }

    return model, results





#######################################################
# WAVELETS

def process_recording_w(
    recording,
    window_size,
    step_size,
    fs=125
):

    data = recording[:]

    # -----------------------------------------
    # Extract channels
    # -----------------------------------------

    ppg = data[:, 0]
    abp = data[:, 1]
    ppg = robust_minmax_normalize(ppg)

    # -----------------------------------------
    # Recording-level PPG normalization
    # -----------------------------------------

    if ppg is None:
        return None, None, None
 

    X = []
    y = []
    delays = []

    # -----------------------------------------
    # Windowing
    # -----------------------------------------

    for start in range(
        0,
        len(ppg) - window_size + 1,
        step_size
    ):

        ppg_window = ppg[
            start:start + window_size
        ]

        abp_window = abp[
            start:start + window_size
        ]

        # -------------------------------------
        # Extract SBP / DBP
        # -------------------------------------

        sbp, dbp = extract_bp_from_abp(
            abp_window
        )

        if sbp is None:
            continue

        # -------------------------------------
        # Extract beat-level delays
        # -------------------------------------

        beat_delays = get_delay_time(
            ppg_window,
            fs
        )

        # -------------------------------------
        # No valid delay detected
        # -------------------------------------

        if len(beat_delays) == 0:
            continue

        # -------------------------------------
        # One representative delay per window
        # -------------------------------------

        window_delay = np.median(
            beat_delays
        )

        # -------------------------------------
        # Store
        # -------------------------------------

        X.append(ppg_window)

        y.append([
            sbp,
            dbp
        ])

        delays.append(
            window_delay
        )

    # -----------------------------------------
    # No valid windows
    # -----------------------------------------

    if len(X) == 0:
        return None, None, None

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(delays, dtype=np.float32)
    )




def process_split_w(
    record_list,
    window_size,
    step_size,
    fs=125
):

    X_all = []
    y_all = []
    delay_all = []

    skipped_recordings = 0

    # -----------------------------------------
    # Process each recording
    # -----------------------------------------

    for f, ref in tqdm(
        record_list,
        desc="Processing recordings"
    ):

        recording = f[ref]

        X, y, delays = process_recording_w(
            recording,
            window_size,
            step_size,
            fs
        )

        if X is None:

            skipped_recordings += 1

            continue

        X_all.append(X)
        y_all.append(y)
        delay_all.append(delays)

    # -----------------------------------------
    # Combine all recordings
    # -----------------------------------------

    X_all = np.concatenate(
        X_all,
        axis=0
    )

    y_all = np.concatenate(
        y_all,
        axis=0
    )

    delay_all = np.concatenate(
        delay_all,
        axis=0
    )

    print(
        "Skipped recordings:",
        skipped_recordings
    )

    return (
        X_all,
        y_all,
        delay_all
    )


def prepare_UCI_dataset_w(
    data_path,
    window_size,
    step_size,
    fs=125,
    test_size=0.20,
    val_size=0.10,
    random_state=42
):

    # ==================================================
    # 1. Load recording references
    # ==================================================

    recordings, files = load_UCI_recordings(
        data_path
    )

    # ==================================================
    # 2. SPLIT RECORDINGS FIRST
    # ==================================================

    (
        train_records,
        val_records,
        test_records
    ) = split_recordings(
        recordings,
        test_size,
        val_size,
        random_state
    )

    # ==================================================
    # 3. Process TRAIN recordings
    # ==================================================

    print("\nProcessing TRAIN recordings...")

    (
        X_train,
        y_train,
        delay_train
    ) = process_split_w(
        train_records,
        window_size,
        step_size,
        fs
    )

    # ==================================================
    # 4. Process VALIDATION recordings
    # ==================================================

    print("\nProcessing VALIDATION recordings...")

    (
        X_val,
        y_val,
        delay_val
    ) = process_split_w(
        val_records,
        window_size,
        step_size,
        fs
    )

    # ==================================================
    # 5. Process TEST recordings
    # ==================================================

    print("\nProcessing TEST recordings...")

    (
        X_test,
        y_test,
        delay_test
    ) = process_split_w(
        test_records,
        window_size,
        step_size,
        fs
    )

    # ==================================================
    # 6. Add CNN channel dimension
    # ==================================================

    X_train = X_train[:, None, :]
    X_val = X_val[:, None, :]
    X_test = X_test[:, None, :]

    # ==================================================
    # 7. Convert to tensors
    # ==================================================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    delay_train = torch.tensor(
        delay_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.float32
    )

    delay_val = torch.tensor(
        delay_val,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )

    delay_test = torch.tensor(
        delay_test,
        dtype=torch.float32
    )

    # ==================================================
    # 8. Print shapes
    # ==================================================

    print(
        "\n======================================"
    )

    print(
        "DATASET SHAPES"
    )

    print(
        "======================================"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "delay_train:",
        delay_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "y_val:",
        y_val.shape
    )

    print(
        "delay_val:",
        delay_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    print(
        "delay_test:",
        delay_test.shape
    )

    return (
        X_train,
        y_train,
        delay_train,

        X_val,
        y_val,
        delay_val,

        X_test,
        y_test,
        delay_test
    )
