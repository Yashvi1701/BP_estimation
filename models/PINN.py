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
    fs=125
):

    if sys_idx is None:
        return None

    # ------------------------------------------
    # Calculate smoothed PPG, VPG and APG
    # ------------------------------------------

    smooth_ppg, vpg, apg = calculate_derivatives(
        ppg_window,
        fs
    )

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

    delays = []

    # -----------------------------------------
    # Process each systolic peak
    # -----------------------------------------

    for sys_idx in sys_indices:

        notch_idx = get_dicrotic_notch(
            ppg_window,
            sys_idx,
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

FS = 125

WINDOW_SIZE = 1000
STEP_SIZE = 500

TEST_SIZE = 0.20
VAL_SIZE = 0.10

RANDOM_STATE = 42

def extract_bp_from_abp(abp_window):

    # -----------------------------------------
    # Detect systolic peaks in ABP
    # -----------------------------------------

    peaks, _ = find_peaks(
        abp_window,
        distance=50,
        prominence=5
    )

    # Need at least two peaks to calculate
    # the pressure between beats
    if len(peaks) < 2:
        return None, None

    # -----------------------------------------
    # SBP
    # -----------------------------------------

    sbp_values = abp_window[peaks]

    # -----------------------------------------
    # DBP
    # Minimum pressure between consecutive
    # systolic peaks
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
    # Use median across beats
    # -----------------------------------------

    sbp = np.median(sbp_values)

    dbp = np.median(dbp_values)

    # -----------------------------------------
    # Sanity checks
    # -----------------------------------------

    if sbp < 50 or sbp > 250:
        return None, None

    if dbp < 20 or dbp > 150:
        return None, None

    return sbp, dbp


def process_recording(
    recording,
    window_size=1000,
    step_size=500,
    fs=125
):

    data = recording[:]

    # -----------------------------------------
    # Extract channels
    # -----------------------------------------

    ppg = data[:, 0]
    abp = data[:, 1]

    # -----------------------------------------
    # Recording-level PPG normalization
    # -----------------------------------------

    mean = np.mean(ppg)
    std = np.std(ppg)

    if std == 0:
        return None, None, None

    ppg = (ppg - mean) / std

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
    window_size=1000,
    step_size=500,
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
    window_size=1000,
    step_size=500,
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