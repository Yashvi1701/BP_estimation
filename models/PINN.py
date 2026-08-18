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

def create_small_resnet1d():
    
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
            
            # SBP + DBP regression head
            self.fc = nn.Linear(256, 2)
        
        
        def _make_layer(self, out_channels, blocks, stride):
            
            layers = []
            
            # First block
            layers.append(
                BasicBlock1D(
                    self.in_channels,
                    out_channels,
                    stride
                )
            )
            
            self.in_channels = out_channels
            
            # Remaining blocks
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
            
            # Input: [B, 1, 1000]
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            # Residual stages
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            # Global average pooling
            x = self.avgpool(x)
            
            # [B, 256, 1] -> [B, 256]
            x = torch.flatten(x, start_dim=1)
            
            # Dropout
            x = self.dropout(x)
            
            # Regression
            # [:, 0] = SBP
            # [:, 1] = DBP
            x = self.fc(x)
            
            return x
    
    
    return SmallResNet1D()


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

    
def create_data_loaders_pinn(
    X_train, y_train, delay_train,
    X_val, y_val, delay_val,
    X_test, y_test, delay_test,
    batch_size=256
):
    
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
    
    return train_loader, val_loader, test_loader

def windkessel_physics_loss(
    predictions,
    delays,
    RC
):

    # -----------------------------------------
    # Network predictions
    # -----------------------------------------

    sbp_pred = predictions[:, 0]
    dbp_pred = predictions[:, 1]


    # -----------------------------------------
    # Windkessel-inspired DBP
    #
    # DBP_WK = SBP * exp(-delay / RC)
    # -----------------------------------------

    dbp_wk = (
        sbp_pred *
        torch.exp(
            -delays / RC
        )
    )


    # -----------------------------------------
    # Physics consistency loss
    # -----------------------------------------

    physics_loss = torch.mean(
        (
            dbp_wk - dbp_pred
        ) ** 2
    )


    return physics_loss

import copy
def pinn_loss(
    predictions,
    targets,
    delays,
    RC,
    lambda_phys=0.01
):

    # -----------------------------------------
    # Supervised loss
    # -----------------------------------------

    bp_loss = nn.functional.mse_loss(
        predictions,
        targets
    )


    # -----------------------------------------
    # Windkessel loss
    # -----------------------------------------

    physics_loss = windkessel_physics_loss(
        predictions,
        delays,
        RC
    )


    # -----------------------------------------
    # Total loss
    # -----------------------------------------

    total = (
        bp_loss
        +
        lambda_phys * physics_loss
    )


    return (
        total,
        bp_loss,
        physics_loss
    )


def train_model_pinn(
    model,
    train_loader,
    val_loader,
    optimizer,
    RC,
    device,
    epochs=50,
    lambda_phys=0.001,
    patience=10
):

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):

        # ==================================================
        # TRAIN
        # ==================================================

        model.train()

        train_total = 0.0
        train_bp = 0.0
        train_phys = 0.0

        train_sbp_errors = []
        train_dbp_errors = []

        for X, y, delay in train_loader:

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            delay = delay.to(device, non_blocking=True)

            predictions = model(X)

            total_loss, bp_loss, physics_loss = pinn_loss(
                predictions,
                y,
                delay,
                RC,
                lambda_phys=lambda_phys
            )

            optimizer.zero_grad()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            bs = X.size(0)

            train_total += total_loss.item() * bs
            train_bp += bp_loss.item() * bs
            train_phys += physics_loss.item() * bs

            train_sbp_errors.append(
                (predictions[:, 0] - y[:, 0])
                .detach()
                .cpu()
            )

            train_dbp_errors.append(
                (predictions[:, 1] - y[:, 1])
                .detach()
                .cpu()
            )

        # ==================================================
        # TRAIN METRICS
        # ==================================================

        n_train = len(train_loader.dataset)

        train_total /= n_train
        train_bp /= n_train
        train_phys /= n_train

        train_sbp_errors = torch.cat(train_sbp_errors)
        train_dbp_errors = torch.cat(train_dbp_errors)

        train_sbp_rmse = torch.sqrt(
            torch.mean(train_sbp_errors ** 2)
        ).item()

        train_dbp_rmse = torch.sqrt(
            torch.mean(train_dbp_errors ** 2)
        ).item()

        # ==================================================
        # VALIDATION
        # ==================================================

        model.eval()

        val_total = 0.0
        val_bp = 0.0
        val_phys = 0.0

        val_sbp_errors = []
        val_dbp_errors = []

        with torch.no_grad():

            for X, y, delay in val_loader:

                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                delay = delay.to(device, non_blocking=True)

                predictions = model(X)

                total_loss, bp_loss, physics_loss = pinn_loss(
                    predictions,
                    y,
                    delay,
                    RC,
                    lambda_phys=lambda_phys
                )

                bs = X.size(0)

                val_total += total_loss.item() * bs
                val_bp += bp_loss.item() * bs
                val_phys += physics_loss.item() * bs

                val_sbp_errors.append(
                    (predictions[:, 0] - y[:, 0]).cpu()
                )

                val_dbp_errors.append(
                    (predictions[:, 1] - y[:, 1]).cpu()
                )

        # ==================================================
        # VALIDATION METRICS
        # ==================================================

        n_val = len(val_loader.dataset)

        val_total /= n_val
        val_bp /= n_val
        val_phys /= n_val

        val_sbp_errors = torch.cat(val_sbp_errors)
        val_dbp_errors = torch.cat(val_dbp_errors)

        val_sbp_rmse = torch.sqrt(
            torch.mean(val_sbp_errors ** 2)
        ).item()

        val_dbp_rmse = torch.sqrt(
            torch.mean(val_dbp_errors ** 2)
        ).item()

        # ==================================================
        # EARLY STOPPING
        # ==================================================

        if val_total < best_val_loss:

            best_val_loss = val_total

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            status = "Improved"

        else:

            patience_counter += 1

            status = (
                f"No improvement "
                f"({patience_counter}/{patience})"
            )

        # ==================================================
        # PRINT
        # ==================================================

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_total:.2f} "
            f"(BP: {train_bp:.2f}, "
            f"Phys: {train_phys:.2f}) | "
            f"Train SBP RMSE: {train_sbp_rmse:.2f} | "
            f"Train DBP RMSE: {train_dbp_rmse:.2f} | "
            f"Val Loss: {val_total:.2f} "
            f"(BP: {val_bp:.2f}, "
            f"Phys: {val_phys:.2f}) | "
            f"Val SBP RMSE: {val_sbp_rmse:.2f} | "
            f"Val DBP RMSE: {val_dbp_rmse:.2f} | "
            f"{status}"
        )

        # ==================================================
        # EARLY STOP
        # ==================================================

        if patience_counter >= patience:

            print("Early stopping.")

            break

    # ==================================================
    # RESTORE BEST MODEL
    # ==================================================

    if best_model_state is not None:

        model.load_state_dict(best_model_state)

    return model, best_val_loss

def evaluate_model_pinn(model, test_loader, device):

    model.eval()

    test_sbp_errors = []
    test_dbp_errors = []

    test_predictions = []
    test_targets = []

    with torch.no_grad():

        for X, y, delay in test_loader:

            X = X.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            predictions = model(X)

            # Errors
            test_sbp_errors.append(
                (predictions[:, 0] - y[:, 0]).cpu()
            )

            test_dbp_errors.append(
                (predictions[:, 1] - y[:, 1]).cpu()
            )

            # Store predictions and targets
            test_predictions.append(
                predictions.cpu()
            )

            test_targets.append(
                y.cpu()
            )

    # Concatenate batches
    test_sbp_errors = torch.cat(test_sbp_errors)
    test_dbp_errors = torch.cat(test_dbp_errors)

    test_predictions = torch.cat(test_predictions)
    test_targets = torch.cat(test_targets)

    # MAE
    test_sbp_mae = torch.mean(
        torch.abs(test_sbp_errors)
    ).item()

    test_dbp_mae = torch.mean(
        torch.abs(test_dbp_errors)
    ).item()

    # RMSE
    test_sbp_rmse = torch.sqrt(
        torch.mean(test_sbp_errors ** 2)
    ).item()

    test_dbp_rmse = torch.sqrt(
        torch.mean(test_dbp_errors ** 2)
    ).item()

    # Print results
    print("\n================================")
    print("WINDKESSEL PINN TEST RESULTS")
    print("================================")

    print(
        f"SBP MAE  : {test_sbp_mae:.2f} mmHg"
    )

    print(
        f"SBP RMSE : {test_sbp_rmse:.2f} mmHg"
    )

    print(
        f"DBP MAE  : {test_dbp_mae:.2f} mmHg"
    )

    print(
        f"DBP RMSE : {test_dbp_rmse:.2f} mmHg"
    )

    return {
        "sbp_mae": test_sbp_mae,
        "sbp_rmse": test_sbp_rmse,
        "dbp_mae": test_dbp_mae,
        "dbp_rmse": test_dbp_rmse,
        "predictions": test_predictions,
        "targets": test_targets
    }

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def plot_mae(results):

    predictions = results["predictions"].numpy()
    targets = results["targets"].numpy()

    # ------------------------------------------
    # SBP
    # ------------------------------------------

    sbp_actual = targets[:, 0]
    sbp_pred = predictions[:, 0]

    sbp_errors = sbp_pred - sbp_actual
    sbp_abs_errors = np.abs(sbp_errors)

    sbp_mae = np.mean(sbp_abs_errors)
    sbp_variance = np.var(sbp_errors)

    # ------------------------------------------
    # DBP
    # ------------------------------------------

    dbp_actual = targets[:, 1]
    dbp_pred = predictions[:, 1]

    dbp_errors = dbp_pred - dbp_actual
    dbp_abs_errors = np.abs(dbp_errors)

    dbp_mae = np.mean(dbp_abs_errors)
    dbp_variance = np.var(dbp_errors)

    # ------------------------------------------
    # Print statistics
    # ------------------------------------------

    print("\n================================")
    print("TEST ERROR STATISTICS")
    print("================================")

    print(f"SBP MAE      : {sbp_mae:.2f} mmHg")
    print(f"SBP Variance : {sbp_variance:.2f} mmHg²")
    print(f"SBP Std Dev  : {np.sqrt(sbp_variance):.2f} mmHg")

    print()

    print(f"DBP MAE      : {dbp_mae:.2f} mmHg")
    print(f"DBP Variance : {dbp_variance:.2f} mmHg²")
    print(f"DBP Std Dev  : {np.sqrt(dbp_variance):.2f} mmHg")

    # ==========================================
    # SBP Scatter
    # ==========================================

    plt.figure(figsize=(7, 7))

    plt.scatter(
        sbp_actual,
        sbp_pred,
        alpha=0.3,
        s=10
    )

    # Perfect prediction line
    min_val = min(sbp_actual.min(), sbp_pred.min())
    max_val = max(sbp_actual.max(), sbp_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual SBP (mmHg)")
    plt.ylabel("Predicted SBP (mmHg)")
    plt.title(
        f"SBP: Actual vs Predicted\n"
        f"MAE = {sbp_mae:.2f} mmHg | "
        f"Variance = {sbp_variance:.2f}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ==========================================
    # DBP Scatter
    # ==========================================

    plt.figure(figsize=(7, 7))

    plt.scatter(
        dbp_actual,
        dbp_pred,
        alpha=0.3,
        s=10
    )

    # Perfect prediction line
    min_val = min(dbp_actual.min(), dbp_pred.min())
    max_val = max(dbp_actual.max(), dbp_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual DBP (mmHg)")
    plt.ylabel("Predicted DBP (mmHg)")
    plt.title(
        f"DBP: Actual vs Predicted\n"
        f"MAE = {dbp_mae:.2f} mmHg | "
        f"Variance = {dbp_variance:.2f}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()