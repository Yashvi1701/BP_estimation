import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def process_recording(recording, WINDOW_SIZE, STEP_SIZE ):

    data = recording[:]

    ppg = data[:,0]
    abp = data[:,1]
    ppg = remove_baseline_wander(ppg)
    ppg = robust_minmax_normalize(ppg)
    if ppg is None:
        return None, None
    ppg = ppg.astype(np.float32)  

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

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) 




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


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. RESNET BLOCK
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

        # If dimensions change, transform the residual
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
        else:
            self.downsample = nn.Identity()

    def forward(self, x):

        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# ============================================================
# 2. RESNET1D MODEL
# ============================================================

class ResNet1D(nn.Module):

    def __init__(self):

        super().__init__()

        # Initial convolution
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

        # ResNet stages
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

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

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

        x = self.fc(x)

        return x


# ============================================================
# 3. TRAIN RESNET
# ============================================================

def train_resnet1d(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=256,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-3,
    patience=10,
    device=None
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("Device:", device)

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    if not torch.is_tensor(X_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_train = X_train.float()

    if not torch.is_tensor(y_train):
        y_train = torch.tensor(y_train, dtype=torch.float32)
    else:
        y_train = y_train.float()

    if not torch.is_tensor(X_val):
        X_val = torch.tensor(X_val, dtype=torch.float32)
    else:
        X_val = X_val.float()

    if not torch.is_tensor(y_val):
        y_val = torch.tensor(y_val, dtype=torch.float32)
    else:
        y_val = y_val.float()

    if not torch.is_tensor(X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    else:
        X_test = X_test.float()

    if not torch.is_tensor(y_test):
        y_test = torch.tensor(y_test, dtype=torch.float32)
    else:
        y_test = y_test.float()

    # --------------------------------------------------------
    # Add channel dimension if necessary
    # (N, 1000) -> (N, 1, 1000)
    # --------------------------------------------------------

    if X_train.ndim == 2:
        X_train = X_train.unsqueeze(1)

    if X_val.ndim == 2:
        X_val = X_val.unsqueeze(1)

    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(1)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet1D().to(device)

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

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

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],

        "train_sbp_rmse": [],
        "train_dbp_rmse": [],

        "val_sbp_rmse": [],
        "val_dbp_rmse": []
    }

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_preds = []
        train_targets = []
        train_loss_total = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_total += (
                loss.item() * X_batch.size(0)
            )

            train_preds.append(
                predictions.detach().cpu().numpy()
            )

            train_targets.append(
                y_batch.detach().cpu().numpy()
            )

        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_loss = (
            train_loss_total /
            len(train_dataset)
        )

        train_preds = np.concatenate(
            train_preds,
            axis=0
        )

        train_targets = np.concatenate(
            train_targets,
            axis=0
        )

        train_sbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 0],
                train_preds[:, 0]
            )
        )

        train_dbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 1],
                train_preds[:, 1]
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_preds = []
        val_targets = []
        val_loss_total = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(
                    device,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    device,
                    non_blocking=True
                )

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss_total += (
                    loss.item() * X_batch.size(0)
                )

                val_preds.append(
                    predictions.cpu().numpy()
                )

                val_targets.append(
                    y_batch.cpu().numpy()
                )

        # ----------------------------------------------------
        # VALIDATION METRICS
        # ----------------------------------------------------

        val_loss = (
            val_loss_total /
            len(val_dataset)
        )

        val_preds = np.concatenate(
            val_preds,
            axis=0
        )

        val_targets = np.concatenate(
            val_targets,
            axis=0
        )

        val_sbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 0],
                val_preds[:, 0]
            )
        )

        val_dbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 1],
                val_preds[:, 1]
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_sbp_rmse"].append(
            train_sbp_rmse
        )

        history["train_dbp_rmse"].append(
            train_dbp_rmse
        )

        history["val_sbp_rmse"].append(
            val_sbp_rmse
        )

        history["val_dbp_rmse"].append(
            val_dbp_rmse
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train SBP RMSE: {train_sbp_rmse:.3f} | "
            f"Train DBP RMSE: {train_dbp_rmse:.3f} | "
            f"Val SBP RMSE: {val_sbp_rmse:.3f} | "
            f"Val DBP RMSE: {val_dbp_rmse:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:

                print(
                    f"\nEarly stopping at epoch "
                    f"{epoch+1}"
                )

                break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is not None:

        model.load_state_dict(best_state)

    # ========================================================
    # TEST
    # ========================================================

    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            predictions = model(X_batch)

            test_preds.append(
                predictions.cpu().numpy()
            )

            test_targets.append(
                y_batch.numpy()
            )

    y_pred = np.concatenate(
        test_preds,
        axis=0
    )

    y_true = np.concatenate(
        test_targets,
        axis=0
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("RESNET1D TEST RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(
        y_true[:, 0],
        y_pred[:, 0]
    )

    print("\nDBP")
    print_metrics(
        y_true[:, 1],
        y_pred[:, 1]
    )

    # --------------------------------------------------------
    # Regression plots
    # --------------------------------------------------------

    plot_regression_results(
        y_true[:, 0],
        y_pred[:, 0],
        model_name="ResNet1D",
        target_name="SBP"
    )

    plot_regression_results(
        y_true[:, 1],
        y_pred[:, 1],
        model_name="ResNet1D",
        target_name="DBP"
    )

    # ========================================================
    # RETURN
    # ========================================================

    results = {
        "predictions": y_pred,
        "targets": y_true,

        "sbp_true": y_true[:, 0],
        "sbp_pred": y_pred[:, 0],

        "dbp_true": y_true[:, 1],
        "dbp_pred": y_pred[:, 1],

        "best_val_loss": best_val_loss,

        "history": history
    }

    return model, results



import os
import glob
import numpy as np
import torch
from tqdm.auto import tqdm


def evaluate_resnet_vitaldb(
    model,
    base_dir,
    batch_size=256,
    device=None
):
    
    # ============================================================
    # DEVICE
    # ============================================================

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    # ============================================================
    # FIND VITALDB FILES
    # ============================================================

    npz_files = sorted(
        glob.glob(
            os.path.join(base_dir, "case_*.npz")
        )
    )

    print("Number of VitalDB cases:", len(npz_files))

    if len(npz_files) == 0:
        raise ValueError(
            f"No case_*.npz files found in:\n{base_dir}"
        )

    # ============================================================
    # STORAGE
    # ============================================================

    sbp_true_all = []
    sbp_pred_all = []

    dbp_true_all = []
    dbp_pred_all = []

    case_ids = []

    # ============================================================
    # PREDICTION
    # ============================================================

    with torch.no_grad():

        for file in tqdm(
            npz_files,
            desc="ResNet1D VitalDB prediction",
            unit="case"
        ):

            data = np.load(file)

            X = data["X"]
            y = data["y"]

            # ----------------------------------------------------
            # VitalDB may be:
            # (N, 1000)
            #
            # ResNet expects:
            # (N, 1, 1000)
            # ----------------------------------------------------

            if X.ndim == 2:
                X = X[:, np.newaxis, :]

            # ----------------------------------------------------
            # True BP
            # ----------------------------------------------------

            sbp_true = y[:, 0]
            dbp_true = y[:, 1]

            case_id = os.path.basename(file).replace(
                ".npz", ""
            )

            case_sbp_pred = []
            case_dbp_pred = []

            # ----------------------------------------------------
            # BATCHED PREDICTION
            # ----------------------------------------------------

            for start in range(
                0,
                len(X),
                batch_size
            ):

                end = min(
                    start + batch_size,
                    len(X)
                )

                X_batch = X[start:end]

                X_tensor = torch.tensor(
                    X_batch,
                    dtype=torch.float32
                ).to(
                    device,
                    non_blocking=True
                )

                predictions = model(X_tensor)

                predictions = predictions.cpu().numpy()

                case_sbp_pred.append(
                    predictions[:, 0]
                )

                case_dbp_pred.append(
                    predictions[:, 1]
                )

            # ----------------------------------------------------
            # JOIN BATCHES FOR THIS CASE
            # ----------------------------------------------------

            case_sbp_pred = np.concatenate(
                case_sbp_pred
            )

            case_dbp_pred = np.concatenate(
                case_dbp_pred
            )

            # ----------------------------------------------------
            # STORE
            # ----------------------------------------------------

            sbp_true_all.append(sbp_true)
            sbp_pred_all.append(case_sbp_pred)

            dbp_true_all.append(dbp_true)
            dbp_pred_all.append(case_dbp_pred)

            case_ids.extend(
                [case_id] * len(sbp_true)
            )

    # ============================================================
    # CONCATENATE ALL CASES
    # ============================================================

    sbp_true_all = np.concatenate(
        sbp_true_all
    )

    sbp_pred_all = np.concatenate(
        sbp_pred_all
    )

    dbp_true_all = np.concatenate(
        dbp_true_all
    )

    dbp_pred_all = np.concatenate(
        dbp_pred_all
    )

    # ============================================================
    # RESULTS
    # ============================================================

    print("\n" + "=" * 60)
    print("RESNET1D")
    print("VITALDB ZERO-SHOT RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(
        sbp_true_all,
        sbp_pred_all
    )

    print("\nDBP")
    print_metrics(
        dbp_true_all,
        dbp_pred_all
    )

    # ============================================================
    # PLOTS
    # ============================================================

    plot_regression_results(
        sbp_true_all,
        sbp_pred_all,
        model_name="ResNet1D - VitalDB Zero-Shot",
        target_name="SBP"
    )

    plot_regression_results(
        dbp_true_all,
        dbp_pred_all,
        model_name="ResNet1D - VitalDB Zero-Shot",
        target_name="DBP"
    )

    # ============================================================
    # RETURN
    # ============================================================

    results = {
        "sbp_true": sbp_true_all,
        "sbp_pred": sbp_pred_all,

        "dbp_true": dbp_true_all,
        "dbp_pred": dbp_pred_all,

        "case_ids": case_ids
    }

    return results



import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. SQUEEZE-AND-EXCITATION BLOCK
# ============================================================

class SEBlock1D(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Squeeze
        # (B, C, L) -> (B, C, 1)
        y = self.pool(x)

        # (B, C, 1) -> (B, C)
        y = y.view(y.size(0), -1)

        # Excitation
        y = self.fc(y)

        # (B, C) -> (B, C, 1)
        y = y.unsqueeze(2)

        # Reweight channels
        return x * y

# ============================================================
# 2. RESNET BLOCK + SE
# ============================================================

class SEBasicBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        reduction=16
    ):

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

        # ----------------------------------------------------
        # SE BLOCK
        # ----------------------------------------------------

        self.se = SEBlock1D(
            channels=out_channels,
            reduction=reduction
        )

        # ----------------------------------------------------
        # Residual projection if dimensions change
        # ----------------------------------------------------

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

        else:

            self.downsample = nn.Identity()

    def forward(self, x):

        identity = self.downsample(x)

        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)

        # ----------------------------------------------------
        # SE ACTIVATION
        # ----------------------------------------------------

        out = self.se(out)

        # Residual connection
        out += identity

        out = self.relu(out)

        return out


# ============================================================
# 3. RESNET1D + SE
# ============================================================

class ResNet1D_SE(nn.Module):

    def __init__(self, reduction=16):

        super().__init__()

        # ----------------------------------------------------
        # Initial convolution
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ResNet stages
        # ----------------------------------------------------

        self.layer1 = nn.Sequential(
            SEBasicBlock1D(
                32,
                32,
                reduction=reduction
            ),
            SEBasicBlock1D(
                32,
                32,
                reduction=reduction
            )
        )

        self.layer2 = nn.Sequential(
            SEBasicBlock1D(
                32,
                64,
                stride=2,
                reduction=reduction
            ),
            SEBasicBlock1D(
                64,
                64,
                reduction=reduction
            )
        )

        self.layer3 = nn.Sequential(
            SEBasicBlock1D(
                64,
                128,
                stride=2,
                reduction=reduction
            ),
            SEBasicBlock1D(
                128,
                128,
                reduction=reduction
            )
        )

        self.layer4 = nn.Sequential(
            SEBasicBlock1D(
                128,
                256,
                stride=2,
                reduction=reduction
            ),
            SEBasicBlock1D(
                256,
                256,
                reduction=reduction
            )
        )

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------
        # Regression head
        # ----------------------------------------------------

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

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

        x = self.fc(x)

        return x


# ============================================================
# 4. TRAIN RESNET + SE
# ============================================================

def train_resnet1d_se(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=256,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-3,
    patience=10,
    reduction=16,
    device=None
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("Device:", device)

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    if not torch.is_tensor(X_train):
        X_train = torch.tensor(
            X_train,
            dtype=torch.float32
        )
    else:
        X_train = X_train.float()

    if not torch.is_tensor(y_train):
        y_train = torch.tensor(
            y_train,
            dtype=torch.float32
        )
    else:
        y_train = y_train.float()

    if not torch.is_tensor(X_val):
        X_val = torch.tensor(
            X_val,
            dtype=torch.float32
        )
    else:
        X_val = X_val.float()

    if not torch.is_tensor(y_val):
        y_val = torch.tensor(
            y_val,
            dtype=torch.float32
        )
    else:
        y_val = y_val.float()

    if not torch.is_tensor(X_test):
        X_test = torch.tensor(
            X_test,
            dtype=torch.float32
        )
    else:
        X_test = X_test.float()

    if not torch.is_tensor(y_test):
        y_test = torch.tensor(
            y_test,
            dtype=torch.float32
        )
    else:
        y_test = y_test.float()

    # --------------------------------------------------------
    # Add channel dimension
    # (N, 1000) -> (N, 1, 1000)
    # --------------------------------------------------------

    if X_train.ndim == 2:
        X_train = X_train.unsqueeze(1)

    if X_val.ndim == 2:
        X_val = X_val.unsqueeze(1)

    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(1)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet1D_SE(
        reduction=reduction
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],

        "train_sbp_rmse": [],
        "train_dbp_rmse": [],

        "val_sbp_rmse": [],
        "val_dbp_rmse": []
    }

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss_total = 0.0

        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_total += (
                loss.item() * X_batch.size(0)
            )

            train_preds.append(
                predictions.detach().cpu().numpy()
            )

            train_targets.append(
                y_batch.detach().cpu().numpy()
            )

        train_loss = (
            train_loss_total /
            len(train_dataset)
        )

        train_preds = np.concatenate(
            train_preds,
            axis=0
        )

        train_targets = np.concatenate(
            train_targets,
            axis=0
        )

        train_sbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 0],
                train_preds[:, 0]
            )
        )

        train_dbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 1],
                train_preds[:, 1]
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss_total = 0.0

        val_preds = []
        val_targets = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(
                    device,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    device,
                    non_blocking=True
                )

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss_total += (
                    loss.item() * X_batch.size(0)
                )

                val_preds.append(
                    predictions.cpu().numpy()
                )

                val_targets.append(
                    y_batch.cpu().numpy()
                )

        val_loss = (
            val_loss_total /
            len(val_dataset)
        )

        val_preds = np.concatenate(
            val_preds,
            axis=0
        )

        val_targets = np.concatenate(
            val_targets,
            axis=0
        )

        val_sbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 0],
                val_preds[:, 0]
            )
        )

        val_dbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 1],
                val_preds[:, 1]
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_sbp_rmse"].append(
            train_sbp_rmse
        )

        history["train_dbp_rmse"].append(
            train_dbp_rmse
        )

        history["val_sbp_rmse"].append(
            val_sbp_rmse
        )

        history["val_dbp_rmse"].append(
            val_dbp_rmse
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train SBP RMSE: {train_sbp_rmse:.3f} | "
            f"Train DBP RMSE: {train_dbp_rmse:.3f} | "
            f"Val SBP RMSE: {val_sbp_rmse:.3f} | "
            f"Val DBP RMSE: {val_dbp_rmse:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:

                print(
                    f"\nEarly stopping at epoch "
                    f"{epoch+1}"
                )

                break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is not None:
        model.load_state_dict(best_state)

    # ========================================================
    # TEST
    # ========================================================

    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            predictions = model(X_batch)

            test_preds.append(
                predictions.cpu().numpy()
            )

            test_targets.append(
                y_batch.numpy()
            )

    y_pred = np.concatenate(
        test_preds,
        axis=0
    )

    y_true = np.concatenate(
        test_targets,
        axis=0
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print("\n" + "=" * 60)
    print("RESNET1D + SE TEST RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(
        y_true[:, 0],
        y_pred[:, 0]
    )

    print("\nDBP")
    print_metrics(
        y_true[:, 1],
        y_pred[:, 1]
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_regression_results(
        y_true[:, 0],
        y_pred[:, 0],
        model_name="ResNet1D + SE",
        target_name="SBP"
    )

    plot_regression_results(
        y_true[:, 1],
        y_pred[:, 1],
        model_name="ResNet1D + SE",
        target_name="DBP"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {
        "predictions": y_pred,
        "targets": y_true,

        "sbp_true": y_true[:, 0],
        "sbp_pred": y_pred[:, 0],

        "dbp_true": y_true[:, 1],
        "dbp_pred": y_pred[:, 1],

        "best_val_loss": best_val_loss,

        "history": history
    }

    return model, results



import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. CBAM - CHANNEL ATTENTION
# ============================================================

class ChannelAttention1D(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = nn.Sequential(
            nn.Conv1d(
                channels,
                hidden,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.mlp(
            self.avg_pool(x)
        )

        max_out = self.mlp(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg_out + max_out
        )

        return x * attention


# ============================================================
# 2. CBAM - TEMPORAL ATTENTION
# ============================================================

class TemporalAttention1D(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Average across channels
        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        # Maximum across channels
        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        # (B, 2, L)
        attention_input = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(attention_input)
        )

        return x * attention


# ============================================================
# 3. COMPLETE CBAM
# ============================================================

class CBAM1D(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16,
        temporal_kernel=7
    ):

        super().__init__()

        self.channel_attention = ChannelAttention1D(
            channels,
            reduction
        )

        self.temporal_attention = TemporalAttention1D(
            temporal_kernel
        )

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.temporal_attention(x)

        return x


# ============================================================
# 4. RESNET BLOCK + CBAM
# ============================================================

class CBAMBasicBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        reduction=16
    ):

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
            padding=3,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        # CBAM
        self.cbam = CBAM1D(
            out_channels,
            reduction=reduction,
            temporal_kernel=7
        )

        # Residual projection
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

        else:

            self.downsample = nn.Identity()

    def forward(self, x):

        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # CBAM
        out = self.cbam(out)

        # Residual
        out = out + identity

        out = self.relu(out)

        return out


# ============================================================
# 5. RESNET1D + CBAM MODEL
# ============================================================

class ResNet1D_CBAM(nn.Module):

    def __init__(self, reduction=16):

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
            CBAMBasicBlock1D(
                32,
                32,
                reduction=reduction
            ),
            CBAMBasicBlock1D(
                32,
                32,
                reduction=reduction
            )
        )

        self.layer2 = nn.Sequential(
            CBAMBasicBlock1D(
                32,
                64,
                stride=2,
                reduction=reduction
            ),
            CBAMBasicBlock1D(
                64,
                64,
                reduction=reduction
            )
        )

        self.layer3 = nn.Sequential(
            CBAMBasicBlock1D(
                64,
                128,
                stride=2,
                reduction=reduction
            ),
            CBAMBasicBlock1D(
                128,
                128,
                reduction=reduction
            )
        )

        self.layer4 = nn.Sequential(
            CBAMBasicBlock1D(
                128,
                256,
                stride=2,
                reduction=reduction
            ),
            CBAMBasicBlock1D(
                256,
                256,
                reduction=reduction
            )
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

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

        x = self.fc(x)

        return x


# ============================================================
# 6. TRAIN RESNET + CBAM
# ============================================================

def train_resnet1d_cbam(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=256,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-3,
    patience=10,
    reduction=16,
    device=None
):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("Device:", device)

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train = torch.as_tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.as_tensor(
        y_train,
        dtype=torch.float32
    )

    X_val = torch.as_tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.as_tensor(
        y_val,
        dtype=torch.float32
    )

    X_test = torch.as_tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.as_tensor(
        y_test,
        dtype=torch.float32
    )

    # Add channel dimension
    if X_train.ndim == 2:
        X_train = X_train.unsqueeze(1)

    if X_val.ndim == 2:
        X_val = X_val.unsqueeze(1)

    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(1)

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet1D_CBAM(
        reduction=reduction
    ).to(device)

    criterion = nn.MSELoss()

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

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sbp_rmse": [],
        "train_dbp_rmse": [],
        "val_sbp_rmse": [],
        "val_dbp_rmse": []
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(epochs):

        model.train()

        train_loss_total = 0
        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_total += (
                loss.item() * X_batch.size(0)
            )

            train_preds.append(
                predictions.detach().cpu().numpy()
            )

            train_targets.append(
                y_batch.detach().cpu().numpy()
            )

        train_loss = (
            train_loss_total /
            len(X_train)
        )

        train_preds = np.concatenate(
            train_preds
        )

        train_targets = np.concatenate(
            train_targets
        )

        train_sbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 0],
                train_preds[:, 0]
            )
        )

        train_dbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 1],
                train_preds[:, 1]
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_total = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(
                    device,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    device,
                    non_blocking=True
                )

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss_total += (
                    loss.item() * X_batch.size(0)
                )

                val_preds.append(
                    predictions.cpu().numpy()
                )

                val_targets.append(
                    y_batch.cpu().numpy()
                )

        val_loss = (
            val_loss_total /
            len(X_val)
        )

        val_preds = np.concatenate(
            val_preds
        )

        val_targets = np.concatenate(
            val_targets
        )

        val_sbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 0],
                val_preds[:, 0]
            )
        )

        val_dbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 1],
                val_preds[:, 1]
            )
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_sbp_rmse"].append(
            train_sbp_rmse
        )

        history["train_dbp_rmse"].append(
            train_dbp_rmse
        )

        history["val_sbp_rmse"].append(
            val_sbp_rmse
        )

        history["val_dbp_rmse"].append(
            val_dbp_rmse
        )

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train SBP RMSE: {train_sbp_rmse:.3f} | "
            f"Train DBP RMSE: {train_dbp_rmse:.3f} | "
            f"Val SBP RMSE: {val_sbp_rmse:.3f} | "
            f"Val DBP RMSE: {val_dbp_rmse:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:

                print(
                    f"\nEarly stopping at epoch "
                    f"{epoch+1}"
                )

                break

    # ========================================================
    # BEST MODEL
    # ========================================================

    if best_state is not None:
        model.load_state_dict(best_state)

    # ========================================================
    # TEST
    # ========================================================

    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            predictions = model(X_batch)

            test_preds.append(
                predictions.cpu().numpy()
            )

            test_targets.append(
                y_batch.numpy()
            )

    y_pred = np.concatenate(test_preds)
    y_true = np.concatenate(test_targets)

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("RESNET1D + CBAM TEST RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(
        y_true[:, 0],
        y_pred[:, 0]
    )

    print("\nDBP")
    print_metrics(
        y_true[:, 1],
        y_pred[:, 1]
    )

    plot_regression_results(
        y_true[:, 0],
        y_pred[:, 0],
        model_name="ResNet1D + CBAM",
        target_name="SBP"
    )

    plot_regression_results(
        y_true[:, 1],
        y_pred[:, 1],
        model_name="ResNet1D + CBAM",
        target_name="DBP"
    )

    results = {
        "predictions": y_pred,
        "targets": y_true,
        "sbp_true": y_true[:, 0],
        "sbp_pred": y_pred[:, 0],
        "dbp_true": y_true[:, 1],
        "dbp_pred": y_pred[:, 1],
        "best_val_loss": best_val_loss,
        "history": history
    }

    return model, results



# ============================================================
# 1. DILATED RESNET BLOCK
# ============================================================

class DilatedBasicBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1
    ):

        super().__init__()

        padding = dilation * 3

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        # ----------------------------------------------------
        # Residual projection
        # ----------------------------------------------------

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.downsample = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(
                    out_channels
                )
            )

        else:

            self.downsample = nn.Identity()

    def forward(self, x):

        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out


# ============================================================
# 2. DILATED RESNET1D
# ============================================================

class ResNet1D_Dilated(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Initial convolution
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            1,
            32,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU(
            inplace=True
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ----------------------------------------------------
        # Residual layers
        # ----------------------------------------------------

        self.layer1 = nn.Sequential(

            DilatedBasicBlock1D(
                32,
                32,
                dilation=1
            ),

            DilatedBasicBlock1D(
                32,
                32,
                dilation=1
            )
        )

        self.layer2 = nn.Sequential(

            DilatedBasicBlock1D(
                32,
                64,
                stride=2,
                dilation=2
            ),

            DilatedBasicBlock1D(
                64,
                64,
                dilation=2
            )
        )

        self.layer3 = nn.Sequential(

            DilatedBasicBlock1D(
                64,
                128,
                stride=2,
                dilation=4
            ),

            DilatedBasicBlock1D(
                128,
                128,
                dilation=4
            )
        )

        self.layer4 = nn.Sequential(

            DilatedBasicBlock1D(
                128,
                256,
                stride=2,
                dilation=8
            ),

            DilatedBasicBlock1D(
                256,
                256,
                dilation=8
            )
        )

        # ----------------------------------------------------
        # Pooling + regression
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

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

        x = self.fc(x)

        return x



# ============================================================
# 3. TRAIN DILATED RESNET
# ============================================================

def train_resnet1d_dilated(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=256,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-3,
    patience=10,
    device=None
):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("Device:", device)

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train = torch.as_tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.as_tensor(
        y_train,
        dtype=torch.float32
    )

    X_val = torch.as_tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.as_tensor(
        y_val,
        dtype=torch.float32
    )

    X_test = torch.as_tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.as_tensor(
        y_test,
        dtype=torch.float32
    )

    # Add channel dimension
    if X_train.ndim == 2:
        X_train = X_train.unsqueeze(1)

    if X_val.ndim == 2:
        X_val = X_val.unsqueeze(1)

    if X_test.ndim == 2:
        X_test = X_test.unsqueeze(1)

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val,
            y_val
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        TensorDataset(
            X_test,
            y_test
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet1D_Dilated().to(device)

    criterion = nn.MSELoss()

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

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sbp_rmse": [],
        "train_dbp_rmse": [],
        "val_sbp_rmse": [],
        "val_dbp_rmse": []
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(epochs):

        model.train()

        train_loss_total = 0

        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_total += (
                loss.item()
                * X_batch.size(0)
            )

            train_preds.append(
                predictions.detach().cpu().numpy()
            )

            train_targets.append(
                y_batch.detach().cpu().numpy()
            )

        train_loss = (
            train_loss_total /
            len(X_train)
        )

        train_preds = np.concatenate(
            train_preds
        )

        train_targets = np.concatenate(
            train_targets
        )

        train_sbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 0],
                train_preds[:, 0]
            )
        )

        train_dbp_rmse = np.sqrt(
            mean_squared_error(
                train_targets[:, 1],
                train_preds[:, 1]
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_total = 0

        val_preds = []
        val_targets = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(
                    device,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    device,
                    non_blocking=True
                )

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss_total += (
                    loss.item()
                    * X_batch.size(0)
                )

                val_preds.append(
                    predictions.cpu().numpy()
                )

                val_targets.append(
                    y_batch.cpu().numpy()
                )

        val_loss = (
            val_loss_total /
            len(X_val)
        )

        val_preds = np.concatenate(
            val_preds
        )

        val_targets = np.concatenate(
            val_targets
        )

        val_sbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 0],
                val_preds[:, 0]
            )
        )

        val_dbp_rmse = np.sqrt(
            mean_squared_error(
                val_targets[:, 1],
                val_preds[:, 1]
            )
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_sbp_rmse"].append(
            train_sbp_rmse
        )

        history["train_dbp_rmse"].append(
            train_dbp_rmse
        )

        history["val_sbp_rmse"].append(
            val_sbp_rmse
        )

        history["val_dbp_rmse"].append(
            val_dbp_rmse
        )

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train SBP RMSE: {train_sbp_rmse:.3f} | "
            f"Train DBP RMSE: {train_dbp_rmse:.3f} | "
            f"Val SBP RMSE: {val_sbp_rmse:.3f} | "
            f"Val DBP RMSE: {val_dbp_rmse:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:

                print(
                    f"\nEarly stopping at epoch "
                    f"{epoch+1}"
                )

                break

    # ========================================================
    # RESTORE BEST
    # ========================================================

    if best_state is not None:
        model.load_state_dict(best_state)

    # ========================================================
    # TEST
    # ========================================================

    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            predictions = model(X_batch)

            test_preds.append(
                predictions.cpu().numpy()
            )

            test_targets.append(
                y_batch.numpy()
            )

    y_pred = np.concatenate(test_preds)
    y_true = np.concatenate(test_targets)

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("RESNET1D + DILATED CONV TEST RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(
        y_true[:, 0],
        y_pred[:, 0]
    )

    print("\nDBP")
    print_metrics(
        y_true[:, 1],
        y_pred[:, 1]
    )

    plot_regression_results(
        y_true[:, 0],
        y_pred[:, 0],
        model_name="ResNet1D + Dilated Conv",
        target_name="SBP"
    )

    plot_regression_results(
        y_true[:, 1],
        y_pred[:, 1],
        model_name="ResNet1D + Dilated Conv",
        target_name="DBP"
    )

    results = {
        "predictions": y_pred,
        "targets": y_true,
        "sbp_true": y_true[:, 0],
        "sbp_pred": y_pred[:, 0],
        "dbp_true": y_true[:, 1],
        "dbp_pred": y_pred[:, 1],
        "best_val_loss": best_val_loss,
        "history": history
    }

    return model, results



def plot_loss_vs_epoch(history, model_name="Basic CNN"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name} - Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


