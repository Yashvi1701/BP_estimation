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

def train_basic_cnn(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=256,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    device=None
):
    """
    Basic 1D CNN for SBP + DBP estimation from PPG.

    Expected X shape:
        (N, 1, signal_length)
    or
        (N, signal_length)

    Expected y shape:
        (N, 2)
        column 0 = SBP
        column 1 = DBP
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 1. Convert to tensors
    # ---------------------------------------------------------
    X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    X_val   = torch.tensor(np.asarray(X_val), dtype=torch.float32)
    X_test  = torch.tensor(np.asarray(X_test), dtype=torch.float32)

    y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32)
    y_val   = torch.tensor(np.asarray(y_val), dtype=torch.float32)
    y_test  = torch.tensor(np.asarray(y_test), dtype=torch.float32)

    # Add channel dimension if necessary
    if X_train.ndim == 2:
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    # ---------------------------------------------------------
    # 2. DataLoaders
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3. Basic CNN
    # ---------------------------------------------------------
    class BasicCNN(nn.Module):

        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(

                # Input: (B, 1, 1000)
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                # (B, 32, 500)
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),

                # (B, 64, 250)
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(),

                nn.AdaptiveAvgPool1d(1)
            )

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.fc(x)
            return x

    model = BasicCNN().to(device)

    print("\nDevice:", device)
    print(model)

    # ---------------------------------------------------------
    # 4. Loss + optimizer
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 5. Training
    # ---------------------------------------------------------
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sbp_rmse": [],
        "train_dbp_rmse": [],
        "val_sbp_rmse": [],
        "val_dbp_rmse": []
    }

    for epoch in range(epochs):

        # ---------------- TRAIN ----------------
        model.train()

        train_loss = 0.0
        train_sbp_sq = 0.0
        train_dbp_sq = 0.0
        train_n = 0

        for xb, yb in train_loader:

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_n = len(xb)

            train_loss += loss.item() * batch_n
            train_sbp_sq += torch.sum(
                (pred[:, 0] - yb[:, 0]) ** 2
            ).item()
            train_dbp_sq += torch.sum(
                (pred[:, 1] - yb[:, 1]) ** 2
            ).item()

            train_n += batch_n

        train_loss /= train_n

        train_sbp_rmse = np.sqrt(train_sbp_sq / train_n)
        train_dbp_rmse = np.sqrt(train_dbp_sq / train_n)

        # ---------------- VALIDATION ----------------
        model.eval()

        val_loss = 0.0
        val_sbp_sq = 0.0
        val_dbp_sq = 0.0
        val_n = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb)

                loss = criterion(pred, yb)

                batch_n = len(xb)

                val_loss += loss.item() * batch_n

                val_sbp_sq += torch.sum(
                    (pred[:, 0] - yb[:, 0]) ** 2
                ).item()

                val_dbp_sq += torch.sum(
                    (pred[:, 1] - yb[:, 1]) ** 2
                ).item()

                val_n += batch_n

        val_loss /= val_n

        val_sbp_rmse = np.sqrt(val_sbp_sq / val_n)
        val_dbp_rmse = np.sqrt(val_dbp_sq / val_n)

        scheduler.step(val_loss)

        # -------------------------------------------------
        # Save history
        # -------------------------------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_sbp_rmse"].append(train_sbp_rmse)
        history["train_dbp_rmse"].append(train_dbp_rmse)

        history["val_sbp_rmse"].append(val_sbp_rmse)
        history["val_dbp_rmse"].append(val_dbp_rmse)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train SBP RMSE: {train_sbp_rmse:.3f} | "
            f"Train DBP RMSE: {train_dbp_rmse:.3f} | "
            f"Val SBP RMSE: {val_sbp_rmse:.3f} | "
            f"Val DBP RMSE: {val_dbp_rmse:.3f} | "
            f"LR: {current_lr:.2e}")

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # ---------------------------------------------------------
    # 6. Restore best model
    # ---------------------------------------------------------
    model.load_state_dict(best_state)

    # ---------------------------------------------------------
    # 7. Test prediction
    # ---------------------------------------------------------
    model.eval()

    predictions = []

    with torch.no_grad():

        for xb, _ in test_loader:

            xb = xb.to(device, non_blocking=True)

            pred = model(xb)

            predictions.append(pred.cpu().numpy())

    y_pred = np.concatenate(predictions)

    y_true = y_test.numpy()

    # ---------------------------------------------------------
    # 8. Metrics
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASIC CNN TEST RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(y_true[:, 0], y_pred[:, 0])

    print("\nDBP")
    print_metrics(y_true[:, 1], y_pred[:, 1])

    # ---------------------------------------------------------
    # 9. Regression plots
    # ---------------------------------------------------------
    plot_regression_results(
        y_true[:, 0],
        y_pred[:, 0],
        model_name="Basic CNN",
        target_name="SBP"
    )

    plot_regression_results(
        y_true[:, 1],
        y_pred[:, 1],
        model_name="Basic CNN",
        target_name="DBP"
    )

    results = {
        "model": model,
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
        "best_val_loss": best_val_loss
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



import os
import glob
import numpy as np
import torch
from tqdm.auto import tqdm


def evaluate_cnn_vitaldb(
    model,
    base_dir,
    batch_size=256,
    device=None
):
    """
    Zero-shot evaluation of a CNN trained on UCI
    on VitalDB.

    No retraining / fine-tuning is performed.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    npz_files = sorted(
        glob.glob(os.path.join(base_dir, "case_*.npz"))
    )

    print("Number of VitalDB cases:", len(npz_files))

    if len(npz_files) == 0:
        raise ValueError(
            f"No case_*.npz files found in:\n{base_dir}"
        )

    sbp_true_all = []
    sbp_pred_all = []

    dbp_true_all = []
    dbp_pred_all = []

    case_ids = []

    # ---------------------------------------------------------
    # Process VitalDB cases
    # ---------------------------------------------------------
    with torch.no_grad():

        for file in tqdm(
            npz_files,
            desc="CNN VitalDB prediction",
            unit="case"
        ):

            data = np.load(file)

            X = data["X"]
            y = data["y"]

            # -------------------------------------------------
            # Ensure CNN input shape = (N, 1, 1000)
            # -------------------------------------------------
            if X.ndim == 2:
                X = X[:, np.newaxis, :]

            sbp_true = y[:, 0]
            dbp_true = y[:, 1]

            case_id = os.path.basename(file).replace(
                ".npz", ""
            )

            case_sbp_pred = []
            case_dbp_pred = []

            # -------------------------------------------------
            # Predict in batches
            # -------------------------------------------------
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

                pred = model(X_tensor)

                pred = pred.cpu().numpy()

                case_sbp_pred.append(
                    pred[:, 0]
                )

                case_dbp_pred.append(
                    pred[:, 1]
                )

            # -------------------------------------------------
            # Combine predictions for this case
            # -------------------------------------------------
            case_sbp_pred = np.concatenate(
                case_sbp_pred
            )

            case_dbp_pred = np.concatenate(
                case_dbp_pred
            )

            sbp_true_all.append(sbp_true)
            sbp_pred_all.append(case_sbp_pred)

            dbp_true_all.append(dbp_true)
            dbp_pred_all.append(case_dbp_pred)

            case_ids.extend(
                [case_id] * len(sbp_true)
            )

    # ---------------------------------------------------------
    # Combine all VitalDB cases
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASIC CNN")
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

    # ---------------------------------------------------------
    # Regression plots
    # ---------------------------------------------------------
    plot_regression_results(
        sbp_true_all,
        sbp_pred_all,
        model_name="Basic CNN - VitalDB Zero-Shot",
        target_name="SBP"
    )

    plot_regression_results(
        dbp_true_all,
        dbp_pred_all,
        model_name="Basic CNN - VitalDB Zero-Shot",
        target_name="DBP"
    )

    # ---------------------------------------------------------
    # Return results
    # ---------------------------------------------------------
    results = {
        "sbp_true": sbp_true_all,
        "sbp_pred": sbp_pred_all,
        "dbp_true": dbp_true_all,
        "dbp_pred": dbp_pred_all,
        "case_ids": case_ids
    }

    return results