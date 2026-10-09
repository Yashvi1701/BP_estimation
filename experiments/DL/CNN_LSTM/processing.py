import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import os
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




import copy
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error


class CNN_LSTM_Baseline(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # LSTM temporal feature learning
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        # SBP and DBP regression
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):

        # Input: (batch, 1, sequence_length)
        x = self.cnn(x)

        # (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)

        # LSTM output: (batch, time, hidden_size)
        x, _ = self.lstm(x)

        # Use the last time-step representation
        x = x[:, -1, :]

        # Output: (batch, 2) -> SBP, DBP
        return self.regressor(x)


def train_cnn_lstm_baseline(
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

    def to_tensor(x):
        if not torch.is_tensor(x):
            return torch.tensor(x, dtype=torch.float32)
        return x.float()

    X_train, y_train = to_tensor(X_train), to_tensor(y_train)
    X_val, y_val = to_tensor(X_val), to_tensor(y_val)
    X_test, y_test = to_tensor(X_test), to_tensor(y_test)

    # Add channel dimension if needed
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

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNN_LSTM_Baseline().to(device)

    # --------------------------------------------------------
    # Loss, optimizer and scheduler
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

        train_loss_total = 0.0
        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_total += loss.item() * X_batch.size(0)

            train_preds.append(predictions.detach().cpu().numpy())
            train_targets.append(y_batch.detach().cpu().numpy())

        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_loss = train_loss_total / len(train_dataset)

        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

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

                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                val_loss_total += loss.item() * X_batch.size(0)

                val_preds.append(predictions.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        # ----------------------------------------------------
        # VALIDATION METRICS
        # ----------------------------------------------------

        val_loss = val_loss_total / len(val_dataset)

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

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

        history["train_sbp_rmse"].append(train_sbp_rmse)
        history["train_dbp_rmse"].append(train_dbp_rmse)

        history["val_sbp_rmse"].append(val_sbp_rmse)
        history["val_dbp_rmse"].append(val_dbp_rmse)

        # ----------------------------------------------------
        # Print one line per epoch
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
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
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
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

            X_batch = X_batch.to(device, non_blocking=True)

            predictions = model(X_batch)

            test_preds.append(predictions.cpu().numpy())
            test_targets.append(y_batch.numpy())

    y_pred = np.concatenate(test_preds, axis=0)
    y_true = np.concatenate(test_targets, axis=0)

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("CNN-LSTM TEST RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(y_true[:, 0], y_pred[:, 0])

    print("\nDBP")
    print_metrics(y_true[:, 1], y_pred[:, 1])

    # --------------------------------------------------------
    # Regression plots
    # --------------------------------------------------------

    plot_regression_results(
        y_true[:, 0],
        y_pred[:, 0],
        model_name="CNN-LSTM",
        target_name="SBP"
    )

    plot_regression_results(
        y_true[:, 1],
        y_pred[:, 1],
        model_name="CNN-LSTM",
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



def plot_loss_vs_epoch(history, model_name):
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




import glob
def evaluate_cnn_lstm_vitaldb(
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
            desc="CNN-LSTM VitalDB prediction",
            unit="case"
        ):

            with np.load(file) as data:
                X = data["X"]
                y = data["y"]

                # ------------------------------------------------
                # Input shape: (N, 1000) -> (N, 1, 1000)
                # ------------------------------------------------

                if X.ndim == 2:
                    X = X[:, np.newaxis, :]

                if X.ndim != 3 or X.shape[1] != 1:
                    raise ValueError(
                        f"Unexpected X shape {X.shape} in {file}. "
                        "Expected (N, 1, sequence_length)."
                    )

                if y.ndim != 2 or y.shape[1] != 2:
                    raise ValueError(
                        f"Unexpected y shape {y.shape} in {file}. "
                        "Expected (N, 2) for SBP and DBP."
                    )

                if len(X) != len(y):
                    raise ValueError(
                        f"X and y have different sample counts in {file}"
                    )

                # ------------------------------------------------
                # TRUE BP
                # ------------------------------------------------

                sbp_true = y[:, 0]
                dbp_true = y[:, 1]

                case_id = os.path.basename(file).replace(
                    ".npz", ""
                )

                case_sbp_pred = []
                case_dbp_pred = []

                # ------------------------------------------------
                # BATCHED PREDICTION
                # ------------------------------------------------

                for start in range(0, len(X), batch_size):

                    end = min(start + batch_size, len(X))

                    X_batch = X[start:end]

                    X_tensor = torch.as_tensor(
                        X_batch,
                        dtype=torch.float32,
                        device=device
                    )

                    predictions = model(X_tensor)

                    predictions = predictions.cpu().numpy()

                    case_sbp_pred.append(predictions[:, 0])
                    case_dbp_pred.append(predictions[:, 1])

                # ------------------------------------------------
                # JOIN BATCHES FOR THIS CASE
                # ------------------------------------------------

                case_sbp_pred = np.concatenate(case_sbp_pred)
                case_dbp_pred = np.concatenate(case_dbp_pred)

                # ------------------------------------------------
                # STORE
                # ------------------------------------------------

                sbp_true_all.append(sbp_true)
                sbp_pred_all.append(case_sbp_pred)

                dbp_true_all.append(dbp_true)
                dbp_pred_all.append(case_dbp_pred)

                case_ids.extend([case_id] * len(sbp_true))

    # ============================================================
    # CONCATENATE ALL CASES
    # ============================================================

    sbp_true_all = np.concatenate(sbp_true_all)
    sbp_pred_all = np.concatenate(sbp_pred_all)

    dbp_true_all = np.concatenate(dbp_true_all)
    dbp_pred_all = np.concatenate(dbp_pred_all)

    # ============================================================
    # RESULTS
    # ============================================================

    print("\n" + "=" * 60)
    print("CNN-LSTM")
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
        model_name="CNN-LSTM - VitalDB Zero-Shot",
        target_name="SBP"
    )

    plot_regression_results(
        dbp_true_all,
        dbp_pred_all,
        model_name="CNN-LSTM - VitalDB Zero-Shot",
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




import torch
import torch.nn as nn
import numpy as np
import copy

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error


class MultiScaleCNN_LSTM(nn.Module):

    def __init__(self):
        super().__init__()

        # ================================================
        # MULTI-SCALE CNN BLOCK
        # ================================================

        # Branch 1: Small temporal patterns
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Branch 2: Medium temporal patterns
        self.branch7 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Branch 3: Larger temporal patterns
        self.branch15 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Concatenated channels: 32 * 3 = 96
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # ================================================
        # FURTHER CNN FEATURE EXTRACTION
        # ================================================

        self.cnn = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # ================================================
        # LSTM
        # ================================================

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        # ================================================
        # REGRESSION HEAD
        # ================================================

        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):

        # Input: (batch, 1, sequence_length)

        # Multi-scale parallel convolutions
        x3 = self.branch3(x)
        x7 = self.branch7(x)
        x15 = self.branch15(x)

        # Concatenate along channel dimension
        x = torch.cat([x3, x7, x15], dim=1)

        # (batch, 96, time)
        x = self.pool1(x)

        # Further feature extraction
        x = self.cnn(x)

        # (batch, 128, time) -> (batch, time, 128)
        x = x.transpose(1, 2)

        # Temporal modeling
        x, _ = self.lstm(x)

        # Last LSTM output
        x = x[:, -1, :]

        # Predict SBP and DBP
        return self.regressor(x)



def train_multiscale_cnn_lstm(
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

    # ========================================================
    # DEVICE
    # ========================================================

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("Device:", device)

    # ========================================================
    # CONVERT TO TENSORS
    # ========================================================

    def to_tensor(x):
        if not torch.is_tensor(x):
            return torch.tensor(x, dtype=torch.float32)
        return x.float()

    X_train, y_train = to_tensor(X_train), to_tensor(y_train)
    X_val, y_val = to_tensor(X_val), to_tensor(y_val)
    X_test, y_test = to_tensor(X_test), to_tensor(y_test)

    # Add channel dimension if needed
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

    # ========================================================
    # DATA LOADERS
    # ========================================================

    pin_memory = device.type == "cuda"

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = MultiScaleCNN_LSTM().to(device)

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

    # ========================================================
    # HISTORY
    # ========================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sbp_rmse": [],
        "train_dbp_rmse": [],
        "val_sbp_rmse": [],
        "val_dbp_rmse": []
    }

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(epochs):

        # -----------------------------------------------
        # TRAIN
        # -----------------------------------------------

        model.train()

        train_loss_total = 0.0
        train_preds = []
        train_targets = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_total += loss.item() * X_batch.size(0)

            train_preds.append(
                predictions.detach().cpu().numpy()
            )

            train_targets.append(
                y_batch.detach().cpu().numpy()
            )

        train_loss = train_loss_total / len(train_dataset)

        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)

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

        # -----------------------------------------------
        # VALIDATION
        # -----------------------------------------------

        model.eval()

        val_loss_total = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                predictions = model(X_batch)

                loss = criterion(predictions, y_batch)

                val_loss_total += loss.item() * X_batch.size(0)

                val_preds.append(predictions.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_loss = val_loss_total / len(val_dataset)

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

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

        # -----------------------------------------------
        # SCHEDULER
        # -----------------------------------------------

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # -----------------------------------------------
        # SAVE HISTORY
        # -----------------------------------------------

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_sbp_rmse"].append(train_sbp_rmse)
        history["train_dbp_rmse"].append(train_dbp_rmse)

        history["val_sbp_rmse"].append(val_sbp_rmse)
        history["val_dbp_rmse"].append(val_dbp_rmse)

        # -----------------------------------------------
        # PRINT METRICS
        # -----------------------------------------------

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

        # -----------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
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

            X_batch = X_batch.to(device, non_blocking=True)

            predictions = model(X_batch)

            test_preds.append(predictions.cpu().numpy())
            test_targets.append(y_batch.numpy())

    y_pred = np.concatenate(test_preds, axis=0)
    y_true = np.concatenate(test_targets, axis=0)

    # ========================================================
    # FINAL TEST RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("MULTI-SCALE CNN-LSTM TEST RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(y_true[:, 0], y_pred[:, 0])

    print("\nDBP")
    print_metrics(y_true[:, 1], y_pred[:, 1])

    # ========================================================
    # REGRESSION PLOTS
    # ========================================================

    plot_regression_results(
        y_true[:, 0],
        y_pred[:, 0],
        model_name="Multi-scale CNN-LSTM",
        target_name="SBP"
    )

    plot_regression_results(
        y_true[:, 1],
        y_pred[:, 1],
        model_name="Multi-scale CNN-LSTM",
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





import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM_Attention(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # LSTM temporal feature extractor
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # SBP and DBP regression head
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x: (batch, 1, signal_length)
        x = self.cnn(x)

        # (batch, channels, time) -> (batch, time, features)
        x = x.transpose(1, 2)

        # (batch, time, 64)
        lstm_out, _ = self.lstm(x)

        # Attention score for each time step
        scores = self.attention(lstm_out).squeeze(-1)

        # Normalize scores across time
        attention_weights = F.softmax(scores, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1),
            dim=1
        )

        # Output: [SBP, DBP]
        return self.regressor(context)




import copy
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_cnn_lstm_attention(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size=256,
    epochs=50,
    lr=1e-3,
    patience=10,
    device=None
):
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def make_X(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (N, 1, L), got {X.shape}")
        return torch.tensor(X, dtype=torch.float32)

    def make_y(y):
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"Expected y shape (N, 2), got {y.shape}")
        return torch.tensor(y, dtype=torch.float32)

    train_ds = TensorDataset(make_X(X_train), make_y(y_train))
    val_ds = TensorDataset(make_X(X_val), make_y(y_val))
    test_ds = TensorDataset(make_X(X_test), make_y(y_test))

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        pin_memory=pin_memory
    )

    model = CNN_LSTM_Attention(dropout=0.3).to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
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
    best_epoch = 0
    epochs_without_improvement = 0

    def run_epoch(loader, training=False):
        model.train(training)

        total_loss = 0.0
        targets, predictions = [], []

        with torch.set_grad_enabled(training):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                pred = model(xb)
                loss = criterion(pred, yb)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                total_loss += loss.item() * xb.size(0)
                targets.append(yb.detach().cpu().numpy())
                predictions.append(pred.detach().cpu().numpy())

        y_true = np.concatenate(targets)
        y_pred = np.concatenate(predictions)
        avg_loss = total_loss / len(loader.dataset)

        sbp_rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        dbp_rmse = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

        return avg_loss, sbp_rmse, dbp_rmse, y_true, y_pred

    for epoch in range(1, epochs + 1):
        train_loss, train_sbp, train_dbp, _, _ = run_epoch(
            train_loader, training=True
        )

        val_loss, val_sbp, val_dbp, _, _ = run_epoch(
            val_loader, training=False
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_sbp_rmse"].append(train_sbp)
        history["train_dbp_rmse"].append(train_dbp)
        history["val_sbp_rmse"].append(val_sbp)
        history["val_dbp_rmse"].append(val_dbp)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train RMSE SBP/DBP: {train_sbp:.2f}/{train_dbp:.2f} | "
            f"Val RMSE SBP/DBP: {val_sbp:.2f}/{val_dbp:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore checkpoint with the lowest validation MSE
    model.load_state_dict(best_state)

    # Evaluate the test set only after model selection
    _, _, _, y_true, y_pred = run_epoch(test_loader, training=False)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation MSE: {best_val_loss:.4f}")

    for i, name in enumerate(["SBP", "DBP"]):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        print(
            f"Test {name}: RMSE={rmse:.3f}, "
            f"MAE={mae:.3f}, R²={r2:.4f}"
        )

    results = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "y_test": y_true,
        "y_pred": y_pred
    }

    return model, results