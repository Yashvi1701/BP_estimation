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



import os
import glob
import numpy as np
import torch
from tqdm.auto import tqdm


def evaluate_convtran_vitaldb(
    model,
    base_dir,
    batch_size=64,
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

    with torch.inference_mode():

        for file in tqdm(
            npz_files,
            desc="ConvTran VitalDB prediction",
            unit="case"
        ):

            with np.load(file) as data:
                X = data["X"]
                y = data["y"]

            # ----------------------------------------------------
            # INPUT SHAPE
            # Expected: (N, 1, 1000)
            # ----------------------------------------------------

            if X.ndim == 2:
                X = X[:, np.newaxis, :]

            if X.ndim != 3:
                raise ValueError(
                    f"Unexpected X shape {X.shape} in {file}. "
                    "Expected (N, 1, 1000)."
                )

            if X.shape[1] != 1:
                raise ValueError(
                    f"Unexpected channel dimension in {file}: "
                    f"{X.shape}. Expected (N, 1, 1000)."
                )

            if y.ndim != 2 or y.shape[1] != 2:
                raise ValueError(
                    f"Unexpected y shape {y.shape} in {file}. "
                    "Expected (N, 2) with [SBP, DBP]."
                )

            if len(X) != len(y):
                raise ValueError(
                    f"X/y sample count mismatch in {file}: "
                    f"{len(X)} vs {len(y)}."
                )

            case_id = os.path.basename(file).replace(
                ".npz", ""
            )

            sbp_true = y[:, 0]
            dbp_true = y[:, 1]

            case_sbp_pred = []
            case_dbp_pred = []

            # ----------------------------------------------------
            # BATCHED PREDICTION
            # ----------------------------------------------------

            for start in range(0, len(X), batch_size):

                end = min(start + batch_size, len(X))
                X_batch = X[start:end]

                X_tensor = torch.as_tensor(
                    X_batch,
                    dtype=torch.float32,
                    device=device
                )

                predictions = model(X_tensor)

                # Handle models that return a tuple/list
                if isinstance(predictions, (tuple, list)):
                    predictions = predictions[0]

                if predictions.ndim != 2 or predictions.shape[1] != 2:
                    raise ValueError(
                        "ConvTran must output shape (batch, 2). "
                        f"Got {tuple(predictions.shape)}."
                    )

                predictions = predictions.detach().cpu().numpy()

                case_sbp_pred.append(predictions[:, 0])
                case_dbp_pred.append(predictions[:, 1])

            # ----------------------------------------------------
            # JOIN BATCHES FOR THIS CASE
            # ----------------------------------------------------

            case_sbp_pred = np.concatenate(case_sbp_pred)
            case_dbp_pred = np.concatenate(case_dbp_pred)

            # ----------------------------------------------------
            # STORE
            # ----------------------------------------------------

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
    print("CONVTRAN")
    print("VITALDB ZERO-SHOT RESULTS")
    print("=" * 60)

    print("\nSBP")
    print_metrics(sbp_true_all, sbp_pred_all)

    print("\nDBP")
    print_metrics(dbp_true_all, dbp_pred_all)

    # ============================================================
    # PLOTS
    # ============================================================

    plot_regression_results(
        sbp_true_all,
        sbp_pred_all,
        model_name="ConvTran - VitalDB Zero-Shot",
        target_name="SBP"
    )

    plot_regression_results(
        dbp_true_all,
        dbp_pred_all,
        model_name="ConvTran - VitalDB Zero-Shot",
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


import matplotlib.pyplot as plt

def plot_convtran_loss(history):
    """
    Plot training and validation loss against epochs.

    history: dictionary containing train_loss and val_loss lists.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("ConvTran: Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

