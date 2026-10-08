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



def evaluate_minirocket_vitaldb(
    base_dir,
    rocket,
    best_en_sbp,
    best_en_dbp,
    model,
    batch_size=2000
):
    """
    Zero-shot VitalDB evaluation using a fitted MiniROCKET
    transformer and separately trained ElasticNet SBP/DBP models.

    MiniROCKET is NOT refitted on VitalDB.

    Parameters
    ----------
    base_dir : str
        Directory containing case_*.npz files.

    rocket : fitted MiniROCKET transformer
        MiniROCKET fitted only on the UCI training data.

    best_en_sbp : fitted model
        ElasticNet model trained for SBP.

    best_en_dbp : fitted model
        ElasticNet model trained for DBP.

    batch_size : int
        Number of VitalDB windows processed at once.

    Returns
    -------
    results : dict
        VitalDB true and predicted SBP/DBP values.
    """

    import os
    import glob
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    # ============================================================
    # 1. FIND VITALDB CASES
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
    # 2. STORAGE FOR ALL PREDICTIONS
    # ============================================================

    case_ids = []

    sbp_true_all = []
    sbp_pred_all = []

    dbp_true_all = []
    dbp_pred_all = []

    # ============================================================
    # 3. PROCESS EACH CASE
    # ============================================================

    for file in tqdm(
        npz_files,
        desc="MiniROCKET VitalDB prediction",
        unit="case"
    ):

        # --------------------------------------------------------
        # Load case
        # --------------------------------------------------------

        data = np.load(file)

        X = data["X"]
        y = data["y"]
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        case_id = os.path.basename(file).replace(
            ".npz",
            ""
        )

        # print(
        #     f"\nProcessing {case_id} "
        #     f"| X shape = {X.shape}"
        # )

        # --------------------------------------------------------
        # True targets
        # --------------------------------------------------------

        sbp_true = y[:, 0]
        dbp_true = y[:, 1]

        case_sbp_pred = []
        case_dbp_pred = []

        # --------------------------------------------------------
        # MiniROCKET + prediction in batches
        # --------------------------------------------------------

        for start in range(
            0,
            len(X),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X)
            )

            X_batch = X[start:end].astype(np.float32)
            X_rocket_batch = rocket.transform(X_batch).to_numpy(dtype=np.float32)

            # ====================================================
            # PREDICT SBP
            # ====================================================

            sbp_pred_batch = best_en_sbp.predict(
                X_rocket_batch
            )

            # ====================================================
            # PREDICT DBP
            # ====================================================

            dbp_pred_batch = best_en_dbp.predict(
                X_rocket_batch
            )

            case_sbp_pred.append(
                sbp_pred_batch
            )

            case_dbp_pred.append(
                dbp_pred_batch
            )

            # X_rocket_batch is discarded here
            # before moving to the next batch

        # --------------------------------------------------------
        # Combine predictions for this case
        # --------------------------------------------------------

        case_sbp_pred = np.concatenate(
            case_sbp_pred
        )

        case_dbp_pred = np.concatenate(
            case_dbp_pred
        )

        # --------------------------------------------------------
        # Store results
        # --------------------------------------------------------

        sbp_true_all.append(
            sbp_true
        )

        sbp_pred_all.append(
            case_sbp_pred
        )

        dbp_true_all.append(
            dbp_true
        )

        dbp_pred_all.append(
            case_dbp_pred
        )

        case_ids.extend(
            [case_id] * len(sbp_true)
        )

    # ============================================================
    # 4. COMBINE ALL CASES
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
    # 5. PRINT METRICS
    # ============================================================

    print("\n")
    print("=" * 60)
    print(f"MiniROCKET + {model}")
    print("VITALDB ZERO-SHOT RESULTS")
    print("=" * 60)

    print("\nSBP")
    print("-" * 30)

    print_metrics(
        sbp_true_all,
        sbp_pred_all
    )

    print("\nDBP")
    print("-" * 30)

    print_metrics(
        dbp_true_all,
        dbp_pred_all
    )

    # ============================================================
    # 6. REGRESSION PLOTS
    # ============================================================

    plot_regression_results(
        sbp_true_all,
        sbp_pred_all,
        model_name=f"MiniROCKET + {model}",
        target_name="SBP"
    )

    plot_regression_results(
        dbp_true_all,
        dbp_pred_all,
        model_name=f"MiniROCKET + {model}",
        target_name="DBP"
    )

    # ============================================================
    # 7. RESULTS DATAFRAME
    # ============================================================

    df_results = pd.DataFrame({
        "case_id": case_ids,
        "SBP_true": sbp_true_all,
        "SBP_pred": sbp_pred_all,
        "DBP_true": dbp_true_all,
        "DBP_pred": dbp_pred_all
    })

    print("\nFinal prediction shape:")
    print(df_results.shape)

    print("\nFirst 5 predictions:")
    print(df_results.head())

    # ============================================================
    # 8. RETURN
    # ============================================================

    return {
        "df": df_results,
        "sbp_true": sbp_true_all,
        "sbp_pred": sbp_pred_all,
        "dbp_true": dbp_true_all,
        "dbp_pred": dbp_pred_all,
        "case_ids": case_ids
    }



import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class _FeatDS(Dataset):
    def __init__(self, X, y, mu, sd, y_mean, y_std):
        self.X, self.y = X, y
        self.mu, self.sd, self.y_mean, self.y_std = mu, sd, y_mean, y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = (np.asarray(self.X[i], dtype=np.float32) - self.mu) / self.sd
        t = (self.y[i] - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.tensor(t, dtype=torch.float32)


class MLPHead:
    """Predicts one target (col 0 = SBP, col 1 = DBP). Has .predict() like sklearn."""
    def __init__(self, model, col, mu, sd, y_mean, y_std, device, idx=None):
        self.m, self.col = model, col
        self.mu, self.sd, self.y_mean, self.y_std = mu, sd, y_mean, y_std
        self.device, self.idx = device, idx   # idx = optional selected-feature indices

    def predict(self, X, bs=4096):
        X = np.asarray(X, dtype=np.float32)
        if self.idx is not None:
            X = X[:, self.idx]
        X = (X - self.mu) / self.sd
        self.m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs]).to(self.device)
                out.append(self.m(xb).cpu().numpy())
        out = np.concatenate(out) * self.y_std + self.y_mean
        return out[:, self.col]

def train_mlp(
    X_train, y_train, X_val, y_val,
    hidden=(512, 128),
    in_dropout=0.2,
    dropout=(0.3, 0.2),
    lr=1e-3,
    weight_decay=1e-2,
    batch_size=512,
    max_epochs=30,
    patience=5,
    feature_idx=None,
    save_path="mlp_best.pt",
    stat_batch=5000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- feature scaler stats (streamed, low memory) ----
    n = len(X_train)
    d = len(feature_idx) if feature_idx is not None else X_train.shape[1]
    s = np.zeros(d); ss = np.zeros(d)
    for i in range(0, n, stat_batch):
        xb = np.asarray(X_train[i:i + stat_batch], dtype=np.float64)
        if feature_idx is not None:
            xb = xb[:, feature_idx]
        s += xb.sum(0); ss += (xb ** 2).sum(0)
    mu = (s / n).astype(np.float32)
    sd = np.sqrt(np.maximum(ss / n - (s / n) ** 2, 1e-12)).astype(np.float32)
    sd[sd == 0] = 1

    y_mean = y_train.mean(0).astype(np.float32)
    y_std = y_train.std(0).astype(np.float32)
    y_std_t = torch.tensor(y_std, device=device)   # for un-scaling to mmHg

    def view(X):
        return X if feature_idx is None else _SelView(X, feature_idx)

    train_dl = DataLoader(_FeatDS(view(X_train), y_train, mu, sd, y_mean, y_std),
                          batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(_FeatDS(view(X_val), y_val, mu, sd, y_mean, y_std),
                        batch_size=2048)

    # ---- model ----
    layers = [nn.Dropout(in_dropout)]
    prev = d
    for h, dr in zip(hidden, dropout):
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dr)]
        prev = h
    layers.append(nn.Linear(prev, 2))
    model = nn.Sequential(*layers).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    loss_fn = nn.HuberLoss()

    def run_eval(loader):
        """Returns huber loss and per-target MSE in mmHg^2 (eval mode, no dropout)."""
        model.eval()
        loss_sum, se = 0.0, torch.zeros(2, device=device)
        cnt = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss_sum += loss_fn(out, yb).item() * len(xb)
                se += (((out - yb) * y_std_t) ** 2).sum(0)   # error in mmHg
                cnt += len(xb)
        mse = (se / cnt).cpu().numpy()                       # [SBP, DBP]
        return loss_sum / cnt, mse

    best_val, bad = float("inf"), 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # metrics in eval mode (train metrics use dropout-off for a fair comparison)
        tr_loss, tr_mse = run_eval(train_dl)
        vl, va_mse = run_eval(val_dl)
        sched.step(vl)

        tr_rmse, va_rmse = np.sqrt(tr_mse), np.sqrt(va_mse)
        print(
            f"epoch {epoch:02d} | "
            f"TRAIN loss {tr_loss:.4f} MSE(SBP {tr_mse[0]:.2f}, DBP {tr_mse[1]:.2f}) "
            f"RMSE(SBP {tr_rmse[0]:.2f}, DBP {tr_rmse[1]:.2f}) | "
            f"VAL loss {vl:.4f} MSE(SBP {va_mse[0]:.2f}, DBP {va_mse[1]:.2f}) "
            f"RMSE(SBP {va_rmse[0]:.2f}, DBP {va_rmse[1]:.2f})"
        )

        if vl < best_val:
            best_val, bad = vl, 0
            torch.save(model.state_dict(), save_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    sbp_head = MLPHead(model, 0, mu, sd, y_mean, y_std, device, idx=feature_idx)
    dbp_head = MLPHead(model, 1, mu, sd, y_mean, y_std, device, idx=feature_idx)
    return sbp_head, dbp_head
class _SelView:
    """Lazy column-selection view over a memmap (no copy of the full array)."""
    def __init__(self, X, idx):
        self.X, self.idx = X, idx
        self.shape = (X.shape[0], len(idx))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return np.asarray(self.X[i])[..., self.idx]


import numpy as np

def select_top_pearson_features(X, y, k=1000):
    """
    Select top-k features based on absolute Pearson correlation.

    X : pandas DataFrame or numpy array
        Shape: (n_samples, n_features)
    y : numpy array or torch tensor
        Shape: (n_samples,)
    """

    # Convert y to NumPy
    if hasattr(y, "cpu"):
        y = y.cpu().numpy()

    y = np.asarray(y, dtype=np.float32)

    # Process X as NumPy
    if hasattr(X, "to_numpy"):
        X = X.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    # Center
    X_mean = X.mean(axis=0)
    y_mean = y.mean()

    X_centered = X - X_mean
    y_centered = y - y_mean

    # Pearson correlation
    numerator = np.sum(X_centered * y_centered[:, None], axis=0)

    denominator = (
        np.sqrt(np.sum(X_centered ** 2, axis=0))
        * np.sqrt(np.sum(y_centered ** 2))
    )

    corr = numerator / np.maximum(denominator, 1e-12)

    # Rank by absolute correlation
    top_idx = np.argsort(np.abs(corr))[::-1][:k]

    return top_idx, corr[top_idx]