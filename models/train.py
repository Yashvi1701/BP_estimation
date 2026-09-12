import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import copy
import numpy as np
import torch


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    scheduler=None,
    patience=10,
    min_delta=0.01
):

    train_losses = []
    val_losses = []

    train_sbp_losses = []
    train_dbp_losses = []

    val_sbp_losses = []
    val_dbp_losses = []

    # =====================
    # Early Stopping
    # =====================

    best_val_loss = float("inf")
    patience_counter = 0

    # Keep best model in RAM
    best_model_state = None

    for epoch in range(epochs):

        # =====================
        # Training
        # =====================

        model.train()

        running_loss = 0.0
        running_sbp_loss = 0.0
        running_dbp_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            # ---------------------
            # SBP Loss
            # ---------------------

            sbp_loss = criterion(
                predictions[:, 0],
                y_batch[:, 0]
            )

            # ---------------------
            # DBP Loss
            # ---------------------

            dbp_loss = criterion(
                predictions[:, 1],
                y_batch[:, 1]
            )

            # ---------------------
            # Total Loss
            # ---------------------

            loss = sbp_loss + dbp_loss

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_sbp_loss += sbp_loss.item()
            running_dbp_loss += dbp_loss.item()

        # =====================
        # Training Metrics
        # =====================

        train_loss = (
            running_loss /
            len(train_loader)
        )

        train_sbp_loss = (
            running_sbp_loss /
            len(train_loader)
        )

        train_dbp_loss = (
            running_dbp_loss /
            len(train_loader)
        )

        # =====================
        # Validation
        # =====================

        model.eval()

        val_loss = 0.0
        val_sbp_loss = 0.0
        val_dbp_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)

                # ---------------------
                # SBP Loss
                # ---------------------

                sbp_loss = criterion(
                    predictions[:, 0],
                    y_batch[:, 0]
                )

                # ---------------------
                # DBP Loss
                # ---------------------

                dbp_loss = criterion(
                    predictions[:, 1],
                    y_batch[:, 1]
                )

                # ---------------------
                # Total Loss
                # ---------------------

                loss = sbp_loss + dbp_loss

                val_loss += loss.item()
                val_sbp_loss += sbp_loss.item()
                val_dbp_loss += dbp_loss.item()

        # =====================
        # Validation Metrics
        # =====================

        val_loss /= len(val_loader)

        val_sbp_loss /= len(val_loader)

        val_dbp_loss /= len(val_loader)

        # =====================
        # Scheduler
        # =====================

        if scheduler is not None:

            scheduler.step(val_loss)

        # =====================
        # Store Losses
        # =====================

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_sbp_losses.append(train_sbp_loss)
        train_dbp_losses.append(train_dbp_loss)

        val_sbp_losses.append(val_sbp_loss)
        val_dbp_losses.append(val_dbp_loss)

        # =====================
        # RMSE
        # =====================

        train_sbp_rmse = np.sqrt(
            train_sbp_loss
        )

        train_dbp_rmse = np.sqrt(
            train_dbp_loss
        )

        val_sbp_rmse = np.sqrt(
            val_sbp_loss
        )

        val_dbp_rmse = np.sqrt(
            val_dbp_loss
        )

        # =====================
        # Early Stopping
        # =====================

        if val_loss < best_val_loss - min_delta:

            # Meaningful improvement

            best_val_loss = val_loss

            patience_counter = 0

            # ---------------------------------
            # Save best weights ONLY IN RAM
            # ---------------------------------

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            improvement_status = "Improved"

        else:

            patience_counter += 1

            improvement_status = (
                f"No improvement "
                f"({patience_counter}/{patience})"
            )

        # =====================
        # Print
        # =====================

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_loss:.2f} "
            f"(SBP RMSE: {train_sbp_rmse:.2f}, "
            f"DBP RMSE: {train_dbp_rmse:.2f}) | "
            f"Val MSE: {val_loss:.2f} "
            f"(SBP RMSE: {val_sbp_rmse:.2f}, "
            f"DBP RMSE: {val_dbp_rmse:.2f}) | "
            f"{improvement_status}"
        )

        # =====================
        # Stop Training
        # =====================

        if patience_counter >= patience:

            print(
                f"\nEarly stopping triggered "
                f"at epoch {epoch + 1}."
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )

            break

    # =====================
    # Restore Best Model
    # =====================

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

        print(
            "\nBest model restored from memory."
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.4f}"
        )

    # =====================
    # History
    # =====================

    history = {

        "train_loss": train_losses,

        "val_loss": val_losses,

        "train_sbp_loss": train_sbp_losses,

        "train_dbp_loss": train_dbp_losses,

        "val_sbp_loss": val_sbp_losses,

        "val_dbp_loss": val_dbp_losses
    }

    return (
        model,
        history,
        train_losses,
        val_losses
    )


from matplotlib import pyplot as plt


def evaluate_model(model, test_loader, device):

    model.eval()

    test_sbp_errors = []
    test_dbp_errors = []

    test_predictions = []
    test_targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            predictions = model(X)

            # =========================================
            # Errors
            # =========================================

            sbp_errors = (
                predictions[:, 0] - y[:, 0]
            ).cpu()

            dbp_errors = (
                predictions[:, 1] - y[:, 1]
            ).cpu()

            test_sbp_errors.append(
                sbp_errors
            )

            test_dbp_errors.append(
                dbp_errors
            )

            # =========================================
            # Store predictions and targets
            # =========================================

            test_predictions.append(
                predictions.cpu()
            )

            test_targets.append(
                y.cpu()
            )

    # =========================================
    # Concatenate batches
    # =========================================

    test_sbp_errors = torch.cat(
        test_sbp_errors
    )

    test_dbp_errors = torch.cat(
        test_dbp_errors
    )

    test_predictions = torch.cat(
        test_predictions
    )

    test_targets = torch.cat(
        test_targets
    )

    # =========================================
    # MAE
    # =========================================

    test_sbp_mae = torch.mean(
        torch.abs(test_sbp_errors)
    ).item()

    test_dbp_mae = torch.mean(
        torch.abs(test_dbp_errors)
    ).item()

    # =========================================
    # RMSE
    # =========================================

    test_sbp_rmse = torch.sqrt(
        torch.mean(
            test_sbp_errors ** 2
        )
    ).item()

    test_dbp_rmse = torch.sqrt(
        torch.mean(
            test_dbp_errors ** 2
        )
    ).item()

    # =========================================
    # Variance
    # =========================================

    test_sbp_variance = torch.var(
        test_sbp_errors,
        unbiased=False
    ).item()

    test_dbp_variance = torch.var(
        test_dbp_errors,
        unbiased=False
    ).item()

    # =========================================
    # Standard deviation
    # =========================================

    test_sbp_std = torch.std(
        test_sbp_errors,
        unbiased=False
    ).item()

    test_dbp_std = torch.std(
        test_dbp_errors,
        unbiased=False
    ).item()

    
    return {
        "sbp_mae": test_sbp_mae,
        "sbp_rmse": test_sbp_rmse,
        "sbp_variance": test_sbp_variance,
        "sbp_std": test_sbp_std,

        "dbp_mae": test_dbp_mae,
        "dbp_rmse": test_dbp_rmse,
        "dbp_variance": test_dbp_variance,
        "dbp_std": test_dbp_std,

        "predictions": test_predictions,
        "targets": test_targets
    }




def train_bp_model(
    bp_model,
    train_feature_loader,
    val_feature_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    patience=10,
    min_delta=0.01,
    save_path="best_papagei_bp_regressor.pt"
):

    # ==================================================
    # EARLY STOPPING
    # ==================================================

    patience_counter = 0
    best_val_loss = float("inf")

    # ==================================================
    # METRIC STORAGE
    # ==================================================

    train_losses = []
    val_losses = []

    train_sbp_rmse = []
    train_dbp_rmse = []

    val_sbp_rmse = []
    val_dbp_rmse = []

    # ==================================================
    # TRAINING LOOP
    # ==================================================

    for epoch in range(epochs):

        # ==================================================
        # TRAIN
        # ==================================================

        bp_model.train()

        train_running_loss = 0.0
        train_sbp_squared_error = 0.0
        train_dbp_squared_error = 0.0
        train_samples = 0

        for X, y in train_feature_loader:

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            predictions = bp_model(X)

            # Loss
            loss = criterion(predictions, y)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # ------------------------------------------
            # Loss
            # ------------------------------------------

            train_running_loss += (
                loss.item() * X.size(0)
            )

            # ------------------------------------------
            # SBP squared error
            # ------------------------------------------

            train_sbp_squared_error += torch.sum(
                (predictions[:, 0] - y[:, 0]) ** 2
            ).item()

            # ------------------------------------------
            # DBP squared error
            # ------------------------------------------

            train_dbp_squared_error += torch.sum(
                (predictions[:, 1] - y[:, 1]) ** 2
            ).item()

            train_samples += X.size(0)

        # ==================================================
        # TRAIN METRICS
        # ==================================================

        train_loss = (
            train_running_loss /
            train_samples
        )

        train_sbp_rmse_epoch = np.sqrt(
            train_sbp_squared_error /
            train_samples
        )

        train_dbp_rmse_epoch = np.sqrt(
            train_dbp_squared_error /
            train_samples
        )

        # ==================================================
        # VALIDATION
        # ==================================================

        bp_model.eval()

        val_running_loss = 0.0
        val_sbp_squared_error = 0.0
        val_dbp_squared_error = 0.0
        val_samples = 0

        with torch.no_grad():

            for X, y in val_feature_loader:

                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Forward pass
                predictions = bp_model(X)

                # Loss
                loss = criterion(predictions, y)

                # ------------------------------------------
                # Loss
                # ------------------------------------------

                val_running_loss += (
                    loss.item() * X.size(0)
                )

                # ------------------------------------------
                # SBP squared error
                # ------------------------------------------

                val_sbp_squared_error += torch.sum(
                    (predictions[:, 0] - y[:, 0]) ** 2
                ).item()

                # ------------------------------------------
                # DBP squared error
                # ------------------------------------------

                val_dbp_squared_error += torch.sum(
                    (predictions[:, 1] - y[:, 1]) ** 2
                ).item()

                val_samples += X.size(0)

        # ==================================================
        # VALIDATION METRICS
        # ==================================================

        val_loss = (
            val_running_loss /
            val_samples
        )

        val_sbp_rmse_epoch = np.sqrt(
            val_sbp_squared_error /
            val_samples
        )

        val_dbp_rmse_epoch = np.sqrt(
            val_dbp_squared_error /
            val_samples
        )

        # ==================================================
        # STORE METRICS
        # ==================================================

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_sbp_rmse.append(train_sbp_rmse_epoch)
        train_dbp_rmse.append(train_dbp_rmse_epoch)

        val_sbp_rmse.append(val_sbp_rmse_epoch)
        val_dbp_rmse.append(val_dbp_rmse_epoch)

        # ==================================================
        # SAVE BEST MODEL + EARLY STOPPING
        # ==================================================

        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                bp_model.state_dict(),
                save_path
            )

            improvement_status = "Improved"

        else:

            patience_counter += 1

            improvement_status = (
                f"No improvement "
                f"({patience_counter}/{patience})"
            )

        # ==================================================
        # PRINT
        # ==================================================

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train SBP RMSE: {train_sbp_rmse_epoch:.2f} | "
            f"Train DBP RMSE: {train_dbp_rmse_epoch:.2f} | "
            f"Val SBP RMSE: {val_sbp_rmse_epoch:.2f} | "
            f"Val DBP RMSE: {val_dbp_rmse_epoch:.2f} | "
            f"{improvement_status}"
        )

        # ==================================================
        # EARLY STOPPING
        # ==================================================

        if patience_counter >= patience:

            print(
                f"\nEarly stopping triggered "
                f"at epoch {epoch + 1}."
            )

            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )

            break

    # ==================================================
    # LOAD BEST MODEL
    # ==================================================

    bp_model.load_state_dict(
        torch.load(
            save_path,
            map_location=device
        )
    )

    print("\nBest model loaded successfully.")

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    # ==================================================
    # RETURN RESULTS
    # ==================================================

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_sbp_rmse": train_sbp_rmse,
        "train_dbp_rmse": train_dbp_rmse,
        "val_sbp_rmse": val_sbp_rmse,
        "val_dbp_rmse": val_dbp_rmse,
    }

    return bp_model, history, best_val_loss


from sklearn.metrics import mean_absolute_error, mean_squared_error
def extract_papagei_features(
    dataloader,
    model,
    device
):

    model.eval()

    all_features = []
    all_targets = []

    with torch.no_grad():

        for X, y in tqdm(
            dataloader,
            desc="Extracting PaPaGei features"
        ):

            X = X.to(
                device,
                non_blocking=True
            )

            # PaPaGei forward pass
            out_class, out_moe1, out_moe2, out = model(X)

            # -----------------------------------------
            # Use the 512-D backbone representation
            # -----------------------------------------
            features = out

            all_features.append(
                features.cpu().numpy()
            )

            all_targets.append(
                y.numpy()
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    targets = np.concatenate(
        all_targets,
        axis=0
    )

    return features, targets


def plot_mae(results):

    predictions = results["predictions"].numpy()
    targets = results["targets"].numpy()

    # ==========================================
    # Calculate absolute errors
    # ==========================================

    sbp_errors = np.abs(
        predictions[:, 0] - targets[:, 0]
    )

    dbp_errors = np.abs(
        predictions[:, 1] - targets[:, 1]
    )

    # ==========================================
    # MAE and variance
    # ==========================================

    sbp_mae = np.mean(sbp_errors)
    dbp_mae = np.mean(dbp_errors)

    sbp_variance = np.var(sbp_errors)
    dbp_variance = np.var(dbp_errors)

    sbp_std = np.std(sbp_errors)
    dbp_std = np.std(dbp_errors)

    # ==========================================
    # Print statistics
    # ==========================================

    print("\n================================")
    print("ABSOLUTE ERROR STATISTICS")
    print("================================")

    print(f"SBP MAE      : {sbp_mae:.2f} mmHg")
    print(f"SBP Variance : {sbp_variance:.2f} mmHg²")
    print(f"SBP Std Dev  : {sbp_std:.2f} mmHg")

    print()

    print(f"DBP MAE      : {dbp_mae:.2f} mmHg")
    print(f"DBP Variance : {dbp_variance:.2f} mmHg²")
    print(f"DBP Std Dev  : {dbp_std:.2f} mmHg")


    # ==========================================
    # SBP MAE Distribution
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.hist(
        sbp_errors,
        bins=50,
        alpha=0.7
    )

    plt.axvline(
        sbp_mae,
        linestyle="--",
        linewidth=2,
        label=f"MAE = {sbp_mae:.2f} mmHg"
    )

    plt.xlabel("Absolute Error (mmHg)")
    plt.ylabel("Number of Samples")

    plt.title(
        "SBP Absolute Error Distribution"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.show()


    # ==========================================
    # DBP MAE Distribution
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.hist(
        dbp_errors,
        bins=50,
        alpha=0.7
    )

    plt.axvline(
        dbp_mae,
        linestyle="--",
        linewidth=2,
        label=f"MAE = {dbp_mae:.2f} mmHg"
    )

    plt.xlabel("Absolute Error (mmHg)")
    plt.ylabel("Number of Samples")

    plt.title(
        "DBP Absolute Error Distribution"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.show()


    # ==========================================================
    # SBP MAE BY TRUE SBP RANGE
    # ==========================================================

    true_sbp = targets[:, 0]

    # SBP bins
    sbp_bins = [
        60, 80, 100, 120, 140,
        160, 180, 200, 220, 240, 260
    ]

    sbp_labels = [
        "60–80",
        "80–100",
        "100–120",
        "120–140",
        "140–160",
        "160–180",
        "180–200",
        "200–220",
        "220–240",
        "240–260"
    ]

    sbp_bin_mae = []
    sbp_bin_counts = []

    for i in range(len(sbp_bins) - 1):

        lower = sbp_bins[i]
        upper = sbp_bins[i + 1]

        mask = (
            (true_sbp >= lower) &
            (true_sbp < upper)
        )

        count = np.sum(mask)

        sbp_bin_counts.append(count)

        if count == 0:
            sbp_bin_mae.append(np.nan)
        else:
            sbp_bin_mae.append(
                np.mean(sbp_errors[mask])
            )


    # ==========================================
    # Print SBP bin statistics
    # ==========================================

    print("\n================================")
    print("SBP MAE BY TRUE SBP RANGE")
    print("================================")

    for label, count, mae in zip(
        sbp_labels,
        sbp_bin_counts,
        sbp_bin_mae
    ):

        if np.isnan(mae):
            print(
                f"{label} mmHg "
                f"| Samples: {count:6d} "
                f"| MAE: N/A"
            )
        else:
            print(
                f"{label} mmHg "
                f"| Samples: {count:6d} "
                f"| MAE: {mae:.2f} mmHg"
            )


    # ==========================================
    # Plot SBP bin MAE
    # ==========================================

    plt.figure(figsize=(10, 5))

    plt.bar(
        sbp_labels,
        sbp_bin_mae
    )

    plt.axhline(
        sbp_mae,
        linestyle="--",
        linewidth=2,
        label=f"Overall MAE = {sbp_mae:.2f} mmHg"
    )

    plt.xlabel("True SBP Range (mmHg)")
    plt.ylabel("MAE (mmHg)")

    plt.title(
        "SBP MAE Across Blood Pressure Ranges"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


    # ==========================================================
    # DBP MAE BY TRUE DBP RANGE
    # ==========================================================

    true_dbp = targets[:, 1]

    # DBP bins
    dbp_bins = [
        20, 30, 40, 50, 60, 70,
        80, 90, 100, 110, 120,
        130, 140, 150
    ]

    dbp_labels = [
        "20–30",
        "30–40",
        "40–50",
        "50–60",
        "60–70",
        "70–80",
        "80–90",
        "90–100",
        "100–110",
        "110–120",
        "120–130",
        "130–140",
        "140–150"
    ]

    dbp_bin_mae = []
    dbp_bin_counts = []

    for i in range(len(dbp_bins) - 1):

        lower = dbp_bins[i]
        upper = dbp_bins[i + 1]

        mask = (
            (true_dbp >= lower) &
            (true_dbp < upper)
        )

        count = np.sum(mask)

        dbp_bin_counts.append(count)

        if count == 0:
            dbp_bin_mae.append(np.nan)
        else:
            dbp_bin_mae.append(
                np.mean(dbp_errors[mask])
            )


    # ==========================================
    # Print DBP bin statistics
    # ==========================================

    print("\n================================")
    print("DBP MAE BY TRUE DBP RANGE")
    print("================================")

    for label, count, mae in zip(
        dbp_labels,
        dbp_bin_counts,
        dbp_bin_mae
    ):

        if np.isnan(mae):
            print(
                f"{label} mmHg "
                f"| Samples: {count:6d} "
                f"| MAE: N/A"
            )
        else:
            print(
                f"{label} mmHg "
                f"| Samples: {count:6d} "
                f"| MAE: {mae:.2f} mmHg"
            )


    # ==========================================
    # Plot DBP bin MAE
    # ==========================================

    plt.figure(figsize=(11, 5))

    plt.bar(
        dbp_labels,
        dbp_bin_mae
    )

    plt.axhline(
        dbp_mae,
        linestyle="--",
        linewidth=2,
        label=f"Overall MAE = {dbp_mae:.2f} mmHg"
    )

    plt.xlabel("True DBP Range (mmHg)")
    plt.ylabel("MAE (mmHg)")

    plt.title(
        "DBP MAE Across Blood Pressure Ranges"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class VitalDBCheckpointDataset(Dataset):

    def __init__(self, checkpoint_dir):

        self.files = sorted(
            glob.glob(
                os.path.join(
                    checkpoint_dir,
                    "case_*.npz"
                )
            )
        )

        if len(self.files) == 0:
            raise ValueError(
                f"No checkpoint files found in: "
                f"{checkpoint_dir}"
            )

        print(
            "Checkpoint files found:",
            len(self.files)
        )

        # -----------------------------------------
        # Store number of windows in each case
        # -----------------------------------------

        self.lengths = []

        for file_path in self.files:

            data = np.load(file_path)

            self.lengths.append(
                len(data["X"])
            )

        # -----------------------------------------
        # Cumulative lengths
        # -----------------------------------------

        self.cumulative_lengths = np.cumsum(
            self.lengths
        )

        self.total_samples = int(
            self.cumulative_lengths[-1]
        )

        print(
            "Total test windows:",
            self.total_samples
        )

    def __len__(self):

        return self.total_samples

    def __getitem__(self, idx):

        # -----------------------------------------
        # Find which case contains this sample
        # -----------------------------------------

        file_idx = np.searchsorted(
            self.cumulative_lengths,
            idx,
            side="right"
        )

        if file_idx == 0:

            local_idx = idx

        else:

            local_idx = (
                idx -
                self.cumulative_lengths[file_idx - 1]
            )

        file_path = self.files[file_idx]

        # -----------------------------------------
        # Load case
        # -----------------------------------------

        data = np.load(file_path)

        X = data["X"][local_idx]
        y = data["y"][local_idx]

        # -----------------------------------------
        # Add channel dimension
        # (1000,) -> (1, 1000)
        # -----------------------------------------

        X = X[None, :]

        # -----------------------------------------
        # Convert to tensors
        # -----------------------------------------

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return X, y


def load_vitaldb_test_loader(
    checkpoint_dir="/data1/yashvi_bhuva/BP_estimation_using_PPG/VitalDB/vitaldb_checkpoints",
    batch_size=1024,
    num_workers=4
):

    # =========================================
    # Dataset
    # =========================================

    test_dataset = VitalDBCheckpointDataset(
        checkpoint_dir
    )

    # =========================================
    # DataLoader
    # =========================================
    test_dataset =test_dataset[:10]
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(
        "Number of test samples:",
        len(test_dataset)
    )

    print(
        "Number of test batches:",
        len(test_loader)
    )

    return test_loader