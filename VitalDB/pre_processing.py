from scipy.signal import find_peaks
import numpy as np
from tqdm import tqdm
import h5py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly


def extract_bp_from_abp(
    abp_window,
    fs=500
):
    # -----------------------------------------
    # Minimum distance between systolic peaks
    # 0.4 sec
    # -----------------------------------------

    min_distance = int(0.4 * fs)

    peaks, _ = find_peaks(
        abp_window,
        distance=min_distance,
        prominence=5
    )

    # Need enough beats
    if len(peaks) < 2:
        return None, None

    # -----------------------------------------
    # SBP
    # -----------------------------------------

    sbp_values = abp_window[peaks]

    # -----------------------------------------
    # DBP
    # Minimum between consecutive systolic peaks
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
    # Sanity checks
    # -----------------------------------------

    if sbp < 50 or sbp > 250:
        return None, None

    if dbp < 20 or dbp > 150:
        return None, None

    return sbp, dbp





def downsample_signal(
    signal,
    original_fs=500,
    target_fs=125
):

    return resample_poly(
        signal,
        up=1,
        down=4
    ).astype(np.float32)


def process_recording(
    ppg,
    art,
    WINDOW_SIZE=4000,
    STEP_SIZE=2000
):

    std = np.std(ppg)

    if std == 0:
        return None, None

    ppg = (
        ppg - np.mean(ppg)
    ) / std

    X = []
    y = []

    # -----------------------------------------
    # Windowing at 500 Hz
    # 4000 samples = 8 seconds
    # -----------------------------------------

    for start in range(
        0,
        len(ppg) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        ppg_window = ppg[
            start:start + WINDOW_SIZE
        ]

        art_window = art[
            start:start + WINDOW_SIZE
        ]

        # -------------------------------------
        # Extract SBP / DBP from 500-Hz ART
        # -------------------------------------

        sbp, dbp = extract_bp_from_abp(
            art_window,
            fs=500
        )

        if sbp is None:
            continue

        # -------------------------------------
        # Downsample PPG
        # 500 Hz → 125 Hz
        # -------------------------------------

        ppg_window_125 = downsample_signal(
            ppg_window,
            original_fs=500,
            target_fs=125
        )

        # -------------------------------------
        # Safety check
        # -------------------------------------

        if len(ppg_window_125) != 1000:
            continue

        X.append(
            ppg_window_125
        )

        y.append([
            sbp,
            dbp
        ])

    if len(X) == 0:
        return None, None

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )

def process_split(
    case_ids,
    WINDOW_SIZE=4000,
    STEP_SIZE=2000
):

    X_all = []
    y_all = []

    skipped = 0

    for case_id in tqdm(case_ids):

        try:

            # Load VitalDB case
            vf = vitaldb.VitalFile(
                case_id
            )

            # Get PPG + ART
            data = vf.to_numpy(
                [
                    "SNUADC/PLETH",
                    "SNUADC/ART"
                ],
                interval = 1/500
            )

            ppg = data[:, 0]
            art = data[:, 1]

            # Process recording
            X, y = process_recording(
                ppg,
                art,
                WINDOW_SIZE,
                STEP_SIZE
            )

            if X is None:
                skipped += 1
                continue

            X_all.append(X)
            y_all.append(y)

        except Exception as e:

            skipped += 1

            print(
                f"Error in case {case_id}: {e}"
            )

    if len(X_all) == 0:
        return None, None

    X_all = np.concatenate(
        X_all,
        axis=0
    )

    y_all = np.concatenate(
        y_all,
        axis=0
    )

    print(
        "Skipped cases:",
        skipped
    )

    print(
        "X shape:",
        X_all.shape
    )

    print(
        "y shape:",
        y_all.shape
    )

    return X_all, y_all