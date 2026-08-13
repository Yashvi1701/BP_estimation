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

    if refine_start >= refine_end:
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


def get_decay_time(ppg_window, fs=125):
    """
    Returns diastolic decay time (systolic peak -> dicrotic notch), in seconds.
    Returns None if either landmark could not be detected (caller should
    skip / fall back to a dataset-median value for that sample).
    """
    sys_idx = get_systolic_peak(ppg_window, fs)
    notch_idx = get_dicrotic_notch(ppg_window, sys_idx, fs)
    if sys_idx is None or notch_idx is None:
        return None
    return (notch_idx - sys_idx) / fs


def extract_decay_times(ppg_windows, fs=125, fallback='median'):
    """
    Runs decay-time extraction over a full array of PPG windows (N, L).
    Failed detections are filled with the median of successful ones.
    Returns: np.array of shape (N,)
    """
    decay_times = []
    for w in ppg_windows:
        dt = get_decay_time(w, fs)
        decay_times.append(dt)
 
    decay_times = np.array([d if d is not None else np.nan for d in decay_times])
    valid = decay_times[~np.isnan(decay_times)]
    if fallback == 'median' and len(valid) > 0:
        med = np.median(valid)
        decay_times = np.where(np.isnan(decay_times), med, decay_times)
    return decay_times.astype(np.float32)