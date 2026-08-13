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



def get_delay_time(ppg_window, fs=125):
    """
    Detect all systolic peaks and estimate the
    peak-to-dicrotic-notch delay for each beat.

    Returns:
        delays : np.ndarray
            Delay for each successfully detected beat.
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
    # Process each systolic peak separately
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
    fs=125,
    min_delay=0.08,
    max_delay=0.35
):
    """
    Extracts peak-to-notch delay for every PPG window.

    Invalid detections are NOT replaced with the median.
    They are stored as NaN.

    Returns:
        delay_times : np.ndarray of shape (N,)
    """

    delay_times = []


    for w in ppg_windows:

        dt = get_delay_time(
            w,
            fs
        )


        # -------------------------------------
        # Detection failed
        # -------------------------------------

        if dt is None:

            delay_times.append(np.nan)

            continue


        # -------------------------------------
        # Reject physiologically unreasonable
        # delays
        # -------------------------------------

        if (
            dt < min_delay
            or
            dt > max_delay
        ):

            delay_times.append(np.nan)

        else:

            delay_times.append(dt)


    return np.array(
        delay_times,
        dtype=np.float32
    )