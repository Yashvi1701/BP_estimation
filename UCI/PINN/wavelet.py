import pywt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import h5py
import os

def wavelet_denoise(
    ppg,
    wavelet,
    level
):

    ppg = np.asarray(
        ppg,
        dtype=np.float64
    )

    # -----------------------------------------
    # Check signal
    # -----------------------------------------

    if len(ppg) < 2:
        return None

    if not np.all(
        np.isfinite(ppg)
    ):
        return None

    # -----------------------------------------
    # DWT decomposition
    # -----------------------------------------

    coeffs = pywt.wavedec(
        ppg,
        wavelet=wavelet,
        level=level
    )

    # coeffs structure:
    #
    # [A10, D10, D9, ..., D2, D1]
    #
    # A10 = lowest-frequency approximation
    # D1  = highest-frequency detail

    # -----------------------------------------
    # Remove the lowest-frequency
    # approximation component
    #
    # This removes very slow baseline
    # variation.
    # -----------------------------------------

    coeffs[0] = np.zeros_like(
        coeffs[0]
    )

    # -----------------------------------------
    # Reconstruct signal
    # -----------------------------------------

    denoised = pywt.waverec(
        coeffs,
        wavelet=wavelet
    )

    # waverec can return slightly more samples
    # because of boundary handling.
    
    denoised = denoised[
        :len(ppg)
    ]

    return denoised.astype(
        np.float32
    )

def analyze_wavelet_denoising(
    ppg,
    denoised_ppg,
    fs=125,
    plot_seconds=30
):
    """
    Compare original and wavelet-denoised PPG.

    Parameters
    ----------
    ppg : array-like
        Original PPG signal.

    denoised_ppg : array-like
        Wavelet-denoised PPG signal.

    fs : int
        Sampling frequency in Hz.

    plot_seconds : int or float
        Duration of signal to show in the first plot.

    Returns
    -------
    results : dict
        Dictionary containing comparison statistics.
    """

    # ==================================================
    # Convert to numpy arrays
    # ==================================================

    ppg = np.asarray(ppg).squeeze()
    denoised_ppg = np.asarray(denoised_ppg).squeeze()


    # ==================================================
    # Check shapes
    # ==================================================

    if ppg.shape != denoised_ppg.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"PPG {ppg.shape} vs "
            f"Denoised {denoised_ppg.shape}"
        )


    # ==================================================
    # Basic statistics
    # ==================================================

    original_mean = np.mean(ppg)
    original_std = np.std(ppg)
    original_min = np.min(ppg)
    original_max = np.max(ppg)

    denoised_mean = np.mean(denoised_ppg)
    denoised_std = np.std(denoised_ppg)
    denoised_min = np.min(denoised_ppg)
    denoised_max = np.max(denoised_ppg)


    # ==================================================
    # Removed component
    # ==================================================

    removed_component = (
        ppg - denoised_ppg
    )


    # ==================================================
    # Correlation
    # ==================================================

    correlation = np.corrcoef(
        ppg,
        denoised_ppg
    )[0, 1]


    # ==================================================
    # Residual statistics
    #
    # Remove mean first so that the DC/baseline
    # difference does not dominate the residual.
    # ==================================================

    ppg_centered = (
        ppg - original_mean
    )

    denoised_centered = (
        denoised_ppg - denoised_mean
    )

    residual = (
        ppg_centered -
        denoised_centered
    )

    residual_std = np.std(
        residual
    )

    residual_ratio = (
        residual_std /
        original_std
    )


    # ==================================================
    # Print results
    # ==================================================

    print("=" * 50)
    print("WAVELET DENOISING ANALYSIS")
    print("=" * 50)

    print("\nShape:")
    print("Original shape :", ppg.shape)
    print("Denoised shape :", denoised_ppg.shape)

    print("\nOriginal:")
    print(
        f"Mean : {original_mean:.6f}"
    )
    print(
        f"Std  : {original_std:.6f}"
    )
    print(
        f"Min  : {original_min:.6f}"
    )
    print(
        f"Max  : {original_max:.6f}"
    )

    print("\nDenoised:")
    print(
        f"Mean : {denoised_mean:.6f}"
    )
    print(
        f"Std  : {denoised_std:.6f}"
    )
    print(
        f"Min  : {denoised_min:.6f}"
    )
    print(
        f"Max  : {denoised_max:.6f}"
    )

    print("\nSimilarity:")
    print(
        f"Correlation : {correlation:.5f}"
    )

    print("\nResidual:")
    print(
        f"Residual std : {residual_std:.6f}"
    )
    print(
        f"Residual / signal std : "
        f"{residual_ratio * 100:.3f}%"
    )


    # ==================================================
    # Plot 1: Original vs denoised
    # ==================================================

    N = min(
        int(plot_seconds * fs),
        len(ppg)
    )

    plt.figure(
        figsize=(15, 5)
    )

    plt.plot(
        ppg[:N],
        label="Original PPG",
        alpha=0.7
    )

    plt.plot(
        denoised_ppg[:N],
        label="Denoised PPG",
        linewidth=1.5
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Original vs db8 Wavelet-Denoised PPG"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.show()


    # ==================================================
    # Plot 2: Removed component
    # ==================================================

    plt.figure(
        figsize=(15, 4)
    )

    plt.plot(
        removed_component
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Component Removed by Wavelet Denoising"
    )

    plt.grid(
        alpha=0.3
    )

    plt.show()


    # ==================================================
    # Plot 3: Centered residual
    # ==================================================

    plt.figure(
        figsize=(15, 4)
    )

    plt.plot(
        residual
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Wavelet Residual After Removing DC Offset"
    )

    plt.grid(
        alpha=0.3
    )

    plt.show()


    # ==================================================
    # Return results
    # ==================================================

    results = {

        "original_mean":
            original_mean,

        "original_std":
            original_std,

        "original_min":
            original_min,

        "original_max":
            original_max,

        "denoised_mean":
            denoised_mean,

        "denoised_std":
            denoised_std,

        "denoised_min":
            denoised_min,

        "denoised_max":
            denoised_max,

        "correlation":
            correlation,

        "residual_std":
            residual_std,

        "residual_ratio":
            residual_ratio
    }

    return results


def plot_dwt_levels(ppg, wavelet, level, fs=125):
    """
    Decomposes a PPG signal using DWT and plots every level
    (A_level, D_level, ..., D1) as stacked subplots, so you can
    visually inspect which layer holds baseline drift, which
    holds the pulse waveform, and which is just noise.
    """
 
    ppg = np.asarray(ppg, dtype=np.float64)
 
    coeffs = pywt.wavedec(ppg, wavelet=wavelet, level=level)
    # coeffs = [A_level, D_level, D_(level-1), ..., D1]
 
    labels = [f"A{level}"] + [f"D{j}" for j in range(level, 0, -1)]
 
    # approximate frequency band each layer represents (Hz)
    bands = []
    bands.append((0, fs / 2**(level + 1)))  # A_level band
    for j in range(level, 0, -1):
        low = fs / 2**(j + 1)
        high = fs / 2**j
        bands.append((low, high))
 
    n_plots = len(coeffs) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=False)
 
    # original signal on top
    axes[0].plot(ppg, color="black", linewidth=0.8)
    axes[0].set_title("Original PPG signal")
 
    for ax, c, label, (lo, hi) in zip(axes[1:], coeffs, labels, bands):
        ax.plot(c, linewidth=0.8)
        ax.set_title(f"{label}  |  ~{lo:.3f}-{hi:.3f} Hz  |  n={len(c)}")
 
    plt.tight_layout()
    plt.savefig("dwt_levels.png", dpi=150)
    plt.show()
 
    return coeffs



 def remove_wavelet_components(
    ppg,
    wavelet="db8",
    level=8,
    remove_components=("A",)
):
    """
    Apply DWT to a PPG signal and remove selected wavelet components.

    Parameters
    ----------
    ppg : numpy array
        1D PPG signal.

    wavelet : str
        Wavelet name, e.g.
        "db4", "db8", "sym4", "coif5".

    level : int
        DWT decomposition level.

    remove_components : tuple/list
        Components to remove.

        Examples:
            ("A",)
                Remove approximation only.

            ("A", "D8")
                Remove A8 and D8 when level=8.

            ("A", "D7", "D8")
                Remove A8, D8 and D7 when level=8.

            ("D1", "D2")
                Remove high-frequency components.

    Returns
    -------
    denoised_ppg : numpy array
        Reconstructed PPG after removing the selected components.
    """

    # ==================================================
    # Convert to numpy
    # ==================================================

    ppg = np.asarray(
        ppg,
        dtype=np.float64
    )

    # ==================================================
    # Check input
    # ==================================================

    if ppg.ndim != 1:
        ppg = ppg.squeeze()

    if len(ppg) < 2:
        raise ValueError(
            "PPG signal is too short."
        )

    if not np.all(np.isfinite(ppg)):
        raise ValueError(
            "PPG contains NaN or Inf values."
        )

    # ==================================================
    # Check decomposition level
    # ==================================================

    wavelet_obj = pywt.Wavelet(wavelet)

    max_level = pywt.dwt_max_level(
        len(ppg),
        wavelet_obj.dec_len
    )

    if level < 1 or level > max_level:
        raise ValueError(
            f"Invalid level={level} for signal length "
            f"{len(ppg)} and wavelet={wavelet}. "
            f"Maximum allowed level is {max_level}."
        )

    # ==================================================
    # DWT decomposition
    # ==================================================

    coeffs = pywt.wavedec(
        ppg,
        wavelet=wavelet,
        level=level
    )

    # Structure:
    #
    # [A_level,
    #  D_level,
    #  D_(level-1),
    #  ...
    #  D2,
    #  D1]

    # ==================================================
    # Normalize component names
    # ==================================================

    remove_components = [
        component.upper()
        for component in remove_components
    ]

    # ==================================================
    # Remove selected components
    # ==================================================

    # ----------------------------------------------
    # Approximation component
    # ----------------------------------------------

    if "A" in remove_components:
        coeffs[0] = np.zeros_like(
            coeffs[0]
        )

    elif f"A{level}" in remove_components:
        coeffs[0] = np.zeros_like(
            coeffs[0]
        )

    # ----------------------------------------------
    # Detail components
    # ----------------------------------------------

    for j in range(1, level + 1):

        component_name = f"D{j}"

        if component_name in remove_components:

            # D_level is coeffs[1]
            # D_(level-1) is coeffs[2]
            # ...
            # D1 is coeffs[-1]

            coeff_index = level - j + 1

            coeffs[coeff_index] = np.zeros_like(
                coeffs[coeff_index]
            )

    # ==================================================
    # Reconstruct signal
    # ==================================================

    denoised_ppg = pywt.waverec(
        coeffs,
        wavelet=wavelet
    )

    # waverec can return extra samples
    denoised_ppg = denoised_ppg[
        :len(ppg)
    ]

    return denoised_ppg.astype(
        np.float64
    )