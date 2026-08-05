from scipy.signal import find_peaks
import numpy as np
from tqdm import tqdm


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


    return sbp, dbp

def process_recording(recording, WINDOW_SIZE = 1000, STEP_SIZE = 500 ):

    data = recording[:]

    ppg = data[:,0]
    abp = data[:,1]

    std = np.std(ppg)

    if std == 0:
        return None, None

    ppg = (ppg - np.mean(ppg))/std

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

    return np.array(X),np.array(y)




def process_split(record_list):

    X_all = []
    y_all = []

    skipped = 0

    for f, ref in tqdm(record_list):

        recording = f[ref]

        X, y = process_recording(recording, WINDOW_SIZE=1000, STEP_SIZE=500)

        if X is None:
            skipped += 1
            continue

        X_all.append(X)
        y_all.append(y)

    X_all = np.concatenate(X_all)

    y_all = np.concatenate(y_all)

    print("Skipped recordings:", skipped)

    return X_all, y_all