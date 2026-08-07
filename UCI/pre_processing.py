from scipy.signal import find_peaks
import numpy as np
from tqdm import tqdm
import h5py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader



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



def load_UCI_dataset(
    data_path="../UCI/data",
    test_size=0.2,
    val_size=0.1,
    random_state=42
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

    X_train, y_train = process_split(train_records)

    X_val, y_val = process_split(val_records)

    X_test, y_test = process_split(test_records)



    # ============================
    # Add channel dimension
    # CNN input:
    # (batch, channels, samples)
    # ============================

    X_train = X_train[:,None,:]

    X_val = X_val[:,None,:]

    X_test = X_test[:,None,:]



    # ============================
    # Convert to tensors
    # ============================

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )



    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )


    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )





class PPGDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]



def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=64
):

    # ==========================
    # Create datasets
    # ==========================

    train_dataset = PPGDataset(
        X_train,
        y_train
    )


    val_dataset = PPGDataset(
        X_val,
        y_val
    )


    test_dataset = PPGDataset(
        X_test,
        y_test
    )


    # ==========================
    # Create dataloaders
    # ==========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )