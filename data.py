

import os
import numpy as np
from scipy import signal
import wfdb

# 44 records used according to standard de Chazal / AAMI benchmarks
DS1_RECS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 
            201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

DS2_RECS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 
            213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# Fixed 4 validation records taken out of DS1
VAL_RECS = [106, 203, 215, 223]
TRAIN_RECS = [r for r in DS1_RECS if r not in VAL_RECS]
ALL_RECORDS = DS1_RECS + DS2_RECS

# AAMI 5-class mapping
# N: Normal / bundle branch block
# S: Supraventricular ectopic beat
# V: Ventricular ectopic beat
# F: Fusion beat
# Q: Unknown / unclassifiable beat
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4
}
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

# Clinical illustrative cost matrix (rows: true class, cols: predicted class)
COST_MATRIX = np.array([
    [0, 1, 3, 2, 1], # True N
    [2, 0, 3, 2, 1], # True S
    [5, 4, 0, 3, 2], # True V
    [3, 2, 4, 0, 2], # True F
    [1, 1, 2, 1, 0]  # True Q
])


def filter_signal(sig, fs=360):
    """
    0.5 - 40 Hz Butterworth band-pass filter using forward-backward filtfilt.
    Removes baseline wander (<0.5 Hz) and high-frequency powerline/muscle noise (>40 Hz).
    """
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = signal.butter(3, [low, high], btype='bandpass')
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def cut_beats(sig, peaks, fs=360):
    """
    Cuts beats centered on R-peaks:
    - 90 samples before R-peak (0.25 s)
    - 162 samples after R-peak (0.45 s)
    - Total beat length = 252 samples
    - Z-score normalization for each beat
    - Returns:
      beats: (N, 252) array
      rr: (N, 2) array containing [prev_rr, next_rr] in seconds
      keep_mask: boolean array of length len(peaks) indicating kept peaks
    """
    beats = []
    rr_list = []
    keep_mask = np.zeros(len(peaks), dtype=bool)

    # Need at least 3 peaks to compute prev and next RR
    if len(peaks) < 3:
        return np.empty((0, 252)), np.empty((0, 2)), keep_mask

    for i in range(1, len(peaks) - 1):
        p = int(peaks[i])
        start = p - 90
        end = p + 162

        # Check boundary limits
        if start < 0 or end > len(sig):
            continue

        beat = sig[start:end]
        std = np.std(beat)
        if std == 0:
            norm_beat = beat - np.mean(beat)
        else:
            norm_beat = (beat - np.mean(beat)) / std

        prev_rr = (peaks[i] - peaks[i - 1]) / float(fs)
        next_rr = (peaks[i + 1] - peaks[i]) / float(fs)

        beats.append(norm_beat)
        rr_list.append([prev_rr, next_rr])
        keep_mask[i] = True

    if len(beats) == 0:
        return np.empty((0, 252)), np.empty((0, 2)), keep_mask

    return np.array(beats, dtype=np.float32), np.array(rr_list, dtype=np.float32), keep_mask


def process_record(rec_id, data_dir='mitdb'):
    
    rec_path = os.path.join(data_dir, str(rec_id))
    record = wfdb.rdrecord(rec_path)

    # Find MLII channel index
    ch = None
    for idx, name in enumerate(record.sig_name):
        if 'MLII' in name.upper():
            ch = idx
            break
    if ch is None:
        # Fallback to channel 0 if MLII not named
        ch = 0

    raw_sig = record.p_signal[:, ch]
    filt_sig = filter_signal(raw_sig, fs=record.fs)

    ann = wfdb.rdann(rec_path, 'atr')
    samples = np.array(ann.sample)
    symbols = np.array(ann.symbol)

    # Filter to only valid AAMI beat annotations
    valid_mask = np.isin(symbols, list(AAMI_MAP.keys()))
    valid_samples = samples[valid_mask]
    valid_symbols = symbols[valid_mask]

    mapped_labels = np.array([AAMI_MAP[s] for s in valid_symbols], dtype=int)

    # Cut beats and get keep mask
    beats, rr, keep_mask = cut_beats(filt_sig, valid_samples, fs=record.fs)
    labels = mapped_labels[keep_mask]
    recs = np.full(len(labels), rec_id, dtype=int)

    return beats, rr, labels, recs


def build_dataset(data_dir='mitdb', cache_file='data_cache.npz'):
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        data = np.load(cache_file)
        return data['X'], data['rr'], data['y'], data['rec']

    print(f"Cache {cache_file} not found. Preparing dataset...")
    os.makedirs(data_dir, exist_ok=True)

    # Check which records are already downloaded
    needed_records = [r for r in ALL_RECORDS if not os.path.exists(os.path.join(data_dir, f"{r}.hea"))]
    if needed_records:
        print(f"Downloading {len(needed_records)} records from PhysioNet mitdb...")
        wfdb.dl_database('mitdb', dl_dir=data_dir, records=[str(r) for r in needed_records])

    all_X = []
    all_rr = []
    all_y = []
    all_rec = []

    for r in ALL_RECORDS:
        print(f"Processing record {r}...")
        beats, rr, labels, recs = process_record(r, data_dir=data_dir)
        all_X.append(beats)
        all_rr.append(rr)
        all_y.append(labels)
        all_rec.append(recs)

    X = np.vstack(all_X)
    rr = np.vstack(all_rr)
    y = np.concatenate(all_y)
    rec = np.concatenate(all_rec)

    print(f"Total extracted beats: {len(y)}")
    print(f"Saving to {cache_file}...")
    np.savez_compressed(cache_file, X=X, rr=rr, y=y, rec=rec)
    return X, rr, y, rec


def get_splits(rec_array, y_array=None):
    train_mask = np.isin(rec_array, TRAIN_RECS)
    val_mask = np.isin(rec_array, VAL_RECS)
    test_mask = np.isin(rec_array, DS2_RECS)

    if y_array is not None:
        print("\n--- Split Distributions ---")
        for split_name, mask in [('Train (DS1-val)', train_mask), ('Val (4 recs)', val_mask), ('Test (DS2)', test_mask)]:
            counts = [np.sum(y_array[mask] == c) for c in range(len(CLASS_NAMES))]
            counts_str = ", ".join([f"{CLASS_NAMES[c]}: {counts[c]}" for c in range(len(CLASS_NAMES))])
            print(f"{split_name:16s} (total {np.sum(mask):6d}): {counts_str}")

    return train_mask, val_mask, test_mask


if __name__ == '__main__':
    X, rr, y, rec = build_dataset()
    get_splits(rec, y)
