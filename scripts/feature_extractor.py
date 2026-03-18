"""
feature_extractor.py
--------------------
Shared audio feature extraction for all 5 models.
Extracts MFCCs, spectral features, and mel-spectrograms from WAV files.
"""

import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
DURATION      = 4.0          # seconds — clips are padded/trimmed to this
N_MFCC        = 40
N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 2048

CLASSES = [
    "gunshot",
    "rifle_fire",
    "vehicle",
    "aircraft",
    "comms_signal",
    "ambient",          # safe / background
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """Load a WAV file, mono, resampled, fixed length."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_flat_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a flat 1-D feature vector for classical ML models
    (SVM, Random Forest, XGBoost).

    Returns a vector of shape (193,):
        - 40 MFCC means + 40 MFCC stds
        - 40 delta-MFCC means
        - 40 delta2-MFCC means
        - 7 spectral features (centroid, bandwidth, rolloff, ZCR, RMS, contrast_mean, flatness)
        - 1 tempo estimate  → 168 dims without tempo, add tempo = 169... 
        Actually: 40+40+40+40+7+1 = 168. Let's keep 168.
    """
    # MFCCs
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                        n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_mean   = np.mean(mfcc, axis=1)
    mfcc_std    = np.std(mfcc,  axis=1)
    delta_mean  = np.mean(mfcc_delta,  axis=1)
    delta2_mean = np.mean(mfcc_delta2, axis=1)

    # Spectral
    centroid  = np.mean(librosa.feature.spectral_centroid(y=y,  sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y,   sr=sr))
    zcr       = np.mean(librosa.feature.zero_crossing_rate(y))
    rms       = np.mean(librosa.feature.rms(y=y))
    contrast  = np.mean(librosa.feature.spectral_contrast(y=y,  sr=sr))
    flatness  = np.mean(librosa.feature.spectral_flatness(y=y))

    spectral = np.array([centroid, bandwidth, rolloff, zcr, rms, contrast, flatness])

    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta2_mean, spectral])


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Return a 2-D mel-spectrogram (N_MELS x T) in dB scale.
    Used by CNN model after reshaping to (1, N_MELS, T).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def extract_sequence_features(y: np.ndarray, sr: int = SAMPLE_RATE,
                               n_frames: int = 128) -> np.ndarray:
    """
    Return a 2-D sequence (n_frames x N_MFCC) for LSTM.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)  # (40, T)
    mfcc = mfcc.T  # (T, 40)
    if mfcc.shape[0] < n_frames:
        pad = np.zeros((n_frames - mfcc.shape[0], N_MFCC))
        mfcc = np.vstack([mfcc, pad])
    else:
        mfcc = mfcc[:n_frames]
    return mfcc  # (128, 40)


# ─── DATASET LOADER ──────────────────────────────────────────────────────────

def load_dataset(dataset_dir: str, feature_type: str = "flat",
                 verbose: bool = True, test_csv: str = None):
    """
    Supports TWO dataset formats automatically:

    FORMAT A — CSV-based (your actual dataset):
        dataset_dir/
            train.csv          columns: path, label  (path = "training/001/0.wav")
            files/
                training/
                    001/  *.wav
                    002/  *.wav

    FORMAT B — folder-per-class (original format):
        dataset_dir/
            gunshot/   *.wav
            vehicle/   *.wav

    Returns X (np.ndarray), y (np.ndarray of int labels), label_names (list)
    """
    import pandas as pd

    # ── Detect FORMAT A: CSV exists ──────────────────────────────────────────
    csv_path = None
    for candidate in ["train.csv", "Train.csv", "training.csv"]:
        p = os.path.join(dataset_dir, candidate)
        if os.path.exists(p):
            csv_path = p
            break

    if test_csv:
        return _load_from_csv(test_csv, dataset_dir, feature_type, False)

    if test_csv:
        return _load_from_csv(test_csv, dataset_dir, feature_type, verbose)

    if csv_path:
        return _load_from_csv(csv_path, dataset_dir, feature_type, verbose)

    # ── Detect FORMAT B: class-named subfolders ──────────────────────────────
    return _load_from_folders(dataset_dir, feature_type, verbose)


def _load_from_csv(csv_path: str, dataset_dir: str, feature_type: str,
                   verbose: bool):
    """Load dataset from train.csv with path + label columns."""
    import pandas as pd

    df = pd.read_csv(csv_path, header=None)

    # Auto-detect columns — find which column has paths and which has labels
    # CSV may or may not have a header row
    # Try to read with header first
    df_h = pd.read_csv(csv_path)
    if "path" in df_h.columns and "label" in df_h.columns:
        df = df_h
    else:
        # No header — assign names by detecting which col looks like paths
        df = pd.read_csv(csv_path, header=None)
        path_col  = None
        label_col = None
        for col in df.columns:
            sample = str(df[col].iloc[0])
            if "/" in sample or "\\" in sample or sample.endswith(".wav"):
                path_col = col
            elif str(sample).isdigit():
                label_col = col
        if path_col is None or label_col is None:
            # fallback: assume col 0 = path, col 1 = label
            path_col, label_col = 0, 1
        df = df.rename(columns={path_col: "path", label_col: "label"})

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "path"])
    df["label"] = df["label"].astype(int)

    unique_labels = sorted(df["label"].unique())
    n_classes     = len(unique_labels)
    label_map     = {orig: idx for idx, orig in enumerate(unique_labels)}

    # Build human-readable class names
    CLASS_NAME_MAP = {
        0: "gunshot",
        1: "rifle_fire",
        2: "vehicle",
        3: "aircraft",
        4: "comms_signal",
        5: "explosion",
        6: "ambient",
    }
    label_names = [CLASS_NAME_MAP.get(orig, f"class_{orig}") for orig in unique_labels]

    if verbose:
        print(f"\n📂 CSV dataset: {csv_path}")
        print(f"   Total rows : {len(df)}")
        print(f"   Classes    : {dict(zip(unique_labels, label_names))}")
        for orig in unique_labels:
            count = (df["label"] == orig).sum()
            print(f"   [{orig}] {CLASS_NAME_MAP.get(orig,'?'):15s}: {count} samples")

    # Find the audio root — look for a "files" subfolder or use dataset_dir
    audio_root = dataset_dir
    for candidate in ["files", "audio", "wavs", "data"]:
        p = os.path.join(dataset_dir, candidate)
        if os.path.isdir(p):
            audio_root = p
            break

    X, y = [], []
    skipped = 0

    for _, row in df.iterrows():
        rel_path  = str(row["path"]).replace("\\", "/")
        label_idx = label_map[int(row["label"])]

        # Try multiple path resolutions
        candidates = [
            os.path.join(audio_root, rel_path),
            os.path.join(dataset_dir, rel_path),
            os.path.join(audio_root, rel_path.replace("/", os.sep)),
            os.path.join(dataset_dir, rel_path.replace("/", os.sep)),
        ]

        fpath = None
        for c in candidates:
            if os.path.exists(c):
                fpath = c
                break

        if fpath is None:
            skipped += 1
            continue

        try:
            audio = load_audio(fpath)
            if feature_type == "flat":
                feat = extract_flat_features(audio)
            elif feature_type == "mel":
                feat = extract_mel_spectrogram(audio)
            elif feature_type == "seq":
                feat = extract_sequence_features(audio)
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            X.append(feat)
            y.append(label_idx)
        except Exception as e:
            skipped += 1
            if verbose and skipped <= 5:
                print(f"   ⚠️  Skipped {fpath}: {e}")

    if verbose:
        print(f"\n   ✅ Loaded: {len(X)} samples | Skipped: {skipped}")

    if verbose:
        from collections import Counter
        counts = Counter(y)
        print(f'   Class counts: {dict(counts)}')

    return np.array(X), np.array(y), label_names


def _load_from_folders(dataset_dir: str, feature_type: str, verbose: bool):
    """Original loader: one subfolder per class."""
    X, y = [], []
    found_classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if verbose:
        print(f"\n📂 Folder-based dataset. Classes: {found_classes}")

    for label_idx, cls_name in enumerate(found_classes):
        cls_dir = os.path.join(dataset_dir, cls_name)
        files   = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]
        if verbose:
            print(f"  [{label_idx}] {cls_name}: {len(files)} files")
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                audio = load_audio(fpath)
                if feature_type == "flat":
                    feat = extract_flat_features(audio)
                elif feature_type == "mel":
                    feat = extract_mel_spectrogram(audio)
                elif feature_type == "seq":
                    feat = extract_sequence_features(audio)
                else:
                    raise ValueError(f"Unknown feature_type: {feature_type}")
                X.append(feat)
                y.append(label_idx)
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Skipped {fname}: {e}")

    return np.array(X), np.array(y), found_classes


if __name__ == "__main__":
    # Quick sanity check
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extractor.py <dataset_dir>")
        sys.exit(1)
    X, y, names = load_dataset(sys.argv[1], feature_type="flat")
    print(f"\n✅ Flat features:  X={X.shape}, y={y.shape}, classes={names}")
    X2, y2, _ = load_dataset(sys.argv[1], feature_type="mel", verbose=False)
    print(f"✅ Mel features:   X={X2.shape}")
    X3, y3, _ = load_dataset(sys.argv[1], feature_type="seq", verbose=False)
    print(f"✅ Seq features:   X={X3.shape}")
