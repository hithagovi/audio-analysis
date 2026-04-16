"""
feature_extractor.py
--------------------
Shared audio feature extraction for all 5 models.
Extracts MFCCs, spectral features, and mel-spectrograms from WAV files.
"""

import os
import numpy as np
import librosa
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
# Default clip length (seconds) used for feature extraction.
# Override without code changes via:
#   TDV3_CLIP_SECONDS=25
#   TDV3_CLIP_SECONDS=60
DURATION      = float(os.getenv("TDV3_CLIP_SECONDS", "4.0"))
SEGMENTS_PER_FILE = int(os.getenv("TDV3_SEGMENTS_PER_FILE", "1"))
SEGMENT_OFFSET_MODE = os.getenv("TDV3_SEGMENT_OFFSET_MODE", "start")
_stride = os.getenv("TDV3_SEGMENT_STRIDE_SECONDS", "").strip()
SEGMENT_STRIDE_SECONDS = float(_stride) if _stride else None
SEGMENT_SEED = int(os.getenv("TDV3_SEGMENT_SEED", "42"))
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

def _get_audio_duration_seconds(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION, offset: float = 0.0):
    """Load a WAV file, mono, resampled, fixed-length window (pad/trim)."""
    if duration is None or duration <= 0:
        raise ValueError("duration must be a positive number of seconds")
    if offset is None:
        offset = 0.0

    y, _ = librosa.load(path, sr=sr, mono=True, offset=float(offset), duration=float(duration))
    target_len = int(round(sr * float(duration)))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def load_audio_segments(
    path: str,
    sr: int = SAMPLE_RATE,
    segment_duration: float = DURATION,
    segments_per_file: int = SEGMENTS_PER_FILE,
    stride: Optional[float] = SEGMENT_STRIDE_SECONDS,
    offset_mode: str = SEGMENT_OFFSET_MODE,
    seed: int = SEGMENT_SEED,
):
    """
    Load one or more fixed-length windows from an audio file.

    Useful when your source files are long (e.g., 25s/60s) but models operate on
    shorter windows (default 4s).

    offset_mode:
      - "start":    always from 0s (backwards-compatible)
      - "linspace": evenly spaced windows across the file
      - "random":   random offsets (seeded)
    """
    if segments_per_file < 0:
        raise ValueError("segments_per_file must be >= 0")
    if segment_duration is None or segment_duration <= 0:
        raise ValueError("segment_duration must be a positive number of seconds")

    total_dur = _get_audio_duration_seconds(path)
    if not total_dur or total_dur <= segment_duration:
        return [load_audio(path, sr=sr, duration=segment_duration, offset=0.0)]

    max_offset = max(0.0, float(total_dur) - float(segment_duration))
    mode = (offset_mode or "start").lower().strip()

    if stride is not None and stride > 0:
        stride = float(stride)
        if segments_per_file == 0:
            n = int(np.floor(max_offset / stride)) + 1
            offsets = [i * stride for i in range(max(1, n))]
        else:
            offsets = [i * stride for i in range(segments_per_file)]
        offsets = [min(o, max_offset) for o in offsets]
    elif mode == "start":
        if segments_per_file <= 1:
            offsets = [0.0]
        else:
            offsets = [i * float(segment_duration) for i in range(segments_per_file)]
            offsets = [min(o, max_offset) for o in offsets]
    elif mode == "linspace":
        n = 1 if segments_per_file <= 1 else segments_per_file
        offsets = np.linspace(0.0, max_offset, num=n).tolist()
    elif mode == "random":
        n = 1 if segments_per_file <= 1 else segments_per_file
        rng = np.random.default_rng(int(seed))
        offsets = rng.uniform(0.0, max_offset, size=n).tolist()
    else:
        raise ValueError(f"Unknown offset_mode: {offset_mode}")

    # Remove duplicates (can happen when offsets get clipped to max_offset).
    seen = set()
    deduped = []
    for o in offsets:
        key = round(float(o), 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(o))
    offsets = deduped

    segments = []
    for off in offsets:
        segments.append(load_audio(path, sr=sr, duration=segment_duration, offset=float(off)))
    return segments


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

def load_dataset(
    dataset_dir: str,
    feature_type: str = "flat",
    verbose: bool = True,
    test_csv: str = None,
    segments_per_file: int = SEGMENTS_PER_FILE,
    segment_duration: float = DURATION,
    segment_stride: Optional[float] = SEGMENT_STRIDE_SECONDS,
    segment_offset_mode: str = SEGMENT_OFFSET_MODE,
    segment_seed: int = SEGMENT_SEED,
):
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
        return _load_from_csv(
            test_csv,
            dataset_dir,
            feature_type,
            verbose=False,
            segments_per_file=segments_per_file,
            segment_duration=segment_duration,
            segment_stride=segment_stride,
            segment_offset_mode=segment_offset_mode,
            segment_seed=segment_seed,
        )

    if csv_path:
        return _load_from_csv(
            csv_path,
            dataset_dir,
            feature_type,
            verbose,
            segments_per_file=segments_per_file,
            segment_duration=segment_duration,
            segment_stride=segment_stride,
            segment_offset_mode=segment_offset_mode,
            segment_seed=segment_seed,
        )

    # ── Detect FORMAT B: class-named subfolders ──────────────────────────────
    return _load_from_folders(
        dataset_dir,
        feature_type,
        verbose,
        segments_per_file=segments_per_file,
        segment_duration=segment_duration,
        segment_stride=segment_stride,
        segment_offset_mode=segment_offset_mode,
        segment_seed=segment_seed,
    )


def _load_from_csv(csv_path: str, dataset_dir: str, feature_type: str,
                   verbose: bool,
                   segments_per_file: int = 1,
                   segment_duration: float = DURATION,
                   segment_stride: Optional[float] = None,
                   segment_offset_mode: str = "start",
                   segment_seed: int = 42):
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
            segments = load_audio_segments(
                fpath,
                sr=SAMPLE_RATE,
                segment_duration=segment_duration,
                segments_per_file=segments_per_file,
                stride=segment_stride,
                offset_mode=segment_offset_mode,
                seed=segment_seed,
            )
            for audio in segments:
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


def _load_from_folders(
    dataset_dir: str,
    feature_type: str,
    verbose: bool,
    segments_per_file: int = SEGMENTS_PER_FILE,
    segment_duration: float = DURATION,
    segment_stride: Optional[float] = SEGMENT_STRIDE_SECONDS,
    segment_offset_mode: str = SEGMENT_OFFSET_MODE,
    segment_seed: int = SEGMENT_SEED,
):
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
                segments = load_audio_segments(
                    fpath,
                    sr=SAMPLE_RATE,
                    segment_duration=segment_duration,
                    segments_per_file=segments_per_file,
                    stride=segment_stride,
                    offset_mode=segment_offset_mode,
                    seed=segment_seed,
                )
                for audio in segments:
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
    import argparse

    ap = argparse.ArgumentParser(description="Extract features from an audio dataset.")
    ap.add_argument("dataset_dir", help="Dataset folder (supports train.csv or class subfolders).")
    ap.add_argument("--segments-per-file", type=int, default=1)
    ap.add_argument("--segment-duration", type=float, default=DURATION)
    ap.add_argument("--segment-stride", type=float, default=None)
    ap.add_argument("--offset-mode", type=str, default="start", choices=["start", "linspace", "random"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seg_kwargs = dict(
        segments_per_file=args.segments_per_file,
        segment_duration=args.segment_duration,
        segment_stride=args.segment_stride,
        segment_offset_mode=args.offset_mode,
        segment_seed=args.seed,
    )

    X, y, names = load_dataset(args.dataset_dir, feature_type="flat", **seg_kwargs)
    print(f"\n✅ Flat features:  X={X.shape}, y={y.shape}, classes={names}")
    X2, y2, _ = load_dataset(args.dataset_dir, feature_type="mel", verbose=False, **seg_kwargs)
    print(f"✅ Mel features:   X={X2.shape}")
    X3, y3, _ = load_dataset(args.dataset_dir, feature_type="seq", verbose=False, **seg_kwargs)
    print(f"✅ Seq features:   X={X3.shape}")
