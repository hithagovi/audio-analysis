"""
run_comparison.py  [v2]
-----------------------
Algorithm comparison runner for Threat Detection Research (Phase 2 — 80% target).

Loads train.csv + test.csv from the dataset, limits to 2000 samples total
(stratified), extracts features at 60s clip length, then trains and evaluates
5 algorithms side-by-side (v2 parameters, saved to models_v2/):
  CNN · BiLSTM · SVM · Random Forest · XGBoost

Usage (from threat-detection-v3 root):
  # Full run (80% target, 150 epochs):
  $env:TDV3_CLIP_SECONDS = "60"
  C:\\Projects\\ALGORITHM\\threat-detection-v2\\.venv-cuda\\Scripts\\python.exe scripts/run_comparison.py `
      --data-dir C:\\Projects\\ALGORITHM\\threat_detection_system\\data\\audio `
      --max-train 1715 --max-test 285 --epochs 150

  # Quick smoke test:
  $env:TDV3_CLIP_SECONDS = "25"
  ... python scripts/run_comparison.py --data-dir ... --max-train 40 --max-test 10 --epochs 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure scripts/ is on path ────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
os.chdir(REPO_ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Return at most max_rows rows from df, stratified by 'label'."""
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    from sklearn.utils import resample
    groups = []
    labels = df["label"].unique()
    per_class = max(1, max_rows // len(labels))
    for lbl in sorted(labels):
        grp = df[df["label"] == lbl]
        n   = min(len(grp), per_class)
        groups.append(grp.sample(n=n, random_state=seed))
    sampled = pd.concat(groups).sample(frac=1, random_state=seed)
    # Trim to exact max if rounding pushed over
    return sampled.head(max_rows).reset_index(drop=True)


def load_csv_features(csv_path: str, audio_root: str, feature_type: str,
                      max_rows: int = None, clip_seconds: float = 60.0,
                      label_offset: int = 0, verbose: bool = True):
    """
    Load CSV → audio → features.  Returns (X, y, label_names).
    label_offset: add to raw CSV label values (use 0 always; labels are already 0-indexed).
    """
    from feature_extractor import (
        load_audio_segments, extract_flat_features,
        extract_mel_spectrogram, extract_sequence_features,
        SAMPLE_RATE,
    )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label", "path"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    CLASS_NAME_MAP = {
        0: "Gunshot (Rifles)",
        1: "Automatic Fire",
        2: "Sniper Shot",
        3: "Tank Fire",
        4: "Artillery",
        5: "Vehicle Engine",
        6: "Aircraft",
    }

    if max_rows:
        df = stratified_sample(df, max_rows)

    unique_labels = sorted(df["label"].unique())
    label_map    = {orig: idx for idx, orig in enumerate(unique_labels)}
    label_names  = [CLASS_NAME_MAP.get(orig, f"class_{orig}") for orig in unique_labels]

    if verbose:
        print(f"\n📂 {Path(csv_path).name}: {len(df)} samples | clip={clip_seconds}s")
        for orig in unique_labels:
            print(f"   [{orig}] {CLASS_NAME_MAP.get(orig,'?'):20s}: {(df['label']==orig).sum()}")

    # Find audio root (try 'files' subfolder)
    audio_root = Path(audio_root)
    for candidate in ["files", "audio", "wavs"]:
        p = audio_root / candidate
        if p.is_dir():
            audio_root = p
            break

    X, y, skipped = [], [], 0
    for _, row in df.iterrows():
        rel = str(row["path"]).replace("\\", "/")
        fpath = None
        for candidate in [audio_root / rel, Path(str(audio_root).rstrip("/files")) / rel]:
            if Path(candidate).exists():
                fpath = str(candidate)
                break
        if fpath is None:
            skipped += 1
            continue
        try:
            segs = load_audio_segments(fpath, sr=SAMPLE_RATE,
                                       segment_duration=clip_seconds,
                                       segments_per_file=1,
                                       offset_mode="start")
            for audio in segs:
                if feature_type == "flat":
                    feat = extract_flat_features(audio)
                elif feature_type == "mel":
                    feat = extract_mel_spectrogram(audio)
                elif feature_type == "seq":
                    feat = extract_sequence_features(audio)
                else:
                    raise ValueError(f"Unknown feature_type: {feature_type}")
                X.append(feat)
                y.append(label_map[int(row["label"])])
        except Exception as e:
            skipped += 1
            if verbose and skipped <= 3:
                print(f"   ⚠️  Skipped {rel}: {e}")

    if verbose:
        print(f"   ✅ Loaded {len(X)} | Skipped {skipped}")

    return np.array(X), np.array(y), label_names


def print_comparison_table(results: dict):
    """Print a formatted comparison table."""
    cols = ["Algorithm", "Accuracy%", "F1%", "Precision%", "Recall%", "Time(s)"]
    widths = [22, 11, 8, 12, 9, 9]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    sep    = "  ".join("─" * w for w in widths)

    print("\n")
    print("═" * 80)
    print("  📊 ALGORITHM COMPARISON — THREAT DETECTION (60s Audio, Train/Test CSV Split)")
    print("═" * 80)
    print(header)
    print(sep)

    best_acc = max(v.get("accuracy", 0) for v in results.values())
    for name, m in results.items():
        acc  = m.get("accuracy", 0)
        flag = " ◄ BEST" if acc == best_acc else ""
        row  = [
            name,
            f"{acc:.2f}{flag}",
            f"{m.get('f1', 0):.2f}",
            f"{m.get('precision', 0):.2f}",
            f"{m.get('recall', 0):.2f}",
            f"{m.get('train_time', 0):.1f}",
        ]
        print("  ".join(f"{v:<{w}}" for v, w in zip(row, widths)))

    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Run all algorithm comparison for threat detection research."
    )
    ap.add_argument(
        "--data-dir",
        default=str(Path(REPO_ROOT).parent / "threat_detection_system" / "data" / "audio"),
        help="Path to folder containing train.csv, test.csv, and files/",
    )
    ap.add_argument("--max-train", type=int, default=1715,
                    help="Max samples from train.csv (stratified). Default: 1715")
    ap.add_argument("--max-test",  type=int, default=285,
                    help="Max samples from test.csv  (stratified). Default: 285")
    ap.add_argument("--epochs",    type=int, default=150,
                    help="Epochs for CNN and LSTM. Default: 150")
    ap.add_argument("--clip",      type=float,
                    default=float(os.getenv("TDV3_CLIP_SECONDS", "60")),
                    help="Audio clip length in seconds. Default: 60 (or TDV3_CLIP_SECONDS)")
    ap.add_argument("--skip",      nargs="*", default=[],
                    help="Algorithms to skip, e.g. --skip lstm logreg")
    ap.add_argument("--random-split", action="store_true",
                    help="Use train.csv only with 80/20 random split (no test.csv audio needed)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Shorthand: total samples when using --random-split (80%% train, 20%% test)")
    args = ap.parse_args()

    data_dir   = Path(args.data_dir)
    train_csv  = str(data_dir / "train.csv")
    test_csv   = str(data_dir / "test.csv")
    models_dir = REPO_ROOT / "models_v2"
    models_dir.mkdir(parents=True, exist_ok=True)

    clip = args.clip
    random_split = args.random_split
    if args.max_samples:
        # --max-samples 500 → 400 train + 100 test
        args.max_train = int(args.max_samples * 0.8)
        args.max_test  = int(args.max_samples * 0.2)
        if not random_split:
            random_split = True  # max-samples implies random split

    print(f"\n{'═'*60}")
    print(f"  🔬 THREAT DETECTION — ALGORITHM COMPARISON  [v2, target ≥80%]")
    print(f"  Clip length : {clip}s")
    print(f"  Save dir    : models_v2/")
    if random_split:
        print(f"  Samples     : {args.max_train + args.max_test} (80/20 random split from train.csv)")
    else:
        print(f"  Max train   : {args.max_train} samples")
        print(f"  Max test    : {args.max_test}  samples")
    print(f"  CNN/LSTM epochs: {args.epochs}")
    print(f"{'═'*60}")

    # Override CNN/LSTM epochs via module-level constants
    import model_cnn, model_lstm
    model_cnn.EPOCHS  = args.epochs
    model_lstm.EPOCHS = args.epochs

    import model_logreg, model_svm, model_rf, model_xgb

    skip = [s.lower() for s in args.skip]

    # ── Feature extraction ────────────────────────────────────────────────────
    total_rows = args.max_train + args.max_test if random_split else args.max_train

    print("\n[1/3] Extracting FLAT features (for SVM / RF / XGB / LogReg)...")
    t0 = time.time()
    if random_split:
        from sklearn.model_selection import train_test_split as sk_split
        X_flat_all, y_all, class_names = load_csv_features(
            train_csv, str(data_dir), "flat", max_rows=total_rows, clip_seconds=clip)
        X_flat_tr, X_flat_te, y_tr, y_te = sk_split(
            X_flat_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    else:
        X_flat_tr, y_tr, class_names = load_csv_features(
            train_csv, str(data_dir), "flat", max_rows=args.max_train, clip_seconds=clip)
        X_flat_te, y_te, _           = load_csv_features(
            test_csv,  str(data_dir), "flat", max_rows=args.max_test,  clip_seconds=clip, verbose=False)
    print(f"   Flat features done in {time.time()-t0:.1f}s — "
          f"X_train={X_flat_tr.shape}, X_test={X_flat_te.shape}")

    print("\n[2/3] Extracting MEL features (for CNN)...")
    t0 = time.time()
    if random_split:
        X_mel_all, _, _ = load_csv_features(
            train_csv, str(data_dir), "mel", max_rows=total_rows, clip_seconds=clip, verbose=False)
        X_mel_tr, X_mel_te, _, _ = sk_split(
            X_mel_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    else:
        X_mel_tr, _, _ = load_csv_features(
            train_csv, str(data_dir), "mel", max_rows=args.max_train, clip_seconds=clip, verbose=False)
        X_mel_te, _, _ = load_csv_features(
            test_csv,  str(data_dir), "mel", max_rows=args.max_test,  clip_seconds=clip, verbose=False)
    print(f"   Mel features done in {time.time()-t0:.1f}s — X_mel_train={X_mel_tr.shape}")

    print("\n[3/3] Extracting SEQ features (for BiLSTM)...")
    t0 = time.time()
    if random_split:
        X_seq_all, _, _ = load_csv_features(
            train_csv, str(data_dir), "seq", max_rows=total_rows, clip_seconds=clip, verbose=False)
        X_seq_tr, X_seq_te, _, _ = sk_split(
            X_seq_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    else:
        X_seq_tr, _, _ = load_csv_features(
            train_csv, str(data_dir), "seq", max_rows=args.max_train, clip_seconds=clip, verbose=False)
        X_seq_te, _, _ = load_csv_features(
            test_csv,  str(data_dir), "seq", max_rows=args.max_test,  clip_seconds=clip, verbose=False)
    print(f"   Seq features done in {time.time()-t0:.1f}s — X_seq_train={X_seq_tr.shape}")

    print(f"\n   Classes ({len(class_names)}): {class_names}")

    # ── Train all algorithms ──────────────────────────────────────────────────
    results = {}

    def timed_run(name, fn, *a, **kw):
        if name.lower() in skip:
            print(f"\n   [SKIP] {name}")
            return
        t = time.time()
        _, metrics = fn(*a, **kw)
        metrics["train_time"] = round(time.time() - t, 1)
        results[name] = metrics

    timed_run("LogReg",        model_logreg.run, X_flat_tr, y_tr, class_names, save=True,
              X_test=X_flat_te, y_test=y_te)

    timed_run("SVM",           model_svm.run,    X_flat_tr, y_tr, class_names, save=True,
              X_test=X_flat_te, y_test=y_te)

    timed_run("Random Forest", model_rf.run,     X_flat_tr, y_tr, class_names, save=True,
              X_test=X_flat_te, y_test=y_te)

    timed_run("XGBoost",       model_xgb.run,    X_flat_tr, y_tr, class_names, save=True,
              X_test=X_flat_te, y_test=y_te)

    timed_run("CNN",           model_cnn.run,    X_mel_tr,  y_tr, class_names, save=True,
              X_test=X_mel_te,  y_test=y_te)

    timed_run("BiLSTM",        model_lstm.run,   X_seq_tr,  y_tr, class_names, save=True,
              X_test=X_seq_te,  y_test=y_te)

    # ── Print comparison table ────────────────────────────────────────────────
    print_comparison_table(results)

    # Per-class breakdown
    print("  📋 Per-Class Accuracy Breakdown")
    print("  " + "─" * 70)
    header = f"  {'Class':<22}" + "".join(f"{n[:10]:<12}" for n in results)
    print(header)
    print("  " + "─" * 70)
    for cls in class_names:
        row = f"  {cls:<22}"
        for m in results.values():
            val = m.get("per_class", {}).get(cls, "-")
            row += f"{str(val)+'%':<12}" if val != "-" else f"{'N/A':<12}"
        print(row)
    print()

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "clip_seconds": clip,
        "max_train": args.max_train,
        "max_test":  args.max_test,
        "class_names": class_names,
        "results": {
            k: {
                "accuracy": v.get("accuracy"),
                "f1":       v.get("f1"),
                "precision":v.get("precision"),
                "recall":   v.get("recall"),
                "train_time_s": v.get("train_time"),
                "per_class": v.get("per_class"),
                "conf_matrix": v.get("conf_matrix"),
            }
            for k, v in results.items()
        },
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = models_dir / "comparison_results_v2.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  💾 Results saved → {out_path}")

    best = max(results, key=lambda k: results[k].get("accuracy", 0))
    print(f"\n  🏆 Best model: {best.upper()} ({results[best].get('accuracy')}% accuracy)\n")

    (models_dir / "class_names.json").write_text(
        json.dumps({"class_names": class_names}, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
