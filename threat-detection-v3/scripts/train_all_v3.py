"""
train_all_v3.py
---------------
V3 Training Pipeline — uses the PROPER train.csv / test.csv split.

Key improvements over v2:
  • Uses train.csv for training and test.csv for honest evaluation
    (no data leakage between train/test clips)
  • Extracts 7 overlapping windows per 60-second clip (stride=8s)
    → ~5x more training data than v2
  • Saves to models_v3/ directory
  • Lighter model architectures (right-sized for dataset)

Usage:
    python scripts/train_all_v3.py data/audio
"""

import os
import sys
import time
import json
from pathlib import Path


class Tee:
    """Mirror stdout to both console and a log file in real-time."""
    def __init__(self, log_path):
        self._console = sys.__stdout__
        self._file    = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, msg):
        self._console.write(msg)
        self._console.flush()
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._console.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_all_v3.py <dataset_dir>")
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "scripts"))

    # Live log - open training_v3_log.txt in VS Code to watch progress
    log_path = repo_root / "training_v3_log.txt"
    tee = Tee(str(log_path))
    sys.stdout = tee

    dataset_dir = sys.argv[1]
    train_csv   = os.path.join(dataset_dir, "train.csv")
    test_csv    = os.path.join(dataset_dir, "test.csv")
    models_dir  = repo_root / "models_v3"
    models_dir.mkdir(parents=True, exist_ok=True)

    assert os.path.exists(train_csv), f"train.csv not found: {train_csv}"
    assert os.path.exists(test_csv),  f"test.csv not found:  {test_csv}"

    from feature_extractor import load_dataset
    import model_cnn, model_lstm, model_rf, model_svm, model_xgb

    print("=" * 70)
    print("  V3 TRAINING  — Proper train.csv / test.csv split")
    print("=" * 70)
    print(f"  Dataset  : {dataset_dir}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Test CSV : {test_csv}")
    print(f"  Output   : {models_dir}")
    print(f"  Log file : {log_path}  <-- open in VS Code to watch live")
    print()

    # ── Segment extraction config ────────────────────────────────────────────
    # Each WAV is 60 seconds. stride=8s → 7 non-overlapping windows per clip.
    # This gives ~5x more samples than v2 (which used 1 segment per row).
    SEG_DURATION = 4.0
    SEG_STRIDE   = 8.0   # 8s stride → 7 windows from 60s clip

    print(f"  Segment duration : {SEG_DURATION}s")
    print(f"  Stride           : {SEG_STRIDE}s  (~7 windows per 60s clip)")
    print()

    # ── Load features ────────────────────────────────────────────────────────
    print("━" * 70)
    print("  LOADING FEATURES")
    print("━" * 70)

    print("\n[1/6] Training mel spectrograms ...")
    X_mel_tr, y_tr, class_names = load_dataset(
        dataset_dir, feature_type="mel", verbose=True,
        segment_duration=SEG_DURATION, segment_stride=SEG_STRIDE,
        segments_per_file=0,
    )

    print(f"\n[2/6] Test mel spectrograms ...")
    X_mel_te, y_te, _ = load_dataset(
        dataset_dir, feature_type="mel", verbose=True,
        test_csv=test_csv,
        segment_duration=SEG_DURATION, segment_stride=SEG_STRIDE,
        segments_per_file=0,
    )

    print(f"\n[3/6] Training flat (MFCC) features ...")
    X_flat_tr, _, _ = load_dataset(
        dataset_dir, feature_type="flat", verbose=False,
        segment_duration=SEG_DURATION, segment_stride=SEG_STRIDE,
        segments_per_file=0,
    )

    print(f"\n[4/6] Test flat (MFCC) features ...")
    X_flat_te, _, _ = load_dataset(
        dataset_dir, feature_type="flat", verbose=False,
        test_csv=test_csv,
        segment_duration=SEG_DURATION, segment_stride=SEG_STRIDE,
        segments_per_file=0,
    )

    print(f"\n[5/6] Training MFCC sequences (LSTM) ...")
    X_seq_tr, _, _ = load_dataset(
        dataset_dir, feature_type="seq", verbose=False,
        segment_duration=SEG_DURATION, segment_stride=SEG_STRIDE,
        segments_per_file=0,
    )

    print(f"\n[6/6] Test MFCC sequences (LSTM) ...")
    X_seq_te, _, _ = load_dataset(
        dataset_dir, feature_type="seq", verbose=False,
        test_csv=test_csv,
        segment_duration=SEG_DURATION, segment_stride=SEG_STRIDE,
        segments_per_file=0,
    )

    print()
    print(f"  Classes      : {class_names}")
    print(f"  Train samples: {len(X_mel_tr)} mel | {len(X_flat_tr)} flat | {len(X_seq_tr)} seq")
    print(f"  Test samples : {len(X_mel_te)} mel | {len(X_flat_te)} flat | {len(X_seq_te)} seq")
    print()

    results   = {}
    cnn_path  = str(models_dir / "cnn_model.pt")
    lstm_path = str(models_dir / "lstm_model.pt")
    rf_path   = str(models_dir / "rf_model.pkl")
    svm_path  = str(models_dir / "svm_model.pkl")
    xgb_path  = str(models_dir / "xgb_model.pkl")

    # ── CNN ──────────────────────────────────────────────────────────────────
    print("━" * 70)
    t0 = time.time()
    _, metrics = model_cnn.run(
        X_mel_tr, y_tr, class_names, save=True,
        X_test=X_mel_te, y_test=y_te,
        save_path=cnn_path,
    )
    results["cnn"] = metrics
    print(f"  CNN     | {(time.time()-t0)/60:.1f} min | acc={metrics['accuracy']}%  f1={metrics['f1']}%")

    # ── BiLSTM ───────────────────────────────────────────────────────────────
    print("━" * 70)
    t0 = time.time()
    _, metrics = model_lstm.run(
        X_seq_tr, y_tr, class_names, save=True,
        X_test=X_seq_te, y_test=y_te,
        save_path=lstm_path,
    )
    results["lstm"] = metrics
    print(f"  BiLSTM  | {(time.time()-t0)/60:.1f} min | acc={metrics['accuracy']}%  f1={metrics['f1']}%")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("━" * 70)
    t0 = time.time()
    _, metrics = model_rf.run(
        X_flat_tr, y_tr, class_names, save=True,
        X_test=X_flat_te, y_test=y_te,
        save_path=rf_path,
    )
    results["rf"] = metrics
    print(f"  RF      | {(time.time()-t0)/60:.1f} min | acc={metrics['accuracy']}%  f1={metrics['f1']}%")

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("━" * 70)
    t0 = time.time()
    _, metrics = model_svm.run(
        X_flat_tr, y_tr, class_names, save=True,
        X_test=X_flat_te, y_test=y_te,
        save_path=svm_path,
    )
    results["svm"] = metrics
    print(f"  SVM     | {(time.time()-t0)/60:.1f} min | acc={metrics['accuracy']}%  f1={metrics['f1']}%")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("━" * 70)
    t0 = time.time()
    _, metrics = model_xgb.run(
        X_flat_tr, y_tr, class_names, save=True,
        X_test=X_flat_te, y_test=y_te,
        save_path=xgb_path,
    )
    results["xgb"] = metrics
    print(f"  XGBoost | {(time.time()-t0)/60:.1f} min | acc={metrics['accuracy']}%  f1={metrics['f1']}%")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINAL RESULTS  (V3 — honest train/test split, stride extraction)")
    print("=" * 70)
    for name, m in results.items():
        print(f"  {name.upper():8s}  acc={m['accuracy']:5.1f}%  "
              f"f1={m['f1']:5.1f}%  prec={m['precision']:5.1f}%  rec={m['recall']:5.1f}%")

    # Save results JSON
    results_path = models_dir / "v3_results.json"
    results_path.write_text(
        json.dumps({
            "class_names": class_names,
            "results": results,
            "train_samples": len(X_mel_tr),
            "test_samples":  len(X_mel_te),
            "seg_duration":  SEG_DURATION,
            "seg_stride":    SEG_STRIDE,
            "saved_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Results saved -> {results_path}")
    print("  All models saved to models_v3/")
    print()

    tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
