"""
train_cnn_lstm.py
-----------------
Resume-training script: trains ONLY the CNN and BiLSTM models
and saves them into models_v2/ (RF, SVM, XGB are already saved).

Usage:
    python scripts/train_cnn_lstm.py <dataset_dir>

Example:
    python scripts/train_cnn_lstm.py data/audio
"""

import os
import sys
import time
import json
from pathlib import Path


class Tee:
    """Mirror stdout to both console and a log file in real-time."""
    def __init__(self, log_path):
        self._console = sys.stdout
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
        print("Usage: python scripts/train_cnn_lstm.py <dataset_dir>")
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "scripts"))

    # ── Live log file (open in VS Code to watch progress) ───────────────────
    log_path = repo_root / "training_log.txt"
    tee = Tee(str(log_path))
    sys.stdout = tee
    print(f"  Log file : {log_path}  ← open this in VS Code to watch live\n")

    dataset_dir = sys.argv[1]
    models_v2_dir = repo_root / "models_v2"
    models_v2_dir.mkdir(parents=True, exist_ok=True)

    from feature_extractor import load_dataset
    import model_cnn, model_lstm

    print("═" * 70)
    print("  RESUME TRAINING — CNN + BiLSTM  →  models_v2/")
    print("═" * 70)
    print(f"  Dataset : {dataset_dir}")
    print(f"  Output  : {models_v2_dir}\n")

    # ── Feature extraction ──────────────────────────────────────────────────
    print("Loading mel spectrograms for CNN …")
    X_mel, y, class_names = load_dataset(dataset_dir, "mel", verbose=True)

    print("\nLoading MFCC sequences for BiLSTM …")
    X_seq, _, _ = load_dataset(dataset_dir, "seq", verbose=False)

    print(f"\n  Classes : {class_names}")

    results = {}

    # ── CNN ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    _, metrics = model_cnn.run(X_mel, y, class_names, save=True)
    results["cnn"] = metrics
    print(f"  CNN done in {(time.time()-t0)/60:.1f} min  |  "
          f"acc={metrics['accuracy']}%  f1={metrics['f1']}%")

    # ── BiLSTM ──────────────────────────────────────────────────────────────
    t0 = time.time()
    _, metrics = model_lstm.run(X_seq, y, class_names, save=True)
    results["lstm"] = metrics
    print(f"  LSTM done in {(time.time()-t0)/60:.1f} min  |  "
          f"acc={metrics['accuracy']}%  f1={metrics['f1']}%")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FINAL RESULTS")
    print("═" * 70)
    for name, m in results.items():
        print(f"  {name.upper():8s}  acc={m['accuracy']}%  "
              f"f1={m['f1']}%  prec={m['precision']}%  rec={m['recall']}%")

    # Save partial results alongside existing models
    results_path = models_v2_dir / "cnn_lstm_results.json"
    results_path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "results": results,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Results saved → {results_path}")
    print("  ✅ models_v2/cnn_model.pt  and  models_v2/lstm_model.pt  are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
