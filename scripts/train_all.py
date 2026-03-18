"""
train_all.py
------------
Offline training runner: trains all algorithms once and writes artifacts into `models/`.
This enables "drop any audio → predict" in new environments without uploading a dataset in the UI.

Usage:
  python scripts/train_all.py <dataset_dir>
"""

import json
import os
import sys
import time
from pathlib import Path


def _compact_metrics(m: dict) -> dict:
    return {
        "accuracy": m.get("accuracy"),
        "f1": m.get("f1"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "conf_matrix": m.get("conf_matrix"),
        "per_class": m.get("per_class"),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_all.py <dataset_dir>")
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "scripts"))

    dataset_dir = sys.argv[1]
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    from feature_extractor import load_dataset
    import model_logreg, model_svm, model_rf, model_xgb, model_cnn, model_lstm

    print("═" * 70)
    print("TRAIN ALL MODELS")
    print("═" * 70)
    print(f"Dataset: {dataset_dir}")
    print(f"Output : {models_dir}")

    # Feature extraction
    X_flat, y, class_names = load_dataset(dataset_dir, "flat", verbose=True)
    X_mel, _, _ = load_dataset(dataset_dir, "mel", verbose=False)
    X_seq, _, _ = load_dataset(dataset_dir, "seq", verbose=False)

    (models_dir / "class_names.json").write_text(
        json.dumps({"class_names": class_names}, indent=2),
        encoding="utf-8",
    )

    results = {}

    # Baselines + tree models (flat features)
    _, metrics = model_logreg.run(X_flat, y, class_names, save=True)
    results["logreg"] = metrics

    _, metrics = model_svm.run(X_flat, y, class_names, save=True)
    results["svm"] = metrics

    _, metrics = model_rf.run(X_flat, y, class_names, save=True)
    results["rf"] = metrics

    _, metrics = model_xgb.run(X_flat, y, class_names, save=True)
    results["xgb"] = metrics

    # Deep models
    _, metrics = model_cnn.run(X_mel, y, class_names, save=True)
    results["cnn"] = metrics

    _, metrics = model_lstm.run(X_seq, y, class_names, save=True)
    results["lstm"] = metrics

    best = max(results, key=lambda k: results[k].get("accuracy", 0))
    print("\n" + "═" * 70)
    print(f"Best model: {best.upper()} ({results[best].get('accuracy')}%)")

    (models_dir / "last_results.json").write_text(
        json.dumps(
            {
                "class_names": class_names,
                "results": {k: _compact_metrics(v) for k, v in results.items()},
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved: {models_dir / 'class_names.json'}")
    print(f"Saved: {models_dir / 'last_results.json'}")
    print("Model files are saved by each model under `models/`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
