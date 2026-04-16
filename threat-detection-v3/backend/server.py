"""
server.py - Threat Detection v3
--------------------------------
FastAPI backend with:
- Direct folder path training (no ZIP needed)
- Real-time training metrics streamed to UI
- All 5 model comparison
- Port 8002
"""

import os, sys, json, time, shutil, tempfile, traceback, threading
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(_BASE, "..", "scripts"))
sys.path.insert(0, _SCRIPTS)


def _best_models_dir() -> str:
    """Return the highest-version models dir that has trained files."""
    v3 = os.path.normpath(os.path.join(_BASE, "..", "models_v3"))
    v2 = os.path.normpath(os.path.join(_BASE, "..", "models_v2"))
    v1 = os.path.normpath(os.path.join(_BASE, "..", "models"))
    for candidate in [v3, v2, v1]:
        if os.path.isdir(candidate) and any(
            f.endswith((".pt", ".pkl")) for f in os.listdir(candidate)
        ):
            return candidate
    return v1

from feature_extractor import (
    load_dataset,
    load_audio_segments,
    extract_flat_features,
    extract_mel_spectrogram,
    extract_sequence_features,
)
import model_svm, model_rf, model_xgb, model_cnn, model_lstm

app = FastAPI(title="Threat Detection API v3", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Global state ──────────────────────────────────────────────────────────────
training_state = {
    "status": "idle",
    "progress": 0,
    "current_model": "",
    "log": [],
    "results": {},
    "class_names": [],
    "error": "",
    "epoch_history": {},   # algo → list of {epoch, train_acc, val_acc, train_loss, val_loss}
    "dataset_stats": {},
}
trained_models = {}

def log(msg):
    print(msg)
    training_state["log"].append(msg)

# ── Training pipeline ─────────────────────────────────────────────────────────
def run_training(dataset_dir: str):
    global trained_models
    state = training_state
    state.update({"status":"running","progress":0,"log":[],"results":{},
                  "error":"","epoch_history":{},"class_names":[]})
    trained_models = {}

    try:
        train_segments_per_file = int(os.getenv("TD_TRAIN_SEGMENTS_PER_FILE", "1"))
        train_segment_duration  = float(os.getenv("TD_TRAIN_SEGMENT_DURATION", "4.0"))
        train_offset_mode       = os.getenv("TD_TRAIN_SEGMENT_OFFSET_MODE", "start")
        _stride_env             = os.getenv("TD_TRAIN_SEGMENT_STRIDE", "").strip()
        train_segment_stride    = float(_stride_env) if _stride_env else None
        train_segment_seed      = int(os.getenv("TD_TRAIN_SEGMENT_SEED", "42"))

        seg_kwargs = dict(
            segments_per_file=train_segments_per_file,
            segment_duration=train_segment_duration,
            segment_stride=train_segment_stride,
            segment_offset_mode=train_offset_mode,
            segment_seed=train_segment_seed,
        )

        log("📂 Loading dataset…")
        state["current_model"] = "Feature Extraction"
        state["progress"] = 3

        X_flat, y, class_names = load_dataset(dataset_dir, "flat", verbose=True, **seg_kwargs)
        state["class_names"] = class_names

        from collections import Counter
        counts = Counter(y.tolist())
        state["dataset_stats"] = {
            "total": len(y),
            "classes": {class_names[i]: int(v) for i, v in counts.items()},
            "feature_dim": int(X_flat.shape[1]),
        }

        log(f"   Classes : {class_names}")
        log(f"   Samples : {len(y)} | Features: {X_flat.shape[1]}")
        state["progress"] = 10

        # Test CSV
        X_flat_test = y_test = X_mel_test = y_mel_test = X_seq_test = y_seq_test = None
        for name in ["test.csv","Test.csv"]:
            p = os.path.join(dataset_dir, name)
            if os.path.exists(p):
                log("   ✅ Found test.csv — proper train/test split!")
                X_flat_test, y_test,     _ = load_dataset(dataset_dir,"flat", verbose=False, test_csv=p, **seg_kwargs)
                X_mel_test,  y_mel_test, _ = load_dataset(dataset_dir,"mel",  verbose=False, test_csv=p, **seg_kwargs)
                X_seq_test,  y_seq_test, _ = load_dataset(dataset_dir,"seq",  verbose=False, test_csv=p, **seg_kwargs)
                log(f"   Test samples: {len(y_test)}")
                break

        state["progress"] = 14
        X_mel, _, _ = load_dataset(dataset_dir, "mel", verbose=False, **seg_kwargs)
        state["progress"] = 19
        X_seq, _, _ = load_dataset(dataset_dir, "seq", verbose=False, **seg_kwargs)
        state["progress"] = 23

        results = {}
        models_dir = os.path.normpath(os.path.join(_BASE, "..", "models"))
        os.makedirs(models_dir, exist_ok=True)

        # ── SVM ──────────────────────────────────────────────────
        state["current_model"] = "SVM"
        log("\n⚙️  Training SVM…")
        m, metrics = model_svm.run(X_flat, y, class_names, save=True,
                                    X_test=X_flat_test, y_test=y_test)
        trained_models["svm"] = m
        results["svm"] = metrics
        log(f"   SVM ✅  {metrics['accuracy']}%")
        state["progress"] = 36

        # ── RF ───────────────────────────────────────────────────
        state["current_model"] = "Random Forest"
        log("\n⚙️  Training Random Forest…")
        m, metrics = model_rf.run(X_flat, y, class_names, save=True,
                                   X_test=X_flat_test, y_test=y_test)
        trained_models["rf"] = m
        results["rf"] = metrics
        log(f"   RF  ✅  {metrics['accuracy']}%")
        state["progress"] = 49

        # ── XGB ──────────────────────────────────────────────────
        state["current_model"] = "XGBoost"
        log("\n⚙️  Training XGBoost…")
        bundle, metrics = model_xgb.run(X_flat, y, class_names, save=True,
                                         X_test=X_flat_test, y_test=y_test)
        trained_models["xgb"] = bundle
        results["xgb"] = metrics
        log(f"   XGB ✅  {metrics['accuracy']}%")
        state["progress"] = 62

        # ── CNN ──────────────────────────────────────────────────
        state["current_model"] = "CNN"
        log("\n⚙️  Training CNN…")
        m, metrics = model_cnn.run(X_mel, y, class_names, save=True,
                                    X_test=X_mel_test, y_test=y_mel_test)
        trained_models["cnn"] = m
        results["cnn"] = metrics
        if "history" in metrics:
            state["epoch_history"]["cnn"] = metrics["history"]
        log(f"   CNN ✅  {metrics['accuracy']}%")
        state["progress"] = 80

        # ── LSTM ─────────────────────────────────────────────────
        state["current_model"] = "LSTM"
        log("\n⚙️  Training BiLSTM…")
        m, metrics = model_lstm.run(X_seq, y, class_names, save=True,
                                     X_test=X_seq_test, y_test=y_seq_test)
        trained_models["lstm"] = m
        results["lstm"] = metrics
        if "history" in metrics:
            state["epoch_history"]["lstm"] = metrics["history"]
        log(f"   LSTM✅  {metrics['accuracy']}%")
        state["progress"] = 96

        best = max(results, key=lambda k: results[k]["accuracy"])
        log(f"\n🏆 Best: {best.upper()} ({results[best]['accuracy']}%)")

        # Save slim metrics
        slim = {}
        for algo, mt in results.items():
            slim[algo] = {k: mt.get(k) for k in
                ["accuracy","f1","precision","recall","conf_matrix","per_class"]}

        with open(os.path.join(models_dir, "metrics.json"), "w") as f:
            json.dump({"results": slim, "class_names": class_names,
                       "dataset_stats": state["dataset_stats"]}, f)

        state.update({"results": slim, "status": "done",
                       "progress": 100, "current_model": ""})

    except Exception as e:
        tb = traceback.format_exc()
        log(f"\n❌ {e}\n{tb}")
        state.update({"status":"error","error":str(e),"current_model":""})


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health(): return {"status":"ok","version":"3.0"}


@app.get("/status")
def status():
    return {
        "status":        training_state["status"],
        "progress":      training_state["progress"],
        "current_model": training_state["current_model"],
        "log":           training_state["log"][-40:],
        "class_names":   training_state["class_names"],
        "dataset_stats": training_state["dataset_stats"],
        "epoch_history": training_state["epoch_history"],
        "error":         training_state["error"],
    }


@app.get("/results")
def results():
    if training_state["status"] != "done":
        raise HTTPException(400, "Training not complete")
    return {
        "results":       training_state["results"],
        "class_names":   training_state["class_names"],
        "dataset_stats": training_state["dataset_stats"],
        "epoch_history": training_state["epoch_history"],
    }


def _load_models_from_dir(models_dir: str):
    """Load all model files from a directory into trained_models."""
    global trained_models
    import joblib, torch

    if not os.path.exists(models_dir):
        raise HTTPException(400, f"Models directory not found: {models_dir}")

    loaded = []

    paths = {"svm": "svm_model.pkl", "rf": "rf_model.pkl", "xgb": "xgb_model.pkl"}
    for key, fname in paths.items():
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            trained_models[key] = joblib.load(p)
            loaded.append(key)

    for key, fname in [("cnn", "cnn_model.pt"), ("lstm", "lstm_model.pt")]:
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu")
            if key == "cnn":
                from model_cnn import ThreatCNN
                mdl = ThreatCNN(ckpt["n_classes"])
            else:
                from model_lstm import ThreatLSTM
                mdl = ThreatLSTM(ckpt["n_classes"])
            mdl.load_state_dict(ckpt["model_state"])
            trained_models[key] = mdl
            if ckpt.get("class_names"):
                training_state["class_names"] = ckpt["class_names"]
            loaded.append(key)
        else:
            print(f"  [!] Missing model file: {p}")

    # Load metrics if present (comparison_results_v2.json, metrics.json, or cnn_lstm_results.json)
    for mname in ["v3_results.json", "cnn_lstm_results.json", "comparison_results_v2.json", "metrics.json"]:
        metrics_path = os.path.join(models_dir, mname)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                saved = json.load(f)
            training_state["results"]       = saved.get("results", {})
            training_state["class_names"]   = saved.get("class_names",
                                               training_state["class_names"])
            training_state["dataset_stats"] = saved.get("dataset_stats", {})
            training_state["status"]        = "done"
            print(f"  [+] Loaded metrics from {mname}")
            break

    if not loaded:
        raise HTTPException(400, f"No saved model files found in {models_dir}")

    print(f"  [+] Successfully loaded {len(loaded)}/5 models from {models_dir}")
    training_state["status"] = "done"
    return loaded


@app.get("/load")
def load_saved():
    """
    Load models — automatically picks models_v2/ if it has trained files,
    otherwise falls back to models/.
    """
    models_dir = _best_models_dir()
    loaded = _load_models_from_dir(models_dir)
    return {
        "loaded":      loaded,
        "source":      os.path.basename(models_dir),
        "class_names": training_state["class_names"],
    }


@app.get("/load_v2")
def load_v2():
    """
    Force-load from models_v2/ (the new high-accuracy models).
    Call this after the v2 training run finishes.
    """
    models_dir = os.path.normpath(os.path.join(_BASE, "..", "models_v2"))
    loaded = _load_models_from_dir(models_dir)
    return {
        "loaded":      loaded,
        "source":      "models_v2",
        "class_names": training_state["class_names"],
    }


@app.get("/load_v1")
def load_v1():
    """Force-load from the original models/ directory."""
    models_dir = os.path.normpath(os.path.join(_BASE, "..", "models"))
    loaded = _load_models_from_dir(models_dir)
    return {
        "loaded":      loaded,
        "source":      "models (v1)",
        "class_names": training_state["class_names"],
    }


@app.post("/train/folder")
async def train_folder(background_tasks: BackgroundTasks,
                        dataset_path: str = None):
    """Train directly from a folder path — no ZIP needed."""
    if training_state["status"] == "running":
        raise HTTPException(400, "Training already in progress")
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(400, f"Path not found: {dataset_path}")

    background_tasks.add_task(run_training, dataset_path)
    return {"message": "Training started", "path": dataset_path}


@app.post("/train")
async def train_zip(background_tasks: BackgroundTasks,
                     file: UploadFile = File(...)):
    """Train from uploaded ZIP file."""
    if training_state["status"] == "running":
        raise HTTPException(400, "Training already in progress")

    import zipfile
    tmp_dir    = tempfile.mkdtemp()
    zip_path   = os.path.join(tmp_dir, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    extract_dir = os.path.join(tmp_dir, "dataset")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Find dataset root
    dataset_root = extract_dir
    entries = os.listdir(extract_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        dataset_root = os.path.join(extract_dir, entries[0])
    for root, dirs, files in os.walk(dataset_root):
        if "train.csv" in files or "Train.csv" in files:
            dataset_root = root
            break

    background_tasks.add_task(run_training, dataset_root)
    return {"message": "Training started"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    segments_per_file: Optional[int] = None,
    segment_duration: Optional[float] = None,
    segment_stride: Optional[float] = None,
    offset_mode: Optional[str] = None,
):
    if not trained_models:
        raise HTTPException(400, "No trained models. Run /train first.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(await file.read())
    tmp.close()
    path = tmp.name

    predictions = {}
    class_names = training_state.get("class_names", [])

    try:
        if segments_per_file is None:
            segments_per_file = int(
                os.getenv(
                    "TD_PREDICT_SEGMENTS_PER_FILE",
                    os.getenv("TDV3_SEGMENTS_PER_FILE", "1"),
                )
            )
        if segment_duration is None:
            segment_duration = float(
                os.getenv(
                    "TD_PREDICT_SEGMENT_DURATION",
                    os.getenv("TDV3_CLIP_SECONDS", "4.0"),
                )
            )
        if offset_mode is None:
            offset_mode = os.getenv(
                "TD_PREDICT_SEGMENT_OFFSET_MODE",
                os.getenv("TDV3_SEGMENT_OFFSET_MODE", "start"),
            )
        if segment_stride is None:
            _ps = os.getenv(
                "TD_PREDICT_SEGMENT_STRIDE",
                os.getenv("TDV3_SEGMENT_STRIDE_SECONDS", ""),
            ).strip()
            segment_stride = float(_ps) if _ps else None
        seg_seed = int(
            os.getenv(
                "TD_PREDICT_SEGMENT_SEED",
                os.getenv("TDV3_SEGMENT_SEED", "42"),
            )
        )
        # Scan multiple windows across the file if segments_per_file is not specified
        if segments_per_file is None:
            segments_per_file = 5
            offset_mode = "linspace"
            print(f"  [*] Auto-scanning full file: {segments_per_file} windows (linspace)")

        segments = load_audio_segments(
            path,
            segment_duration=float(segment_duration),
            segments_per_file=int(segments_per_file),
            stride=segment_stride,
            offset_mode=offset_mode,
            seed=seg_seed,
        )

        flat_batch = mel_batch = seq_batch = None
        if any(k in trained_models for k in ["svm", "rf", "xgb"]):
            flat_batch = np.stack([extract_flat_features(s) for s in segments], axis=0)
        if "cnn" in trained_models:
            mel_batch = np.stack([extract_mel_spectrogram(s) for s in segments], axis=0)
        if "lstm" in trained_models:
            seq_batch = np.stack([extract_sequence_features(s) for s in segments], axis=0)

        if "svm" in trained_models:
            p = trained_models["svm"].predict_proba(flat_batch).mean(axis=0)
            predictions["svm"] = {"pred": int(np.argmax(p)), "proba": p.tolist()}

        if "rf" in trained_models:
            p = trained_models["rf"].predict_proba(flat_batch).mean(axis=0)
            predictions["rf"] = {"pred": int(np.argmax(p)), "proba": p.tolist()}

        if "xgb" in trained_models:
            b = trained_models["xgb"]
            # b is always a dict {"model": ..., "scaler": ...}
            # (returned directly by model_xgb.run() and saved/loaded the same way)
            xgb_scaler = b["scaler"] if isinstance(b, dict) else b[0]
            xgb_model  = b["model"]  if isinstance(b, dict) else b[1]
            s = xgb_scaler.transform(flat_batch)
            p = xgb_model.predict_proba(s).mean(axis=0)
            predictions["xgb"] = {"pred": int(np.argmax(p)), "proba": p.tolist()}

        if "cnn" in trained_models:
            import torch
            dev = next(trained_models["cnn"].parameters()).device
            # Per-sample normalisation (matches V3 training exactly)
            mu  = mel_batch.mean(axis=(1, 2), keepdims=True)
            std = mel_batch.std(axis=(1, 2),  keepdims=True) + 1e-8
            m_n = (mel_batch - mu) / std
            t   = torch.tensor(m_n[:, None, :, :], dtype=torch.float32).to(dev)
            with torch.no_grad():
                p = torch.softmax(trained_models["cnn"](t), dim=1).cpu().numpy().mean(axis=0)
            predictions["cnn"] = {"pred": int(np.argmax(p)), "proba": p.tolist()}

        if "lstm" in trained_models:
            import torch
            dev = next(trained_models["lstm"].parameters()).device
            # Per-sample normalisation for LSTM (matches SeqDataset in model_lstm.py)
            mu  = seq_batch.mean(axis=(1), keepdims=True)
            std = seq_batch.std(axis=(1),  keepdims=True) + 1e-8
            s_n = (seq_batch - mu) / std
            t   = torch.tensor(s_n, dtype=torch.float32).to(dev)
            with torch.no_grad():
                p = torch.softmax(trained_models["lstm"](t), dim=1).cpu().numpy().mean(axis=0)
            predictions["lstm"] = {"pred": int(np.argmax(p)), "proba": p.tolist()}

    finally:
        os.unlink(path)

    return {"predictions": predictions, "class_names": class_names, "segments_used": len(segments)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)