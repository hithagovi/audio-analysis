"""
server.py
---------
FastAPI backend for the Audio Threat Detection UI.
Endpoints:
  POST /train        — upload dataset ZIP → train all 5 models → return metrics
  POST /predict      — upload single WAV  → run all trained models → return results
  GET  /status       — training progress
  GET  /health       — heartbeat
"""

import os, sys, json, time, shutil, zipfile, tempfile, threading, traceback
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make scripts importable — works from any working directory on Windows/Mac/Linux
_BASE = os.path.dirname(os.path.abspath(__file__))          # .../backend/
_SCRIPTS = os.path.join(_BASE, "..", "scripts")             # .../scripts/
sys.path.insert(0, os.path.normpath(_SCRIPTS))

from feature_extractor import load_dataset, load_audio, extract_flat_features, \
                              extract_mel_spectrogram, extract_sequence_features
import model_svm, model_rf, model_xgb, model_cnn, model_lstm

app = FastAPI(title="Threat Detection API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────

training_state = {
    "status": "idle",          # idle | running | done | error
    "progress": 0,             # 0-100
    "current_model": "",
    "log": [],
    "results": {},
    "class_names": [],
    "error": "",
}
trained_models = {}            # algo_id → model object


def log(msg: str):
    print(msg)
    training_state["log"].append(msg)


# ─── TRAINING PIPELINE ────────────────────────────────────────────────────────

def run_training(dataset_dir: str):
    global trained_models
    state = training_state
    state["status"]  = "running"
    state["progress"] = 0
    state["log"]      = []
    state["results"]  = {}
    state["error"]    = ""
    trained_models    = {}

    try:
        # ── 1. Feature extraction ─────────────────────────────────
        log("📂 Loading dataset…")
        state["current_model"] = "Feature Extraction"
        state["progress"] = 5

        X_flat, y, class_names = load_dataset(dataset_dir, "flat", verbose=True)
        state["class_names"] = class_names
        log(f"   Classes: {class_names}")
        log(f"   Train samples: {len(y)} | Features: {X_flat.shape[1]}")
        state["progress"] = 12

        # Load test.csv if it exists for proper evaluation
        X_flat_test, y_test = None, None
        X_mel_test,  y_mel_test = None, None
        X_seq_test,  y_seq_test = None, None

        for test_name in ["test.csv", "Test.csv"]:
            test_csv_path = os.path.join(dataset_dir, test_name)
            if os.path.exists(test_csv_path):
                log(f"   ✅ Found test.csv — using proper train/test split!")
                X_flat_test, y_test,     _ = load_dataset(dataset_dir, "flat", verbose=False, test_csv=test_csv_path)
                X_mel_test,  y_mel_test, _ = load_dataset(dataset_dir, "mel",  verbose=False, test_csv=test_csv_path)
                X_seq_test,  y_seq_test, _ = load_dataset(dataset_dir, "seq",  verbose=False, test_csv=test_csv_path)
                log(f"   Test samples: {len(y_test)}")
                break
        else:
            log("   No test.csv found — using 80/20 split")

        state["progress"] = 15

        X_mel, _, _ = load_dataset(dataset_dir, "mel", verbose=False)
        log(f"   Mel shape: {X_mel.shape}")
        state["progress"] = 22

        X_seq, _, _ = load_dataset(dataset_dir, "seq", verbose=False)
        log(f"   Seq shape: {X_seq.shape}")
        state["progress"] = 28

        results = {}

        # ── 2. SVM ────────────────────────────────────────────────
        state["current_model"] = "SVM"
        log("\n⚙️  Training SVM…")
        model, metrics = model_svm.run(X_flat, y, class_names, save=True,
                                        X_test=X_flat_test, y_test=y_test)
        trained_models["svm"] = model
        results["svm"]        = metrics
        log(f"   SVM ✅  Accuracy: {metrics['accuracy']}%")
        state["progress"] = 42

        # ── 3. Random Forest ──────────────────────────────────────
        state["current_model"] = "Random Forest"
        log("\n⚙️  Training Random Forest…")
        model, metrics = model_rf.run(X_flat, y, class_names, save=True,
                                       X_test=X_flat_test, y_test=y_test)
        trained_models["rf"] = model
        results["rf"]        = metrics
        log(f"   RF  ✅  Accuracy: {metrics['accuracy']}%")
        state["progress"] = 55

        # ── 4. XGBoost ────────────────────────────────────────────
        state["current_model"] = "XGBoost"
        log("\n⚙️  Training XGBoost…")
        bundle, metrics = model_xgb.run(X_flat, y, class_names, save=True,
                                        X_test=X_flat_test, y_test=y_test)
        trained_models["xgb"] = bundle
        results["xgb"]        = metrics
        log(f"   XGB ✅  Accuracy: {metrics['accuracy']}%")
        state["progress"] = 68

        # ── 5. CNN ────────────────────────────────────────────────
        state["current_model"] = "CNN"
        log("\n⚙️  Training CNN…")
        model, metrics = model_cnn.run(X_mel, y, class_names, save=True,
                                        X_test=X_mel_test, y_test=y_mel_test)
        trained_models["cnn"] = model
        results["cnn"]        = metrics
        log(f"   CNN ✅  Accuracy: {metrics['accuracy']}%")
        state["progress"] = 84

        # ── 6. LSTM ───────────────────────────────────────────────
        state["current_model"] = "LSTM"
        log("\n⚙️  Training BiLSTM…")
        model, metrics = model_lstm.run(X_seq, y, class_names, save=True,
                                         X_test=X_seq_test, y_test=y_seq_test)
        trained_models["lstm"] = model
        results["lstm"]        = metrics
        log(f"   LSTM✅  Accuracy: {metrics['accuracy']}%")
        state["progress"] = 98

        # ── Summary ───────────────────────────────────────────────
        best = max(results, key=lambda k: results[k]["accuracy"])
        log(f"\n🏆 Best model: {best.upper()}  ({results[best]['accuracy']}%)")

        state["results"]  = results
        state["status"]   = "done"
        state["progress"] = 100
        state["current_model"] = ""

        # Save slim metrics to disk so they survive server restarts
        import json
        models_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models"))
        os.makedirs(models_dir, exist_ok=True)
        slim = {}
        for algo, m in results.items():
            slim[algo] = {
                "accuracy":    m.get("accuracy"),
                "f1":          m.get("f1"),
                "precision":   m.get("precision"),
                "recall":      m.get("recall"),
                "conf_matrix": m.get("conf_matrix"),
                "per_class":   m.get("per_class"),
            }
        with open(os.path.join(models_dir, "metrics.json"), "w") as f:
            json.dump({"results": slim, "class_names": class_names}, f)
        log("💾 Metrics saved to models/metrics.json")

    except Exception as e:
        tb = traceback.format_exc()
        log(f"\n❌ Error: {e}\n{tb}")
        state["status"]  = "error"
        state["error"]   = str(e)
        state["current_model"] = ""


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}


@app.get("/load")
def load_saved_models():
    """Load previously trained models from disk and restore results."""
    import joblib, torch
    global trained_models

    models_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models"))

    if not os.path.exists(models_dir):
        raise HTTPException(400, "No models directory found. Train first.")

    loaded = []
    results = {}

    # SVM
    svm_path = os.path.join(models_dir, "svm_model.pkl")
    if os.path.exists(svm_path):
        m = joblib.load(svm_path)
        trained_models["svm"] = m
        loaded.append("svm")

    # RF
    rf_path = os.path.join(models_dir, "rf_model.pkl")
    if os.path.exists(rf_path):
        m = joblib.load(rf_path)
        trained_models["rf"] = m
        loaded.append("rf")

    # XGB
    xgb_path = os.path.join(models_dir, "xgb_model.pkl")
    if os.path.exists(xgb_path):
        m = joblib.load(xgb_path)
        trained_models["xgb"] = m
        loaded.append("xgb")

    # CNN
    cnn_path = os.path.join(models_dir, "cnn_model.pt")
    if os.path.exists(cnn_path):
        sys.path.insert(0, os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "scripts")))
        from model_cnn import ThreatCNN
        import torch
        ckpt = torch.load(cnn_path, map_location="cpu")
        mdl  = ThreatCNN(ckpt["n_classes"])
        mdl.load_state_dict(ckpt["model_state"])
        trained_models["cnn"] = mdl
        class_names = ckpt.get("class_names", [])
        if class_names:
            training_state["class_names"] = class_names
        loaded.append("cnn")

    # LSTM
    lstm_path = os.path.join(models_dir, "lstm_model.pt")
    if os.path.exists(lstm_path):
        from model_lstm import ThreatLSTM
        ckpt = torch.load(lstm_path, map_location="cpu")
        mdl  = ThreatLSTM(ckpt["n_classes"])
        mdl.load_state_dict(ckpt["model_state"])
        trained_models["lstm"] = mdl
        loaded.append("lstm")

    # Load saved metrics if available
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            import json
            saved = json.load(f)
            training_state["results"]     = saved.get("results", {})
            training_state["class_names"] = saved.get("class_names", training_state["class_names"])
            training_state["status"]      = "done"

    if not loaded:
        raise HTTPException(400, "No saved model files found in models/ folder.")

    training_state["status"] = "done"

    # If no metrics.json, create placeholder so UI shows comparison tab
    if not training_state["results"]:
        placeholder = {}
        for algo in loaded:
            placeholder[algo] = {
                "accuracy":    0.0,
                "f1":          0.0,
                "precision":   0.0,
                "recall":      0.0,
                "conf_matrix": [],
                "per_class":   {},
            }
        training_state["results"] = placeholder

    return {"loaded": loaded, "class_names": training_state["class_names"]}


@app.get("/status")
def status():
    return {
        "status":        training_state["status"],
        "progress":      training_state["progress"],
        "current_model": training_state["current_model"],
        "log":           training_state["log"][-30:],   # last 30 lines
        "class_names":   training_state["class_names"],
        "error":         training_state["error"],
    }


@app.get("/results")
def results():
    if training_state["status"] != "done":
        raise HTTPException(400, "Training not complete")
    # Strip heavy arrays (y_proba, y_pred, report) — only send what UI needs
    slim = {}
    for algo, m in training_state["results"].items():
        slim[algo] = {
            "accuracy":    m.get("accuracy"),
            "f1":          m.get("f1"),
            "precision":   m.get("precision"),
            "recall":      m.get("recall"),
            "conf_matrix": m.get("conf_matrix"),
            "per_class":   m.get("per_class"),
        }
    return {
        "results":     slim,
        "class_names": training_state["class_names"],
    }


@app.post("/train")
async def train_endpoint(background_tasks: BackgroundTasks,
                          file: UploadFile = File(...)):
    """
    Accept a ZIP of the dataset (folder-per-class structure) and start training.
    The ZIP should contain:
        gunshot/  *.wav
        rifle_fire/ *.wav
        ...
    """
    if training_state["status"] == "running":
        raise HTTPException(400, "Training already in progress")

    # Save ZIP to temp dir, extract
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "dataset.zip")

    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)

    extract_dir = os.path.join(tmp_dir, "dataset")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Find actual dataset root — walk to find train.csv or class folders
    dataset_root = extract_dir
    entries = os.listdir(extract_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        dataset_root = os.path.join(extract_dir, entries[0])

    # Walk deeper to find train.csv
    for root, dirs, files in os.walk(dataset_root):
        if "train.csv" in files or "Train.csv" in files:
            dataset_root = root
            break

    background_tasks.add_task(run_training, dataset_root)
    return {"message": "Training started", "tmp_dir": tmp_dir}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept a single WAV file, run all trained models, return predictions.
    """
    if not trained_models:
        raise HTTPException(400, "No trained models available. Run /train first.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(await file.read())
    tmp.close()
    audio_path = tmp.name

    predictions = {}
    class_names = training_state.get("class_names", [])

    try:
        # Flat features for SVM/RF/XGB
        audio    = load_audio(audio_path)
        flat     = extract_flat_features(audio)
        mel      = extract_mel_spectrogram(audio)
        seq      = extract_sequence_features(audio)

        # SVM
        if "svm" in trained_models:
            proba = trained_models["svm"].predict_proba(flat.reshape(1, -1))[0]
            predictions["svm"] = {"pred": int(np.argmax(proba)), "proba": proba.tolist()}

        # RF
        if "rf" in trained_models:
            proba = trained_models["rf"].predict_proba(flat.reshape(1, -1))[0]
            predictions["rf"] = {"pred": int(np.argmax(proba)), "proba": proba.tolist()}

        # XGBoost
        if "xgb" in trained_models:
            bundle = trained_models["xgb"]
            m, sc  = bundle["model"], bundle["scaler"]
            scaled = sc.transform(flat.reshape(1, -1))
            proba  = m.predict_proba(scaled)[0]
            predictions["xgb"] = {"pred": int(np.argmax(proba)), "proba": proba.tolist()}

        # CNN
        if "cnn" in trained_models:
            import torch
            device = next(trained_models["cnn"].parameters()).device
            mel_n  = (mel - mel.mean()) / (mel.std() + 1e-8)
            t      = torch.tensor(mel_n[None, None, :, :], dtype=torch.float32).to(device)
            with torch.no_grad():
                out   = trained_models["cnn"](t)
                proba = torch.softmax(out, dim=1).cpu().numpy()[0]
            predictions["cnn"] = {"pred": int(np.argmax(proba)), "proba": proba.tolist()}

        # LSTM
        if "lstm" in trained_models:
            import torch
            device = next(trained_models["lstm"].parameters()).device
            t      = torch.tensor(seq[None, :, :], dtype=torch.float32).to(device)
            with torch.no_grad():
                out   = trained_models["lstm"](t)
                proba = torch.softmax(out, dim=1).cpu().numpy()[0]
            predictions["lstm"] = {"pred": int(np.argmax(proba)), "proba": proba.tolist()}

    finally:
        os.unlink(audio_path)

    return {"predictions": predictions, "class_names": class_names}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
