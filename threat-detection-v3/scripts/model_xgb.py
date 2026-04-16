"""
model_xgb.py  [v2]
------------------
XGBoost — tuned parameters for ≥80% accuracy.

Key changes vs v1:
  • Saves to  models_v2/xgb_model.pkl  (old models untouched)
  • n_estimators   : 1000 → 2000  (more boosting rounds)
  • learning_rate  : 0.03 → 0.01  (slower, more precise learning)
  • max_depth      : 6 → 8        (capture richer audio feature interactions)
  • min_child_weight : 3 → 1      (allow splits on small classes)
  • early_stopping_rounds : 50    (auto-stop when val loss stops improving)
"""

import numpy as np
import joblib
import os
from typing import Optional
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models_v2","xgb_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models_v2","xgb_encoder.pkl")


def build_model(n_classes):
    return XGBClassifier(
        n_estimators=2000,         # ↑ from 1000 — more boosting rounds
        max_depth=8,               # ↑ from 6 — richer feature interactions
        learning_rate=0.01,        # ↓ from 0.03 — slower, more precise
        subsample=0.75,            # slight tweak for regularisation
        colsample_bytree=0.8,
        colsample_bylevel=0.8,     # NEW: column subsampling per level
        min_child_weight=1,        # ↓ from 3 — allows splits on small classes
        gamma=0.05,                # slightly lower — more aggressive splits
        reg_alpha=0.05,
        reg_lambda=1.5,            # ↑ L2 slight increase
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=50,  # NEW: auto-stop when val loss plateaus
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        device="cuda",             # GPU if available
    )


def train(X_train, y_train, n_classes, X_val=None, y_val=None):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = build_model(n_classes)

    # Use a 10% validation split for early stopping if not provided
    if X_val is None:
        from sklearn.model_selection import train_test_split as tts
        X_tr2, X_val2, y_tr2, y_val2 = tts(
            X_scaled, y_train, test_size=0.10, random_state=42, stratify=y_train)
        eval_set = [(X_tr2, y_tr2), (scaler.transform(X_val2) if False else X_val2, y_val2)]
        # Use the full training data for final fit after finding best n_estimators
        model.fit(X_scaled, y_train,
                  eval_set=[(X_val2, y_val2)],
                  verbose=100)
    else:
        X_val_scaled = scaler.transform(X_val)
        model.fit(X_scaled, y_train,
                  eval_set=[(X_val_scaled, y_val)],
                  verbose=100)
    return model, scaler


def evaluate(model, scaler, X_test, y_test, class_names):
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    y_proba  = model.predict_proba(X_scaled)

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1":          round(f1_score(y_test, y_pred, average="weighted") * 100, 2),
        "precision":   round(precision_score(y_test, y_pred, average="weighted",
                                             zero_division=0) * 100, 2),
        "recall":      round(recall_score(y_test, y_pred, average="weighted",
                                          zero_division=0) * 100, 2),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "per_class":   {
            cls: round(
                accuracy_score(y_test[y_test == i], y_pred[y_test == i]) * 100, 2
            )
            for i, cls in enumerate(class_names)
            if len(y_test[y_test == i]) > 0
        },
        "report":  classification_report(y_test, y_pred,
                                          target_names=class_names,
                                          labels=list(range(len(class_names))),
                                          zero_division=0,
                                          output_dict=True),
        "y_pred":  y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }
    return metrics


def run(X, y, class_names, save=True, X_test=None, y_test=None, save_path=None):
    effective_path = save_path or MODEL_PATH

    print("\n" + "=" * 50)
    print("  XGBoost — Extreme Gradient Boosting  [v3]")
    print("=" * 50)

    n_classes = len(class_names)
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, y_train = X, y

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print("  Training XGBoost v3 (up to 2000 rounds, early-stop @50, lr=0.01)...")

    model, scaler = train(X_train, y_train, n_classes)
    metrics       = evaluate(model, scaler, X_test, y_test, class_names)

    print(f"  Final Accuracy : {metrics['accuracy']}%")
    print(f"  Final F1 Score : {metrics['f1']}%")

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(effective_path)), exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler}, effective_path)
        print(f"  Saved -> {effective_path}")

    return {"model": model, "scaler": scaler}, metrics


def predict_file(
    audio_path,
    bundle=None,
    segments_per_file: int = 1,
    segment_duration: float = 4.0,
    stride: Optional[float] = None,
    offset_mode: str = "start",
    seed: int = 42,
):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import load_audio_segments, extract_flat_features

    if bundle is None:
        bundle = joblib.load(MODEL_PATH)
    model, scaler = bundle["model"], bundle["scaler"]

    segments = load_audio_segments(
        audio_path,
        segment_duration=segment_duration,
        segments_per_file=segments_per_file,
        stride=stride,
        offset_mode=offset_mode,
        seed=seed,
    )
    feats  = np.stack([extract_flat_features(s) for s in segments], axis=0)
    scaled = scaler.transform(feats)
    proba  = model.predict_proba(scaled).mean(axis=0)
    pred   = int(np.argmax(proba))
    return pred, proba.tolist()