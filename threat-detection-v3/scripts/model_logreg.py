"""
model_logreg.py
---------------
Train & evaluate a Logistic Regression baseline on MFCC + spectral features.
"""

import numpy as np
import joblib
import os
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "logreg_model.pkl")


def build_model(n_classes: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])


def train(X_train, y_train):
    n_classes = len(np.unique(y_train))
    model = build_model(n_classes)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, class_names):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

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
        "report": classification_report(y_test, y_pred,
                                         target_names=class_names,
                                         labels=list(range(len(class_names))),
                                         zero_division=0,
                                         output_dict=True),
        "y_pred":  y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }
    return metrics


def run(X, y, class_names, save=True, X_test=None, y_test=None):
    print("\n" + "═" * 50)
    print("  Logistic Regression (Baseline)")
    print("═" * 50)

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, y_train = X, y

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print("  Training Logistic Regression...")

    model   = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, class_names)

    print(f"  ✅ Accuracy : {metrics['accuracy']}%")
    print(f"     F1 Score : {metrics['f1']}%")
    print(f"     Precision: {metrics['precision']}%")
    print(f"     Recall   : {metrics['recall']}%")

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"  💾 Saved → {MODEL_PATH}")

    return model, metrics


def predict_file(
    audio_path,
    model=None,
    segments_per_file: int = 1,
    segment_duration: float = 4.0,
    stride: Optional[float] = None,
    offset_mode: str = "start",
    seed: int = 42,
):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import load_audio_segments, extract_flat_features

    if model is None:
        model = joblib.load(MODEL_PATH)

    segments = load_audio_segments(
        audio_path,
        segment_duration=segment_duration,
        segments_per_file=segments_per_file,
        stride=stride,
        offset_mode=offset_mode,
        seed=seed,
    )
    feats = np.stack([extract_flat_features(s) for s in segments], axis=0)
    proba = model.predict_proba(feats).mean(axis=0)
    pred  = int(np.argmax(proba))
    return pred, proba.tolist()
