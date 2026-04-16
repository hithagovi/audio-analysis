"""
model_rf.py  [v2]
-----------------
Random Forest — tuned parameters for ≥80% accuracy.

Key changes vs v1:
  • Saves to  models_v2/rf_model.pkl  (old models untouched)
  • n_estimators   : 1500 → 2000   (more trees = lower variance)
  • max_depth      : None → 40     (prevents extreme overfitting)
  • min_samples_split : 4 → 2      (finer splits for rare dual-sound classes)
  • min_samples_leaf  : 2 → 1      (leaf nodes can be single samples)
  • max_features   : sqrt → log2   (stronger randomization between trees)
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models_v2","rf_model.pkl")


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500,              # faster + still accurate with more data
            max_depth=None,                # fully grown trees (data is larger now)
            min_samples_split=2,
            min_samples_leaf=2,            # slight smoothing
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ))
    ])


def train(X_train, y_train):
    model = build_model()
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
        "report":  classification_report(y_test, y_pred,
                                          target_names=class_names, output_dict=True),
        "y_pred":  y_pred.tolist(),
        "y_proba": y_proba.tolist(),

        # Feature importance (top 20)
        "feature_importances": (
            model.named_steps["rf"].feature_importances_[:20].tolist()
        ),
    }
    return metrics


def run(X, y, class_names, save=True, X_test=None, y_test=None, save_path=None):
    effective_path = save_path or MODEL_PATH

    print("\n" + "=" * 50)
    print("  Random Forest Classifier  [v3]")
    print("=" * 50)

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, y_train = X, y

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print("  Training Random Forest (500 trees)...")

    model   = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test, class_names)

    print(f"  Final Accuracy : {metrics['accuracy']}%")
    print(f"  Final F1 Score : {metrics['f1']}%")

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(effective_path)), exist_ok=True)
        joblib.dump(model, effective_path)
        print(f"  Saved -> {effective_path}")

    return model, metrics


def predict_file(audio_path, model=None):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import load_audio, extract_flat_features

    if model is None:
        model = joblib.load(MODEL_PATH)

    audio = load_audio(audio_path)
    feat  = extract_flat_features(audio).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    pred  = int(np.argmax(proba))
    return pred, proba.tolist()