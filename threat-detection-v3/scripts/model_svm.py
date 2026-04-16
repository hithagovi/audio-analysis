"""
model_svm.py  [v2]
------------------
SVM — tuned for ≥80% accuracy on audio threat detection.

Key changes vs v1:
  • Saves to  models_v2/svm_model.pkl  (old models untouched)
  • kernel : default → RBF    (handles non-linear class boundaries)
  • C      : default → 50     (higher margin penalty, less slack)
  • gamma  : default → scale  (auto-scales with feature variance)
  • class_weight : None → balanced  (handles class imbalance)
"""

import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models_v2","svm_model.pkl")


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",          # RBF handles non-linear feature spaces
            C=50,                  # ↑ high penalty → tighter margin fit
            gamma="scale",         # auto-scales with feature std
            class_weight="balanced",
            probability=True,      # enables predict_proba for scoring
            cache_size=2000,       # larger cache → faster training
            random_state=42
        ))
    ])


def train(X_train, y_train):
    print("  Training SVM (RBF kernel, C=50) — this may take a few minutes...")
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
    }
    return metrics


def run(X, y, class_names, save=True, X_test=None, y_test=None, save_path=None):
    effective_path = save_path or MODEL_PATH

    print("\n" + "=" * 50)
    print("  SVM — RBF Kernel Classifier  [v3]")
    print("=" * 50)

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, y_train = X, y

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

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
    proba = model.predict_proba(feat)[0]  # SVC(probability=True) required
    pred  = int(np.argmax(proba))
    return pred, proba.tolist()