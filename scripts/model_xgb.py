"""
model_xgb.py
------------
Train & evaluate XGBoost on MFCC + spectral features.
"""

import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models","xgb_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models","xgb_encoder.pkl")


def build_model(n_classes):
    return XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",       # fast histogram method
    )


def train(X_train, y_train, n_classes):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = build_model(n_classes)
    model.fit(
        X_scaled, y_train,
        eval_set=[(X_scaled, y_train)],
        verbose=False
    )
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
                                          target_names=class_names, output_dict=True),
        "y_pred":  y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }
    return metrics


def run(X, y, class_names, save=True, X_test=None, y_test=None):
    print("\n" + "═" * 50)
    print("  XGBoost — Extreme Gradient Boosting")
    print("═" * 50)

    n_classes = len(class_names)
    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, y_train = X, y

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print("  Training XGBoost (400 estimators, ~2 min)...")

    model, scaler = train(X_train, y_train, n_classes)
    metrics       = evaluate(model, scaler, X_test, y_test, class_names)

    print(f"  ✅ Accuracy : {metrics['accuracy']}%")
    print(f"     F1 Score : {metrics['f1']}%")
    print(f"     Precision: {metrics['precision']}%")
    print(f"     Recall   : {metrics['recall']}%")

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
        print(f"  💾 Saved → {MODEL_PATH}")

    return (model, scaler), metrics


def predict_file(audio_path, bundle=None):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import load_audio, extract_flat_features

    if bundle is None:
        bundle = joblib.load(MODEL_PATH)
    model, scaler = bundle["model"], bundle["scaler"]

    audio  = load_audio(audio_path)
    feat   = extract_flat_features(audio).reshape(1, -1)
    scaled = scaler.transform(feat)
    proba  = model.predict_proba(scaled)[0]
    pred   = int(np.argmax(proba))
    return pred, proba.tolist()
