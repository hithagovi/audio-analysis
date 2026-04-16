"""
model_cnn.py  [v3]
------------------
2-D CNN on mel spectrograms — right-sized for ~2,200 training clips.

Key changes vs v2:
  • Lighter 3-block architecture (512-ch → 256-ch max): far less overfitting
  • SpecAugment: time/frequency masking during training (better generalisation)
  • Bigger batch: 8 → 32 (faster, more stable gradients with more data)
  • Fewer epochs: 150 → 80 (sufficient with proper train/test split)
  • save_path param: saves to models_v3/ instead of models_v2/
  • Fixed inference normalisation (now matches training exactly)
"""

import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

# Default path (v2 compat) — overridden by save_path in run()
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models_v2", "cnn_model.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32     # ↑ from 8 — stable gradients, faster training
EPOCHS     = 80     # ↓ from 150 — sufficient with 5x more data & proper split
LR         = 1e-3


# ─── SPEC AUGMENT ─────────────────────────────────────────────────────────────

def spec_augment(x: torch.Tensor,
                 freq_mask_param: int = 20,
                 time_mask_param: int = 30,
                 num_freq_masks: int  = 2,
                 num_time_masks: int  = 2) -> torch.Tensor:
    """
    Apply SpecAugment to a batch of mel spectrograms.
    x: (B, 1, F, T)
    """
    clone = x.clone()
    _, _, freq_bins, time_steps = clone.shape
    for _ in range(num_freq_masks):
        f  = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, freq_bins - f))
        clone[:, :, f0:f0 + f, :] = 0
    for _ in range(num_time_masks):
        t  = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, time_steps - t))
        clone[:, :, :, t0:t0 + t] = 0
    return clone


# ─── DATASET ─────────────────────────────────────────────────────────────────

class MelDataset(Dataset):
    """Wraps a mel-spectrogram array (N, 128, T) into a Dataset."""
    def __init__(self, X, y):
        # Per-sample normalisation — matches inference normalisation exactly
        mu  = X.mean(axis=(1, 2), keepdims=True)
        std = X.std(axis=(1, 2),  keepdims=True) + 1e-8
        X   = (X - mu) / std
        # Add channel dim -> (N, 1, 128, T)
        self.X = torch.tensor(X[:, np.newaxis, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):          return len(self.y)
    def __getitem__(self, i):   return self.X[i], self.y[i]


# ─── MODEL ───────────────────────────────────────────────────────────────────

class ThreatCNN(nn.Module):
    """
    Lighter 3-block 2-D CNN for mel spectrogram classification.
    Right-sized for ~2,200 unique source clips.
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — low-level edges / onsets
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
            # Block 2 — harmonic / tonal patterns
            nn.Conv2d(64,  128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.20),
            # Block 3 — mid-level spectro-temporal patterns
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(512, 256),          nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── TRAIN / EVAL HELPERS ────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, use_augment=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if use_augment:
            xb = spec_augment(xb)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out  = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


def evaluate(model, loader, class_names):
    model.eval()
    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb    = xb.to(DEVICE)
            out   = model(xb)
            proba = torch.softmax(out, dim=1).cpu().numpy()
            preds = proba.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().tolist())
            all_proba.extend(proba.tolist())

    y_test = np.array(all_labels)
    y_pred = np.array(all_preds)

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
        "y_pred":  all_preds,
        "y_proba": all_proba,
    }
    return metrics


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def run(X, y, class_names, save=True, X_test=None, y_test=None, save_path=None):
    """
    X : np.ndarray shape (N, 128, T) — mel spectrograms from load_dataset(..."mel")
    y : np.ndarray shape (N,)
    save_path: override save location (default = MODEL_PATH for v2 compat)
    """
    effective_path = save_path or MODEL_PATH

    print("\n" + "=" * 50)
    print(f"  2-D CNN  [v3]  [{DEVICE}]")
    print("=" * 50)

    n_classes = len(class_names)

    if X_test is not None and y_test is not None:
        train_ds = MelDataset(X, y)
        test_ds  = MelDataset(X_test, y_test)
        n_train, n_test = len(train_ds), len(test_ds)
    else:
        dataset = MelDataset(X, y)
        n_test  = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_test
        train_ds, test_ds = random_split(
            dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {n_train} | Test: {n_test}")

    model     = ThreatCNN(n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model, test_loader, criterion)
        scheduler.step(epoch - 1)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            if save:
                os.makedirs(os.path.dirname(os.path.abspath(effective_path)), exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "n_classes":   n_classes,
                    "class_names": class_names,
                    "best_val_acc": best_val_acc,
                    "version":     "v3",
                }, effective_path)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"loss {tr_loss:.4f} -> {vl_loss:.4f} | "
                  f"acc {tr_acc*100:.1f}% -> {vl_acc*100:.1f}% | lr={lr_now:.6f}")

    # Load best checkpoint for final evaluation
    if save and os.path.exists(effective_path):
        ckpt = torch.load(effective_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

    metrics = evaluate(model, test_loader, class_names)
    print(f"  Best val acc (checkpoint): {best_val_acc*100:.1f}%")
    print(f"  Final Accuracy : {metrics['accuracy']}%")
    print(f"  Final F1 Score : {metrics['f1']}%")
    print(f"  Saved -> {effective_path}")

    return model, metrics


def predict_file(audio_path, model=None, class_names=None, model_path=None):
    """Predict from a raw audio file path."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import load_audio, extract_mel_spectrogram

    load_from = model_path or MODEL_PATH

    if model is None:
        ckpt        = torch.load(load_from, map_location=DEVICE)
        n_classes   = ckpt["n_classes"]
        class_names = ckpt.get("class_names", [str(i) for i in range(n_classes)])
        model       = ThreatCNN(n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    audio = load_audio(audio_path)
    mel   = extract_mel_spectrogram(audio)           # (128, T)

    # ── Normalise per-sample (2D) — matches MelDataset exactly ──────────────
    mu  = mel.mean()
    std = mel.std() + 1e-8
    mel = (mel - mu) / std

    tensor = torch.tensor(mel[np.newaxis, np.newaxis, :, :],   # (1, 1, 128, T)
                          dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        out   = model(tensor)
        proba = torch.softmax(out, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(proba))
    return pred, proba.tolist()