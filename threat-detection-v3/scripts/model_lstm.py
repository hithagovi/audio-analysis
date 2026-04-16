"""
model_lstm.py  [v3]
-------------------
BiLSTM + Attention — right-sized for ~2,200 training clips.

Key changes vs v2:
  • Hidden size: 384 -> 128  (prevents overfitting on small dataset)
  • LSTM layers: 3 -> 2
  • Batch size: 8 -> 64       (faster + more stable)
  • Epochs: 150 -> 80         (sufficient with proper split)
  • save_path param: saves to models_v3/
  • Mixup augmentation during training (better inter-class boundaries)
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

# Default path (v2 compat) — overridden by save_path in run()
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models_v2", "lstm_model.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64    # ↑ from 8
EPOCHS      = 80    # ↓ from 150
LR          = 1e-3
N_FRAMES    = 128
N_MFCC      = 40
HIDDEN_SIZE = 128   # ↓ from 384 — right-sized for dataset


# ─── MIXUP HELPER ────────────────────────────────────────────────────────────

def mixup_batch(x, y, n_classes, alpha=0.3):
    """Apply Mixup augmentation to a batch."""
    if alpha <= 0:
        return x, torch.zeros(len(y), n_classes, device=x.device).scatter_(1, y.unsqueeze(1), 1)
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    # One-hot soft labels
    y_oh  = torch.zeros(len(y), n_classes, device=x.device).scatter_(1, y.unsqueeze(1), 1)
    y_mix = lam * y_oh + (1 - lam) * y_oh[idx]
    return x_mix, y_mix


# ─── DATASET ─────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, X, y):
        # Per-dataset normalisation (mean/std across all samples and time)
        mu  = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        X   = (X - mu) / std
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.y   = torch.tensor(y, dtype=torch.long)
        self.mu  = mu
        self.std = std

    def __len__(self):            return len(self.y)
    def __getitem__(self, i):     return self.X[i], self.y[i]


# ─── MODEL ───────────────────────────────────────────────────────────────────

class ThreatLSTM(nn.Module):
    """
    Bidirectional LSTM with additive attention.
    Right-sized (hidden=128, 2 layers) for ~2,200 unique clips.
    """
    def __init__(self, n_classes: int,
                 input_size: int  = N_MFCC,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int  = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.30,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256), nn.GELU(), nn.Dropout(0.40),
            nn.Linear(256, 128),             nn.GELU(), nn.Dropout(0.30),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                               # (B, T, H*2)
        attn   = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
        ctx    = (attn * out).sum(dim=1)                    # (B, H*2)
        return self.classifier(ctx)


# ─── TRAIN / EVAL ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, n_classes, use_mixup=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if use_mixup:
            xb, yb_soft = mixup_batch(xb, yb, n_classes, alpha=0.3)
            optimizer.zero_grad()
            out  = model(xb)
            log_prob = torch.log_softmax(out, dim=-1)
            loss = -(yb_soft * log_prob).sum(dim=-1).mean()
            preds = out.argmax(1)
            hard_y = yb_soft.argmax(1)
        else:
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            preds = out.argmax(1)
            hard_y = yb
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(hard_y)
        correct    += (preds == hard_y).sum().item()
        total      += len(hard_y)
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
    X : np.ndarray shape (N, 128, 40) — MFCC sequences
    save_path: override save location (default = MODEL_PATH for v2 compat)
    """
    effective_path = save_path or MODEL_PATH

    print("\n" + "=" * 50)
    print(f"  BiLSTM + Attention  [v3]  [{DEVICE}]")
    print("=" * 50)

    n_classes = len(class_names)

    if X_test is not None and y_test is not None:
        train_ds = SeqDataset(X, y)
        test_ds  = SeqDataset(X_test, y_test)
        n_train, n_test = len(train_ds), len(test_ds)
    else:
        dataset = SeqDataset(X, y)
        n_test  = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_test
        train_ds, test_ds = random_split(
            dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {n_train} | Test: {n_test}")

    model     = ThreatLSTM(n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, n_classes)
        vl_loss, vl_acc = eval_epoch(model, test_loader, criterion)
        scheduler.step(epoch - 1)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            if save:
                os.makedirs(os.path.dirname(os.path.abspath(effective_path)), exist_ok=True)
                torch.save({
                    "model_state":  model.state_dict(),
                    "n_classes":    n_classes,
                    "class_names":  class_names,
                    "best_val_acc": best_val_acc,
                    "hidden_size":  HIDDEN_SIZE,
                    "version":      "v3",
                }, effective_path)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"loss {tr_loss:.4f} -> {vl_loss:.4f} | "
                  f"acc {tr_acc*100:.1f}% -> {vl_acc*100:.1f}% | lr={lr_now:.6f}")

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
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import load_audio, extract_sequence_features

    load_from = model_path or MODEL_PATH

    if model is None:
        ckpt        = torch.load(load_from, map_location=DEVICE)
        n_classes   = ckpt["n_classes"]
        class_names = ckpt.get("class_names", [str(i) for i in range(n_classes)])
        hidden_size = ckpt.get("hidden_size", HIDDEN_SIZE)
        model       = ThreatLSTM(n_classes, hidden_size=hidden_size).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    audio  = load_audio(audio_path)
    seq    = extract_sequence_features(audio)  # (128, 40)
    tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        out   = model(tensor)
        proba = torch.softmax(out, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(proba))
    return pred, proba.tolist()