"""
model_cnn.py
------------
Train & evaluate a CNN on mel-spectrogram images using PyTorch.
Architecture: 3x Conv2d blocks → Global Average Pool → Linear classifier
"""

import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models","cnn_model.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
N_MELS     = 128


# ─── DATASET ─────────────────────────────────────────────────────────────────

class MelDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (N, N_MELS, T) — normalize to [-1, 1]
        X_norm = (X - X.mean()) / (X.std() + 1e-8)
        self.X = torch.tensor(X_norm[:, np.newaxis, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─── MODEL ───────────────────────────────────────────────────────────────────

class ThreatCNN(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = N_MELS):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.3),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── TRAIN / EVAL ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
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
            xb = xb.to(DEVICE)
            out = model(xb)
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


def run(X, y, class_names, save=True, X_test=None, y_test=None):
    print("\n" + "═" * 50)
    print(f"  CNN — ThreatCNN  [{DEVICE}]")
    print("═" * 50)

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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {n_train} | Test: {n_test}")
    print(f"  Epochs: {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")

    model     = ThreatCNN(n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0
    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            if save:
                os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "n_classes": n_classes,
                    "class_names": class_names,
                }, MODEL_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"loss {tr_loss:.4f} → {vl_loss:.4f} | "
                  f"acc {tr_acc*100:.1f}% → {vl_acc*100:.1f}%")

    print(f"\n  Best val acc: {best_val_acc*100:.2f}%")

    # Reload best model for final eval
    if save and os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])

    metrics = evaluate(model, test_loader, class_names)
    metrics["history"] = history

    print(f"  ✅ Accuracy : {metrics['accuracy']}%")
    print(f"     F1 Score : {metrics['f1']}%")

    return model, metrics


def predict_file(audio_path, model=None, class_names=None):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from feature_extractor import load_audio, extract_mel_spectrogram

    if model is None:
        ckpt        = torch.load(MODEL_PATH, map_location=DEVICE)
        n_classes   = ckpt["n_classes"]
        class_names = ckpt.get("class_names", [str(i) for i in range(n_classes)])
        model       = ThreatCNN(n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    audio = load_audio(audio_path)
    mel   = extract_mel_spectrogram(audio)
    mel   = (mel - mel.mean()) / (mel.std() + 1e-8)
    tensor = torch.tensor(mel[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        out   = model(tensor)
        proba = torch.softmax(out, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(proba))
    return pred, proba.tolist()
