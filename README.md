# 🎯 Acoustic Threat Detection System
### Multi-Algorithm Audio Classification — 5 ML Models, Real Dataset, Full Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

A complete end-to-end audio threat detection system that classifies environmental sounds into **7 threat categories** using **5 different machine learning algorithms** simultaneously. Built with a FastAPI backend and React frontend, the system allows users to upload audio datasets for training, compare model performance, and run live predictions on new audio files.

This project was developed as a comparative study of classical ML vs deep learning approaches for acoustic threat classification in military/surveillance environments.

---

## 🎯 7 Threat Classes

| Label | Class | Icon | Description |
|-------|-------|------|-------------|
| 0 | Gunshot | 🔫 | Pistol/handgun fire |
| 1 | Rifle Fire | 🎯 | Automatic/semi-auto rifle |
| 2 | Vehicle | 🚛 | Ground vehicle movement |
| 3 | Aircraft | ✈️ | Jets, helicopters, drones |
| 4 | Comms Signal | 📡 | Radio/communication signals |
| 5 | Explosion | 💥 | Blast/explosion sounds |
| 6 | Ambient/Safe | 🔇 | Background/safe noise |

---

## 🤖 5 ML Algorithms — Model Parameters

### 1. Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function)
- **C:** 100
- **Gamma:** scale
- **Class Weight:** balanced
- **Max Iterations:** 2000
- **Input:** 167-dim MFCC flat vector

### 2. Random Forest
- **n_estimators:** 1000 trees
- **Max Features:** sqrt
- **Max Depth:** None (fully grown)
- **Class Weight:** balanced
- **n_jobs:** -1 (all CPU cores)
- **Input:** 167-dim MFCC flat vector

### 3. XGBoost
- **n_estimators:** 800
- **Max Depth:** 8
- **Learning Rate:** 0.05
- **Subsample:** 0.8
- **Colsample by Tree:** 0.8
- **Objective:** multi:softprob
- **Input:** 167-dim MFCC flat vector

### 4. CNN (Convolutional Neural Network)
- **Architecture:** 4 Conv2D blocks → Global Avg Pool → FC layers
- **Epochs:** 30
- **Batch Size:** 32
- **Optimizer:** AdamW + CosineAnnealingLR
- **Learning Rate:** 1e-3
- **Dropout:** 0.2–0.5
- **Input:** Mel Spectrogram (128 × 173)
- **Device:** CUDA GPU / CPU fallback

### 5. BiLSTM + Attention
- **Architecture:** 3-layer BiLSTM + Attention + FC
- **Hidden Size:** 256 (×2 bidirectional)
- **Epochs:** 30
- **Batch Size:** 32
- **Optimizer:** AdamW + ReduceLROnPlateau
- **Learning Rate:** 1e-3
- **Gradient Clipping:** max_norm=1.0
- **Input:** MFCC sequence (128 frames × 40 coefficients)
- **Device:** CUDA GPU / CPU fallback

---

## 📊 Results

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| 🚀 XGBoost | **40.84%** | 40.70% | 40.88% | 40.84% |
| 🌲 Random Forest | 40.31% | 40.15% | 41.13% | 40.31% |
| 🧠 CNN | 39.34% | 38.06% | 38.97% | 39.34% |
| 🔄 BiLSTM | 38.64% | 38.82% | 38.26% | 38.64% |
| ⚡ SVM | 36.47% | 35.90% | 38.51% | 36.47% |

> Tested on held-out test.csv (471 samples). Random baseline = 14.3%. All models ~3x better than random.

---

## ⚠️ Known Limitations & Problems

### 1. Small Dataset — Primary Bottleneck
- Only ~680 samples per class (4,788 total training)
- Industry standard: 2,000–5,000 per class for 80%+ accuracy

### 2. Short Audio Clips (25 Seconds)
- Ambient/Safe achieves highest accuracy because background noise is consistent in short clips
- Gunshot achieves lowest (13–29%) — 25 seconds may contain only 1–2 gunshot events, model can't learn enough pattern
- Aircraft and vehicle sounds need longer clips to capture full frequency sweep
- **Fix planned:** Increase to 60 seconds per clip

### 3. Incomplete Dataset
- train.csv references 6,429 files but only 4,788 WAV files exist
- 1,641 files (~25%) missing due to incomplete download from YouTube sources

### 4. Aircraft vs Comms Signal Confusion
- Both share high-frequency spectral components in MFCC features
- Models frequently confuse these two classes
- MFCC doesn't capture directional/doppler aircraft characteristics

### 5. Rifle Fire Dominance
- Rifle fire has 1,293 samples vs ~773 for others
- Models bias toward rifle fire predictions
- Class weighting partially mitigates this

### 6. Python 3.14 CUDA Incompatibility
- PyTorch CUDA doesn't support Python 3.14
- Workaround: separate `.venv-cuda` with Python 3.11 + CUDA 12.4

---

## 🔮 Future Improvements

### Short Term
- [ ] Increase audio clip duration to 60 seconds
- [ ] Collect minimum 2,000 samples per class
- [ ] Fix missing 1,641 audio files
- [ ] Add data augmentation (pitch shift, time stretch, noise injection)

### Medium Term
- [ ] Integrate pretrained YAMNet or VGGish (expected 70–85% accuracy with existing data)
- [ ] Add chroma + tonnetz features for better aircraft separation
- [ ] Implement ensemble voting across all 5 models
- [ ] Add SpecAugment for CNN/LSTM training

### Long Term
- [ ] Real-time microphone stream detection
- [ ] Edge deployment — quantize for Raspberry Pi
- [ ] Confidence threshold alerting system
- [ ] Multi-label classification (overlapping sounds)

---

## 🏗️ Project Structure

```
threat-detection/
├── backend/
│   └── server.py              ← FastAPI server (port 8001)
├── scripts/
│   ├── feature_extractor.py   ← MFCC, Mel, Sequence extraction
│   ├── model_svm.py           ← SVM classifier
│   ├── model_rf.py            ← Random Forest
│   ├── model_xgb.py           ← XGBoost
│   ├── model_cnn.py           ← CNN (PyTorch)
│   └── model_lstm.py          ← BiLSTM + Attention (PyTorch)
├── threat-ui/                 ← Vite React frontend
│   └── src/App.jsx
├── models/                    ← Auto-created after training
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── cnn_model.pt
│   ├── lstm_model.pt
│   └── metrics.json
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Extraction

```
WAV File → Load (22,050 Hz, mono, 4 sec fixed length)
    ↓
FLAT (SVM/RF/XGB): 40 MFCC + 40 std + 40 delta + 40 delta2 + 7 spectral = 167 dims
MEL  (CNN):        128 mel bins × 173 time frames (dB scale)
SEQ  (LSTM):       128 frames × 40 MFCC coefficients
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11
- Node.js 18+
- NVIDIA GPU with CUDA 12.4+ (optional)
- 10GB+ free disk space

### Step 1 — Clone
```bash
git clone https://github.com/yourusername/threat-detection.git
cd threat-detection
```

### Step 2 — Python Environment

**CPU only:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**With NVIDIA GPU:**
```bash
py -3.11 -m venv .venv-cuda

# Windows
.venv-cuda\Scripts\Activate.ps1

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install fastapi uvicorn python-multipart numpy pandas librosa scikit-learn xgboost joblib scipy soundfile
```

### Step 3 — Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Step 4 — Start Backend
```bash
cd backend
python server.py
# Runs on http://localhost:8001
# API docs: http://localhost:8001/docs
```

### Step 5 — Start Frontend
```bash
cd threat-ui
npm install
npm run dev -- --port 5176
# Opens at http://localhost:5176
```

### Step 6 — Prepare Dataset ZIP

Dataset folder must contain:
```
audio/
├── train.csv     (path, label columns)
├── test.csv      (path, label columns)
└── files/
    ├── training/
    │   └── 001/ *.wav ...
    └── test/
        └── 227/ *.wav ...
```

**Zip it:**
```bash
# Windows
Compress-Archive -Path audio -DestinationPath dataset.zip

# Mac/Linux
zip -r dataset.zip audio/
```

### Step 7 — Train via UI
1. Open `http://localhost:5176`
2. Go to **TRAIN** tab → drop `dataset.zip`
3. Wait ~25–35 min (CPU) or ~10–15 min (GPU)
4. Auto-switches to **COMPARISON** tab

### Step 8 — Predict
1. Go to **PREDICT** tab
2. Drop any `.wav` or `.mp3` file
3. Click **IDENTIFY SOUND CLASS**
4. All 5 models predict simultaneously

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Training progress |
| GET | `/results` | Model metrics |
| GET | `/load` | Load saved models |
| POST | `/train` | Upload ZIP → train |
| POST | `/predict` | Upload WAV → predict |

---

## 🛠️ Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'fastapi'` | Activate venv + `pip install -r requirements.txt` |
| `Port 8001 already in use` | `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess -Force` |
| `No module named 'scripts'` | Run `server.py` from inside `backend/` folder |
| `CUDA not available` | Use Python 3.11 + `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Blank UI | Check F12 console, verify backend on correct port |

---


## 🙏 Acknowledgements

- [librosa](https://librosa.org/) — audio processing
- [PyTorch](https://pytorch.org/) — deep learning
- [scikit-learn](https://scikit-learn.org/) — classical ML
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting
- [FastAPI](https://fastapi.tiangolo.com/) — REST API
- [Vite + React](https://vitejs.dev/) — frontend
