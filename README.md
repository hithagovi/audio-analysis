# 🎯 Acoustic Threat Detection System
### Multi-Algorithm Audio Classification — 5 Models, Real Dataset Training

---

## 📁 Project Structure

```
threat-detection/
├── backend/
│   └── server.py              ← FastAPI server (main entry point)
├── scripts/
│   ├── feature_extractor.py   ← Shared audio feature extraction
│   ├── model_svm.py           ← Support Vector Machine
│   ├── model_rf.py            ← Random Forest
│   ├── model_xgb.py           ← XGBoost
│   ├── model_cnn.py           ← CNN (PyTorch)
│   └── model_lstm.py          ← BiLSTM + Attention (PyTorch)
├── frontend/
│   └── App.jsx                ← React UI (connect to API at localhost:8000)
├── models/                    ← Saved trained models (auto-created)
└── requirements.txt
```

---

## ⚙️ SETUP — Step by Step

### Step 1: Prerequisites

Make sure you have installed:
- **Python 3.9 or 3.10** (recommended — TensorFlow doesn't support 3.12 yet)
- **Node.js 18+** (for the React frontend)
- **VSCode** with Python extension

Check versions:
```bash
python --version
node --version
```

---

### Step 2: Clone / Open Project in VSCode

```bash
cd threat-detection
code .
```

Open a **new terminal** inside VSCode (`Ctrl + `` ` ``).

---

### Step 3: Create Python Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 4: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏳ This installs PyTorch, TensorFlow, librosa, FastAPI, XGBoost etc.
> Takes 5–10 minutes on first install.

**If you have a GPU (NVIDIA CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**If GPU install fails, CPU-only works fine:**
```bash
pip install torch torchvision torchaudio
```

---

### Step 5: Start the Backend API Server

```bash
# From the threat-detection/ root folder:
cd backend
python server.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ Keep this terminal open — this is your backend.

Test it's working:
```
http://localhost:8000/health     → should return {"status":"ok"}
http://localhost:8000/docs       → Swagger UI for all endpoints
```

---

### Step 6: Setup React Frontend

Open a **second terminal** in VSCode:

```bash
# Option A: Use with Create React App
npx create-react-app threat-ui
cd threat-ui
cp ../frontend/App.jsx src/App.js

# Install extra deps
npm install

# Start the dev server
npm start
```

```bash
# Option B: Use with Vite (faster, recommended)
npm create vite@latest threat-ui -- --template react
cd threat-ui
cp ../frontend/App.jsx src/App.jsx
npm install
npm run dev
```

Frontend will open at: **http://localhost:3000** (CRA) or **http://localhost:5173** (Vite)

---

### Step 7: Prepare Your Dataset

Organize your WAV files like this:
```
my_dataset/
├── gunshot/
│   ├── shot_001.wav
│   ├── shot_002.wav
│   └── ...
├── rifle_fire/
│   ├── rifle_001.wav
│   └── ...
├── vehicle/
│   └── ...
├── aircraft/
│   └── ...
├── comms_signal/
│   └── ...
└── ambient/
    └── ...
```

> 📌 Folder names = class labels. Use any names you want.
> Recommended: ~800–1200 samples per class minimum.

**Zip the dataset folder:**
```bash
# Windows (PowerShell)
Compress-Archive -Path my_dataset -DestinationPath dataset.zip

# Mac / Linux
zip -r dataset.zip my_dataset/
```

---

### Step 8: Train All 5 Models via the UI

1. Open **http://localhost:3000** in your browser
2. Go to the **"TRAIN MODELS"** tab
3. Drop your `dataset.zip` into the upload zone
4. Watch real-time logs as all 5 models train automatically
5. When done, the UI switches to **"COMPARISON"** tab automatically

---

### Step 9: Run Training from Command Line (alternative)

You can also train directly without the UI:

```bash
# From threat-detection/ root
python scripts/feature_extractor.py /path/to/my_dataset    # verify dataset loads

# Train individual models:
python -c "
import sys; sys.path.insert(0,'scripts')
from feature_extractor import load_dataset
from model_svm import run
X, y, names = load_dataset('/path/to/my_dataset', 'flat')
run(X, y, names)
"
```

---

## 🔌 API Endpoints

| Method | Endpoint    | Description                            |
|--------|-------------|----------------------------------------|
| GET    | /health     | Check server is running                |
| GET    | /status     | Training progress + logs               |
| GET    | /results    | All 5 model metrics after training     |
| POST   | /train      | Upload dataset.zip → start training    |
| POST   | /predict    | Upload WAV file → get all predictions  |

Swagger docs: `http://localhost:8000/docs`

---

## 🧠 Model Architecture Summary

| Model        | Input Features         | Architecture                       | Best For                  |
|--------------|------------------------|------------------------------------|---------------------------|
| **SVM**      | MFCC flat (168-dim)    | RBF kernel, C=10                   | Fast baseline              |
| **RF**       | MFCC flat (168-dim)    | 500 trees, sqrt features           | Explainability            |
| **XGBoost**  | MFCC flat (168-dim)    | 400 estimators, depth=6            | Speed + accuracy balance  |
| **CNN**      | Mel spectrogram (2D)   | 4 conv blocks + GAP + FC           | Best accuracy              |
| **BiLSTM**   | MFCC sequence (128×40) | 3-layer BiLSTM + attention + FC    | Temporal patterns          |

---

## 📦 Exported Model Files

After training, models are saved to `models/`:
```
models/
├── svm_model.pkl       ← joblib (sklearn Pipeline)
├── rf_model.pkl        ← joblib (sklearn Pipeline)
├── xgb_model.pkl       ← joblib (XGBClassifier + scaler dict)
├── cnn_model.pt        ← PyTorch state dict
└── lstm_model.pt       ← PyTorch state dict
```

Load and use later:
```python
# SVM / RF
import joblib
model = joblib.load("models/svm_model.pkl")

# CNN / LSTM
import torch
ckpt = torch.load("models/cnn_model.pt")

# Predict a single file
from scripts.model_cnn import predict_file
pred_idx, probas = predict_file("test.wav")
```

---

## 🔧 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: librosa` | Run `pip install -r requirements.txt` with venv active |
| `CORS error in browser` | Make sure backend is running on port 8000 |
| `Port 8000 in use` | `kill -9 $(lsof -t -i:8000)` or change port in server.py |
| `OutOfMemoryError (GPU)` | Reduce BATCH_SIZE in model_cnn.py / model_lstm.py |
| `No module named 'scripts'` | Run server.py from the `backend/` folder, not root |
| `ZIP extraction failed` | Ensure zip has folder-per-class structure directly inside |

---

## 📊 Expected Training Times (6,000 samples, CPU)

| Model   | Time      |
|---------|-----------|
| SVM     | 2–4 min   |
| RF      | 1–2 min   |
| XGBoost | 3–5 min   |
| CNN     | 15–25 min |
| BiLSTM  | 20–30 min |

> With NVIDIA GPU: CNN and LSTM train 5–10× faster.

---

## 🚀 Quick Start Summary

```bash
# Terminal 1 — Backend
cd threat-detection/backend
source ../venv/bin/activate   # (or venv\Scripts\activate on Windows)
python server.py

# Terminal 2 — Frontend
cd threat-detection/threat-ui
npm start
```

Then open `http://localhost:3000` and drop your dataset ZIP!
