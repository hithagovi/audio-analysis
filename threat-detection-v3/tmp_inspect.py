import pandas as pd
import os
from collections import Counter

CLASS_MAP = {
    0: 'gunshot', 1: 'rifle_fire', 2: 'vehicle',
    3: 'aircraft', 4: 'comms_signal', 5: 'explosion', 6: 'ambient'
}

# ── TRAIN ─────────────────────────────────────────────────────────────────────
train_df = pd.read_csv('data/audio/train.csv')
print("=" * 55)
print("  TRAIN SET")
print("=" * 55)
print("  Total rows (segments):", len(train_df))

train_df['folder'] = train_df['path'].apply(lambda x: str(x).rsplit('/', 1)[0])
unique_clips = train_df['folder'].nunique()
print("  Unique source clips  :", unique_clips)
print("  Avg segments/clip    :", round(len(train_df) / unique_clips, 1))
print()
print("  Class distribution:")
counts = Counter(train_df['label'].tolist())
for k, v in sorted(counts.items()):
    name = CLASS_MAP.get(k, 'class_' + str(k))
    print("    [" + str(k) + "] " + name.ljust(15) + ": " + str(v).rjust(5) + " segs")

# ── TEST ──────────────────────────────────────────────────────────────────────
test_df = pd.read_csv('data/audio/test.csv')
print()
print("=" * 55)
print("  TEST SET")
print("=" * 55)
print("  Total rows:", len(test_df))
print("  Columns   :", test_df.columns.tolist())

if 'path' in test_df.columns:
    test_df['folder'] = test_df['path'].apply(lambda x: str(x).rsplit('/', 1)[0])
    print("  Unique source clips:", test_df['folder'].nunique())

if 'label' in test_df.columns:
    print()
    print("  Class distribution:")
    counts_t = Counter(test_df['label'].tolist())
    for k, v in sorted(counts_t.items()):
        name = CLASS_MAP.get(k, 'class_' + str(k))
        print("    [" + str(k) + "] " + name.ljust(15) + ": " + str(v).rjust(5) + " segs")
else:
    print()
    print("  (No label column — this is an unlabelled submission test set)")
    print()
    print("  First 5 rows:")
    print(test_df.head().to_string())

# ── FILE EXISTENCE CHECK ─────────────────────────────────────────────────────
print()
print("=" * 55)
print("  FILES ON DISK")
print("=" * 55)
sample_train = os.path.join('data/audio/files', train_df['path'].iloc[0])
print("  Sample train file:", sample_train)
print("  Exists           :", os.path.exists(sample_train))

if 'path' in test_df.columns:
    sample_test = os.path.join('data/audio/files', test_df['path'].iloc[0])
    print("  Sample test file :", sample_test)
    print("  Exists           :", os.path.exists(sample_test))

train_wav_count = 0
train_dir = os.path.join('data', 'audio', 'files', 'training')
for d in os.listdir(train_dir):
    dp = os.path.join(train_dir, d)
    if os.path.isdir(dp):
        train_wav_count += len([f for f in os.listdir(dp) if f.endswith('.wav')])
print()
print("  WAV files in training/:", train_wav_count)

test_dir = os.path.join('data', 'audio', 'files', 'test')
if os.path.exists(test_dir):
    test_wav_count = 0
    for d in os.listdir(test_dir):
        dp = os.path.join(test_dir, d)
        if os.path.isdir(dp):
            test_wav_count += len([f for f in os.listdir(dp) if f.endswith('.wav')])
    print("  WAV files in test/    :", test_wav_count)

try:
    import librosa
    dur = librosa.get_duration(path=sample_train)
    print()
    print("  Sample clip duration:", round(dur, 1), "seconds")
except Exception as e:
    print("  Could not check duration:", e)
