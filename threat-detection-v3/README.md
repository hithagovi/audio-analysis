# Threat Detection v3

## Using 25s / 60s audio files

This project’s models train/predict on **fixed-length windows** (not “whole file”).

- Default window length: `TDV3_CLIP_SECONDS` (defaults to `4.0` seconds)
- To use long source files (25s/60s), extract multiple windows per file:
  - `TDV3_SEGMENTS_PER_FILE` (`1` = first window only, `0` = all windows using stride)
  - `TDV3_SEGMENT_STRIDE_SECONDS` (e.g. `4` for non-overlapping 4s windows)
  - `TDV3_SEGMENT_OFFSET_MODE` (`start`, `linspace`, `random`)

### Examples (PowerShell)

Train using **all non-overlapping 4s windows** from each file:

```powershell
$env:TDV3_CLIP_SECONDS="4"
$env:TDV3_SEGMENTS_PER_FILE="0"
$env:TDV3_SEGMENT_STRIDE_SECONDS="4"
python scripts/train_all.py <dataset_dir>
```

Train using **6 evenly spaced windows** per file:

```powershell
$env:TDV3_CLIP_SECONDS="4"
$env:TDV3_SEGMENTS_PER_FILE="6"
$env:TDV3_SEGMENT_OFFSET_MODE="linspace"
python scripts/train_all.py <dataset_dir>
```

Train using **full 60s windows** (heavier; changes model input sizes):

```powershell
$env:TDV3_CLIP_SECONDS="60"
$env:TDV3_SEGMENTS_PER_FILE="1"
python scripts/train_all.py <dataset_dir>
```

## Backend prediction

`backend/server.py` averages model probabilities across extracted windows.

You can control segmentation via query parameters on `/predict`:

- `segments_per_file`
- `segment_duration`
- `segment_stride`
- `offset_mode`

Or set the same `TDV3_*` environment variables above (the backend falls back to them).

