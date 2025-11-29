Exercise Animation Pipeline — README

Purpose

A small Python pipeline that processes gym exercise videos and produces:
- `pose.json`: per-frame 2D joint positions (15-joint human rig) with confidences
- `equipment.json`: equipment classification + per-frame detections/anchor points
- `overlay_preview.mp4`: original video with skeleton + equipment overlay
- `mocap_preview.mp4`: standalone visualization reconstructable from JSON only

Quick Install (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install core dependencies (CPU-only, works everywhere):

```powershell
pip install -r requirements.txt
```

### Optional: YOLO Equipment Detection (CPU)

```powershell
pip install ultralytics
```

### Optional: GPU Acceleration (NVIDIA CUDA)

For significantly faster YOLO inference on NVIDIA GPUs:

```powershell
# Install ultralytics first
pip install ultralytics

# Replace CPU PyTorch with CUDA version (cu124 for CUDA 12.4)
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **Note**: Use `cu121` or `cu118` if your NVIDIA driver doesn't support CUDA 12.4.  
> Check your CUDA version with `nvidia-smi` (shows "CUDA Version" in top right).  
> See https://pytorch.org/get-started/locally for the exact command for your setup.

Verify GPU is working:
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Running the pipeline

Basic (filename/heuristic-based equipment detection):

```powershell
venv\Scripts\python.exe -m src.main --input_folder in --output_folder out --frame_step 3
```

Enable YOLO-based detection (requires `ultralytics`):

```powershell
venv\Scripts\python.exe -m src.main --input_folder in --output_folder out --use_yolo --yolo_model n
```

Skip preview generation (only export JSON data, much faster):

```powershell
venv\Scripts\python.exe -m src.main --input_folder in --output_folder out --skip-overlay --skip-mocap
```

Flags summary
- `--input_folder` (required): folder containing video files (.mp4, .mov, .avi, .mkv)
- `--output_folder` (required): base output path (per-video folders are created)
- `--frame_step` (default: 1): process every Nth frame
- `--min_detection_confidence` (default: 0.5): MediaPipe detection threshold
- `--min_tracking_confidence` (default: 0.5): MediaPipe tracking threshold
- `--use_yolo`: enable YOLOv8 object detection for equipment localization
- `--yolo_model` (n|s|m|l|x): YOLO model size (n=nano fastest)
- `--yolo_weights`: path to local YOLO weights file (avoids network download)
- `--device` (auto|cuda|cpu): compute device, auto uses GPU if available
- `--workers` (default: 1): parallel workers for multi-video processing
- `--skip-overlay`: skip generating overlay preview video (saves time)
- `--skip-mocap`: skip generating mocap preview video (saves time)
- `--quiet`: suppress verbose progress output

Outputs
For each input video `name.ext`, the pipeline creates `out/name/` containing:
- `pose.json`
- `equipment.json`
- `overlay_preview.mp4`
- `mocap_preview.mp4`

Privacy & Offline Guidance
- Inference is local by default: frames are processed in-process and outputs are written locally.
- The only network activity the pipeline may perform is downloading YOLO model weights on first use when `--use_yolo` is enabled.
- To guarantee no network access:
  - Use `--yolo_weights path/to/yolov8n.pt` to load weights from a local file.
  - Download weights beforehand: `pip install ultralytics && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`

GPU Acceleration
- Use `--device cuda` to force GPU usage (requires CUDA + PyTorch with CUDA support).
- `--device auto` (default) will use GPU if available, otherwise CPU.
- YOLO equipment detection automatically uses the selected device.
- MediaPipe pose estimation uses TensorFlow Lite (CPU by default).

Parallel Processing
- Use `--workers N` to process multiple videos in parallel (each worker initializes its own models).
- Recommended for CPU-only setups: `--workers 4` (adjust based on CPU cores).
- For GPU setups: keep `--workers 1` to avoid GPU memory conflicts, or use multiple GPUs.

Programmatic API
```python
from src import run_pipeline, detect_device

# Auto-detect GPU/CPU
device = detect_device("auto")  # Returns "cuda" or "cpu"

# Run pipeline programmatically
result = run_pipeline(
    input_path="in/squat.mp4",
    output_folder="out",
    frame_step=2,
    use_yolo=True,
    device=device,
    skip_overlay=True,
)
print(f"Processed {result['videos_processed']} videos on {result['device']}")
```

Developer notes
- Pose estimation is implemented using MediaPipe (`src/pose_estimator.py`). Joints are normalized (0-1) in `pose.json`.
- Equipment detection uses filename heuristics by default; YOLO integration is available in `src/equipment_detector.py`.
- Output schema examples are documented in `.github/copilot-instructions.md`.

Quick troubleshooting
- If you see duplicate video processing, ensure filenames are not duplicated with different case — the discovery logic deduplicates duplicates but earlier versions might have run twice.
- If `ultralytics` is slow to install, use model `n` (nano) for quicker downloads and inference.