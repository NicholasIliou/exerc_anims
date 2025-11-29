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

2. Install core dependencies:

```powershell
pip install -r requirements.txt
```

Optional: To enable YOLO-based equipment detection (more accurate, slower), install `ultralytics`:

```powershell
pip install ultralytics
```

Notes:
- First-time YOLO usage will download pretrained weights (e.g., `yolov8n.pt`) unless you provide local weights.

Running the pipeline

Basic (filename/heuristic-based equipment detection):

```powershell
venv\Scripts\python.exe -m src.main --input_folder in --output_folder out --frame_step 3
```

Enable YOLO-based detection (requires `ultralytics`):

```powershell
venv\Scripts\python.exe -m src.main --input_folder in --output_folder out --use_yolo --yolo_model n
```

Flags summary
- `--input_folder` (required): folder containing video files (.mp4, .mov, .avi, .mkv)
- `--output_folder` (required): base output path (per-video folders are created)
- `--frame_step` (default: 1): process every Nth frame
- `--min_detection_confidence` (default: 0.5): MediaPipe detection threshold
- `--min_tracking_confidence` (default: 0.5): MediaPipe tracking threshold
- `--use_yolo`: enable YOLOv8 object detection for equipment localization
- `--yolo_model` (n|s|m|l|x): YOLO model size (n=nano fastest)
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
  - Install `ultralytics` and download `yolov8*.pt` weights beforehand on another machine and copy them into the project.
  - Run the pipeline offline or modify `src/equipment_detector.py` to load weights from a local path (I can add a `--yolo_weights` flag on request).

Developer notes
- Pose estimation is implemented using MediaPipe (`src/pose_estimator.py`). Joints are normalized (0-1) in `pose.json`.
- Equipment detection uses filename heuristics by default; YOLO integration is available in `src/equipment_detector.py`.
- Output schema examples are documented in `.github/copilot-instructions.md`.

Quick troubleshooting
- If you see duplicate video processing, ensure filenames are not duplicated with different case — the discovery logic deduplicates duplicates but earlier versions might have run twice.
- If `ultralytics` is slow to install, use model `n` (nano) for quicker downloads and inference.