# Copilot Instructions for Exercise Animation Project

## Project Overview

Python-based computer vision pipeline that processes gym exercise videos to extract motion capture data and equipment detection, outputting structured data for future 3D rigging.

## Architecture

```
src/
  __init__.py           # Package init
  video_loader.py       # VideoLoader class, iterate_frames(), get_video_metadata()
  pose_estimator.py     # PoseEstimator class wrapping MediaPipe, JOINT_NAMES, SKELETON_CONNECTIONS
  equipment_detector.py # EquipmentDetector with classify_equipment_for_video() + anchor point detection
  exporter.py           # DataExporter writes pose.json and equipment.json with consistent schema
  visualizer.py         # Visualizer creates overlay_preview.mp4 and mocap_preview.mp4
  main.py               # CLI entry: python -m src.main --input_folder X --output_folder Y --frame_step N
```

## Output Structure (per video)

For `squat_001.mp4` → creates `output/squat_001/`:
- `pose.json` - Frame-by-frame joint positions with metadata (FPS, resolution, downsampling)
- `equipment.json` - Equipment type + optional per-frame bounding boxes/anchor points
- `overlay_preview.mp4` - Original video with skeleton + equipment boxes overlay
- `mocap_preview.mp4` - Standalone visualization (solid background, reconstructable from JSON alone)

## Human Rig Joints (15 joints)

`head`, `neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `spine`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`

## Equipment Types (string labels)

`barbell`, `dumbbell`, `kettlebell`, `machine`, `bodyweight`, `unknown`

Reference: `exercises.txt` contains 38 indexed exercises with equipment in parentheses.

## Key Design Decisions

1. **Separation of Concerns**: Human rig data and equipment rig data stored separately for flexible 3D attachment later
2. **Graceful Failure**: Missing poses → null coordinates with zero confidence; unknown equipment → `"unknown"` label
3. **Schema Consistency**: Always include required fields even on detection failure
4. **Reproducibility**: Output paths deterministic from video base name
5. **Anchor Points**: Equipment grip positions derived from wrist joint positions for 3D model attachment

## Data Schema Patterns

```python
# pose.json structure
{
  "metadata": {"filename": str, "fps": float, "width": int, "height": int, "frame_step": int, "total_frames": int},
  "joints_schema": {"names": [str], "indices": {str: int}},
  "frames": [{"frame_index": int, "joints": {"joint_name": {"x": float|null, "y": float|null, "confidence": float}}}]
}

# equipment.json structure  
{
  "equipment_type": str,  # one of the 6 labels
  "confidence": float,
  "metadata": {"filename": str, "classifier": str, "equipment_types": [str]},
  "detections": [{"frame_index": int, "bboxes": [...], "anchor_points": [{"x": float, "y": float, "name": str, "confidence": float}]}]
}
```

## Dependencies

- `opencv-python>=4.8.0` - Video I/O and drawing
- `mediapipe>=0.10.0` - Pose estimation
- `numpy>=1.24.0` - Array operations

## CLI Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Basic run (filename-based equipment detection)
python -m src.main --input_folder path/to/videos --output_folder path/to/output --frame_step 3

# With YOLO equipment detection (install: pip install ultralytics)
python -m src.main --input_folder path/to/videos --output_folder path/to/output --use_yolo --yolo_model n

# Skip preview generation to save time (only export JSON data)
python -m src.main --input_folder path/to/videos --output_folder path/to/output --skip-overlay --skip-mocap

# Options
#   --frame_step N                  Process every Nth frame (default: 1)
#   --min_detection_confidence F    Pose detection threshold (default: 0.5)
#   --min_tracking_confidence F     Pose tracking threshold (default: 0.5)
#   --use_yolo                      Enable YOLO-based equipment detection
#   --yolo_model {n,s,m,l,x}        YOLO model size (n=nano/fastest, x=xlarge/best)
#   --skip-overlay                  Skip generating overlay preview video
#   --skip-mocap                    Skip generating mocap preview video
#   --quiet                         Suppress progress output
```