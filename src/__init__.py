"""
Exercise Animation Pipeline

A computer vision pipeline for processing gym exercise videos to extract
motion capture data and equipment detection.

Public API:
    - VideoLoader: Discovers and iterates video files
    - PoseEstimator: MediaPipe-based 2D pose estimation  
    - EquipmentDetector: Equipment classification and localization
    - DataExporter: JSON export for pose and equipment data
    - Visualizer: Preview video generation
    - run_pipeline: High-level function to process videos
    - detect_device: Auto-detect best available device (cuda/cpu)
"""

__version__ = "0.2.0"

from .video_loader import (
    VideoLoader,
    VideoMetadata,
    discover_videos,
    get_video_metadata,
    iterate_frames,
    get_video_basename,
)
from .pose_estimator import (
    PoseEstimator,
    JointPosition,
    JOINT_NAMES,
    SKELETON_CONNECTIONS,
    joints_to_dict,
    get_joint_names,
    get_skeleton_connections,
)
from .equipment_detector import (
    EquipmentDetector,
    EQUIPMENT_TYPES,
    BoundingBox,
    AnchorPoint,
    FrameEquipmentDetection,
    VideoEquipmentResult,
)
from .exporter import DataExporter
from .visualizer import Visualizer


def detect_device(prefer: str = "auto") -> str:
    """
    Detect the best available compute device.
    
    Args:
        prefer: Device preference - "auto", "cuda", "cpu".
                "auto" will try CUDA first, fall back to CPU.
                
    Returns:
        Device string: "cuda" or "cpu"
    """
    if prefer == "cpu":
        return "cpu"
    
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    
    # Check via ultralytics if torch not available directly
    try:
        from ultralytics import YOLO
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    
    if prefer == "cuda":
        print("Warning: CUDA requested but not available. Falling back to CPU.")
    
    return "cpu"


def run_pipeline(
    input_path: str,
    output_folder: str,
    frame_step: int = 1,
    use_yolo: bool = False,
    yolo_model: str = "n",
    yolo_weights: str = None,
    device: str = "auto",
    skip_overlay: bool = False,
    skip_mocap: bool = False,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    High-level function to run the full pipeline on a video or folder.
    
    This is the recommended entry point for programmatic usage.
    
    Args:
        input_path: Path to a video file or folder containing videos.
        output_folder: Base output folder for results.
        frame_step: Process every Nth frame (default: 1 = all frames).
        use_yolo: Enable YOLO-based equipment detection.
        yolo_model: YOLO model size (n/s/m/l/x).
        yolo_weights: Path to local YOLO weights file (optional).
        device: Compute device - "auto", "cuda", or "cpu".
        skip_overlay: Skip generating overlay preview video.
        skip_mocap: Skip generating mocap preview video.
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence: MediaPipe tracking threshold.
        verbose: Print progress information.
        
    Returns:
        Dictionary with processing results:
        {
            "videos_processed": int,
            "videos_failed": int,
            "device": str,
            "results": [{"video": str, "outputs": dict, "error": str|None}, ...]
        }
    """
    from pathlib import Path
    from .main import process_video
    
    # Detect device
    selected_device = detect_device(device)
    if verbose:
        print(f"Using device: {selected_device}")
    
    # Determine if input is a file or folder
    input_path_obj = Path(input_path)
    if input_path_obj.is_file():
        video_paths = [str(input_path_obj)]
    else:
        video_paths = discover_videos(input_path)
    
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {input_path}")
    
    # Initialize components
    pose_estimator = PoseEstimator(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    equipment_detector = EquipmentDetector(
        use_detection_model=use_yolo,
        model_size=yolo_model,
        weights_path=yolo_weights,
        device=selected_device,
    )
    exporter = DataExporter(output_folder)
    visualizer = Visualizer()
    
    results = []
    for video_path in video_paths:
        try:
            output_paths = process_video(
                video_path=video_path,
                output_folder=output_folder,
                pose_estimator=pose_estimator,
                equipment_detector=equipment_detector,
                exporter=exporter,
                visualizer=visualizer,
                frame_step=frame_step,
                verbose=verbose,
                skip_overlay=skip_overlay,
                skip_mocap=skip_mocap,
            )
            results.append({"video": video_path, "outputs": output_paths, "error": None})
        except Exception as e:
            results.append({"video": video_path, "outputs": None, "error": str(e)})
    
    # Cleanup
    pose_estimator.close()
    
    return {
        "videos_processed": sum(1 for r in results if r["error"] is None),
        "videos_failed": sum(1 for r in results if r["error"] is not None),
        "device": selected_device,
        "results": results,
    }


__all__ = [
    # Version
    "__version__",
    # Video loading
    "VideoLoader",
    "VideoMetadata", 
    "discover_videos",
    "get_video_metadata",
    "iterate_frames",
    "get_video_basename",
    # Pose estimation
    "PoseEstimator",
    "JointPosition",
    "JOINT_NAMES",
    "SKELETON_CONNECTIONS",
    "joints_to_dict",
    "get_joint_names",
    "get_skeleton_connections",
    # Equipment detection
    "EquipmentDetector",
    "EQUIPMENT_TYPES",
    "BoundingBox",
    "AnchorPoint",
    "FrameEquipmentDetection",
    "VideoEquipmentResult",
    # Export and visualization
    "DataExporter",
    "Visualizer",
    # High-level API
    "run_pipeline",
    "detect_device",
]
