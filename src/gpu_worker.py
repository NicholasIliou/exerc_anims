"""
GPU worker module for efficient single-GPU multi-video processing.

This module implements a producer-consumer pattern where:
- One GPU worker process owns the YOLO model and handles all GPU inference
- Multiple CPU worker processes handle pose estimation, video I/O, and export
- Communication happens via multiprocessing queues

This avoids GPU memory conflicts when processing multiple videos in parallel.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import traceback

import numpy as np


@dataclass
class DetectionRequest:
    """Request for GPU-based equipment detection."""
    video_id: str
    frame_index: int
    frame: np.ndarray  # BGR frame
    equipment_type: str


@dataclass  
class DetectionResponse:
    """Response from GPU worker with detection results."""
    video_id: str
    frame_index: int
    bboxes: List[Dict]
    anchor_points: List[Dict]
    error: Optional[str] = None


@dataclass
class ClassificationRequest:
    """Request for equipment classification from sample frames."""
    video_id: str
    video_path: str
    sample_frames: List[np.ndarray]


@dataclass
class ClassificationResponse:
    """Response with equipment classification."""
    video_id: str
    equipment_type: str
    confidence: float
    error: Optional[str] = None


# Sentinel to signal worker shutdown
SHUTDOWN_SENTINEL = "SHUTDOWN"


def gpu_worker_process(
    request_queue: Queue,
    response_queue: Queue,
    model_size: str,
    weights_path: Optional[str],
    device: str,
):
    """
    GPU worker process that handles all YOLO inference.
    
    This process owns the GPU model and processes detection/classification
    requests from CPU workers via queues.
    
    Args:
        request_queue: Queue to receive detection/classification requests.
        response_queue: Queue to send responses back.
        model_size: YOLO model size (n/s/m/l/x).
        weights_path: Optional path to local weights.
        device: Device to use (cuda/cpu).
    """
    from .equipment_detector import EquipmentDetector, FrameEquipmentDetection
    
    # Initialize detector on GPU
    detector = EquipmentDetector(
        use_detection_model=True,
        model_size=model_size,
        weights_path=weights_path,
        device=device,
    )
    
    print(f"[GPU Worker] Initialized YOLO on {device}")
    
    try:
        while True:
            request = request_queue.get()
            
            if request == SHUTDOWN_SENTINEL:
                print("[GPU Worker] Received shutdown signal")
                break
            
            try:
                if isinstance(request, ClassificationRequest):
                    # Classify equipment from sample frames
                    equipment_type = detector._classify_from_frames(request.sample_frames)
                    response = ClassificationResponse(
                        video_id=request.video_id,
                        equipment_type=equipment_type,
                        confidence=0.8 if equipment_type != "unknown" else 0.0,
                    )
                    
                elif isinstance(request, DetectionRequest):
                    # Detect equipment in single frame
                    detection = detector.detect_equipment_in_frame(
                        request.frame,
                        request.frame_index,
                        request.equipment_type,
                    )
                    response = DetectionResponse(
                        video_id=request.video_id,
                        frame_index=request.frame_index,
                        bboxes=[box.to_dict() for box in detection.bboxes],
                        anchor_points=[ap.to_dict() for ap in detection.anchor_points],
                    )
                else:
                    # Unknown request type
                    response = DetectionResponse(
                        video_id=getattr(request, 'video_id', 'unknown'),
                        frame_index=getattr(request, 'frame_index', -1),
                        bboxes=[],
                        anchor_points=[],
                        error=f"Unknown request type: {type(request)}",
                    )
                    
            except Exception as e:
                # Return error response
                if isinstance(request, ClassificationRequest):
                    response = ClassificationResponse(
                        video_id=request.video_id,
                        equipment_type="unknown",
                        confidence=0.0,
                        error=str(e),
                    )
                else:
                    response = DetectionResponse(
                        video_id=request.video_id,
                        frame_index=request.frame_index,
                        bboxes=[],
                        anchor_points=[],
                        error=str(e),
                    )
            
            response_queue.put(response)
            
    except Exception as e:
        print(f"[GPU Worker] Fatal error: {e}")
        traceback.print_exc()
    finally:
        print("[GPU Worker] Shutting down")


class GPUWorkerPool:
    """
    Manages a single GPU worker process and provides an interface for 
    submitting detection requests from multiple CPU workers.
    
    Usage:
        with GPUWorkerPool(model_size="n", device="cuda") as pool:
            # Submit classification request
            pool.submit_classification("video1", "path/to/video.mp4", sample_frames)
            
            # Submit detection request  
            pool.submit_detection("video1", 0, frame, "barbell")
            
            # Get responses
            response = pool.get_response()
    """
    
    def __init__(
        self,
        model_size: str = "n",
        weights_path: Optional[str] = None,
        device: str = "cuda",
        max_queue_size: int = 100,
    ):
        self.model_size = model_size
        self.weights_path = weights_path
        self.device = device
        
        # Create queues for communication
        self.request_queue: Queue = mp.Queue(maxsize=max_queue_size)
        self.response_queue: Queue = mp.Queue(maxsize=max_queue_size)
        
        self._worker_process: Optional[Process] = None
        self._started = False
    
    def start(self):
        """Start the GPU worker process."""
        if self._started:
            return
        
        self._worker_process = Process(
            target=gpu_worker_process,
            args=(
                self.request_queue,
                self.response_queue,
                self.model_size,
                self.weights_path,
                self.device,
            ),
            daemon=True,
        )
        self._worker_process.start()
        self._started = True
    
    def stop(self):
        """Stop the GPU worker process."""
        if not self._started:
            return
        
        # Send shutdown signal
        self.request_queue.put(SHUTDOWN_SENTINEL)
        
        # Wait for process to finish
        if self._worker_process is not None:
            self._worker_process.join(timeout=10)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
        
        self._started = False
    
    def submit_classification(
        self,
        video_id: str,
        video_path: str,
        sample_frames: List[np.ndarray],
    ):
        """Submit a classification request."""
        request = ClassificationRequest(
            video_id=video_id,
            video_path=video_path,
            sample_frames=sample_frames,
        )
        self.request_queue.put(request)
    
    def submit_detection(
        self,
        video_id: str,
        frame_index: int,
        frame: np.ndarray,
        equipment_type: str,
    ):
        """Submit a detection request."""
        request = DetectionRequest(
            video_id=video_id,
            frame_index=frame_index,
            frame=frame,
            equipment_type=equipment_type,
        )
        self.request_queue.put(request)
    
    def get_response(self, timeout: Optional[float] = None):
        """Get a response from the GPU worker."""
        return self.response_queue.get(timeout=timeout)
    
    def get_response_nowait(self):
        """Get a response without blocking, or None if queue is empty."""
        try:
            return self.response_queue.get_nowait()
        except:
            return None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def process_video_with_gpu_queue(
    video_path: str,
    output_folder: str,
    gpu_pool: GPUWorkerPool,
    frame_step: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    skip_overlay: bool = False,
    skip_mocap: bool = False,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Process a single video using the GPU worker pool for detection.
    
    Pose estimation (MediaPipe) runs in this process (CPU).
    Equipment detection requests are sent to the GPU worker pool.
    
    Args:
        video_path: Path to video file.
        output_folder: Base output folder.
        gpu_pool: Shared GPU worker pool.
        frame_step: Frame downsampling factor.
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence: MediaPipe tracking threshold.
        skip_overlay: Skip overlay video generation.
        skip_mocap: Skip mocap video generation.
        verbose: Print progress.
        
    Returns:
        Dictionary of output file paths.
    """
    from .video_loader import get_video_metadata, iterate_frames, get_video_basename
    from .pose_estimator import PoseEstimator, joints_to_dict
    from .equipment_detector import (
        EquipmentDetector, VideoEquipmentResult, FrameEquipmentDetection,
        BoundingBox, AnchorPoint
    )
    from .exporter import DataExporter
    from .visualizer import Visualizer
    from tqdm import tqdm
    
    video_id = get_video_basename(video_path)
    
    if verbose:
        print(f"\nProcessing: {video_id}")
    
    # Get metadata
    metadata = get_video_metadata(video_path, frame_step)
    metadata_dict = metadata.to_dict()
    
    # Setup output
    exporter = DataExporter(output_folder)
    video_output_folder = exporter.prepare_output(video_id)
    output_paths = exporter.get_paths(video_output_folder)
    
    if verbose:
        print(f"  Video: {metadata.width}x{metadata.height} @ {metadata.fps:.1f} FPS")
    
    # Initialize pose estimator (CPU)
    pose_estimator = PoseEstimator(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    # Collect frames and poses
    pose_frames = []
    frames_with_poses = []
    sample_frames_for_classification = []
    
    total_frames = max(1, (metadata.total_frames + frame_step - 1) // frame_step)
    frame_iter = iterate_frames(video_path, frame_step)
    
    for frame_index, frame in tqdm(frame_iter, total=total_frames, 
                                    desc=f"Pose {video_id}", disable=not verbose):
        # Pose estimation (CPU)
        joints = pose_estimator.get_pose_for_frame(frame)
        joints_dict = joints_to_dict(joints)
        
        pose_frames.append({
            "frame_index": frame_index,
            "joints": joints_dict,
        })
        frames_with_poses.append((frame_index, frame, joints_dict))
        
        # Collect sample frames for classification
        if len(sample_frames_for_classification) < 10:
            sample_frames_for_classification.append(frame.copy())
    
    pose_estimator.close()
    
    if verbose:
        print(f"  Pose extraction complete: {len(pose_frames)} frames")
        print("  Requesting equipment classification (GPU)...")
    
    # Request classification from GPU worker
    gpu_pool.submit_classification(video_id, video_path, sample_frames_for_classification)
    
    # Wait for classification response
    classification_response = None
    while classification_response is None:
        response = gpu_pool.get_response(timeout=60)
        if isinstance(response, ClassificationResponse) and response.video_id == video_id:
            classification_response = response
            break
    
    equipment_type = classification_response.equipment_type
    if verbose:
        print(f"  Equipment type: {equipment_type}")
    
    # Request per-frame detection from GPU worker (if YOLO is being used)
    if verbose:
        print("  Requesting equipment detection (GPU)...")
    
    # Submit all detection requests
    for frame_index, frame, joints_dict in frames_with_poses:
        gpu_pool.submit_detection(video_id, frame_index, frame, equipment_type)
    
    # Collect detection responses
    detections = []
    detection_responses = {}
    
    pending = len(frames_with_poses)
    while pending > 0:
        response = gpu_pool.get_response(timeout=60)
        if isinstance(response, DetectionResponse) and response.video_id == video_id:
            detection_responses[response.frame_index] = response
            pending -= 1
    
    # Build equipment result
    from .equipment_detector import compute_anchor_points_from_hands_static
    
    for frame_index, frame, joints_dict in frames_with_poses:
        resp = detection_responses.get(frame_index)
        
        bboxes = []
        if resp and resp.bboxes:
            for box_dict in resp.bboxes:
                bboxes.append(BoundingBox(
                    x_min=box_dict['x_min'],
                    y_min=box_dict['y_min'],
                    x_max=box_dict['x_max'],
                    y_max=box_dict['y_max'],
                    confidence=box_dict['confidence'],
                    label=box_dict['label'],
                ))
        
        # Compute anchor points from hand positions
        anchor_points = compute_anchor_points_from_hands_static(
            joints_dict, equipment_type, frame_index
        )
        
        detection = FrameEquipmentDetection(
            frame_index=frame_index,
            bboxes=bboxes,
            anchor_points=anchor_points,
        )
        detections.append(detection)
    
    # Build final equipment result
    equipment_result = VideoEquipmentResult(
        equipment_type=equipment_type,
        confidence=classification_response.confidence,
        classifier_method="yolo_gpu_queue",
        detections=detections,
    )
    equipment_dict = equipment_result.to_dict()
    
    if verbose:
        print("  Exporting data...")
    
    # Export
    exporter.export_pose(video_output_folder, metadata_dict, pose_frames)
    exporter.export_equipment(video_output_folder, equipment_dict, metadata_dict)
    
    # Visualizations
    visualizer = Visualizer()
    pose_data = {"metadata": metadata_dict, "frames": pose_frames}
    
    if not skip_overlay:
        if verbose:
            print("  Creating overlay preview...")
        visualizer.create_overlay_video(
            video_path, pose_data, equipment_dict,
            output_paths["overlay_video"], frame_step, show_progress=verbose
        )
    
    if not skip_mocap:
        if verbose:
            print("  Creating mocap preview...")
        visualizer.create_mocap_video(
            pose_data, equipment_dict,
            output_paths["mocap_video"], show_progress=verbose
        )
    
    if verbose:
        print(f"  Output folder: {video_output_folder}")
    
    return output_paths
