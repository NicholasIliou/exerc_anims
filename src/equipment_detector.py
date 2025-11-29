"""
Equipment detection module for gym exercise videos.

Classifies equipment type and optionally detects/localizes equipment in frames.
Uses a heuristic-based approach with optional YOLO integration for detection.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

# Equipment type labels
EQUIPMENT_TYPES = [
    "barbell",
    "dumbbell", 
    "kettlebell",
    "machine",
    "bodyweight",
    "unknown"
]

# COCO class names that indicate gym equipment
# Map COCO classes to our equipment types
COCO_TO_EQUIPMENT = {
    # Sports equipment that could be gym weights
    "sports ball": "dumbbell",  # Sometimes detected as sports equipment
    "bottle": "dumbbell",       # Water bottles sometimes misclassified
    "cup": "dumbbell",
    # People doing exercises
    "person": "bodyweight",
}

# Keywords in YOLO detections that suggest equipment types
DETECTION_KEYWORDS = {
    "barbell": ["bar", "barbell", "weight"],
    "dumbbell": ["dumbbell", "weight", "db"],
    "kettlebell": ["kettlebell", "kb"],
    "machine": ["machine", "equipment", "bench"],
}


@dataclass
class BoundingBox:
    """Bounding box for detected equipment."""
    x_min: float  # Normalized coordinates (0-1)
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 0.0
    label: str = "equipment"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "confidence": self.confidence,
            "label": self.label
        }


@dataclass
class AnchorPoint:
    """
    Anchor point for equipment-human interaction.
    
    Represents positions where hands contact equipment (grip positions).
    """
    x: float  # Normalized coordinates (0-1)
    y: float
    name: str = "grip"  # e.g., "left_grip", "right_grip", "handle"
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "confidence": self.confidence
        }


@dataclass
class FrameEquipmentDetection:
    """Equipment detection results for a single frame."""
    frame_index: int
    bboxes: List[BoundingBox] = field(default_factory=list)
    anchor_points: List[AnchorPoint] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_index": self.frame_index,
            "bboxes": [box.to_dict() for box in self.bboxes],
            "anchor_points": [ap.to_dict() for ap in self.anchor_points]
        }


@dataclass  
class VideoEquipmentResult:
    """Complete equipment detection result for a video."""
    equipment_type: str
    confidence: float
    classifier_method: str
    detections: List[FrameEquipmentDetection] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "equipment_type": self.equipment_type,
            "confidence": self.confidence,
            "metadata": {
                "classifier": self.classifier_method
            },
            "detections": [det.to_dict() for det in self.detections]
        }


class EquipmentDetector:
    """
    Equipment detector for gym exercise videos.
    
    Uses a combination of:
    1. Filename heuristics (if video name contains equipment keywords)
    2. Simple image analysis (color/shape patterns)
    3. Optional: YOLO-based object detection
    
    Example:
        detector = EquipmentDetector()
        result = detector.classify_equipment_for_video(video_path, frames)
        # result.equipment_type is one of EQUIPMENT_TYPES
    """
    
    def __init__(self, use_detection_model: bool = False, model_size: str = "n"):
        """
        Initialize the equipment detector.
        
        Args:
            use_detection_model: Whether to use YOLO object detection for localization.
                                If False, only classification is performed.
            model_size: YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge).
                       Nano is fastest but less accurate.
        """
        self.use_detection_model = use_detection_model
        self._detection_model = None
        
        if use_detection_model:
            if not YOLO_AVAILABLE:
                print("Warning: ultralytics not available. Install with: pip install ultralytics")
                print("Falling back to filename-only detection.")
                self.use_detection_model = False
            else:
                try:
                    # Load YOLOv8 model (will download on first use)
                    self._detection_model = YOLO(f"yolov8{model_size}.pt")
                    print(f"Loaded YOLOv8{model_size} model for equipment detection")
                except Exception as e:
                    print(f"Warning: Failed to load YOLO model: {e}")
                    print("Falling back to filename-only detection.")
                    self.use_detection_model = False
        
        # Keywords for filename-based classification
        self._equipment_keywords = {
            "barbell": ["barbell", "bb", "bar", "squat"],
            "dumbbell": ["dumbbell", "db", "dumbell"],
            "kettlebell": ["kettlebell", "kb", "kettle"],
            "machine": ["machine", "cable", "pulldown", "press machine", "leg press", 
                       "lat pulldown", "seated", "rowing machine", "treadmill"],
            "bodyweight": ["bodyweight", "bw", "pullup", "pull up", "pushup", 
                          "push up", "dip", "plank", "hiking"]
        }
    
    def classify_equipment_for_video(
        self, 
        video_path: str,
        sample_frames: Optional[List[np.ndarray]] = None
    ) -> str:
        """
        Classify the equipment type for an entire video.
        
        Uses multiple signals:
        1. Filename keywords
        2. Frame analysis (if sample_frames provided)
        
        Args:
            video_path: Path to the video file.
            sample_frames: Optional list of sample frames for analysis.
            
        Returns:
            Equipment type string (one of EQUIPMENT_TYPES).
        """
        # First, try filename-based classification
        filename = Path(video_path).stem.lower()
        
        for equipment_type, keywords in self._equipment_keywords.items():
            for keyword in keywords:
                if keyword in filename:
                    return equipment_type
        
        # If sample frames provided, try visual analysis
        if sample_frames and len(sample_frames) > 0:
            return self._classify_from_frames(sample_frames)
        
        return "unknown"
    
    def _classify_from_frames(self, frames: List[np.ndarray]) -> str:
        """
        Classify equipment type from visual frame analysis using YOLO.
        
        Args:
            frames: List of BGR frames to analyze.
            
        Returns:
            Equipment type string.
        """
        if not self.use_detection_model or self._detection_model is None:
            return "unknown"
        
        equipment_votes = {eq_type: 0 for eq_type in EQUIPMENT_TYPES}
        
        # Sample frames for classification (max 10 to avoid slowdown)
        sample_frames = frames[:min(10, len(frames))]
        
        for frame in sample_frames:
            # Run YOLO detection
            results = self._detection_model(frame, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Check detected classes
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = self._detection_model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Map COCO class to equipment type
                    if class_name in COCO_TO_EQUIPMENT:
                        equipment_votes[COCO_TO_EQUIPMENT[class_name]] += confidence
                    
                    # Check for keyword matches in class name
                    for eq_type, keywords in DETECTION_KEYWORDS.items():
                        if any(kw in class_name.lower() for kw in keywords):
                            equipment_votes[eq_type] += confidence
        
        # If person detected but no equipment, likely bodyweight
        if equipment_votes["bodyweight"] > 0 and sum(equipment_votes.values()) == equipment_votes["bodyweight"]:
            return "bodyweight"
        
        # Return equipment type with highest votes (excluding unknown)
        equipment_votes.pop("unknown", None)
        if equipment_votes:
            best_type = max(equipment_votes.items(), key=lambda x: x[1])
            if best_type[1] > 0.5:  # Require some confidence
                return best_type[0]
        
        return "unknown"
    
    def detect_equipment_in_frame(
        self, 
        frame: np.ndarray,
        frame_index: int,
        equipment_type: str = "unknown"
    ) -> FrameEquipmentDetection:
        """
        Detect and localize equipment in a single frame using YOLO.
        
        Args:
            frame: BGR image as numpy array.
            frame_index: Index of the frame in the video.
            equipment_type: Known equipment type for context.
            
        Returns:
            FrameEquipmentDetection with bounding boxes and anchor points.
        """
        detection = FrameEquipmentDetection(frame_index=frame_index)
        
        if not self.use_detection_model or self._detection_model is None:
            return detection
        
        # Run YOLO detection on this frame
        results = self._detection_model(frame, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            h, w = frame.shape[:2]
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self._detection_model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get normalized bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1_norm = float(x1 / w)
                y1_norm = float(y1 / h)
                x2_norm = float(x2 / w)
                y2_norm = float(y2 / h)
                
                # Only add boxes for relevant objects
                if class_name in COCO_TO_EQUIPMENT or any(
                    any(kw in class_name.lower() for kw in keywords)
                    for keywords in DETECTION_KEYWORDS.values()
                ):
                    detection.bboxes.append(BoundingBox(
                        x_min=x1_norm,
                        y_min=y1_norm,
                        x_max=x2_norm,
                        y_max=y2_norm,
                        confidence=confidence,
                        label=class_name
                    ))
        
        return detection
    
    def compute_anchor_points_from_hands(
        self,
        joints: Dict,
        equipment_type: str,
        frame_index: int
    ) -> List[AnchorPoint]:
        """
        Compute equipment anchor points based on hand positions.
        
        For barbells/dumbbells/kettlebells, grip positions are at the wrists.
        For bodyweight exercises, track hands for visualization.
        
        Args:
            joints: Dictionary of joint positions from pose estimation.
            equipment_type: Type of equipment being used.
            frame_index: Frame index for context.
            
        Returns:
            List of anchor points.
        """
        anchors = []
        
        # For all equipment types that involve hand grips, track wrist positions
        if equipment_type in ["barbell", "dumbbell", "kettlebell", "bodyweight", "unknown"]:
            # Use wrist positions as grip anchor points
            left_wrist = joints.get("left_wrist", {})
            right_wrist = joints.get("right_wrist", {})
            
            if left_wrist.get("x") is not None and left_wrist.get("confidence", 0.0) > 0.3:
                anchors.append(AnchorPoint(
                    x=left_wrist["x"],
                    y=left_wrist["y"],
                    name="left_grip",
                    confidence=left_wrist.get("confidence", 0.0)
                ))
            
            if right_wrist.get("x") is not None and right_wrist.get("confidence", 0.0) > 0.3:
                anchors.append(AnchorPoint(
                    x=right_wrist["x"],
                    y=right_wrist["y"],
                    name="right_grip",
                    confidence=right_wrist.get("confidence", 0.0)
                ))
        
        return anchors
    
    def process_video(
        self,
        video_path: str,
        frames_with_poses: List[Tuple[int, np.ndarray, Dict]]
    ) -> VideoEquipmentResult:
        """
        Process a complete video for equipment detection.
        
        Args:
            video_path: Path to the video file.
            frames_with_poses: List of (frame_index, frame, joints_dict) tuples.
            
        Returns:
            VideoEquipmentResult with classification and detections.
        """
        # Extract just the frames for classification
        sample_frames = [f[1] for f in frames_with_poses[:10]]  # Sample first 10 frames
        
        # Classify equipment type
        equipment_type = self.classify_equipment_for_video(video_path, sample_frames)
        
        # Process each frame for detections
        detections = []
        for frame_index, frame, joints in frames_with_poses:
            detection = self.detect_equipment_in_frame(
                frame, frame_index, equipment_type
            )
            
            # Add anchor points based on hand positions
            if joints:
                anchors = self.compute_anchor_points_from_hands(
                    joints, equipment_type, frame_index
                )
                detection.anchor_points = anchors
            
            detections.append(detection)
        
        return VideoEquipmentResult(
            equipment_type=equipment_type,
            confidence=0.8 if equipment_type != "unknown" else 0.0,
            classifier_method="filename_heuristic",
            detections=detections
        )


def get_equipment_types() -> List[str]:
    """Return the list of supported equipment types."""
    return EQUIPMENT_TYPES.copy()
