"""
Pose estimation module using MediaPipe.

Wraps MediaPipe Pose to extract 2D keypoints for major body joints.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


@dataclass
class JointPosition:
    """2D position and confidence for a single joint."""
    x: Optional[float]  # Normalized x coordinate (0-1) or None if not detected
    y: Optional[float]  # Normalized y coordinate (0-1) or None if not detected
    confidence: float   # Detection confidence (0-1)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence
        }
    
    @staticmethod
    def empty() -> 'JointPosition':
        """Create an empty joint position for failed detections."""
        return JointPosition(x=None, y=None, confidence=0.0)


# Mapping from our standardized joint names to MediaPipe landmark indices
# MediaPipe Pose has 33 landmarks, we extract the key ones
MEDIAPIPE_JOINT_MAPPING = {
    # Head and neck region
    "head": 0,           # nose (approximation for head)
    "neck": None,        # Will be computed as midpoint of shoulders
    
    # Upper body
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    
    # Spine / core
    "spine": None,       # Will be computed as midpoint of hips
    
    # Lower body
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Ordered list of joint names for consistent indexing
JOINT_NAMES = [
    "head", "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "spine",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# Skeleton connections for visualization (pairs of joint names)
SKELETON_CONNECTIONS = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "spine"),
    ("spine", "left_hip"),
    ("spine", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class PoseEstimator:
    """
    Pose estimation wrapper using MediaPipe.
    
    Extracts 2D keypoints for major body joints from video frames.
    
    Example:
        estimator = PoseEstimator()
        joints = estimator.get_pose_for_frame(frame)
        for name, joint in joints.items():
            print(f"{name}: ({joint.x}, {joint.y}) conf={joint.confidence}")
    """
    
    def __init__(
        self, 
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initialize the pose estimator.
        
        Args:
            min_detection_confidence: Minimum confidence for person detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            model_complexity: Model complexity (0, 1, or 2). Higher = more accurate but slower.
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def get_pose_for_frame(self, frame: np.ndarray) -> Dict[str, JointPosition]:
        """
        Extract pose keypoints from a single frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format).
            
        Returns:
            Dictionary mapping joint names to JointPosition objects.
            If detection fails, all joints will have None coordinates and 0 confidence.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = frame[:, :, ::-1]
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Initialize all joints as empty
        joints = {name: JointPosition.empty() for name in JOINT_NAMES}
        
        if results.pose_landmarks is None:
            return joints
        
        landmarks = results.pose_landmarks.landmark
        
        # Extract direct landmarks
        for joint_name, landmark_idx in MEDIAPIPE_JOINT_MAPPING.items():
            if landmark_idx is not None:
                lm = landmarks[landmark_idx]
                joints[joint_name] = JointPosition(
                    x=lm.x,
                    y=lm.y,
                    confidence=lm.visibility
                )
        
        # Compute derived landmarks (neck and spine)
        # Neck = midpoint of shoulders
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        joints["neck"] = JointPosition(
            x=(left_shoulder.x + right_shoulder.x) / 2,
            y=(left_shoulder.y + right_shoulder.y) / 2,
            confidence=min(left_shoulder.visibility, right_shoulder.visibility)
        )
        
        # Spine = midpoint of hips
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        joints["spine"] = JointPosition(
            x=(left_hip.x + right_hip.x) / 2,
            y=(left_hip.y + right_hip.y) / 2,
            confidence=min(left_hip.visibility, right_hip.visibility)
        )
        
        return joints
    
    def close(self):
        """Release resources."""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def joints_to_dict(joints: Dict[str, JointPosition]) -> Dict[str, dict]:
    """
    Convert joints dictionary to JSON-serializable format.
    
    Args:
        joints: Dictionary mapping joint names to JointPosition objects.
        
    Returns:
        Dictionary with same keys but JointPosition converted to dicts.
    """
    return {name: joint.to_dict() for name, joint in joints.items()}


def get_joint_names() -> List[str]:
    """Return the ordered list of joint names."""
    return JOINT_NAMES.copy()


def get_skeleton_connections() -> List[tuple]:
    """Return the list of skeleton bone connections."""
    return SKELETON_CONNECTIONS.copy()
