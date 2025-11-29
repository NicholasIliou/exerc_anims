"""
Visualization module for pose and equipment data.

Creates overlay_preview.mp4 (with original video) and mocap_preview.mp4 (standalone).
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

import cv2
import numpy as np


# Color scheme (BGR format for OpenCV)
COLORS = {
    "skeleton": (0, 255, 0),        # Green for skeleton lines
    "joints": (0, 0, 255),          # Red for joint points
    "bbox": (255, 165, 0),          # Orange for equipment boxes
    "anchor": (255, 0, 255),        # Magenta for anchor points
    "text": (255, 255, 255),        # White for text
    "background": (40, 40, 40),     # Dark gray background for mocap
}

# Skeleton connections (same as pose_estimator but defined here to avoid import issues)
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


def draw_skeleton(
    frame: np.ndarray,
    joints: Dict[str, Dict[str, Any]],
    color: Tuple[int, int, int] = COLORS["skeleton"],
    joint_color: Tuple[int, int, int] = COLORS["joints"],
    thickness: int = 2,
    joint_radius: int = 4,
    min_confidence: float = 0.3
) -> np.ndarray:
    """
    Draw skeleton on a frame.
    
    Args:
        frame: BGR image to draw on.
        joints: Dictionary of joint positions (normalized coordinates).
        color: Line color in BGR.
        joint_color: Joint point color in BGR.
        thickness: Line thickness.
        joint_radius: Radius of joint circles.
        min_confidence: Minimum confidence to draw a joint.
        
    Returns:
        Frame with skeleton drawn.
    """
    height, width = frame.shape[:2]
    drawn = frame.copy()
    
    # Draw skeleton lines
    for joint1_name, joint2_name in SKELETON_CONNECTIONS:
        joint1 = joints.get(joint1_name, {})
        joint2 = joints.get(joint2_name, {})
        
        if (joint1.get("x") is None or joint2.get("x") is None or
            joint1.get("confidence", 0) < min_confidence or
            joint2.get("confidence", 0) < min_confidence):
            continue
        
        pt1 = (int(joint1["x"] * width), int(joint1["y"] * height))
        pt2 = (int(joint2["x"] * width), int(joint2["y"] * height))
        
        cv2.line(drawn, pt1, pt2, color, thickness)
    
    # Draw joints
    for joint_name, joint in joints.items():
        if joint.get("x") is None or joint.get("confidence", 0) < min_confidence:
            continue
        
        pt = (int(joint["x"] * width), int(joint["y"] * height))
        cv2.circle(drawn, pt, joint_radius, joint_color, -1)
    
    return drawn


def draw_equipment(
    frame: np.ndarray,
    detections: Dict[str, Any],
    equipment_type: str,
    bbox_color: Tuple[int, int, int] = COLORS["bbox"],
    anchor_color: Tuple[int, int, int] = COLORS["anchor"],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw equipment detections on a frame.
    
    Args:
        frame: BGR image to draw on.
        detections: Detection data for this frame.
        equipment_type: Type of equipment for label.
        bbox_color: Bounding box color.
        anchor_color: Anchor point color.
        thickness: Line thickness.
        
    Returns:
        Frame with equipment drawn.
    """
    height, width = frame.shape[:2]
    drawn = frame.copy()
    
    # Draw bounding boxes
    for bbox in detections.get("bboxes", []):
        x_min = int(bbox["x_min"] * width)
        y_min = int(bbox["y_min"] * height)
        x_max = int(bbox["x_max"] * width)
        y_max = int(bbox["y_max"] * height)
        
        cv2.rectangle(drawn, (x_min, y_min), (x_max, y_max), bbox_color, thickness)
        
        label = bbox.get("label", equipment_type)
        cv2.putText(drawn, label, (x_min, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
    
    # Draw anchor points
    for anchor in detections.get("anchor_points", []):
        if anchor.get("x") is None:
            continue
        
        pt = (int(anchor["x"] * width), int(anchor["y"] * height))
        cv2.circle(drawn, pt, 6, anchor_color, -1)
        cv2.putText(drawn, anchor.get("name", "grip"), (pt[0] + 8, pt[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, anchor_color, 1)
    
    return drawn


def draw_equipment_simple(
    frame: np.ndarray,
    anchors: List[Dict[str, Any]],
    equipment_type: str,
    color: Tuple[int, int, int] = COLORS["bbox"]
) -> np.ndarray:
    """
    Draw simple equipment representation for mocap preview.
    
    Args:
        frame: BGR image to draw on.
        anchors: List of anchor point dictionaries.
        equipment_type: Type of equipment.
        color: Drawing color.
        
    Returns:
        Frame with equipment drawn.
    """
    height, width = frame.shape[:2]
    drawn = frame.copy()
    
    valid_anchors = [a for a in anchors if a.get("x") is not None]
    
    if equipment_type == "barbell" and len(valid_anchors) >= 2:
        # Draw barbell as a line between grips with end circles (weight plates)
        pt1 = (int(valid_anchors[0]["x"] * width), int(valid_anchors[0]["y"] * height))
        pt2 = (int(valid_anchors[1]["x"] * width), int(valid_anchors[1]["y"] * height))
        
        # Draw bar
        cv2.line(drawn, pt1, pt2, color, 6)
        
        # Draw weight plates (circles at ends)
        cv2.circle(drawn, pt1, 20, color, 4)
        cv2.circle(drawn, pt2, 20, color, 4)
        
        # Draw inner plates
        cv2.circle(drawn, pt1, 12, color, 2)
        cv2.circle(drawn, pt2, 12, color, 2)
    
    elif equipment_type == "dumbbell" and valid_anchors:
        # Draw dumbbells as weights at each grip
        for anchor in valid_anchors:
            pt = (int(anchor["x"] * width), int(anchor["y"] * height))
            # Dumbbell bell (two circles connected)
            cv2.circle(drawn, (pt[0] - 15, pt[1]), 12, color, 3)
            cv2.circle(drawn, (pt[0] + 15, pt[1]), 12, color, 3)
            # Handle
            cv2.line(drawn, (pt[0] - 15, pt[1]), (pt[0] + 15, pt[1]), color, 4)
    
    elif equipment_type == "kettlebell" and valid_anchors:
        # Draw kettlebells as circles with handles
        for anchor in valid_anchors:
            pt = (int(anchor["x"] * width), int(anchor["y"] * height))
            # Kettlebell body
            cv2.circle(drawn, (pt[0], pt[1] + 15), 18, color, 3)
            # Handle arc
            cv2.ellipse(drawn, pt, (10, 12), 0, 180, 360, color, 3)
    
    elif equipment_type == "bodyweight" and valid_anchors:
        # Draw hand markers for bodyweight exercises
        for anchor in valid_anchors:
            pt = (int(anchor["x"] * width), int(anchor["y"] * height))
            cv2.circle(drawn, pt, 8, color, 2)
            # Draw small cross to indicate hand position
            cv2.line(drawn, (pt[0] - 5, pt[1]), (pt[0] + 5, pt[1]), color, 2)
            cv2.line(drawn, (pt[0], pt[1] - 5), (pt[0], pt[1] + 5), color, 2)
    
    elif equipment_type == "machine" and valid_anchors:
        # Draw machine interface points (handles/grips)
        for anchor in valid_anchors:
            pt = (int(anchor["x"] * width), int(anchor["y"] * height))
            cv2.rectangle(drawn, (pt[0] - 10, pt[1] - 10), (pt[0] + 10, pt[1] + 10), color, 3)
    
    elif equipment_type == "unknown" and valid_anchors:
        # Draw simple markers for unknown equipment
        for anchor in valid_anchors:
            pt = (int(anchor["x"] * width), int(anchor["y"] * height))
            cv2.circle(drawn, pt, 6, color, -1)
    
    return drawn


class Visualizer:
    """
    Video visualizer for pose and equipment data.
    
    Creates two types of preview videos:
    1. overlay_preview.mp4 - Original video with skeleton/equipment overlay
    2. mocap_preview.mp4 - Standalone visualization on solid background
    
    Example:
        viz = Visualizer()
        viz.create_overlay_video(video_path, pose_data, equipment_data, output_path)
        viz.create_mocap_video(pose_data, equipment_data, output_path, fps, (width, height))
    """
    
    def __init__(
        self,
        skeleton_color: Tuple[int, int, int] = COLORS["skeleton"],
        joint_color: Tuple[int, int, int] = COLORS["joints"],
        bbox_color: Tuple[int, int, int] = COLORS["bbox"],
        background_color: Tuple[int, int, int] = COLORS["background"]
    ):
        """
        Initialize the visualizer.
        
        Args:
            skeleton_color: Color for skeleton lines.
            joint_color: Color for joint points.
            bbox_color: Color for equipment boxes.
            background_color: Background color for mocap preview.
        """
        self.skeleton_color = skeleton_color
        self.joint_color = joint_color
        self.bbox_color = bbox_color
        self.background_color = background_color
    
    def create_overlay_video(
        self,
        video_path: str,
        pose_data: Dict[str, Any],
        equipment_data: Dict[str, Any],
        output_path: str,
        frame_step: int = 1
    ) -> None:
        """
        Create overlay preview video.
        
        Draws skeleton and equipment detections on original video frames.
        
        Args:
            video_path: Path to original video.
            pose_data: Pose data dictionary (from pose.json).
            equipment_data: Equipment data dictionary (from equipment.json).
            output_path: Path for output video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Build frame lookup for pose and equipment data
        pose_frames = {f["frame_index"]: f for f in pose_data.get("frames", [])}
        equipment_frames = {d["frame_index"]: d for d in equipment_data.get("detections", [])}
        equipment_type = equipment_data.get("equipment_type", "unknown")
        
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw skeleton if we have pose data for this frame
            if frame_index in pose_frames:
                joints = pose_frames[frame_index].get("joints", {})
                frame = draw_skeleton(
                    frame, joints,
                    color=self.skeleton_color,
                    joint_color=self.joint_color
                )
            
            # Draw equipment if we have detection data
            if frame_index in equipment_frames:
                frame = draw_equipment(
                    frame, equipment_frames[frame_index],
                    equipment_type,
                    bbox_color=self.bbox_color
                )
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_index}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2)
            cv2.putText(frame, f"Equipment: {equipment_type}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2)
            
            out.write(frame)
            frame_index += 1
        
        cap.release()
        out.release()
    
    def create_mocap_video(
        self,
        pose_data: Dict[str, Any],
        equipment_data: Dict[str, Any],
        output_path: str,
        fps: Optional[float] = None,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Create standalone mocap preview video.
        
        Renders skeleton and equipment on a solid background.
        This video can be reconstructed purely from pose.json and equipment.json.
        
        Args:
            pose_data: Pose data dictionary (from pose.json).
            equipment_data: Equipment data dictionary (from equipment.json).
            output_path: Path for output video.
            fps: Frames per second (from metadata if not provided).
            frame_size: (width, height) tuple (from metadata if not provided).
        """
        metadata = pose_data.get("metadata", {})
        
        if fps is None:
            fps = metadata.get("fps", 30.0)
        if frame_size is None:
            frame_size = (metadata.get("width", 640), metadata.get("height", 480))
        
        width, height = frame_size
        frame_step = metadata.get("frame_step", 1)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_step, (width, height))
        
        equipment_type = equipment_data.get("equipment_type", "unknown")
        equipment_frames = {d["frame_index"]: d for d in equipment_data.get("detections", [])}
        
        for frame_data in pose_data.get("frames", []):
            frame_index = frame_data["frame_index"]
            joints = frame_data.get("joints", {})
            
            # Create blank frame with background color
            frame = np.full((height, width, 3), self.background_color, dtype=np.uint8)
            
            # Draw skeleton
            frame = draw_skeleton(
                frame, joints,
                color=self.skeleton_color,
                joint_color=self.joint_color,
                thickness=3,
                joint_radius=6
            )
            
            # Draw simple equipment representation
            if frame_index in equipment_frames:
                anchors = equipment_frames[frame_index].get("anchor_points", [])
                frame = draw_equipment_simple(
                    frame, anchors, equipment_type,
                    color=self.bbox_color
                )
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_index}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2)
            cv2.putText(frame, f"Equipment: {equipment_type}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2)
            
            out.write(frame)
        
        out.release()


def create_preview_videos(
    video_path: str,
    pose_json_path: str,
    equipment_json_path: str,
    overlay_output_path: str,
    mocap_output_path: str
) -> None:
    """
    Create both preview videos from JSON data files.
    
    Convenience function for creating visualizations from saved data.
    
    Args:
        video_path: Path to original video.
        pose_json_path: Path to pose.json.
        equipment_json_path: Path to equipment.json.
        overlay_output_path: Path for overlay preview video.
        mocap_output_path: Path for mocap preview video.
    """
    with open(pose_json_path, 'r') as f:
        pose_data = json.load(f)
    
    with open(equipment_json_path, 'r') as f:
        equipment_data = json.load(f)
    
    viz = Visualizer()
    viz.create_overlay_video(video_path, pose_data, equipment_data, overlay_output_path)
    viz.create_mocap_video(pose_data, equipment_data, mocap_output_path)
