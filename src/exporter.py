"""
Data exporter module for pose and equipment data.

Writes pose.json and equipment.json with consistent schemas.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PoseFrame:
    """Pose data for a single frame."""
    frame_index: int
    joints: Dict[str, Dict[str, Any]]
    
    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "joints": self.joints
        }


def export_pose_data(
    output_path: str,
    metadata: Dict[str, Any],
    frames: List[Dict[str, Any]]
) -> None:
    """
    Export pose data to JSON file.
    
    Schema:
    {
        "metadata": {
            "filename": str,      # Original video filename
            "fps": float,         # Video frames per second
            "width": int,         # Video width in pixels
            "height": int,        # Video height in pixels
            "frame_step": int,    # Downsampling factor
            "total_frames": int   # Total frames in original video
        },
        "joints_schema": {
            "names": [str],       # Ordered list of joint names
            "indices": {str: int} # Mapping of joint name to index
        },
        "frames": [
            {
                "frame_index": int,
                "joints": {
                    "joint_name": {
                        "x": float|null,      # Normalized x coordinate
                        "y": float|null,      # Normalized y coordinate  
                        "confidence": float   # Detection confidence (0-1)
                    }
                }
            }
        ]
    }
    
    Args:
        output_path: Path to output JSON file.
        metadata: Video metadata dictionary.
        frames: List of frame data dictionaries.
    """
    # Import here to avoid circular dependency
    from .pose_estimator import get_joint_names
    
    joint_names = get_joint_names()
    
    pose_data = {
        "metadata": metadata,
        "joints_schema": {
            "names": joint_names,
            "indices": {name: idx for idx, name in enumerate(joint_names)}
        },
        "frames": frames
    }
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, indent=2)


def export_equipment_data(
    output_path: str,
    equipment_result: Dict[str, Any],
    video_metadata: Dict[str, Any]
) -> None:
    """
    Export equipment data to JSON file.
    
    Schema:
    {
        "equipment_type": str,    # One of: barbell, dumbbell, kettlebell, machine, bodyweight, unknown
        "confidence": float,      # Classification confidence (0-1)
        "metadata": {
            "filename": str,      # Original video filename
            "classifier": str,    # Classification method used
            "equipment_types": [str]  # All supported equipment types
        },
        "detections": [
            {
                "frame_index": int,
                "bboxes": [
                    {
                        "x_min": float,      # Normalized coordinates
                        "y_min": float,
                        "x_max": float,
                        "y_max": float,
                        "confidence": float,
                        "label": str
                    }
                ],
                "anchor_points": [
                    {
                        "x": float,          # Normalized coordinates
                        "y": float,
                        "name": str,         # e.g., "left_grip", "right_grip"
                        "confidence": float
                    }
                ]
            }
        ]
    }
    
    Args:
        output_path: Path to output JSON file.
        equipment_result: Equipment detection result dictionary.
        video_metadata: Video metadata for context.
    """
    from .equipment_detector import get_equipment_types
    
    # Merge video metadata into equipment metadata
    equipment_data = {
        "equipment_type": equipment_result.get("equipment_type", "unknown"),
        "confidence": equipment_result.get("confidence", 0.0),
        "metadata": {
            "filename": video_metadata.get("filename", ""),
            "classifier": equipment_result.get("metadata", {}).get("classifier", "unknown"),
            "equipment_types": get_equipment_types()
        },
        "detections": equipment_result.get("detections", [])
    }
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(equipment_data, f, indent=2)


def create_output_folder(output_base: str, video_basename: str) -> str:
    """
    Create output folder for a video's results.
    
    Args:
        output_base: Base output directory.
        video_basename: Video filename without extension.
        
    Returns:
        Path to the created output folder.
    """
    output_folder = Path(output_base) / video_basename
    output_folder.mkdir(parents=True, exist_ok=True)
    return str(output_folder)


def get_output_paths(output_folder: str) -> Dict[str, str]:
    """
    Get standardized output file paths for a video.
    
    Args:
        output_folder: Path to the video's output folder.
        
    Returns:
        Dictionary with keys: pose_json, equipment_json, overlay_video, mocap_video
    """
    folder = Path(output_folder)
    return {
        "pose_json": str(folder / "pose.json"),
        "equipment_json": str(folder / "equipment.json"),
        "overlay_video": str(folder / "overlay_preview.mp4"),
        "mocap_video": str(folder / "mocap_preview.mp4")
    }


class DataExporter:
    """
    High-level data exporter for video processing results.
    
    Example:
        exporter = DataExporter(output_base="./output")
        output_folder = exporter.prepare_output("squat_001")
        exporter.export_pose(output_folder, metadata, frames)
        exporter.export_equipment(output_folder, equipment_result, metadata)
    """
    
    def __init__(self, output_base: str):
        """
        Initialize the exporter.
        
        Args:
            output_base: Base directory for all outputs.
        """
        self.output_base = output_base
        Path(output_base).mkdir(parents=True, exist_ok=True)
    
    def prepare_output(self, video_basename: str) -> str:
        """
        Prepare output folder for a video.
        
        Args:
            video_basename: Video filename without extension.
            
        Returns:
            Path to the output folder.
        """
        return create_output_folder(self.output_base, video_basename)
    
    def get_paths(self, output_folder: str) -> Dict[str, str]:
        """Get output file paths for a video."""
        return get_output_paths(output_folder)
    
    def export_pose(
        self,
        output_folder: str,
        metadata: Dict[str, Any],
        frames: List[Dict[str, Any]]
    ) -> str:
        """
        Export pose data.
        
        Args:
            output_folder: Video's output folder.
            metadata: Video metadata.
            frames: List of frame pose data.
            
        Returns:
            Path to the created pose.json file.
        """
        paths = self.get_paths(output_folder)
        export_pose_data(paths["pose_json"], metadata, frames)
        return paths["pose_json"]
    
    def export_equipment(
        self,
        output_folder: str,
        equipment_result: Dict[str, Any],
        video_metadata: Dict[str, Any]
    ) -> str:
        """
        Export equipment data.
        
        Args:
            output_folder: Video's output folder.
            equipment_result: Equipment detection result.
            video_metadata: Video metadata.
            
        Returns:
            Path to the created equipment.json file.
        """
        paths = self.get_paths(output_folder)
        export_equipment_data(
            paths["equipment_json"], 
            equipment_result, 
            video_metadata
        )
        return paths["equipment_json"]
