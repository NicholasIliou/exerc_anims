"""
Video loader module for gym exercise video processing.

Handles video file discovery and frame iteration with optional downsampling.
"""

import os
from pathlib import Path
from typing import Iterator, Tuple, List, Optional
from dataclasses import dataclass

import cv2
import numpy as np


# Supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


@dataclass
class VideoMetadata:
    """Metadata for a video file."""
    filename: str
    filepath: str
    fps: float
    width: int
    height: int
    total_frames: int
    frame_step: int
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "frame_step": self.frame_step
        }


def discover_videos(input_folder: str) -> List[str]:
    """
    Discover all video files in the input folder.
    
    Args:
        input_folder: Path to folder containing video files.
        
    Returns:
        List of absolute paths to video files.
    """
    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    videos = set()
    for ext in VIDEO_EXTENSIONS:
        videos.update(folder.glob(f"*{ext}"))
        videos.update(folder.glob(f"*{ext.upper()}"))
    
    return sorted([str(v.absolute()) for v in videos])


def get_video_metadata(video_path: str, frame_step: int = 1) -> VideoMetadata:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to the video file.
        frame_step: Downsampling factor (process every Nth frame).
        
    Returns:
        VideoMetadata object with video properties.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    try:
        metadata = VideoMetadata(
            filename=Path(video_path).name,
            filepath=str(Path(video_path).absolute()),
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            frame_step=frame_step
        )
    finally:
        cap.release()
    
    return metadata


def iterate_frames(
    video_path: str, 
    frame_step: int = 1
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Iterate over video frames with optional downsampling.
    
    Args:
        video_path: Path to the video file.
        frame_step: Process every Nth frame (1 = all frames).
        
    Yields:
        Tuple of (frame_index, frame_image) where frame_image is BGR numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    try:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % frame_step == 0:
                yield frame_index, frame
            
            frame_index += 1
    finally:
        cap.release()


def get_video_basename(video_path: str) -> str:
    """
    Get the base name of a video file (without extension).
    
    Used to create deterministic output folder names.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Base name without extension (e.g., "squat_001" from "squat_001.mp4").
    """
    return Path(video_path).stem


class VideoLoader:
    """
    High-level video loader for processing multiple videos.
    
    Example:
        loader = VideoLoader("path/to/videos", frame_step=3)
        for video_path, metadata in loader:
            for frame_idx, frame in loader.iterate_video(video_path):
                # process frame
    """
    
    def __init__(self, input_folder: str, frame_step: int = 1):
        """
        Initialize the video loader.
        
        Args:
            input_folder: Path to folder containing video files.
            frame_step: Downsampling factor (process every Nth frame).
        """
        self.input_folder = input_folder
        self.frame_step = frame_step
        self.video_paths = discover_videos(input_folder)
    
    def __iter__(self) -> Iterator[Tuple[str, VideoMetadata]]:
        """Iterate over all discovered videos with their metadata."""
        for video_path in self.video_paths:
            metadata = get_video_metadata(video_path, self.frame_step)
            yield video_path, metadata
    
    def __len__(self) -> int:
        """Return the number of discovered videos."""
        return len(self.video_paths)
    
    def iterate_video(self, video_path: str) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over frames of a specific video.
        
        Args:
            video_path: Path to the video file.
            
        Yields:
            Tuple of (frame_index, frame_image).
        """
        return iterate_frames(video_path, self.frame_step)
