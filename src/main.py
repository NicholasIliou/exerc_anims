"""
Main entry point for the exercise animation pipeline.

Processes gym exercise videos to extract pose and equipment data.

Usage:
    python main.py --input_folder path/to/videos --output_folder path/to/output --frame_step 3
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from .video_loader import VideoLoader, get_video_basename
from .pose_estimator import PoseEstimator, joints_to_dict
from .equipment_detector import EquipmentDetector
from .exporter import DataExporter
from .visualizer import Visualizer


def process_video(
    video_path: str,
    output_folder: str,
    pose_estimator: PoseEstimator,
    equipment_detector: EquipmentDetector,
    exporter: DataExporter,
    visualizer: Visualizer,
    frame_step: int = 1,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Process a single video through the full pipeline.
    
    Args:
        video_path: Path to the video file.
        output_folder: Base output folder.
        pose_estimator: Initialized pose estimator.
        equipment_detector: Initialized equipment detector.
        exporter: Data exporter instance.
        visualizer: Visualizer instance.
        frame_step: Frame downsampling factor.
        verbose: Whether to print progress.
        
    Returns:
        Dictionary of output file paths.
    """
    from .video_loader import get_video_metadata, iterate_frames
    
    video_basename = get_video_basename(video_path)
    if verbose:
        print(f"\nProcessing: {video_basename}")
    
    # Get video metadata
    metadata = get_video_metadata(video_path, frame_step)
    metadata_dict = metadata.to_dict()
    
    # Prepare output folder
    video_output_folder = exporter.prepare_output(video_basename)
    output_paths = exporter.get_paths(video_output_folder)
    
    if verbose:
        print(f"  Video: {metadata.width}x{metadata.height} @ {metadata.fps:.1f} FPS")
        print(f"  Total frames: {metadata.total_frames}, processing every {frame_step} frame(s)")
    
    # Process frames
    pose_frames: List[Dict[str, Any]] = []
    frames_with_poses: List[Tuple[int, np.ndarray, Dict]] = []
    
    frame_count = 0
    for frame_index, frame in iterate_frames(video_path, frame_step):
        # Extract pose
        joints = pose_estimator.get_pose_for_frame(frame)
        joints_dict = joints_to_dict(joints)
        
        pose_frames.append({
            "frame_index": frame_index,
            "joints": joints_dict
        })
        
        frames_with_poses.append((frame_index, frame, joints_dict))
        frame_count += 1
        
        if verbose and frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")
    
    if verbose:
        print(f"  Pose extraction complete: {frame_count} frames")
    
    # Process equipment detection
    if verbose:
        print("  Detecting equipment...")
    
    equipment_result = equipment_detector.process_video(video_path, frames_with_poses)
    equipment_dict = equipment_result.to_dict()
    
    if verbose:
        print(f"  Equipment type: {equipment_result.equipment_type}")
    
    # Export data
    if verbose:
        print("  Exporting data...")
    
    exporter.export_pose(video_output_folder, metadata_dict, pose_frames)
    exporter.export_equipment(video_output_folder, equipment_dict, metadata_dict)
    
    # Create visualizations
    if verbose:
        print("  Creating overlay preview...")
    
    pose_data = {
        "metadata": metadata_dict,
        "frames": pose_frames
    }
    
    visualizer.create_overlay_video(
        video_path,
        pose_data,
        equipment_dict,
        output_paths["overlay_video"],
        frame_step
    )
    
    if verbose:
        print("  Creating mocap preview...")
    
    visualizer.create_mocap_video(
        pose_data,
        equipment_dict,
        output_paths["mocap_video"]
    )
    
    if verbose:
        print(f"  Output folder: {video_output_folder}")
    
    return output_paths


def main(args: List[str] = None) -> int:
    """
    Main entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None).
        
    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Process gym exercise videos to extract pose and equipment data."
    )
    parser.add_argument(
        "--input_folder", "-i",
        required=True,
        help="Path to folder containing input videos."
    )
    parser.add_argument(
        "--output_folder", "-o",
        required=True,
        help="Path to output folder for results."
    )
    parser.add_argument(
        "--frame_step", "-s",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)."
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum pose detection confidence (0-1)."
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum pose tracking confidence (0-1)."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output."
    )
    parser.add_argument(
        "--use_yolo",
        action="store_true",
        help="Enable YOLO-based equipment detection (requires ultralytics package)."
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO model size: n(ano), s(mall), m(edium), l(arge), x(large). Default: n"
    )
    
    parsed_args = parser.parse_args(args)
    
    verbose = not parsed_args.quiet
    
    if verbose:
        print("=" * 60)
        print("Exercise Animation Pipeline")
        print("=" * 60)
        print(f"Input folder:  {parsed_args.input_folder}")
        print(f"Output folder: {parsed_args.output_folder}")
        print(f"Frame step:    {parsed_args.frame_step}")
    
    # Initialize components
    try:
        video_loader = VideoLoader(parsed_args.input_folder, parsed_args.frame_step)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if len(video_loader) == 0:
        print("No video files found in input folder.", file=sys.stderr)
        return 1
    
    if verbose:
        print(f"\nFound {len(video_loader)} video(s)")
    
    # Initialize processing components
    pose_estimator = PoseEstimator(
        min_detection_confidence=parsed_args.min_detection_confidence,
        min_tracking_confidence=parsed_args.min_tracking_confidence
    )
    equipment_detector = EquipmentDetector(
        use_detection_model=parsed_args.use_yolo,
        model_size=parsed_args.yolo_model
    )
    exporter = DataExporter(parsed_args.output_folder)
    visualizer = Visualizer()
    
    # Process each video
    results = []
    for video_path, metadata in video_loader:
        try:
            output_paths = process_video(
                video_path=video_path,
                output_folder=parsed_args.output_folder,
                pose_estimator=pose_estimator,
                equipment_detector=equipment_detector,
                exporter=exporter,
                visualizer=visualizer,
                frame_step=parsed_args.frame_step,
                verbose=verbose
            )
            results.append((video_path, output_paths, None))
        except Exception as e:
            print(f"Error processing {video_path}: {e}", file=sys.stderr)
            results.append((video_path, None, str(e)))
    
    # Cleanup
    pose_estimator.close()
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Processing Complete")
        print("=" * 60)
        
        success_count = sum(1 for _, paths, error in results if error is None)
        print(f"Successfully processed: {success_count}/{len(results)} videos")
        
        for video_path, paths, error in results:
            basename = get_video_basename(video_path)
            if error:
                print(f"  ✗ {basename}: {error}")
            else:
                print(f"  ✓ {basename}")
    
    return 0 if all(error is None for _, _, error in results) else 1


if __name__ == "__main__":
    sys.exit(main())
