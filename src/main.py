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
from tqdm import tqdm


def process_video(
    video_path: str,
    output_folder: str,
    pose_estimator: PoseEstimator,
    equipment_detector: EquipmentDetector,
    exporter: DataExporter,
    visualizer: Visualizer,
    frame_step: int = 1,
    verbose: bool = True,
    skip_overlay: bool = False,
    skip_mocap: bool = False
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
        skip_overlay: Skip generating overlay preview video.
        skip_mocap: Skip generating mocap preview video.
        
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
    # Use tqdm to show per-video frame progress when verbose
    total_frames_to_process = max(1, (metadata.total_frames + frame_step - 1) // frame_step)
    frame_iter = iterate_frames(video_path, frame_step)
    for frame_index, frame in tqdm(frame_iter, total=total_frames_to_process, desc=f"Frames {video_basename}", unit="frame", disable=not verbose):
        # Extract pose
        joints = pose_estimator.get_pose_for_frame(frame)
        joints_dict = joints_to_dict(joints)

        pose_frames.append({
            "frame_index": frame_index,
            "joints": joints_dict
        })

        frames_with_poses.append((frame_index, frame, joints_dict))
        frame_count += 1
    
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
    pose_data = {
        "metadata": metadata_dict,
        "frames": pose_frames
    }
    
    if not skip_overlay:
        if verbose:
            print("  Creating overlay preview...")
        
        visualizer.create_overlay_video(
            video_path,
            pose_data,
            equipment_dict,
            output_paths["overlay_video"],
            frame_step,
            show_progress=verbose
        )
    elif verbose:
        print("  Skipping overlay preview (--skip-overlay)")
    
    if not skip_mocap:
        if verbose:
            print("  Creating mocap preview...")
        
        visualizer.create_mocap_video(
            pose_data,
            equipment_dict,
            output_paths["mocap_video"],
            show_progress=verbose
        )
    elif verbose:
        print("  Skipping mocap preview (--skip-mocap)")
    
    if verbose:
        print(f"  Output folder: {video_output_folder}")
    
    return output_paths


def _process_video_worker(
    video_path: str,
    output_folder: str,
    frame_step: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    use_yolo: bool,
    yolo_model: str,
    yolo_weights: str,
    device: str,
    skip_overlay: bool,
    skip_mocap: bool,
) -> Dict[str, str]:
    """
    Worker function for parallel video processing.
    
    Each worker initializes its own models to avoid GPU resource conflicts.
    This function is called by ProcessPoolExecutor.
    """
    # Initialize components in worker process
    pose_estimator = PoseEstimator(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    equipment_detector = EquipmentDetector(
        use_detection_model=use_yolo,
        model_size=yolo_model,
        weights_path=yolo_weights,
        device=device
    )
    exporter = DataExporter(output_folder)
    visualizer = Visualizer()
    
    try:
        output_paths = process_video(
            video_path=video_path,
            output_folder=output_folder,
            pose_estimator=pose_estimator,
            equipment_detector=equipment_detector,
            exporter=exporter,
            visualizer=visualizer,
            frame_step=frame_step,
            verbose=False,  # Suppress per-video output in workers
            skip_overlay=skip_overlay,
            skip_mocap=skip_mocap
        )
        return output_paths
    finally:
        pose_estimator.close()


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
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Skip generating overlay preview video (saves processing time)."
    )
    parser.add_argument(
        "--skip-mocap",
        action="store_true",
        help="Skip generating mocap preview video (saves processing time)."
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default=None,
        help="Path to local YOLO weights file (avoids network download)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device: auto (GPU if available), cuda, or cpu. Default: auto"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers for processing multiple videos. Default: 1 (sequential)."
    )
    
    parsed_args = parser.parse_args(args)
    
    verbose = not parsed_args.quiet
    
    # Detect compute device
    from . import detect_device
    device = detect_device(parsed_args.device)
    
    if verbose:
        print("=" * 60)
        print("Exercise Animation Pipeline")
        print("=" * 60)
        print(f"Input folder:  {parsed_args.input_folder}")
        print(f"Output folder: {parsed_args.output_folder}")
        print(f"Frame step:    {parsed_args.frame_step}")
        print(f"Device:        {device}")
        if parsed_args.workers > 1:
            print(f"Workers:       {parsed_args.workers}")
    
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
        model_size=parsed_args.yolo_model,
        weights_path=parsed_args.yolo_weights,
        device=device
    )
    exporter = DataExporter(parsed_args.output_folder)
    visualizer = Visualizer()
    
    # Process videos (parallel or sequential)
    results = []
    
    if parsed_args.workers > 1 and len(video_loader) > 1:
        # Parallel processing using multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        # For parallel processing, we need to process videos in separate processes
        # Each process will initialize its own pose estimator and equipment detector
        # to avoid GPU resource conflicts
        
        if verbose:
            print(f"\nProcessing {len(video_loader)} video(s) with {parsed_args.workers} workers...")
        
        # Collect video paths
        video_paths = [vp for vp, _ in video_loader]
        
        # Use a process pool - note: each worker will init its own models
        with ProcessPoolExecutor(max_workers=parsed_args.workers) as executor:
            futures = {}
            for video_path in video_paths:
                future = executor.submit(
                    _process_video_worker,
                    video_path,
                    parsed_args.output_folder,
                    parsed_args.frame_step,
                    parsed_args.min_detection_confidence,
                    parsed_args.min_tracking_confidence,
                    parsed_args.use_yolo,
                    parsed_args.yolo_model,
                    parsed_args.yolo_weights,
                    device,
                    parsed_args.skip_overlay,
                    parsed_args.skip_mocap,
                )
                futures[future] = video_path
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Videos", disable=not verbose):
                video_path = futures[future]
                try:
                    output_paths = future.result()
                    results.append((video_path, output_paths, None))
                except Exception as e:
                    print(f"Error processing {video_path}: {e}", file=sys.stderr)
                    results.append((video_path, None, str(e)))
    else:
        # Sequential processing (original behavior)
        for video_path, metadata in tqdm(video_loader, total=len(video_loader), desc="Videos", disable=not verbose):
            try:
                output_paths = process_video(
                    video_path=video_path,
                    output_folder=parsed_args.output_folder,
                    pose_estimator=pose_estimator,
                    equipment_detector=equipment_detector,
                    exporter=exporter,
                    visualizer=visualizer,
                    frame_step=parsed_args.frame_step,
                    verbose=verbose,
                    skip_overlay=parsed_args.skip_overlay,
                    skip_mocap=parsed_args.skip_mocap
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
