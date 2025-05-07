#!/usr/bin/env python3

"""
OAK-D ArUco Detection System

This is the main integration module that combines all components into a
complete ArUco marker detection system for the OAK-D camera with OpenCV 4.10
compatibility.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np
import time
import os
import argparse
from typing import Tuple, List, Optional, Dict, Any, Union

# Import our custom modules
from aruco_detector import ArUcoDetector
from oak_d_camera import OakDCamera
from marker_tracker import MarkerTrackerManager
from calibration_manager import CalibrationManager


class OakDArUcoSystem:
    """
    Main system class that integrates all components for ArUco marker detection
    with the OAK-D camera.
    """
    
    def __init__(self, 
                 target_id: Optional[int] = None,
                 resolution_mode: str = "adaptive",
                 use_cuda: bool = False,
                 performance_mode: bool = False,
                 calib_dir: str = "aruco/camera_calibration",
                 marker_size: float = 0.3048,  # 12 inches in meters
                 enable_validation: bool = True,
                 headless: bool = False):
        """
        Initialize the OAK-D ArUco detection system
        
        Args:
            target_id: ID of the target marker to prioritize (optional)
            resolution_mode: Resolution mode ("low", "medium", "high", or "adaptive")
            use_cuda: Whether to use CUDA acceleration if available
            performance_mode: Whether to enable performance optimizations
            calib_dir: Directory for calibration files
            marker_size: Physical size of the marker in meters
            enable_validation: Whether to enable validation to prevent false positives
            headless: Whether to run in headless mode (no GUI)
        """
        self.target_id = target_id
        self.resolution_mode = resolution_mode
        self.use_cuda = use_cuda
        self.performance_mode = performance_mode
        self.calib_dir = calib_dir
        self.marker_size = marker_size
        self.enable_validation = enable_validation
        self.headless = headless
        
        # Initialize components
        print("Initializing OAK-D ArUco Detection System...")
        
        # Initialize calibration manager
        print("Initializing calibration manager...")
        self.calibration_manager = CalibrationManager(calib_dir=calib_dir)
        
        # Get calibration data
        self.camera_matrix = self.calibration_manager.get_camera_matrix()
        self.dist_coeffs = self.calibration_manager.get_dist_coeffs()
        
        # Check if using default calibration
        if self.calibration_manager.is_default_calibration():
            print("Warning: Using default calibration")
            print("For better accuracy, please calibrate your camera")
        
        # Initialize ArUco detector
        print("Initializing ArUco detector...")
        self.aruco_detector = ArUcoDetector(
            dictionary_type=cv2.aruco.DICT_6X6_250,
            marker_size=marker_size,
            use_cuda=use_cuda,
            enable_validation=enable_validation
        )
        
        # Initialize OAK-D camera
        print("Initializing OAK-D camera...")
        self.camera = OakDCamera(
            resolution_mode=resolution_mode,
            enable_spatial=True,
            enable_depth=True
        )
        
        # Initialize marker tracker
        print("Initializing marker tracker...")
        self.tracker_manager = MarkerTrackerManager(
            target_id=target_id,
            max_markers=10
        )
        
        # System state
        self.running = False
        self.frame_count = 0
        self.last_markers_frame = None
        self.last_depth_frame = None
        self.estimated_distance = 5000  # Initial estimate: 5m
        self.skip_frames = 0  # For performance mode
        
        # Performance monitoring
        self.start_time = time.time()
        self.fps_history = []
        self.detection_times = []
        
        print("System initialization complete")
    
    def start(self):
        """
        Start the OAK-D ArUco detection system
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Starting OAK-D ArUco detection system...")
        
        # Start camera
        if not self.camera.start():
            print("Failed to start camera")
            return False
        
        # Update frame size in tracker
        frame_width, frame_height = self.camera.current_resolution
        self.tracker_manager.set_frame_size(frame_width, frame_height)
        
        # Set system state
        self.running = True
        self.start_time = time.time()
        
        print("System started successfully")
        return True
    
    def stop(self):
        """
        Stop the OAK-D ArUco detection system
        """
        print("Stopping OAK-D ArUco detection system...")
        
        # Stop camera
        self.camera.stop()
        
        # Set system state
        self.running = False
        
        # Clean up resources
        if not self.headless:
            cv2.destroyAllWindows()
        
        print("System stopped")
    
    def process_frame(self):
        """
        Process a single frame from the camera
        
        Returns:
            bool: True if frame was processed, False otherwise
        """
        if not self.running:
            return False
        
        # Get RGB and depth frames
        rgb_frame = self.camera.get_rgb_frame()
        depth_frame = self.camera.get_depth_frame()
        spatial_data = self.camera.get_spatial_data()
        
        # Skip frame processing if needed (for performance)
        self.frame_count += 1
        if self.performance_mode and self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
            # Still display the last processed frame if not in headless mode
            if not self.headless:
                if self.last_markers_frame is not None:
                    cv2.imshow("ArUco Detection", self.last_markers_frame)
                    
                if self.last_depth_frame is not None:
                    cv2.imshow("Depth", self.last_depth_frame)
            
            return True
        
        # Process RGB frame if available
        if rgb_frame is not None:
            # Update camera resolution based on estimated distance
            if self.resolution_mode == "adaptive":
                self.camera.update_resolution(self.estimated_distance)
            
            # Detect ArUco markers
            start_time = time.time()
            markers_frame, corners, ids = self.aruco_detector.detect_markers(
                rgb_frame,
                self.camera_matrix,
                self.dist_coeffs,
                self.estimated_distance,
                simple_detection=self.performance_mode
            )
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            # Store for display
            self.last_markers_frame = markers_frame
            
            # Update spatial calculator ROI based on marker positions
            if ids is not None and len(ids) > 0:
                self.camera.update_spatial_calc_roi(corners, ids, self.target_id)
            
            # Estimate pose if markers detected
            distances = {}
            if ids is not None and len(ids) > 0 and self.camera_matrix is not None and self.dist_coeffs is not None:
                # Estimate pose
                rvecs, tvecs = self.aruco_detector.estimate_pose(
                    corners,
                    ids,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                # Calculate distances
                for i, tvec in enumerate(tvecs):
                    # Calculate Euclidean distance in meters
                    distance = np.linalg.norm(tvec)
                    distances[i] = distance
                
                # Update estimated distance (for adaptive parameters)
                if len(distances) > 0:
                    # Use average distance of all markers
                    avg_distance = sum(distances.values()) / len(distances)
                    # Convert to mm for consistency with other components
                    self.estimated_distance = avg_distance * 1000
            
            # Update marker tracker
            self.tracker_manager.update(corners, ids, distances)
            
            # Draw tracked markers
            markers_frame = self.tracker_manager.draw_markers(
                markers_frame,
                draw_ids=True,
                draw_centers=True,
                highlight_target=True
            )
            
            # Draw targeting guidance if target is specified
            if self.target_id is not None:
                markers_frame = self.tracker_manager.draw_targeting_guidance(markers_frame)
            
            # Add performance stats
            fps = len(self.fps_history) / (time.time() - self.start_time) if time.time() - self.start_time > 0 else 0
            self.fps_history.append(time.time())
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
            
            avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
            
            cv2.putText(
                markers_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                markers_frame,
                f"Detection: {avg_detection_time*1000:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                markers_frame,
                f"Distance: {self.estimated_distance/1000:.2f}m",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display the frame if not in headless mode
            if not self.headless:
                cv2.imshow("ArUco Detection", markers_frame)
            
            # Store for next frame
            self.last_markers_frame = markers_frame
        
        # Process depth frame if available
        if depth_frame is not None:
            # Store for display
            self.last_depth_frame = depth_frame
            
            # Display the frame if not in headless mode
            if not self.headless:
                cv2.imshow("Depth", depth_frame)
        
        # Monitor performance and adjust processing
        if self.performance_mode:
            # If we have enough data, adjust skip_frames based on performance
            if len(self.detection_times) > 30:
                avg_time = sum(self.detection_times) / len(self.detection_times)
                
                # If processing is too slow, increase frame skipping
                if avg_time > 0.1:  # More than 100ms per frame
                    self.skip_frames = min(3, self.skip_frames + 1)
                    print(f"Performance adjustment: Skipping {self.skip_frames} frames")
                elif avg_time < 0.03 and self.skip_frames > 0:  # Less than 30ms per frame
                    self.skip_frames = max(0, self.skip_frames - 1)
                    print(f"Performance adjustment: Skipping {self.skip_frames} frames")
        
        return True
    
    def run(self):
        """
        Run the main processing loop
        """
        if not self.start():
            return
        
        print("Running main processing loop...")
        print("Press 'q' to exit")
        
        try:
            while self.running:
                # Process frame
                self.process_frame()
                
                # Check for key press if not in headless mode
                if not self.headless and cv2.waitKey(1) == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Stop system
            self.stop()
    
    def get_tracked_markers(self) -> Dict[int, Dict]:
        """
        Get information about all tracked markers
        
        Returns:
            Dict mapping marker IDs to information dictionaries
        """
        return self.tracker_manager.get_tracked_markers()
    
    def set_target_id(self, target_id: Optional[int]):
        """
        Set the target marker ID
        
        Args:
            target_id: ID of the target marker to prioritize
        """
        self.target_id = target_id
        self.tracker_manager.set_target_id(target_id)
        print(f"Target marker ID set to: {target_id}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics
        
        Returns:
            Dict containing performance statistics
        """
        # Calculate FPS
        fps = len(self.fps_history) / (time.time() - self.start_time) if time.time() - self.start_time > 0 else 0
        
        # Calculate average detection time
        avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
        
        return {
            "fps": fps,
            "avg_detection_time": avg_detection_time,
            "estimated_distance": self.estimated_distance / 1000  # Convert to meters
        }


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='OAK-D ArUco Detection System')
    
    parser.add_argument('--target', '-t', type=int, help='Target marker ID to highlight')
    parser.add_argument('--resolution', '-r', choices=['low', 'medium', 'high', 'adaptive'],
                      default='adaptive', help='Resolution mode (default: adaptive)')
    parser.add_argument('--cuda', '-c', action='store_true', help='Enable CUDA acceleration if available')
    parser.add_argument('--performance', '-p', action='store_true', help='Enable performance optimizations')
    parser.add_argument('--marker-size', '-m', type=float, default=0.3048,
                      help='Marker size in meters (default: 0.3048m / 12 inches)')
    parser.add_argument('--calib-dir', type=str, default='aruco/camera_calibration',
                      help='Directory for calibration files')
    parser.add_argument('--no-validation', action='store_true',
                      help='Disable marker validation (not recommended)')
    parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode (no GUI)')
    
    return parser.parse_args()


def main():
    """
    Main function
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create system
    system = OakDArUcoSystem(
        target_id=args.target,
        resolution_mode=args.resolution,
        use_cuda=args.cuda,
        performance_mode=args.performance,
        calib_dir=args.calib_dir,
        marker_size=args.marker_size,
        enable_validation=not args.no_validation,
        headless=args.headless
    )
    
    # Run system
    system.run()


if __name__ == "__main__":
    main()