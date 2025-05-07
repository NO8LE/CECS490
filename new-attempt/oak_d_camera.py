#!/usr/bin/env python3

"""
OAK-D Camera Interface Module

This module provides an interface for the OAK-D camera using DepthAI,
configured specifically for ArUco marker detection. It handles the
setup of RGB and depth streams, frame acquisition, and resource management.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any, Union
import depthai as dai


class OakDCamera:
    """
    OAK-D camera interface using DepthAI
    
    This class handles the setup and management of the OAK-D camera
    for ArUco marker detection, providing access to RGB and depth
    streams with proper resource management.
    """
    
    # Resolution profiles
    RESOLUTION_PROFILES = {
        "low": (640, 400),      # For close markers (0.5-3m)
        "medium": (1280, 720),  # For mid-range markers (3-8m)
        "high": (1920, 1080)    # For distant markers (8-12m)
    }
    
    def __init__(self, 
                 resolution_mode: str = "adaptive",
                 enable_spatial: bool = True,
                 enable_depth: bool = True):
        """
        Initialize the OAK-D camera interface
        
        Args:
            resolution_mode: Resolution mode ("low", "medium", "high", or "adaptive")
            enable_spatial: Whether to enable spatial calculations (default: True)
            enable_depth: Whether to enable depth stream (default: True)
        """
        self.resolution_mode = resolution_mode
        self.enable_spatial = enable_spatial
        self.enable_depth = enable_depth
        
        # Get initial resolution based on mode
        if self.resolution_mode == "adaptive":
            self.current_resolution = self.RESOLUTION_PROFILES["medium"]
        else:
            if resolution_mode not in self.RESOLUTION_PROFILES:
                print(f"Warning: Unknown resolution mode '{resolution_mode}'. Using 'medium'.")
                self.current_resolution = self.RESOLUTION_PROFILES["medium"]
            else:
                self.current_resolution = self.RESOLUTION_PROFILES[resolution_mode]
        
        # Initialize DepthAI pipeline
        self.pipeline = None
        self.device = None
        
        # Output queues
        self.rgb_queue = None
        self.depth_queue = None
        self.spatial_calc_queue = None
        self.spatial_calc_config_queue = None
        
        # ROI for spatial location calculation
        self.roi_top_left = dai.Point2f(0.4, 0.4)
        self.roi_bottom_right = dai.Point2f(0.6, 0.6)
        
        # Estimated distance (used for adaptive resolution)
        self.estimated_distance = 5000  # Initial estimate: 5m
        
        # Performance monitoring
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Initialize the pipeline
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """
        Initialize the DepthAI pipeline for the OAK-D camera
        """
        # Create pipeline
        self.pipeline = dai.Pipeline()
        
        # Define sources and outputs
        rgb_cam = self.pipeline.create(dai.node.ColorCamera)
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        
        # Create spatial calculator if enabled
        if self.enable_spatial:
            spatial_calc = self.pipeline.create(dai.node.SpatialLocationCalculator)
        
        # Create outputs
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        
        if self.enable_depth:
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
        
        if self.enable_spatial:
            xout_spatial_data = self.pipeline.create(dai.node.XLinkOut)
            xout_spatial_data.setStreamName("spatial_data")
            xin_spatial_calc_config = self.pipeline.create(dai.node.XLinkIn)
            xin_spatial_calc_config.setStreamName("spatial_calc_config")
        
        # Configure RGB camera
        rgb_cam.setPreviewSize(self.current_resolution[0], self.current_resolution[1])
        rgb_cam.setInterleaved(False)
        rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb_cam.setFps(30)
        
        # Configure mono cameras
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # LEFT
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # RIGHT
        
        # Configure stereo depth
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # RGB
        stereo.setOutputSize(rgb_cam.getPreviewWidth(), rgb_cam.getPreviewHeight())
        stereo.setExtendedDisparity(True)  # Extended disparity for longer range
        
        # Configure spatial calculator if enabled
        if self.enable_spatial:
            spatial_calc.inputConfig.setWaitForMessage(False)
            spatial_calc.inputDepth.setBlocking(False)
            
            # Initial config for spatial location calculator
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 15000  # Increased to 15m for long-range detection
            config.roi = dai.Rect(self.roi_top_left, self.roi_bottom_right)
            spatial_calc.initialConfig.addROI(config)
        
        # Link nodes
        rgb_cam.preview.link(xout_rgb.input)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        if self.enable_depth:
            if self.enable_spatial:
                stereo.depth.link(spatial_calc.inputDepth)
                spatial_calc.passthroughDepth.link(xout_depth.input)
            else:
                stereo.depth.link(xout_depth.input)
        
        if self.enable_spatial:
            spatial_calc.out.link(xout_spatial_data.input)
            xin_spatial_calc_config.out.link(spatial_calc.inputConfig)
    
    def start(self):
        """
        Start the OAK-D camera and initialize resources
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Connect to device
            print("Connecting to OAK-D device...")
            self.device = dai.Device(self.pipeline)
            print("OAK-D device connected successfully")
            
            # Get output queues
            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            if self.enable_depth:
                self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            if self.enable_spatial:
                self.spatial_calc_queue = self.device.getOutputQueue(name="spatial_data", maxSize=4, blocking=False)
                self.spatial_calc_config_queue = self.device.getInputQueue("spatial_calc_config")
            
            print("Camera started successfully")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """
        Stop the OAK-D camera and release resources
        """
        if self.device is not None:
            self.device.close()
            self.device = None
            print("Camera stopped and resources released")
    
    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest RGB frame from the camera
        
        Returns:
            np.ndarray: RGB frame or None if not available
        """
        if self.rgb_queue is None:
            return None
        
        in_rgb = self.rgb_queue.tryGet()
        if in_rgb is not None:
            return in_rgb.getCvFrame()
        return None
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest depth frame from the camera
        
        Returns:
            np.ndarray: Depth frame or None if not available
        """
        if not self.enable_depth or self.depth_queue is None:
            return None
        
        in_depth = self.depth_queue.tryGet()
        if in_depth is not None:
            depth_frame = in_depth.getFrame()
            # Normalize and colorize the depth frame for visualization
            depth_frame_colored = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_colored = cv2.equalizeHist(depth_frame_colored)
            depth_frame_colored = cv2.applyColorMap(depth_frame_colored, cv2.COLORMAP_HOT)
            return depth_frame_colored
        return None
    
    def get_spatial_data(self) -> List:
        """
        Get the latest spatial data from the camera
        
        Returns:
            List: Spatial locations or empty list if not available
        """
        if not self.enable_spatial or self.spatial_calc_queue is None:
            return []
        
        in_spatial_data = self.spatial_calc_queue.tryGet()
        if in_spatial_data is not None:
            return in_spatial_data.getSpatialLocations()
        return []
    
    def update_spatial_calc_roi(self, corners: List, marker_ids: Optional[np.ndarray] = None, target_id: Optional[int] = None):
        """
        Update the ROI for spatial location calculation based on marker position
        
        Args:
            corners: Detected marker corners
            marker_ids: Detected marker IDs
            target_id: Target marker ID to prioritize
        """
        if not self.enable_spatial or len(corners) == 0:
            return
        
        # Find the target marker if specified
        target_index = None
        if target_id is not None and marker_ids is not None:
            for i, marker_id in enumerate(marker_ids):
                if marker_id[0] == target_id:
                    target_index = i
                    break
        
        # Use the target marker if found, otherwise use the first marker
        corner_index = target_index if target_index is not None else 0
        corner = corners[corner_index]
        
        # Calculate the center of the selected marker
        center_x = np.mean([corner[0][0][0], corner[0][1][0], corner[0][2][0], corner[0][3][0]])
        center_y = np.mean([corner[0][0][1], corner[0][1][1], corner[0][2][1], corner[0][3][1]])
        
        # Get frame dimensions
        frame_width, frame_height = self.current_resolution
        
        # Use fixed ROI size in pixels (more stable than distance-based)
        roi_width_pixels = 100
        roi_height_pixels = 100
        
        # Convert to normalized coordinates (0-1 range)
        roi_width = roi_width_pixels / frame_width
        roi_height = roi_height_pixels / frame_height
        
        # Calculate ROI around the marker center with bounds check
        # Ensure ROI is fully within bounds and not too close to edges
        x_normalized = max(roi_width/2, min(1.0 - roi_width/2, center_x / frame_width))
        y_normalized = max(roi_height/2, min(1.0 - roi_height/2, center_y / frame_height))
        
        # Apply exponential smoothing to ROI position if we had a previous position
        if hasattr(self, 'prev_roi_x') and hasattr(self, 'prev_roi_y'):
            # Smoothing factor (0-1), lower value = more smoothing
            alpha = 0.3
            x_normalized = alpha * x_normalized + (1 - alpha) * self.prev_roi_x
            y_normalized = alpha * y_normalized + (1 - alpha) * self.prev_roi_y
        
        # Store for next frame
        self.prev_roi_x = x_normalized
        self.prev_roi_y = y_normalized
        
        # Calculate ROI coordinates with safety margins
        self.roi_top_left = dai.Point2f(
            max(0.001, x_normalized - roi_width/2),
            max(0.001, y_normalized - roi_height/2)
        )
        self.roi_bottom_right = dai.Point2f(
            min(0.999, x_normalized + roi_width/2),
            min(0.999, y_normalized + roi_height/2)
        )
        
        try:
            # Send updated config to the device
            cfg = dai.SpatialLocationCalculatorConfig()
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 15000
            
            # Create rectangle - ensure it's valid
            rect = dai.Rect(self.roi_top_left, self.roi_bottom_right)
            
            # Check if width/height are properties or methods (depends on DepthAI version)
            try:
                # Try as methods first
                width = rect.width()
                height = rect.height()
            except:
                # Fall back to properties if methods don't work
                width = rect.width
                height = rect.height
            
            if width > 0 and height > 0:
                config.roi = rect
                cfg.addROI(config)
                self.spatial_calc_config_queue.send(cfg)
            else:
                print(f"Invalid ROI dimensions: width={width}, height={height}")
        except Exception as e:
            print(f"Error updating spatial calculator ROI: {e}")
    
    def update_resolution(self, estimated_distance: float):
        """
        Update camera resolution based on estimated distance
        
        Args:
            estimated_distance: Estimated distance to markers in mm
        """
        if self.resolution_mode != "adaptive":
            return
        
        # Select resolution based on distance
        if estimated_distance > 8000:  # > 8m
            new_resolution = self.RESOLUTION_PROFILES["high"]
        elif estimated_distance > 3000:  # 3-8m
            new_resolution = self.RESOLUTION_PROFILES["medium"]
        else:  # < 3m
            new_resolution = self.RESOLUTION_PROFILES["low"]
        
        # Check if resolution needs to be updated
        if new_resolution != self.current_resolution:
            print(f"Updating resolution to {new_resolution} based on distance {estimated_distance/1000:.2f}m")
            self.current_resolution = new_resolution
            
            # Reinitialize pipeline with new resolution
            self.stop()
            self.initialize_pipeline()
            self.start()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics
        
        Returns:
            Dict containing performance statistics
        """
        if not self.frame_times:
            return {"avg_frame_time": 0.0, "fps": 0.0}
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "avg_frame_time": avg_time,
            "fps": fps
        }
    
    def update_performance_stats(self):
        """
        Update performance statistics
        """
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)


# Example usage
if __name__ == "__main__":
    # Create camera interface
    camera = OakDCamera(resolution_mode="medium")
    
    # Start camera
    if not camera.start():
        print("Failed to start camera")
        exit(1)
    
    try:
        while True:
            # Get RGB and depth frames
            rgb_frame = camera.get_rgb_frame()
            depth_frame = camera.get_depth_frame()
            
            # Update performance stats
            camera.update_performance_stats()
            
            # Display frames if available
            if rgb_frame is not None:
                # Display performance stats
                stats = camera.get_performance_stats()
                cv2.putText(
                    rgb_frame,
                    f"FPS: {stats['fps']:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("RGB", rgb_frame)
            
            if depth_frame is not None:
                cv2.imshow("Depth", depth_frame)
            
            # Check for key press
            if cv2.waitKey(1) == ord('q'):
                break
    
    finally:
        # Stop camera and release resources
        camera.stop()
        cv2.destroyAllWindows()