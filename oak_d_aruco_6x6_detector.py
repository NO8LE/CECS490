#!/usr/bin/env python3

"""
OAK-D ArUco 6x6 Marker Detector

This script uses the Luxonis OAK-D camera to detect 6x6 ArUco markers,
calculate their 3D position, and visualize the results.

Usage:
  python3 oak_d_aruco_6x6_detector.py

Press 'q' to exit the program.
"""

import os
import sys
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

try:
    import cv2
    # Check if aruco module is available
    if not hasattr(cv2, 'aruco'):
        # Try to import aruco from opencv-contrib-python
        try:
            # This is a workaround for some OpenCV installations
            from cv2 import aruco
            # Make aruco available as cv2.aruco
            cv2.aruco = aruco
        except ImportError:
            print("Error: OpenCV ArUco module not found.")
            print("Please install opencv-contrib-python:")
            print("  pip install opencv-contrib-python")
            sys.exit(1)
except ImportError:
    print("Error: OpenCV (cv2) not found.")
    print("Please install OpenCV:")
    print("  pip install opencv-python opencv-contrib-python")
    sys.exit(1)

try:
    import depthai as dai
except ImportError:
    print("Error: DepthAI module not found.")
    print("Please install DepthAI:")
    print("  pip install depthai")
    sys.exit(1)

# ArUco dictionary to use (6x6 with 250 markers)
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250

# ArUco marker side length in meters
MARKER_SIZE = 0.05  # 5 cm

# Camera calibration directory - create if it doesn't exist
CALIB_DIR = "camera_calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

class OakDArUcoDetector:
    def __init__(self):
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.Dictionary.get(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters.create()
        
        # Camera calibration matrices
        self.camera_matrix = None
        self.dist_coeffs = None
        
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
        
        # Initialize the pipeline
        self.initialize_pipeline()
        
        # Load or create camera calibration
        self.load_camera_calibration()
        
    def load_camera_calibration(self):
        """
        Load camera calibration from file or use default values
        """
        calib_file = os.path.join(CALIB_DIR, "calibration.npz")
        
        if os.path.exists(calib_file):
            # Load calibration from file
            print(f"Loading camera calibration from {calib_file}")
            data = np.load(calib_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
        else:
            # Use default calibration (will be less accurate)
            print("Using default camera calibration")
            # Default calibration for OAK-D RGB camera (approximate values)
            self.camera_matrix = np.array([
                [860.0, 0, 640.0],
                [0, 860.0, 360.0],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.zeros((1, 5))
            
            # Save default calibration
            np.savez(calib_file, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
            print(f"Saved default calibration to {calib_file}")
            print("For better accuracy, consider calibrating your camera")
    
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
        spatial_calc = self.pipeline.create(dai.node.SpatialLocationCalculator)
        
        # Create outputs
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_spatial_data = self.pipeline.create(dai.node.XLinkOut)
        xin_spatial_calc_config = self.pipeline.create(dai.node.XLinkIn)
        
        # Set stream names
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        xout_spatial_data.setStreamName("spatial_data")
        xin_spatial_calc_config.setStreamName("spatial_calc_config")
        
        # Properties
        rgb_cam.setPreviewSize(640, 400)
        rgb_cam.setInterleaved(False)
        rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb_cam.setFps(30)
        
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # StereoDepth configuration
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(rgb_cam.getPreviewWidth(), rgb_cam.getPreviewHeight())
        
        # Spatial location calculator configuration
        spatial_calc.setWaitForConfigInput(False)
        spatial_calc.inputDepth.setBlocking(False)
        
        # Initial config for spatial location calculator
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(self.roi_top_left, self.roi_bottom_right)
        spatial_calc.initialConfig.addROI(config)
        
        # Linking
        rgb_cam.preview.link(xout_rgb.input)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(spatial_calc.inputDepth)
        spatial_calc.out.link(xout_spatial_data.input)
        spatial_calc.passthroughDepth.link(xout_depth.input)
        xin_spatial_calc_config.out.link(spatial_calc.inputConfig)
        
    def start(self):
        """
        Start the OAK-D camera and process frames
        """
        # Connect to device and start pipeline
        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()
        
        # Get output queues
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.spatial_calc_queue = self.device.getOutputQueue(name="spatial_data", maxSize=4, blocking=False)
        self.spatial_calc_config_queue = self.device.getInputQueue("spatial_calc_config")
        
        print("Camera started. Press 'q' to exit.")
        
        # Main loop
        while True:
            # Get camera frames
            rgb_frame = self.get_rgb_frame()
            depth_frame = self.get_depth_frame()
            spatial_data = self.get_spatial_data()
            
            if rgb_frame is not None:
                # Process the frame to detect ArUco markers
                markers_frame, marker_corners, marker_ids = self.detect_aruco_markers(rgb_frame)
                
                # If markers are detected, calculate their 3D position
                if marker_ids is not None:
                    # Update ROI for spatial location calculation based on marker position
                    self.update_spatial_calc_roi(marker_corners)
                    
                    # Draw marker information on the frame
                    self.draw_marker_info(markers_frame, marker_corners, marker_ids, spatial_data)
                
                # Display the frames
                cv2.imshow("RGB", markers_frame)
                
                if depth_frame is not None:
                    cv2.imshow("Depth", depth_frame)
            
            # Check for key press
            if cv2.waitKey(1) == ord('q'):
                break
                
        # Clean up
        cv2.destroyAllWindows()
        self.device.close()
        
    def get_rgb_frame(self):
        """
        Get the latest RGB frame from the camera
        """
        in_rgb = self.rgb_queue.tryGet()
        if in_rgb is not None:
            return in_rgb.getCvFrame()
        return None
        
    def get_depth_frame(self):
        """
        Get the latest depth frame from the camera
        """
        in_depth = self.depth_queue.tryGet()
        if in_depth is not None:
            depth_frame = in_depth.getFrame()
            # Normalize and colorize the depth frame for visualization
            depth_frame_colored = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_colored = cv2.equalizeHist(depth_frame_colored)
            depth_frame_colored = cv2.applyColorMap(depth_frame_colored, cv2.COLORMAP_HOT)
            return depth_frame_colored
        return None
        
    def get_spatial_data(self):
        """
        Get the latest spatial data from the camera
        """
        in_spatial_data = self.spatial_calc_queue.tryGet()
        if in_spatial_data is not None:
            return in_spatial_data.getSpatialLocations()
        return []
        
    def detect_aruco_markers(self, frame):
        """
        Detect ArUco markers in the frame
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.aruco_params
        )
        
        # Draw detected markers on the frame
        markers_frame = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(markers_frame, corners, ids)
            
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 
                MARKER_SIZE, 
                self.camera_matrix, 
                self.dist_coeffs
            )
            
            # Draw axis for each marker
            for i in range(len(ids)):
                cv2.aruco.drawAxis(
                    markers_frame, 
                    self.camera_matrix, 
                    self.dist_coeffs, 
                    rvecs[i], 
                    tvecs[i], 
                    0.03  # axis length
                )
                
                # Calculate Euler angles
                rotation_matrix = cv2.Rodrigues(rvecs[i])[0]
                r = R.from_matrix(rotation_matrix)
                euler_angles = r.as_euler('xyz', degrees=True)
                
                # Display rotation information
                cv2.putText(
                    markers_frame,
                    f"Marker {ids[i][0]}: Rot X: {euler_angles[0]:.1f}, Y: {euler_angles[1]:.1f}, Z: {euler_angles[2]:.1f}",
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        
        return markers_frame, corners, ids
        
    def update_spatial_calc_roi(self, corners):
        """
        Update the ROI for spatial location calculation based on marker position
        """
        if len(corners) > 0:
            # Calculate the center of the first detected marker
            corner = corners[0]
            center_x = np.mean([corner[0][0][0], corner[0][1][0], corner[0][2][0], corner[0][3][0]])
            center_y = np.mean([corner[0][0][1], corner[0][1][1], corner[0][2][1], corner[0][3][1]])
            
            # Calculate ROI around the marker center
            roi_size = 0.05  # Size of ROI relative to image size
            self.roi_top_left = dai.Point2f(
                (center_x - 20) / 640,  # Normalize to 0-1 range
                (center_y - 20) / 400
            )
            self.roi_bottom_right = dai.Point2f(
                (center_x + 20) / 640,
                (center_y + 20) / 400
            )
            
            # Send updated config to the device
            cfg = dai.SpatialLocationCalculatorConfig()
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 10000
            config.roi = dai.Rect(self.roi_top_left, self.roi_bottom_right)
            cfg.addROI(config)
            self.spatial_calc_config_queue.send(cfg)
            
    def draw_marker_info(self, frame, corners, ids, spatial_data):
        """
        Draw marker information on the frame
        """
        if len(spatial_data) > 0 and ids is not None:
            for i, spatial_point in enumerate(spatial_data):
                # Get spatial coordinates
                x = spatial_point.spatialCoordinates.x
                y = spatial_point.spatialCoordinates.y
                z = spatial_point.spatialCoordinates.z
                
                # Draw ROI rectangle
                roi = spatial_point.config.roi
                roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                
                # Display spatial coordinates
                cv2.putText(
                    frame,
                    f"X: {x/1000:.2f} m",
                    (xmin + 10, ymin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Y: {y/1000:.2f} m",
                    (xmin + 10, ymin + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Z: {z/1000:.2f} m",
                    (xmin + 10, ymin + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

def main():
    """
    Main function
    """
    print("Initializing OAK-D ArUco 6x6 Marker Detector...")
    detector = OakDArUcoDetector()
    detector.start()

if __name__ == "__main__":
    main()
