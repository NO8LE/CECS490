#!/usr/bin/env python3

"""
OAK-D ArUco 6x6 Marker Detector for Drone Applications

This script uses the Luxonis OAK-D camera to detect 6x6 ArUco markers,
calculate their 3D position, and visualize the results. It is optimized for
drone-based detection at ranges from 0.5m to 12m on the Jetson Orin Nano.

Usage:
  python3 oak_d_aruco_6x6_detector.py [--target MARKER_ID] [--resolution RESOLUTION]

Options:
  --target, -t MARKER_ID       Specify a target marker ID to highlight
  --resolution, -r RESOLUTION  Specify resolution (low, medium, high) (default: adaptive)
  --cuda, -c                   Enable CUDA acceleration if available
  --performance, -p            Enable high performance mode on Jetson
  --stream, -st                Enable video streaming over RTP/UDP
  --stream-ip IP               IP address to stream to (default: 192.168.2.1)
  --stream-port PORT           Port to stream to (default: 5600)
  --stream-bitrate BITRATE     Streaming bitrate in bits/sec (default: 4000000)
  --headless                   Run in headless mode (no GUI windows)
  --quiet, -q                  Suppress progress messages in the console

Examples:
  python3 oak_d_aruco_6x6_detector.py
  python3 oak_d_aruco_6x6_detector.py --target 5
  python3 oak_d_aruco_6x6_detector.py -t 10 -r high -c -p
  python3 oak_d_aruco_6x6_detector.py --stream --stream-ip 192.168.2.1
  python3 oak_d_aruco_6x6_detector.py --headless --quiet --stream

Press 'q' to exit the program.
"""

import os
import sys
import numpy as np
import time
import argparse
from scipy.spatial.transform import Rotation as R
from collections import deque
import threading

# Import PyMAVLink
try:
    from pymavlink import mavutil
    print("PyMAVLink successfully imported")
except ImportError:
    print("Warning: PyMAVLink not found. MAVLink functionality will be disabled.")
    print("To enable MAVLink servo control, install pymavlink:")
    print("  pip install pymavlink")
    mavutil = None

# Import OpenCV
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("Error: OpenCV (cv2) not found.")
    print("Please install OpenCV:")
    print("  pip install opencv-python opencv-contrib-python")
    sys.exit(1)

# Check for CUDA support
USE_CUDA = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
        USE_CUDA = True
        print("CUDA acceleration enabled")
    else:
        print("No CUDA devices found, using CPU")
except:
    print("CUDA not available in OpenCV, using CPU")

# Import ArUco module - OpenCV 4.10
try:
    print("Using cv2.aruco with OpenCV 4.10")
    aruco = cv2.aruco
except Exception as e:
    print(f"Error importing ArUco module: {str(e)}")
    print("Please ensure opencv-contrib-python is installed correctly")
    sys.exit(1)

# Import DepthAI
try:
    import depthai as dai
    print(f"DepthAI version: {dai.__version__}")
except ImportError:
    print("Error: DepthAI module not found.")
    print("Please install DepthAI:")
    print("  pip install depthai")
    sys.exit(1)

# Add the aruco directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aruco'))

# Import the OpenCV 4.10.0 ArUco fixes
try:
    from opencv410_aruco_fix import (
        create_dictionary_fixed,
        generate_marker_fixed,
        create_charuco_board_fixed,
        generate_charuco_board_image_fixed,
        OpenCV410ArUcoFix
    )
    print("OpenCV 4.10.0 ArUco fixes loaded successfully")
    USE_ARUCO_FIX = True
except ImportError:
    print("Warning: OpenCV 4.10.0 ArUco fixes not found. Using standard methods.")
    USE_ARUCO_FIX = False

# Verify ArUco module is working for OpenCV 4.10
try:
    # Define the dictionary type for 6x6 ArUco markers
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    
    # Use the fixed dictionary if available, otherwise use standard method
    if USE_ARUCO_FIX:
        print("Using fixed ArUco dictionary creation for OpenCV 4.10.0")
        aruco_dict = create_dictionary_fixed(ARUCO_DICT_TYPE)
        dictionary_method = "opencv4.10_fixed"
    else:
        try:
            # Preferred method for OpenCV 4.10
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
            print("Using getPredefinedDictionary for ArUco dictionary")
            dictionary_method = "opencv4.10_predefined"
        except Exception as e:
            print(f"Error with getPredefinedDictionary: {e}")
            # Fallback method
            aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE, 6)
            print("Using Dictionary constructor with markerSize=6")
            dictionary_method = "opencv4.10_constructor"
    
    # Verify by creating a detector
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    print(f"ArUco module successfully loaded and verified for OpenCV {cv2.__version__}")
    
except Exception as e:
    print(f"Error verifying ArUco module: {str(e)}")
    print("Please check your OpenCV installation and version.")
    print(f"OpenCV version: {cv2.__version__}")
    sys.exit(1)

# ArUco marker side length in meters
MARKER_SIZE = 0.3048  # 12 inches (0.3048m)

# Resolution profiles
RESOLUTION_PROFILES = {
    "low": (640, 400),      # For close markers (0.5-3m)
    "medium": (1280, 720),  # For mid-range markers (3-8m)
    "high": (1920, 1080)    # For distant markers (8-12m)
}

# Detection parameter profiles for different distances
DETECTION_PROFILES = {
    "close": {  # 0.5-3m
        "adaptiveThreshConstant": 7,
        "minMarkerPerimeterRate": 0.1,
        "maxMarkerPerimeterRate": 4.0,
        "polygonalApproxAccuracyRate": 0.05,
        "cornerRefinementMethod": 1,  # CORNER_REFINE_SUBPIX
        "cornerRefinementWinSize": 5,
        "cornerRefinementMaxIterations": 30,
        "cornerRefinementMinAccuracy": 0.1,
        "perspectiveRemovePixelPerCell": 4,
        "perspectiveRemoveIgnoredMarginPerCell": 0.13
    },
    "medium": {  # 3-8m
        "adaptiveThreshConstant": 9,
        "minMarkerPerimeterRate": 0.05,
        "maxMarkerPerimeterRate": 4.0,
        "polygonalApproxAccuracyRate": 0.08,
        "cornerRefinementMethod": 1,  # CORNER_REFINE_SUBPIX
        "cornerRefinementWinSize": 5,
        "cornerRefinementMaxIterations": 30,
        "cornerRefinementMinAccuracy": 0.1,
        "perspectiveRemovePixelPerCell": 4,
        "perspectiveRemoveIgnoredMarginPerCell": 0.13
    },
    "far": {  # 8-12m
        "adaptiveThreshConstant": 11,
        "minMarkerPerimeterRate": 0.03,
        "maxMarkerPerimeterRate": 4.0,
        "polygonalApproxAccuracyRate": 0.1,
        "cornerRefinementMethod": 1,  # CORNER_REFINE_SUBPIX
        "cornerRefinementWinSize": 5,
        "cornerRefinementMaxIterations": 30,
        "cornerRefinementMinAccuracy": 0.1,
        "perspectiveRemovePixelPerCell": 4,
        "perspectiveRemoveIgnoredMarginPerCell": 0.13
    }
}

# Directory paths
CALIB_DIR = "aruco/camera_calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

# ArUco marker directory
ARUCO_MARKER_DIR = "aruco/aruco_markers"

# Calibration patterns directory
CALIBRATION_PATTERNS_DIR = "aruco/calibration_patterns"

class MarkerTracker:
    """
    Class for tracking ArUco markers across frames
    """
    def __init__(self, corners, marker_id, timestamp):
        self.marker_id = marker_id
        self.corners = corners
        self.last_seen = timestamp
        self.positions = deque(maxlen=10)  # Store last 10 positions
        self.positions.append((corners, timestamp))
        self.velocity = np.zeros((4, 2))  # Velocity of each corner

    def update(self, corners, timestamp):
        """Update tracker with new detection"""
        dt = timestamp - self.last_seen
        if dt > 0:
            # Calculate velocity
            velocity = (corners - self.corners) / dt
            # Apply exponential smoothing to velocity
            alpha = 0.7
            self.velocity = alpha * velocity + (1 - alpha) * self.velocity

        self.corners = corners
        self.last_seen = timestamp
        self.positions.append((corners, timestamp))

    def predict(self, timestamp):
        """Predict marker position at given timestamp"""
        dt = timestamp - self.last_seen
        if dt > 0.5:  # If not seen for more than 0.5 seconds, prediction may be unreliable
            return None

        # Predict using velocity
        predicted_corners = self.corners + self.velocity * dt
        return predicted_corners

    def is_valid(self, timestamp):
        """Check if tracker is still valid"""
        return (timestamp - self.last_seen) < 1.0  # Valid for 1 second

class OakDArUcoDetector:
    def __init__(self, target_id=None, resolution="adaptive", use_cuda=False, high_performance=False,
                 mavlink_connection=None, enable_servo_control=False, enable_streaming=False,
                 stream_ip="192.168.2.1", stream_port=5600, stream_bitrate=4000000,
                 headless=False, quiet=False):
        # Target marker ID to highlight
        self.target_id = target_id
        if self.target_id is not None:
            print(f"Target marker ID: {self.target_id}")

        # Resolution setting
        self.resolution_mode = resolution
        print(f"Resolution mode: {self.resolution_mode}")

        # CUDA setting
        self.use_cuda = use_cuda and USE_CUDA
        if self.use_cuda:
            print("Using CUDA acceleration")

        # Performance mode for Jetson
        self.high_performance = high_performance
        if self.high_performance:
            self.set_power_mode(True)

        # Headless mode (no GUI)
        self.headless = headless
        if self.headless:
            print("Running in headless mode (no GUI windows)")

        # Quiet mode (suppress progress messages)
        self.quiet = quiet
        if self.quiet:
            print("Running in quiet mode (suppressing progress messages)")

        # MAVLink settings
        self.mavlink_connection = mavlink_connection
        self.enable_servo_control = enable_servo_control
        self.mav_connection = None

        # Initialize MAVLink connection if requested
        if self.enable_servo_control:
            self.setup_mavlink_connection()

        # Video streaming settings
        self.enable_streaming = enable_streaming
        self.stream_ip = stream_ip
        self.stream_port = stream_port
        self.stream_bitrate = stream_bitrate
        self.video_writer = None

        # Print streaming settings if enabled
        if self.enable_streaming:
            print(f"Video streaming enabled to {self.stream_ip}:{self.stream_port}")
            print(f"Streaming bitrate: {self.stream_bitrate} bps")

        # PWM servo values
        self.servo_channels = {
            "top_left": 9,     # Channel for top-left corner servo
            "top_right": 10,   # Channel for top-right corner servo
            "bottom_right": 11, # Channel for bottom-right corner servo
            "bottom_left": 12   # Channel for bottom-left corner servo
        }
        # Default PWM values (mid position)
        self.default_pwm = 1500
        # PWM limits
        self.min_pwm = 1000
        self.max_pwm = 2000

        # Initialize ArUco detector for OpenCV 4.10
        print(f"Initializing ArUco detector for OpenCV {cv2.__version__}")
        
        # Use the fixed dictionary creation if available
        if USE_ARUCO_FIX:
            print("Initializing ArUco detector with OpenCV 4.10.0 fixes")
            self.aruco_dict = create_dictionary_fixed(ARUCO_DICT_TYPE)
        else:
            try:
                # Preferred method for OpenCV 4.10
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
                print("Using getPredefinedDictionary for detector initialization")
            except Exception as e:
                print(f"Error with getPredefinedDictionary: {e}")
                # Fallback method
                self.aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE, 6)
                print("Using Dictionary constructor with markerSize=6")

        # Create and configure detector parameters
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Configure parameters with stricter settings to prevent false positives
        # Adaptive thresholding parameters
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7

        # Contour filtering parameters - these are critical for preventing false detections
        self.aruco_params.minMarkerPerimeterRate = 0.05  # Increase minimum size to prevent small false detections
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03  # More accurate corner detection
        self.aruco_params.minCornerDistanceRate = 0.05  # Minimum distance between corners
        self.aruco_params.minDistanceToBorder = 3  # Minimum distance from borders

        # Corner refinement parameters
        self.aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1

        # Error correction parameters - critical for preventing false positives
        self.aruco_params.errorCorrectionRate = 0.6  # Increase error correction rate (default 0.6)
        self.aruco_params.minOtsuStdDev = 5.0  # Minimum standard deviation for Otsu thresholding
        self.aruco_params.perspectiveRemovePixelPerCell = 4  # Pixels per cell for perspective removal
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13  # Margin for perspective removal

        # Create the detector
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        print("Successfully created ArucoDetector for OpenCV 4.10")

        # Parameters have already been configured in the detector initialization above

        # Set initial detection profile (will be updated based on distance)
        self.current_profile = "medium"
        self.apply_detection_profile(self.current_profile)

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

        # Marker tracking
        self.trackers = {}
        self.last_frame_time = time.time()
        self.estimated_distance = 5000  # Initial estimate: 5m

        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.detection_success_rate = deque(maxlen=30)
        self.skip_frames = 0
        self.frame_count = 0

        # Initialize the pipeline
        self.initialize_pipeline()

        # Load or create camera calibration
        self.load_camera_calibration()

        # Initialize video streaming if enabled
        if self.enable_streaming:
            self.initialize_video_streaming()

    def set_power_mode(self, high_performance=False):
        """
        Set Jetson power mode based on processing needs
        """
        try:
            if high_performance:
                # Maximum performance when actively searching for markers
                os.system("sudo nvpmodel -m 0")  # Max performance mode
                os.system("sudo jetson_clocks")   # Max clock speeds
                print("Set Jetson to high performance mode")
            else:
                # Power saving when idle or markers are consistently tracked
                os.system("sudo nvpmodel -m 1")  # Power saving mode
                print("Set Jetson to power saving mode")
        except Exception as e:
            print(f"Failed to set power mode: {e}")

    def apply_detection_profile(self, profile_name):
        """
        Apply detection parameters based on profile
        """
        if self.aruco_params is None:
            return

        profile = DETECTION_PROFILES[profile_name]
        for param, value in profile.items():
            try:
                setattr(self.aruco_params, param, value)
            except Exception as e:
                print(f"Warning: Could not set parameter {param}: {e}")

        self.current_profile = profile_name
        print(f"Applied detection profile: {profile_name}")

    def select_resolution_profile(self, estimated_distance):
        """
        Select optimal resolution based on estimated distance
        """
        if self.resolution_mode != "adaptive":
            return RESOLUTION_PROFILES[self.resolution_mode]

        if estimated_distance > 8000:  # > 8m
            return RESOLUTION_PROFILES["high"]
        elif estimated_distance > 3000:  # 3-8m
            return RESOLUTION_PROFILES["medium"]
        else:  # < 3m
            return RESOLUTION_PROFILES["low"]

    def monitor_performance(self):
        """
        Monitor and adapt processing based on performance
        """
        # If we don't have enough data yet, return
        if len(self.frame_times) < 10:
            return

        # Calculate average processing time
        avg_time = sum(self.frame_times) / len(self.frame_times)

        # Calculate detection success rate (only if we have data)
        if not self.quiet:
            if len(self.detection_success_rate) > 0:
                success_rate = sum(self.detection_success_rate) / len(self.detection_success_rate)
                print(f"Avg processing time: {avg_time*1000:.1f}ms, Success rate: {success_rate*100:.1f}%")
            else:
                # No detections yet
                print(f"Avg processing time: {avg_time*1000:.1f}ms, No markers detected yet")

        # If processing is too slow, reduce resolution or simplify detection
        if avg_time > 0.1:  # More than 100ms per frame
            self.skip_frames = 1  # Process every other frame
            if not self.quiet:
                print("Performance warning: Processing time > 100ms, skipping frames")
        else:
            self.skip_frames = 0  # Process every frame

    def load_camera_calibration(self):
        """
        Load camera calibration from file or use default values
        """
        # Check calibration file in the aruco subdirectory
        calib_file = os.path.join(CALIB_DIR, "calibration.npz")

        # Also check in the original location (root directory) as fallback
        legacy_calib_file = "camera_calibration/calibration.npz"

        if os.path.exists(calib_file):
            # Load calibration from file in aruco subdirectory
            print(f"Loading camera calibration from {calib_file}")
            data = np.load(calib_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']

            # Check if this is a CharucoBoard calibration
            if 'charuco_calibration' in data:
                print("Detected CharucoBoard calibration data")
                if 'squares_x' in data and 'squares_y' in data:
                    print(f"CharucoBoard: {data['squares_x']}x{data['squares_y']} squares")
        elif os.path.exists(legacy_calib_file):
            # Load from legacy location as fallback
            print(f"Loading camera calibration from legacy location: {legacy_calib_file}")
            try:
                data = np.load(legacy_calib_file)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(calib_file), exist_ok=True)

                # Copy to new location
                np.savez(calib_file, **data)
                print(f"Copied calibration data to new location: {calib_file}")
            except Exception as e:
                print(f"Error loading legacy calibration: {e}")
                self._create_default_calibration(calib_file)
        else:
            # Use default calibration if no file exists
            self._create_default_calibration(calib_file)

    def _create_default_calibration(self, calib_file):
        """
        Create default camera calibration when no calibration file exists
        """
        print("No calibration file found. Using default camera calibration.")
        # Default calibration for OAK-D RGB camera (approximate values)
        self.camera_matrix = np.array([
            [860.0, 0, 640.0],
            [0, 860.0, 360.0],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((1, 5))

        # Ensure directory exists
        os.makedirs(os.path.dirname(calib_file), exist_ok=True)

        # Save default calibration
        try:
            np.savez(calib_file, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
            print(f"Saved default calibration to {calib_file}")
        except Exception as e:
            print(f"Failed to save default calibration: {e}")

        print("For better accuracy, consider calibrating your camera using aruco/calibrate_camera.py")

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

        # Get initial resolution based on mode
        if self.resolution_mode == "adaptive":
            initial_res = RESOLUTION_PROFILES["medium"]
        else:
            initial_res = RESOLUTION_PROFILES[self.resolution_mode]

        # Properties
        rgb_cam.setPreviewSize(initial_res[0], initial_res[1])
        rgb_cam.setInterleaved(False)
        rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb_cam.setFps(30)

        # Note: Camera control initialization removed due to compatibility issues with DepthAI 2.24.0.0
        # The camera will use default auto-exposure settings

        # Use CAM_B and CAM_C instead of deprecated LEFT and RIGHT
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Updated from LEFT
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Updated from RIGHT

        # StereoDepth configuration
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Updated from RGB
        stereo.setOutputSize(rgb_cam.getPreviewWidth(), rgb_cam.getPreviewHeight())

        # Extended disparity for longer range
        stereo.setExtendedDisparity(True)

        # Spatial location calculator configuration
        # Use inputConfig.setWaitForMessage() instead of setWaitForConfigInput
        spatial_calc.inputConfig.setWaitForMessage(False)
        spatial_calc.inputDepth.setBlocking(False)

        # Initial config for spatial location calculator
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 15000  # Increased to 15m for long-range detection
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

    def setup_mavlink_connection(self):
        """
        Initialize the MAVLink connection for servo control
        """
        if mavutil is None:
            print("PyMAVLink not available. Servo control disabled.")
            return

        if not self.mavlink_connection:
            print("No MAVLink connection string provided. Servo control disabled.")
            return

        try:
            print(f"Connecting to MAVLink device at {self.mavlink_connection}")
            self.mav_connection = mavutil.mavlink_connection(self.mavlink_connection)

            # Wait for the first heartbeat to ensure connection is established
            print("Waiting for MAVLink heartbeat...")
            self.mav_connection.wait_heartbeat()
            print(f"MAVLink heartbeat received from system {self.mav_connection.target_system}")

            # Set the system and component ID for sending commands
            self.target_system = self.mav_connection.target_system
            self.target_component = self.mav_connection.target_component

            print("MAVLink connection established successfully")
        except Exception as e:
            print(f"Error setting up MAVLink connection: {e}")
            self.mav_connection = None
            self.enable_servo_control = False

    def send_servo_command(self, channel, pwm_value):
        """
        Send a servo command via MAVLink

        Args:
            channel: Servo channel number (1-16)
            pwm_value: PWM value (typically 1000-2000)
        """
        if not self.enable_servo_control or self.mav_connection is None:
            return

        try:
            # Ensure PWM value is within valid range
            pwm_value = max(self.min_pwm, min(self.max_pwm, int(pwm_value)))

            # Send command using DO_SET_SERVO
            self.mav_connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,  # Confirmation
                channel,  # Servo channel
                pwm_value,  # PWM value
                0, 0, 0, 0, 0  # Unused parameters
            )
            print(f"Sent servo command to channel {channel}: PWM={pwm_value}")
        except Exception as e:
            print(f"Error sending servo command: {e}")

    def calculate_corner_pwm(self, corners):
        """
        Calculate PWM values for servos based on marker corner positions

        Args:
            corners: Array of marker corner coordinates

        Returns:
            Dictionary of channel:pwm_value pairs
        """
        # Check if we have valid corners
        if corners is None or len(corners) < 4:
            return {}

        # Get frame center and dimensions
        h, w = self.last_markers_frame.shape[:2] if hasattr(self, 'last_markers_frame') else (720, 1280)
        center_x, center_y = w/2, h/2

        # Calculate PWM values based on corner positions relative to center
        # The idea is to map screen position to servo angle

        # Extract the four corners (assuming they are in a specific order)
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        # Calculate offsets from center (normalized to -1 to 1)
        tl_offset_x = (top_left[0] - center_x) / (w/2)
        tl_offset_y = (top_left[1] - center_y) / (h/2)

        tr_offset_x = (top_right[0] - center_x) / (w/2)
        tr_offset_y = (top_right[1] - center_y) / (h/2)

        br_offset_x = (bottom_right[0] - center_x) / (w/2)
        br_offset_y = (bottom_right[1] - center_y) / (h/2)

        bl_offset_x = (bottom_left[0] - center_x) / (w/2)
        bl_offset_y = (bottom_left[1] - center_y) / (h/2)

        # Convert offsets to PWM values (1000-2000 range with 1500 as center)
        # Use a combination of x and y offsets to determine servo position
        top_left_pwm = int(self.default_pwm + 500 * (tl_offset_x - tl_offset_y))
        top_right_pwm = int(self.default_pwm + 500 * (tr_offset_x + tr_offset_y))
        bottom_right_pwm = int(self.default_pwm + 500 * (br_offset_x - br_offset_y))
        bottom_left_pwm = int(self.default_pwm + 500 * (bl_offset_x + bl_offset_y))

        # Ensure values are within limits
        top_left_pwm = max(self.min_pwm, min(self.max_pwm, top_left_pwm))
        top_right_pwm = max(self.min_pwm, min(self.max_pwm, top_right_pwm))
        bottom_right_pwm = max(self.min_pwm, min(self.max_pwm, bottom_right_pwm))
        bottom_left_pwm = max(self.min_pwm, min(self.max_pwm, bottom_left_pwm))

        return {
            self.servo_channels["top_left"]: top_left_pwm,
            self.servo_channels["top_right"]: top_right_pwm,
            self.servo_channels["bottom_right"]: bottom_right_pwm,
            self.servo_channels["bottom_left"]: bottom_left_pwm
        }

    def control_servos_from_target(self):
        """
        Control servos based on target marker corners
        """
        if not self.enable_servo_control or self.mav_connection is None:
            return

        if not hasattr(self, 'target_corners') or self.target_corners is None:
            return

        # Calculate PWM values from corner positions
        pwm_values = self.calculate_corner_pwm(self.target_corners)

        # Send servo commands
        for channel, pwm in pwm_values.items():
            self.send_servo_command(channel, pwm)

    def initialize_video_streaming(self):
        """
        Initialize the GStreamer pipeline for H.264 RTP streaming
        """
        try:
            # Get initial resolution based on mode
            if self.resolution_mode == "adaptive":
                width, height = RESOLUTION_PROFILES["medium"]
            else:
                width, height = RESOLUTION_PROFILES[self.resolution_mode]

            # List of pipeline configurations to try in order (from most to least complex)
            pipeline_options = [
                # Option 1: Standard x264enc with optimized parameters
                (f"appsrc ! video/x-raw,format=BGR ! videoconvert ! "
                 f"x264enc bitrate={int(self.stream_bitrate/1000)} speed-preset=ultrafast tune=zerolatency ! "
                 f"h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host={self.stream_ip} port={self.stream_port} sync=false"),

                # Option 2: x264enc with minimal parameters
                (f"appsrc ! video/x-raw,format=BGR ! videoconvert ! "
                 f"x264enc bitrate={int(self.stream_bitrate/1000)} ! "
                 f"rtph264pay ! udpsink host={self.stream_ip} port={self.stream_port}"),

                # Option 3: Try with openh264enc instead
                (f"appsrc ! video/x-raw,format=BGR ! videoconvert ! "
                 f"openh264enc bitrate={int(self.stream_bitrate/1000)} ! "
                 f"rtph264pay ! udpsink host={self.stream_ip} port={self.stream_port}"),

                # Option 4: Alternative x264enc pipeline structure
                (f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
                 f"x264enc bitrate={int(self.stream_bitrate/1000)} ! "
                 f"rtph264pay ! udpsink host={self.stream_ip} port={self.stream_port}"),

                # Option 5: JPEG encoding (lower quality but higher compatibility)
                (f"appsrc ! videoconvert ! jpegenc ! "
                 f"rtpjpegpay ! udpsink host={self.stream_ip} port={self.stream_port}"),

                # Option 6: Jetson-specific pipeline (uses omxh264enc if available)
                (f"appsrc ! videoconvert ! omxh264enc bitrate={int(self.stream_bitrate/1000)} ! "
                 f"h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host={self.stream_ip} port={self.stream_port} sync=false"),

                # Option 7: Fallback v4l2 hardware encoder (for Jetson/SBCs)
                (f"appsrc ! videoconvert ! v4l2h264enc ! "
                 f"h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host={self.stream_ip} port={self.stream_port} sync=false"),

                # Option 8: Simplest pipeline, raw video (high bandwidth but maximum compatibility)
                (f"appsrc ! videoconvert ! rtpvrawpay ! udpsink host={self.stream_ip} port={self.stream_port}")
            ]

            # First, check if required GStreamer elements are available
            if not self.quiet:
                print("Checking available GStreamer encoders...")
                os.system("gst-inspect-1.0 x264enc >/dev/null 2>&1 || echo 'x264enc not available'")
                os.system("gst-inspect-1.0 openh264enc >/dev/null 2>&1 || echo 'openh264enc not available'")
                os.system("gst-inspect-1.0 omxh264enc >/dev/null 2>&1 || echo 'omxh264enc not available'")
                os.system("gst-inspect-1.0 v4l2h264enc >/dev/null 2>&1 || echo 'v4l2h264enc not available'")

            # Try each pipeline option until one works
            self.video_writer = None
            for i, pipeline in enumerate(pipeline_options):
                try:
                    if not self.quiet:
                        print(f"Trying pipeline option {i+1}...")

                    # Initialize the VideoWriter with the current GStreamer pipeline
                    writer = cv2.VideoWriter(
                        pipeline,
                        cv2.CAP_GSTREAMER,
                        0,  # Codec is ignored when using GStreamer
                        30.0,  # Target 30 FPS
                        (width, height)
                    )

                    # Check if VideoWriter was successfully initialized
                    if writer.isOpened():
                        self.video_writer = writer
                        if not self.quiet:
                            print(f"Success! Using pipeline option {i+1}")
                        break
                    else:
                        if not self.quiet:
                            print(f"Pipeline option {i+1} failed")
                except Exception as e:
                    if not self.quiet:
                        print(f"Error with pipeline option {i+1}: {e}")

            # Check if any pipeline option worked
            if self.video_writer is None or not self.video_writer.isOpened():
                print("Failed to open video writer pipeline. Streaming will be disabled.")
                print("Make sure GStreamer is properly installed with encoder support.")

                # Detailed diagnostics
                print("\nDiagnostic information:")
                print("Available GStreamer plugins on your system:")
                os.system("gst-inspect-1.0 | grep -E 'x264|h264|enc|jpegenc|v4l2' 2>&1")
                print("\nChecking GStreamer installation:")
                os.system("which gst-launch-1.0 2>&1 || echo 'GStreamer not found in PATH'")
                print("\nGStreamer version:")
                os.system("gst-launch-1.0 --version 2>&1 || echo 'Cannot determine GStreamer version'")
                print("\nIf you need to install GStreamer plugins:")
                print("sudo apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly")

                self.enable_streaming = False
                self.video_writer = None
            else:
                if not self.quiet:
                    print("Video streaming pipeline initialized successfully")
                    print(f"Streaming {width}x{height} @ 30fps to {self.stream_ip}:{self.stream_port}")

                    # Print SDP information for client playback
                    print("\nTo play the stream on the client (GCS), create a file named stream.sdp with these contents:")
                    print("c=IN IP4 0.0.0.0")
                    print(f"m=video {self.stream_port} RTP/AVP 96")
                    print("a=rtpmap:96 H264/90000")
                    print("\nThen use VLC to open this file, or use ffplay with:")
                    print(f"ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental \"udp://@:{self.stream_port}?buffer_size=120000\"")

        except Exception as e:
            print(f"Error initializing video streaming: {e}")
            self.enable_streaming = False
            self.video_writer = None

    def start(self):
        """
        Start the OAK-D camera and process frames
        """
        print("=" * 50)
        print("STARTING CAMERA INITIALIZATION")
        print("=" * 50)

        try:
            # Connect to device
            print("Connecting to OAK-D device...")
            try:
                self.device = dai.Device()
                print("OAK-D device connected successfully")
            except Exception as e:
                print(f"Failed to connect to OAK-D device: {e}")
                print("Please ensure the device is connected and properly recognized by the system")
                return

            # Start pipeline
            print("Starting pipeline...")
            try:
                self.device.startPipeline(self.pipeline)
                print("Pipeline started successfully")
            except Exception as e:
                print(f"Failed to start pipeline: {e}")
                print("Pipeline configuration may be incompatible with the device")
                self.device.close()
                return

            # Get output queues
            print("Setting up I/O queues...")
            try:
                self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                self.spatial_calc_queue = self.device.getOutputQueue(name="spatial_data", maxSize=4, blocking=False)
                self.spatial_calc_config_queue = self.device.getInputQueue("spatial_calc_config")
                print("I/O queues set up successfully")
            except Exception as e:
                print(f"Failed to set up I/O queues: {e}")
                self.device.close()
                return

            print("Camera initialization completed successfully!")
            print("=" * 50)

            if self.headless:
                print("Camera started in headless mode. Use Ctrl+C to exit.")
            else:
                print("Camera started. Press 'q' to exit.")
        except Exception as e:
            print(f"ERROR: Unexpected error during camera initialization: {str(e)}")
            print("=" * 50)
            return

        # Main loop
        running = True
        while running:
            start_time = time.time()

            # Get camera frames
            rgb_frame = self.get_rgb_frame()
            depth_frame = self.get_depth_frame()
            spatial_data = self.get_spatial_data()

            # Skip frame processing if needed (for performance)
            self.frame_count += 1
            if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
                # Still display the last processed frame if not in headless mode
                if not self.headless:
                    if hasattr(self, 'last_markers_frame'):
                        cv2.imshow("RGB", self.last_markers_frame)

                    if depth_frame is not None:
                        cv2.imshow("Depth", depth_frame)

                    # Check for key press
                    if cv2.waitKey(1) == ord('q'):
                        running = False

                continue

            if rgb_frame is not None:
                # Update detection parameters based on estimated distance
                if self.estimated_distance > 8000:  # > 8m
                    if self.current_profile != "far":
                        self.apply_detection_profile("far")
                elif self.estimated_distance > 3000:  # 3-8m
                    if self.current_profile != "medium":
                        self.apply_detection_profile("medium")
                else:  # < 3m
                    if self.current_profile != "close":
                        self.apply_detection_profile("close")

                # Process the frame to detect ArUco markers - disable multi-scale detection to reduce lag
                markers_frame, marker_corners, marker_ids = self.detect_aruco_markers(rgb_frame, simple_detection=True)
                self.last_markers_frame = markers_frame

                # Stream the annotated frame if streaming is enabled
                if self.enable_streaming and self.video_writer is not None:
                    try:
                        self.video_writer.write(markers_frame)
                    except Exception as e:
                        print(f"Error streaming frame: {e}")

                # Update detection success rate - but don't actually use it for performance monitoring
                if False:  # Disabled to prevent lag
                    self.detection_success_rate.append(1.0 if marker_ids is not None and len(marker_ids) > 0 else 0.0)

                # Update marker trackers
                current_time = time.time()
                if marker_ids is not None:
                    for i, marker_id in enumerate(marker_ids):
                        marker_id_val = marker_id[0]
                        if marker_id_val in self.trackers:
                            # Update existing tracker
                            self.trackers[marker_id_val].update(marker_corners[i][0], current_time)
                        else:
                            # Create new tracker
                            self.trackers[marker_id_val] = MarkerTracker(marker_corners[i][0], marker_id_val, current_time)

                # Clean up old trackers
                for marker_id in list(self.trackers.keys()):
                    if not self.trackers[marker_id].is_valid(current_time):
                        del self.trackers[marker_id]

                # If markers are detected, calculate their 3D position
                if marker_ids is not None:
                    # Update ROI for spatial location calculation based on marker position
                    self.update_spatial_calc_roi(marker_corners)

                    # Draw marker information on the frame
                    self.draw_marker_info(markers_frame, marker_corners, marker_ids, spatial_data)

                    # Update estimated distance from spatial data
                    if len(spatial_data) > 0:
                        # Get average distance of all detected markers
                        total_z = 0
                        count = 0
                        for spatial_point in spatial_data:
                            z = spatial_point.spatialCoordinates.z
                            if 100 < z < 20000:  # Valid range: 0.1m to 20m
                                total_z += z
                                count += 1

                        if count > 0:
                            # Update estimated distance with exponential smoothing
                            new_distance = total_z / count
                            alpha = 0.3  # Smoothing factor
                            self.estimated_distance = alpha * new_distance + (1 - alpha) * self.estimated_distance

                # Display the frames (only if not in headless mode)
                if not self.headless:
                    cv2.putText(
                        markers_frame,
                        f"Est. Distance: {self.estimated_distance/1000:.2f}m",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        markers_frame,
                        f"Profile: {self.current_profile}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    cv2.imshow("RGB", markers_frame)

                    if depth_frame is not None:
                        cv2.imshow("Depth", depth_frame)

            # Calculate frame processing time
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)

            # Periodically monitor performance, but only after we have enough frames
            if self.frame_count >= 30 and self.frame_count % 30 == 0:
                self.monitor_performance()

            # Check for key press if not in headless mode
            if not self.headless and cv2.waitKey(1) == ord('q'):
                running = False

        # Clean up
        if not self.headless:
            cv2.destroyAllWindows()

        # Release video writer if streaming was enabled
        if self.enable_streaming and self.video_writer is not None:
            print("Closing video streaming pipeline...")
            self.video_writer.release()

        self.device.close()

        # Reset Jetson power mode if it was changed
        if self.high_performance:
            self.set_power_mode(False)

    def preprocess_image(self, frame):
        """
        Preprocess image to improve marker detection
        """
        if self.use_cuda and USE_CUDA:
            return self.preprocess_image_gpu(frame)
        else:
            return self.preprocess_image_cpu(frame)

    def preprocess_image_cpu(self, frame):
        """
        Preprocess image on CPU
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray

    def preprocess_image_gpu(self, frame):
        """
        Preprocess image on GPU using CUDA
        """
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Convert to grayscale on GPU
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization on GPU
        gpu_gray = cv2.cuda.equalizeHist(gpu_gray)

        # Download result
        gray = gpu_gray.download()
        return gray

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

    def detect_aruco_markers(self, frame, simple_detection=False):
        """
        Detect ArUco markers and CharucoBoard in the frame for OpenCV 4.10

        Args:
            frame: Input image
            simple_detection: If True, use a simplified detection approach to reduce lag
        """
        # Preprocess image - don't use CLAHE in simple mode (faster)
        if simple_detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.preprocess_image(frame)

        # Detect ArUco markers
        corners = []
        ids = None
        rejected = []

        # Using OpenCV 4.10 ArucoDetector API
        try:
            # Make sure we have a detector
            if not hasattr(self, 'aruco_detector') or self.aruco_detector is None:
                # Create detector - use fixed dictionary if available
                if USE_ARUCO_FIX:
                    try:
                        self.aruco_dict = create_dictionary_fixed(ARUCO_DICT_TYPE)
                        print("Using fixed ArUco dictionary creation for detection (OpenCV 4.10.0 fix)")
                    except Exception as e:
                        print(f"Error with fixed dictionary creation: {e}")
                        # Fallback to standard method
                        self.aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE, 6)
                        print("Falling back to Dictionary constructor with markerSize")
                else:
                    try:
                        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
                        print("Using getPredefinedDictionary for detection (primary method)")
                    except Exception as e:
                        print(f"Error with getPredefinedDictionary: {e}")
                        # Fallback to Dictionary constructor
                        self.aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE, 6)
                        print("Using Dictionary constructor with markerSize (fallback)")

                self.aruco_params = cv2.aruco.DetectorParameters()

                # Configure detector parameters for better detection
                self.aruco_params.adaptiveThreshWinSizeMin = 3
                self.aruco_params.adaptiveThreshWinSizeMax = 23
                self.aruco_params.adaptiveThreshWinSizeStep = 10
                self.aruco_params.adaptiveThreshConstant = 7
                self.aruco_params.minMarkerPerimeterRate = 0.01  # Reduced from 0.03 to detect smaller markers
                self.aruco_params.maxMarkerPerimeterRate = 4.0
                self.aruco_params.polygonalApproxAccuracyRate = 0.1  # Increased from 0.05 for better detection
                self.aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX

                # Add error correction parameters
                self.aruco_params.errorCorrectionRate = 0.8  # Increased for better detection

                # Additional parameters for better detection
                if hasattr(self.aruco_params, 'minCornerDistanceRate'):
                    self.aruco_params.minCornerDistanceRate = 0.03  # Relaxed from default
                if hasattr(self.aruco_params, 'minDistanceToBorder'):
                    self.aruco_params.minDistanceToBorder = 1  # Reduced to detect markers near borders

                # Create detector
                self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

            # Use the detector
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

            # Convert corners to the format expected by the rest of the code
            # In OpenCV 4.10, corners might be returned in a different format
            if ids is not None and len(ids) > 0:
                # Ensure corners is a list of arrays with shape (1, 4, 2)
                if not isinstance(corners, list):
                    corners_list = []
                    for i in range(len(ids)):
                        corners_list.append(corners[i].reshape(1, 4, 2))
                    corners = corners_list

                # Apply validation checks to filter out false detections
                valid_indices = []
                valid_corners = []
                valid_ids = []

                # Filter out false positives
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    marker_corners = corners[i]

                    # Check 1: Verify marker has a valid perimeter (minimum size)
                    perimeter = cv2.arcLength(marker_corners[0], True)
                    min_perimeter = gray.shape[0] * 0.01  # Reduced from 4% to 1% of image height

                    # Check 2: Verify corner angles (should be close to 90 degrees)
                    angles_valid = True
                    for j in range(4):
                        p1 = marker_corners[0][j]
                        p2 = marker_corners[0][(j+1) % 4]
                        p3 = marker_corners[0][(j+2) % 4]

                        # Calculate vectors
                        v1 = p1 - p2
                        v2 = p3 - p2

                        # Calculate angle
                        dot = np.sum(v1 * v2)
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)

                        # Avoid division by zero
                        if norm1 > 0 and norm2 > 0:
                            cos_angle = dot / (norm1 * norm2)
                            # Limit to valid range due to numerical errors
                            cos_angle = max(-1.0, min(1.0, cos_angle))
                            angle = np.abs(np.arccos(cos_angle) * 180 / np.pi)

                            # In a perfect square, opposite corners should have angle close to 90 degrees
                            # Much more relaxed angle validation (from 35 to 50 degrees deviation)
                            if abs(angle - 90) > 50:
                                angles_valid = False
                                break
                        else:
                            angles_valid = False
                            break

                    # Check 3: Verify marker has a reasonable aspect ratio
                    width = np.linalg.norm(marker_corners[0][0] - marker_corners[0][1])
                    height = np.linalg.norm(marker_corners[0][1] - marker_corners[0][2])

                    if width > 0 and height > 0:
                        aspect_ratio = max(width/height, height/width)
                        aspect_valid = aspect_ratio < 4.0  # Increased from 2.5 to allow more distorted views
                    else:
                        aspect_valid = False

                    # Consider the marker valid if it passes all checks
                    # For debugging, print information about the marker validation
                    if marker_id < 10:  # Only print for markers with low IDs to avoid spam
                        print(f"Marker ID {marker_id} validation: perimeter={perimeter:.1f} (min={min_perimeter:.1f}), angles_valid={angles_valid}, aspect_ratio={aspect_ratio:.2f} (valid={aspect_valid})")

                    # More relaxed validation - only require perimeter check
                    if perimeter >= min_perimeter:
                        valid_indices.append(i)
                        valid_corners.append(corners[i])
                        valid_ids.append(ids[i])

                # Update with validated markers only
                if len(valid_indices) > 0:
                    corners = valid_corners
                    ids = np.array(valid_ids)
                    print(f"Detected {len(ids)} valid markers out of {len(corners_list)} candidates")
                else:
                    # No valid markers found
                    corners = []
                    ids = np.array([])
                    print("All detected markers were filtered out as false positives")
            else:
                # Initialize empty arrays if no markers detected
                corners = []
                ids = np.array([])
        except Exception as e:
            print(f"Error in ArucoDetector: {str(e)}")
            corners = []
            ids = None

        # Try to detect CharucoBoard only if not in simple detection mode
        charuco_corners = None
        charuco_ids = None
        if not simple_detection and ids is not None and len(ids) >= 4:
            # Check if this looks like a CharucoBoard pattern
            try:
                # Create dictionary for CharucoBoard
                try:
                    charuco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
                    print("Using getPredefinedDictionary for CharucoBoard detection")
                except Exception as e:
                    print(f"Error with getPredefinedDictionary for CharucoBoard: {e}")
                    # Fallback to Dictionary constructor
                    charuco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE, 6)
                    print("Using Dictionary constructor for CharucoBoard detection (fallback)")

                # Create CharucoBoard with the constructor or fixed method
                if USE_ARUCO_FIX:
                    charuco_board = create_charuco_board_fixed(
                        squares_x=6, 
                        squares_y=6, 
                        square_length=0.3048/6, 
                        marker_length=0.3048/6*0.75, 
                        dict_type=ARUCO_DICT_TYPE
                    )
                    print("Using fixed CharucoBoard creation for OpenCV 4.10.0")
                else:
                    charuco_board = cv2.aruco.CharucoBoard(
                        (6, 6),  # (squaresX, squaresY) as a tuple
                        0.3048/6,  # squareLength: Board is 12 inches divided into 6 squares
                        0.3048/6*0.75,  # markerLength: Markers are 75% of square size
                        charuco_dict
                    )

                # Use the CharucoDetector for OpenCV 4.10
                charuco_params = cv2.aruco.CharucoParameters()
                charuco_detector = cv2.aruco.CharucoDetector(charuco_board, charuco_params)

                # Detect the board
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

                # Check if we have enough corners
                ret = charuco_corners is not None and len(charuco_corners) > 4
                if ret:
                    print(f"Detected CharucoBoard with {len(charuco_corners)} corners using CharucoDetector")
                else:
                    charuco_corners = None
                    charuco_ids = None
            except Exception as e:
                print(f"Error detecting CharucoBoard: {e}")
                charuco_corners = None
                charuco_ids = None

        # Skip multi-scale detection in simple mode to reduce lag
        if simple_detection:
            # Skip secondary detection methods
            pass
        else:
            # If no markers found and we're looking for distant markers, try with different parameters
            if (ids is None or len(ids) == 0) and self.current_profile == "far":
                # Try with enhanced parameters for distant detection
                backup_params = self.aruco_params
                try:
                    enhanced_params = cv2.aruco.DetectorParameters.create()
                    enhanced_params.adaptiveThreshConstant = 15
                    enhanced_params.minMarkerPerimeterRate = 0.02
                    enhanced_params.polygonalApproxAccuracyRate = 0.12

                    corners, ids, rejected = cv2.aruco.detectMarkers(
                        gray,
                        self.aruco_dict,
                        parameters=enhanced_params
                    )
                except:
                    # Restore original parameters
                    self.aruco_params = backup_params

            # If still no markers found, try with a scaled version of the image
            if ids is None or len(ids) == 0:
                # Try with 75% scale
                h, w = gray.shape
                scaled_gray = cv2.resize(gray, (int(w*0.75), int(h*0.75)))
                scaled_corners, scaled_ids, _ = cv2.aruco.detectMarkers(
                    scaled_gray,
                    self.aruco_dict,
                    parameters=self.aruco_params
                )

                # If markers found in scaled image, convert coordinates back to original scale
                if scaled_ids is not None and len(scaled_ids) > 0:
                    scale_factor = 1/0.75
                    for i in range(len(scaled_corners)):
                        scaled_corners[i][0][:, 0] *= scale_factor
                        scaled_corners[i][0][:, 1] *= scale_factor
                    corners = scaled_corners
                    ids = scaled_ids

        # Draw detected markers on the frame
        markers_frame = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(markers_frame, corners, ids)

            # If CharucoBoard corners were detected, draw them and estimate pose
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 4:
                # Draw the Charuco corners
                cv2.aruco.drawDetectedCornersCharuco(markers_frame, charuco_corners, charuco_ids)

                try:
                    # Create a CharucoBoard object for pose estimation using OpenCV 4.10
                    # Create or reuse the CharucoBoard from earlier
                    if USE_ARUCO_FIX:
                        charuco_board = create_charuco_board_fixed(
                            squares_x=6, 
                            squares_y=6, 
                            square_length=0.3048/6, 
                            marker_length=0.3048/6*0.75, 
                            dict_type=ARUCO_DICT_TYPE
                        )
                    else:
                        charuco_board = cv2.aruco.CharucoBoard(
                            (6, 6),  # (squaresX, squaresY) as a tuple
                            0.3048/6,  # squareLength: Board is 12 inches divided into 6 squares
                            0.3048/6*0.75,  # markerLength: Markers are 75% of square size
                            charuco_dict
                        )

                    # For OpenCV 4.10, use solvePnP directly
                    # Create object points from the CharucoBoard
                    objPoints = []
                    imgPoints = []

                    # Make sure we have valid charuco_corners and charuco_ids
                    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 4:
                        # Get the 3D coordinates of the corners from the board
                        for i in range(len(charuco_ids)):
                            corner_id = charuco_ids[i][0]
                            # Calculate 3D position based on square size
                            row = corner_id // 7  # For a 6x6 board, there are 7 corners in each direction
                            col = corner_id % 7
                            objPoints.append([col * 0.3048/6, row * 0.3048/6, 0])
                            imgPoints.append(charuco_corners[i][0])

                        # Convert to numpy arrays
                        objPoints = np.array(objPoints, dtype=np.float32)
                        imgPoints = np.array(imgPoints, dtype=np.float32)

                        # Use solvePnP to get pose
                        retval, charuco_rvec, charuco_tvec = cv2.solvePnP(
                            objPoints,
                            imgPoints,
                            self.camera_matrix,
                            self.dist_coeffs
                        )
                    else:
                        retval = False
                        charuco_rvec = None
                        charuco_tvec = None

                    if retval:
                        # Draw the CharucoBoard axes
                        # Try to use drawAxis for OpenCV 4.10
                        try:
                            cv2.aruco.drawAxis(
                                markers_frame,
                                self.camera_matrix,
                                self.dist_coeffs,
                                charuco_rvec,
                                charuco_tvec,
                                0.2  # Larger axis length for the board
                            )
                        except AttributeError:
                            # Fallback for OpenCV 4.10 where drawAxis might be moved
                            try:
                                # Try using cv2.drawFrameAxes instead (newer OpenCV versions)
                                cv2.drawFrameAxes(
                                    markers_frame,
                                    self.camera_matrix,
                                    self.dist_coeffs,
                                    charuco_rvec,
                                    charuco_tvec,
                                    0.2  # Larger axis length for the board
                                )
                            except Exception as axis_err:
                                # If all visualization methods fail, just draw a circle at the center
                                if charuco_corners is not None and len(charuco_corners) > 0:
                                    center = np.mean(charuco_corners, axis=0).astype(int)
                                    cv2.circle(markers_frame, tuple(center[0]), 20, (255, 0, 255), 2)
                                print(f"Using fallback visualization for CharucoBoard - axis drawing failed: {axis_err}")

                        # Calculate distance to CharucoBoard (from tvec)
                        charuco_distance = np.linalg.norm(charuco_tvec) * 1000  # Convert to mm

                        # Draw CharucoBoard distance info
                        cv2.putText(
                            markers_frame,
                            f"ChArUco Distance: {charuco_distance/1000:.2f}m",
                            (10, markers_frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 255),  # Magenta
                            2
                        )

                        # Add caption for ChArUco detection
                        cv2.putText(
                            markers_frame,
                            "ChArUco Board Detected",
                            (10, markers_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 255),  # Magenta
                            2
                        )
                except Exception as e:
                    print(f"Error estimating ChArUco board pose: {e}")

            # Estimate pose of each individual marker using OpenCV 4.10's solvePnP
            try:
                # Ensure we have valid calibration data
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    # For OpenCV 4.10, use solvePnP approach
                    rvecs = []
                    tvecs = []

                    # Process each marker individually
                    for i in range(len(corners)):
                        # Create object points for a square marker
                        objPoints = np.array([
                            [-MARKER_SIZE/2, MARKER_SIZE/2, 0],
                            [MARKER_SIZE/2, MARKER_SIZE/2, 0],
                            [MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                            [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
                        ], dtype=np.float32)

                        # Get image points from corners
                        imgPoints = corners[i][0].astype(np.float32)

                        # Use solvePnP to get pose
                        success, rvec, tvec = cv2.solvePnP(
                            objPoints,
                            imgPoints,
                            self.camera_matrix,
                            self.dist_coeffs
                        )

                        if success:
                            rvecs.append(rvec)
                            tvecs.append(tvec)

                    # Convert to numpy arrays with the same shape as the original function
                    if len(rvecs) > 0:
                        rvecs = np.array(rvecs)
                        tvecs = np.array(tvecs)
                    else:
                        rvecs = None
                        tvecs = None

                    # Check for valid pose data
                    if rvecs is not None and tvecs is not None:
                        # Draw axis for each marker (with smaller axis length to reduce visual clutter)
                        for i in range(len(ids)):
                            # Calculate distance to decide if we should show axes
                            distance = np.linalg.norm(tvecs[i]) * 1000  # in mm
                            max_display_distance = 10000  # Only show axes for markers within 10m

                            # Use distance to determine axis length (smaller at greater distances)
                            axis_length = max(0.05, min(0.15, 0.2 * (1 - distance/15000)))

                            # Only display axes for closer markers to reduce visual clutter
                            if distance < max_display_distance:
                                # Apply smoothing to rotation vectors to reduce jitter
                                if hasattr(self, 'prev_rvecs') and len(self.prev_rvecs) > i:
                                    smooth_factor = 0.7  # Higher values mean less smoothing
                                    rvecs[i] = smooth_factor * rvecs[i] + (1 - smooth_factor) * self.prev_rvecs[i]

                                # Try to use drawAxis - handle API changes in OpenCV 4.10
                                try:
                                    cv2.aruco.drawAxis(
                                        markers_frame,
                                        self.camera_matrix,
                                        self.dist_coeffs,
                                        rvecs[i],
                                        tvecs[i],
                                        axis_length
                                    )
                                except AttributeError:
                                    # Fallback for OpenCV 4.10 where drawAxis might be moved
                                    try:
                                        # Try using cv2.drawFrameAxes instead (newer OpenCV versions)
                                        cv2.drawFrameAxes(
                                            markers_frame,
                                            self.camera_matrix,
                                            self.dist_coeffs,
                                            rvecs[i],
                                            tvecs[i],
                                            axis_length
                                        )
                                    except Exception as axis_err:
                                        # If all visualization methods fail, just draw a circle at the marker center
                                        center = np.mean(corners[i][0], axis=0).astype(int)
                                        cv2.circle(markers_frame, tuple(center), int(axis_length * 10), (0, 255, 0), 2)
                                        if i == 0:  # Only print once to avoid spam
                                            print(f"Using fallback visualization (circle) - axis drawing failed: {axis_err}")

                                # Calculate Euler angles
                                rotation_matrix = cv2.Rodrigues(rvecs[i])[0]
                                r = R.from_matrix(rotation_matrix)
                                euler_angles = r.as_euler('xyz', degrees=True)

                                # Display rotation information (only for a limited number of markers)
                                if i < 3:  # Limit to 3 markers to avoid cluttering the display
                                    cv2.putText(
                                        markers_frame,
                                        f"ID {ids[i][0]}: Rot X: {euler_angles[0]:.1f}, Y: {euler_angles[1]:.1f}, Z: {euler_angles[2]:.1f}",
                                        (10, 90 + i * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 0),
                                        2
                                    )

                        # Store current rotation vectors for next frame's smoothing
                        self.prev_rvecs = rvecs.copy()
                    else:
                        print("Invalid pose data returned")
                else:
                    print("Camera not calibrated, cannot estimate pose")
            except Exception as e:
                print(f"Error estimating pose: {e}")

        return markers_frame, corners, ids

    def update_spatial_calc_roi(self, corners, marker_ids=None):
        """
        Update the ROI for spatial location calculation based on marker position
        Prioritizes the target marker if specified
        """
        if len(corners) == 0:
            return

        # Find the target marker if specified
        target_index = None
        if self.target_id is not None and marker_ids is not None:
            for i, marker_id in enumerate(marker_ids):
                if marker_id[0] == self.target_id:
                    target_index = i
                    break

        # Use the target marker if found, otherwise use the first marker
        corner_index = target_index if target_index is not None else 0
        corner = corners[corner_index]

        # Calculate the center of the selected marker
        center_x = np.mean([corner[0][0][0], corner[0][1][0], corner[0][2][0], corner[0][3][0]])
        center_y = np.mean([corner[0][0][1], corner[0][1][1], corner[0][2][1], corner[0][3][1]])

        # Get frame dimensions from initial resolution
        if self.resolution_mode == "adaptive":
            frame_width, frame_height = RESOLUTION_PROFILES["medium"]
        else:
            frame_width, frame_height = RESOLUTION_PROFILES[self.resolution_mode]

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

        # Print ROI info for debugging
        print(f"ROI: ({self.roi_top_left.x:.3f}, {self.roi_top_left.y:.3f}) to ({self.roi_bottom_right.x:.3f}, {self.roi_bottom_right.y:.3f})")

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

        # If this is the target marker, store its information
        if target_index is not None:
            self.target_found = True
            self.target_center = (center_x, center_y)
            self.target_corners = corner[0]
        else:
            self.target_found = False

    def draw_marker_info(self, frame, corners, ids, spatial_data):
        """
        Draw marker information on the frame
        Prioritizes the target marker if specified
        """
        if len(spatial_data) == 0 or ids is None:
            return frame

        # Create a mapping from marker IDs to spatial data
        marker_to_spatial = {}
        for i, marker_id in enumerate(ids):
            marker_id_val = marker_id[0]
            if i < len(spatial_data):
                marker_to_spatial[marker_id_val] = spatial_data[i]

        # Find the target marker if specified
        target_id_val = None
        if self.target_id is not None:
            for marker_id in ids:
                if marker_id[0] == self.target_id:
                    target_id_val = marker_id[0]
                    break

        # Process all markers, but prioritize the target
        processed_ids = set()

        # First process the target if found
        if target_id_val is not None and target_id_val in marker_to_spatial:
            self._draw_single_marker_info(
                frame,
                corners[np.where(ids == target_id_val)[0][0]],
                target_id_val,
                marker_to_spatial[target_id_val],
                is_target=True
            )
            processed_ids.add(target_id_val)

            # Add targeting guidance (direction to target)
            self._draw_targeting_guidance(frame)

        # Then process all other markers
        for i, marker_id in enumerate(ids):
            marker_id_val = marker_id[0]
            if marker_id_val not in processed_ids and marker_id_val in marker_to_spatial:
                self._draw_single_marker_info(
                    frame,
                    corners[i],
                    marker_id_val,
                    marker_to_spatial[marker_id_val],
                    is_target=False
                )
                processed_ids.add(marker_id_val)

        return frame

    def _draw_single_marker_info(self, frame, corner, marker_id, spatial_point, is_target=False):
        """
        Draw information for a single marker
        """
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

        # Use different color for target marker
        rect_color = (0, 0, 255) if is_target else (255, 255, 0)  # Red for target, yellow for others
        rect_thickness = 3 if is_target else 2

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), rect_color, rect_thickness)

        # Display spatial coordinates
        text_color = (0, 255, 255) if is_target else (255, 255, 255)  # Cyan for target, white for others

        # Add "TARGET" label if this is the target marker
        if is_target:
            cv2.putText(
                frame,
                "TARGET",
                (xmin + 10, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # Draw a more prominent highlight for the target
            # Draw corners with larger points
            for pt in corner[0]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            # Draw a crosshair on the target
            center_x = int(np.mean([corner[0][0][0], corner[0][1][0], corner[0][2][0], corner[0][3][0]]))
            center_y = int(np.mean([corner[0][0][1], corner[0][1][1], corner[0][2][1], corner[0][3][1]]))
            cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
            cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), 2)

        # Display marker ID and distance prominently
        cv2.putText(
            frame,
            f"ID: {marker_id}",
            (xmin + 10, ymin + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )

        cv2.putText(
            frame,
            f"Dist: {z/1000:.2f}m",
            (xmin + 10, ymin + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )

        # Display X, Y coordinates
        cv2.putText(
            frame,
            f"X: {x/1000:.2f}m Y: {y/1000:.2f}m",
            (xmin + 10, ymin + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1
        )

    def _draw_targeting_guidance(self, frame):
        """
        Draw targeting guidance to help navigate to the target
        """
        if not hasattr(self, 'target_found') or not self.target_found:
            return

        # Get frame center
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Calculate offset from center
        if hasattr(self, 'target_center'):
            target_x, target_y = self.target_center
            offset_x = target_x - center_x
            offset_y = target_y - center_y

            # Draw guidance arrow
            arrow_length = min(100, max(20, int(np.sqrt(offset_x**2 + offset_y**2) / 5)))
            angle = np.arctan2(offset_y, offset_x)
            end_x = int(center_x + np.cos(angle) * arrow_length)
            end_y = int(center_y + np.sin(angle) * arrow_length)

            # Only draw if target is not centered
            if abs(offset_x) > 30 or abs(offset_y) > 30:
                # Draw direction arrow
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)

                # Add guidance text
                direction_text = ""
                if abs(offset_y) > 30:
                    direction_text += "UP " if offset_y < 0 else "DOWN "
                if abs(offset_x) > 30:
                    direction_text += "LEFT" if offset_x < 0 else "RIGHT"

                cv2.putText(
                    frame,
                    direction_text,
                    (center_x + 20, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                # Target is centered
                cv2.putText(
                    frame,
                    "TARGET CENTERED",
                    (center_x - 100, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Send servo commands if servo control is enabled
            if self.enable_servo_control and self.mav_connection is not None:
                self.control_servos_from_target()

def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OAK-D ArUco 6x6 Marker Detector for Drone Applications')
    parser.add_argument('--target', '-t', type=int, help='Target marker ID to highlight')
    parser.add_argument('--resolution', '-r', choices=['low', 'medium', 'high', 'adaptive'],
                      default='adaptive', help='Resolution mode (default: adaptive)')
    parser.add_argument('--cuda', '-c', action='store_true', help='Enable CUDA acceleration if available')
    parser.add_argument('--performance', '-p', action='store_true', help='Enable high performance mode on Jetson')
    parser.add_argument('--mavlink', '-m', type=str, help='MAVLink connection string (e.g., udp:localhost:14550)')
    parser.add_argument('--servo-control', '-s', action='store_true', help='Enable servo control via MAVLink')
    parser.add_argument('--stream', '-st', action='store_true', help='Enable video streaming over RTP/UDP')
    parser.add_argument('--stream-ip', type=str, default='192.168.2.1', help='IP address to stream to (default: 192.168.2.1)')
    parser.add_argument('--stream-port', type=int, default=5600, help='Port to stream to (default: 5600)')
    parser.add_argument('--stream-bitrate', type=int, default=4000000, help='Streaming bitrate in bits/sec (default: 4000000)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI windows)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress progress messages in the console')
    args = parser.parse_args()

    print("Initializing OAK-D ArUco 6x6 Marker Detector for Drone Applications...")
    detector = OakDArUcoDetector(
        target_id=args.target,
        resolution=args.resolution,
        use_cuda=args.cuda,
        high_performance=args.performance,
        mavlink_connection=args.mavlink,
        enable_servo_control=args.servo_control,
        enable_streaming=args.stream,
        stream_ip=args.stream_ip,
        stream_port=args.stream_port,
        stream_bitrate=args.stream_bitrate,
        headless=args.headless,
        quiet=args.quiet
    )
    detector.start()

if __name__ == "__main__":
    main()
