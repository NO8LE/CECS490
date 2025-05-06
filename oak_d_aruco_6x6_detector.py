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
  --stream-ip IP               IP address to stream to (default: 192.168.1.100)
  --stream-port PORT           Port to stream to (default: 5000)
  --stream-bitrate BITRATE     Streaming bitrate in bits/sec (default: 4000000)

Examples:
  python3 oak_d_aruco_6x6_detector.py
  python3 oak_d_aruco_6x6_detector.py --target 5
  python3 oak_d_aruco_6x6_detector.py -t 10 -r high -c -p
  python3 oak_d_aruco_6x6_detector.py --stream --stream-ip 192.168.1.100

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

# Import ArUco module - try different approaches for different OpenCV installations
try:
    # First try direct import from cv2
    if hasattr(cv2, 'aruco'):
        print("Using cv2.aruco")
        aruco = cv2.aruco
    else:
        # Try to import as a separate module (older OpenCV versions)
        try:
            from cv2 import aruco
            print("Using from cv2 import aruco")
            # Make it available as cv2.aruco for consistency
            cv2.aruco = aruco
        except ImportError:
            # For Jetson with custom OpenCV builds
            try:
                sys.path.append('/usr/lib/python3/dist-packages/cv2/python-3.10')
                from cv2 import aruco
                print("Using Jetson-specific aruco import")
                cv2.aruco = aruco
            except (ImportError, FileNotFoundError):
                print("Error: OpenCV ArUco module not found.")
                print("Please ensure opencv-contrib-python is installed:")
                print("  pip install opencv-contrib-python")
                print("\nFor Jetson platforms, you might need to install it differently:")
                print("  sudo apt-get install python3-opencv")
                sys.exit(1)
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

# Verify ArUco module is working
try:
    # Try to access a dictionary to verify ArUco is working
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
    print("ArUco module successfully loaded and verified (using Dictionary_get)")
    # Store the method to use later
    dictionary_method = "old"
except Exception as e:
    # If Dictionary_get fails, try the newer API
    try:
        ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
        aruco_dict = cv2.aruco.Dictionary.get(ARUCO_DICT_TYPE)
        print("ArUco module successfully loaded and verified (using Dictionary.get)")
        # Store the method to use later
        dictionary_method = "new"
    except Exception as e2:
        # If both methods fail, try to create a dictionary directly
        try:
            # Try to create a dictionary directly (OpenCV 4.x approach)
            ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
            aruco_dict = cv2.aruco.Dictionary.create(ARUCO_DICT_TYPE)
            print("ArUco module successfully loaded and verified (using Dictionary.create)")
            # Store the method to use later
            dictionary_method = "create"
        except Exception as e3:
            # Last resort: try to create a dictionary with parameters
            try:
                # Try to create a dictionary with parameters (another approach)
                ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
                aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE)
                print("ArUco module successfully loaded and verified (using Dictionary constructor)")
                # Store the method to use later
                dictionary_method = "constructor"
            except Exception as e4:
                print(f"Error verifying ArUco module: {str(e4)}")
                print("ArUco module found but not working correctly")
                print("\nDetailed error information:")
                print(f"Dictionary_get error: {str(e)}")
                print(f"Dictionary.get error: {str(e2)}")
                print(f"Dictionary.create error: {str(e3)}")
                print(f"Dictionary constructor error: {str(e4)}")
                print("\nPlease check your OpenCV installation and version.")
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

# Camera calibration directory - create if it doesn't exist
CALIB_DIR = "camera_calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

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
                 stream_ip="192.168.251.105", stream_port=5000, stream_bitrate=4000000,
                 headless=False):
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
            
        # Initialize ArUco detector using the method that worked during initialization
        if dictionary_method == "old":
            self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
        elif dictionary_method == "new":
            self.aruco_dict = cv2.aruco.Dictionary.get(ARUCO_DICT_TYPE)
        elif dictionary_method == "create":
            self.aruco_dict = cv2.aruco.Dictionary.create(ARUCO_DICT_TYPE)
        elif dictionary_method == "constructor":
            self.aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE)
        else:
            # Fallback to trying all methods
            try:
                self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
            except:
                try:
                    self.aruco_dict = cv2.aruco.Dictionary.get(ARUCO_DICT_TYPE)
                except:
                    try:
                        self.aruco_dict = cv2.aruco.Dictionary.create(ARUCO_DICT_TYPE)
                    except:
                        self.aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE)
        
        # Initialize detector parameters
        try:
            self.aruco_params = cv2.aruco.DetectorParameters.create()
        except:
            try:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            except:
                try:
                    self.aruco_params = cv2.aruco.DetectorParameters()
                except Exception as e:
                    print(f"Error creating detector parameters: {str(e)}")
                    print("Using default parameters")
                    self.aruco_params = None
        
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
        if len(self.detection_success_rate) > 0:
            success_rate = sum(self.detection_success_rate) / len(self.detection_success_rate)
            print(f"Avg processing time: {avg_time*1000:.1f}ms, Success rate: {success_rate*100:.1f}%")
        else:
            # No detections yet
            print(f"Avg processing time: {avg_time*1000:.1f}ms, No markers detected yet")
        
        # If processing is too slow, reduce resolution or simplify detection
        if avg_time > 0.1:  # More than 100ms per frame
            self.skip_frames = 1  # Process every other frame
            print("Performance warning: Processing time > 100ms, skipping frames")
        else:
            self.skip_frames = 0  # Process every frame
        
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
            
            # Check if this is a CharucoBoard calibration
            if 'charuco_calibration' in data:
                print("Detected CharucoBoard calibration data")
                if 'squares_x' in data and 'squares_y' in data:
                    print(f"CharucoBoard: {data['squares_x']}x{data['squares_y']} squares")
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
                
            # Construct the GStreamer pipeline string
            gst_pipeline = (
                f"appsrc ! video/x-raw,format=BGR ! videoconvert ! "
                f"video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                f"nvv4l2h264enc insert-sps-pps=1 bitrate={self.stream_bitrate} preset-level=1 iframeinterval=30 ! "
                f"h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host={self.stream_ip} port={self.stream_port} sync=false"
            )
            
            # Initialize the VideoWriter with GStreamer pipeline
            self.video_writer = cv2.VideoWriter(
                gst_pipeline,
                cv2.CAP_GSTREAMER,
                0,  # Codec is ignored when using GStreamer
                30.0,  # Target 30 FPS
                (width, height)
            )
            
            # Check if VideoWriter was successfully initialized
            if not self.video_writer.isOpened():
                print("Failed to open video writer pipeline. Streaming will be disabled.")
                print("Make sure GStreamer is properly installed and NVENC is available.")
                self.enable_streaming = False
                self.video_writer = None
            else:
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
        # Connect to device and start pipeline
        # Use Device() and startPipeline(pipeline) instead of Device(pipeline)
        self.device = dai.Device()
        self.device.startPipeline(self.pipeline)
        
        # Get output queues
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.spatial_calc_queue = self.device.getOutputQueue(name="spatial_data", maxSize=4, blocking=False)
        self.spatial_calc_config_queue = self.device.getInputQueue("spatial_calc_config")
        
        if self.headless:
            print("Camera started in headless mode. Use Ctrl+C to exit.")
        else:
            print("Camera started. Press 'q' to exit.")
        
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
        Detect ArUco markers and CharucoBoard in the frame
        
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
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.aruco_params
        )
        
        # Try to detect CharucoBoard only if not in simple detection mode
        charuco_corners = None
        charuco_ids = None
        if not simple_detection and ids is not None and len(ids) >= 4:
            # Check if this looks like a CharucoBoard pattern
            try:
                # Create a CharucoBoard object for detection
                if hasattr(cv2.aruco, 'CharucoBoard_create'):
                    # Old API
                    charuco_board = cv2.aruco.CharucoBoard_create(
                        squaresX=6, 
                        squaresY=6, 
                        squareLength=0.3048/6,  # Board is 12 inches divided into 6 squares
                        markerLength=0.3048/6*0.75,  # Markers are 75% of square size
                        dictionary=self.aruco_dict
                    )
                else:
                    # New API
                    charuco_board = cv2.aruco.CharucoBoard.create(
                        squaresX=6, 
                        squaresY=6, 
                        squareLength=0.3048/6,  # Board is 12 inches divided into 6 squares
                        markerLength=0.3048/6*0.75,  # Markers are 75% of square size
                        dictionary=self.aruco_dict
                    )
                
                # Interpolate the corners of the CharucoBoard
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, charuco_board
                )
                
                if ret and len(charuco_corners) > 4:
                    print(f"Detected CharucoBoard with {len(charuco_corners)} corners")
            except Exception as e:
                print(f"Error detecting CharucoBoard: {e}")
        
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
                    # Create a CharucoBoard object for pose estimation
                    if hasattr(cv2.aruco, 'CharucoBoard_create'):
                        # Old API
                        charuco_board = cv2.aruco.CharucoBoard_create(
                            squaresX=6, 
                            squaresY=6, 
                            squareLength=0.3048/6,  # Board is 12 inches divided into 6 squares
                            markerLength=0.3048/6*0.75,  # Markers are 75% of square size
                            dictionary=self.aruco_dict
                        )
                    else:
                        # New API
                        charuco_board = cv2.aruco.CharucoBoard.create(
                            squaresX=6, 
                            squaresY=6, 
                            squareLength=0.3048/6,  # Board is 12 inches divided into 6 squares
                            markerLength=0.3048/6*0.75,  # Markers are 75% of square size
                            dictionary=self.aruco_dict
                        )
                    
                    # Estimate the pose of the CharucoBoard
                    retval, charuco_rvec, charuco_tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids,
                        board=charuco_board,
                        cameraMatrix=self.camera_matrix,
                        distCoeffs=self.dist_coeffs,
                        rvec=None,
                        tvec=None
                    )
                    
                    if retval:
                        # Draw the CharucoBoard axes
                        cv2.aruco.drawAxis(
                            markers_frame,
                            self.camera_matrix,
                            self.dist_coeffs,
                            charuco_rvec,
                            charuco_tvec,
                            0.2  # Larger axis length for the board
                        )
                        
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
            
            # Estimate pose of each individual marker
            try:
                # Ensure we have valid calibration data
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, 
                        MARKER_SIZE, 
                        self.camera_matrix, 
                        self.dist_coeffs
                    )
                    
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
                                
                                cv2.aruco.drawAxis(
                                    markers_frame, 
                                    self.camera_matrix, 
                                    self.dist_coeffs, 
                                    rvecs[i], 
                                    tvecs[i], 
                                    axis_length
                                )
                                
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
    parser.add_argument('--stream-ip', type=str, default='192.168.251.105', help='IP address to stream to (default: 192.168.251.105)')
    parser.add_argument('--stream-port', type=int, default=5000, help='Port to stream to (default: 5000)')
    parser.add_argument('--stream-bitrate', type=int, default=4000000, help='Streaming bitrate in bits/sec (default: 4000000)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI windows)')
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
        headless=args.headless
    )
    detector.start()

if __name__ == "__main__":
    main()
