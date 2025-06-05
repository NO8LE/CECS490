#!/usr/bin/env python3

"""
Autonomous Precision Landing Mission for ArUco Marker Detection

This script implements autonomous ArUco marker-based precision landing using:
- Jetson Orin Nano onboard computer
- OAK-D camera for computer vision
- CubePilot Orange+ running ArduCopter 4.6

The drone will:
1. Take off to search altitude
2. Perform a search pattern to locate the target ArUco marker
3. Validate and confirm the marker identity
4. Center above the marker with precision loiter
5. Perform a controlled descent while maintaining position above the marker
6. Land precisely on the target marker
"""

import os
import sys
import time
import math
import yaml
import argparse
import logging
import cv2
import numpy as np
from enum import Enum
from threading import Thread, Lock

# Import MAVLink controller
from mavlink_controller import MAVLinkController, CoordinateTransformer

# Import OAK-D wrapper if available
try:
    # First check if we need OpenCV 4.10 ArUco fixes
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aruco'))
    try:
        from opencv410_aruco_fix import OpenCV410ArUcoFix
        USE_ARUCO_FIX = True
        print("Using OpenCV 4.10 ArUco fixes")
    except ImportError:
        USE_ARUCO_FIX = False
        print("OpenCV 4.10 ArUco fixes not found, using standard OpenCV ArUco")

    # Now import the OAK-D wrapper
    from oak_d_aruco_wrapper410 import OAKDArUcoWrapper
    OAKD_AVAILABLE = True
    print("OAK-D wrapper imported successfully")
except ImportError as e:
    OAKD_AVAILABLE = False
    print(f"Failed to import OAK-D wrapper: {e}")
    print("Running without OAK-D camera. External detection is required.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("precision_landing_mission.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PrecisionLandingMission")

# Mission state definitions
class MissionState(Enum):
    """Mission state machine states"""
    INIT = "initializing"
    CONNECT = "connecting"
    TAKEOFF = "taking_off"
    SEARCH = "searching"
    TARGET_ACQUIRE = "acquiring_target"
    TARGET_VALIDATE = "validating_target"
    PRECISION_LOITER = "precision_loiter"
    FINAL_APPROACH = "final_approach"
    PRECISION_LAND = "precision_landing"
    LANDED = "landed"
    ABORT = "aborting"
    EMERGENCY = "emergency"

# Default mission configuration
DEFAULT_MISSION_CONFIG = {
    # Area and altitude settings
    'search_altitude': 15.0,          # meters - safe search height
    'landing_start_altitude': 8.0,    # begin precision landing descent
    'final_approach_altitude': 1.0,   # begin flow-based final approach (m)
    'min_altitude': 1.0,              # minimum safe altitude
    'max_altitude': 30.0,             # maximum mission altitude
    'search_area_size': 27.4,         # 30 yards converted to meters
    'search_pattern_spacing': 5.0,    # meters between search legs
    
    # Target and detection settings  
    'target_marker_id': 5,            # specific landing marker ID
    'confirmation_frames': 5,         # frames to confirm detection
    'validation_time': 3.0,           # seconds to validate target
    'center_tolerance': 0.3,          # acceptable center deviation
    'min_marker_size': 50,            # minimum marker size in pixels
    'max_detection_distance': 12000,  # maximum detection range (mm)
    
    # Optical flow settings
    'use_optical_flow': True,         # Enable optical flow sensor integration
    'final_descent_rate': 0.3,        # Final descent rate (m/s) during flow landing
    'flow_quality_threshold': 50,     # Minimum flow quality (0-255)
    'position_variance_threshold': 0.5, # Maximum acceptable position variance (m²)
    
    # Mission timing and safety
    'max_mission_time': 600,          # 10 minute mission timeout
    'search_timeout': 300,            # 5 minute search timeout  
    'landing_timeout': 120,           # 2 minute landing timeout
    'min_battery_voltage': 22.0,      # abort threshold
    'connection_timeout': 5.0,        # MAVLink connection timeout
    
    # Camera settings
    'camera_fps': 30,                 # Camera frame rate
    'camera_resolution': (1280, 720), # Camera resolution
    'marker_size': 0.3048,            # Marker size in meters (12 inches)
    'camera_calibration_file': 'aruco/camera_calibration/calibration.npz',  # Camera calibration file
    
    # MAVLink settings
    'mavlink_baudrate': 921600,       # MAVLink connection baudrate
    'mavlink_device': '/dev/ttyACM0', # MAVLink connection device
    'mavlink_timeout': 10,            # MAVLink connection timeout (seconds)
    'heartbeat_timeout': 3            # Heartbeat timeout (seconds)
}

class SearchPattern:
    """Generate search pattern coordinates for the drone to follow"""
    
    def __init__(self, area_size, spacing, start_pos=(0, 0)):
        """
        Initialize the search pattern generator
        
        Args:
            area_size: Size of the search area in meters
            spacing: Distance between search legs in meters
            start_pos: Starting position (x, y) in meters
        """
        self.area_size = area_size
        self.spacing = spacing
        self.start_pos = start_pos
        self.waypoints = self._generate_lawnmower_pattern()
        self.current_idx = 0
        
    def _generate_lawnmower_pattern(self):
        """Generate a lawnmower pattern for efficient area coverage"""
        # Calculate number of legs
        num_legs = math.ceil(self.area_size / self.spacing)
        
        waypoints = []
        half_size = self.area_size / 2
        start_x, start_y = self.start_pos
        
        # Start at the southwest corner
        x = start_x - half_size
        y = start_y - half_size
        
        # Generate the lawnmower pattern
        for i in range(num_legs):
            # Add waypoint at the current position
            if i % 2 == 0:
                # Even legs go north
                waypoints.append((x, y))
                waypoints.append((x, y + self.area_size))
            else:
                # Odd legs go south
                waypoints.append((x, y + self.area_size))
                waypoints.append((x, y))
                
            # Move east by the spacing distance
            x += self.spacing
            
        return waypoints
        
    def get_next_waypoint(self):
        """Get the next waypoint in the search pattern"""
        if self.current_idx >= len(self.waypoints):
            return None
            
        waypoint = self.waypoints[self.current_idx]
        self.current_idx += 1
        return waypoint
        
    def reset(self):
        """Reset the search pattern to the beginning"""
        self.current_idx = 0
        
    def get_all_waypoints(self):
        """Get all waypoints in the search pattern"""
        return self.waypoints

class PrecisionLandingMission:
    """Main class for autonomous precision landing mission"""
    
    def __init__(self, config_file=None, simulation=False):
        """
        Initialize the precision landing mission
        
        Args:
            config_file: Path to YAML configuration file
            simulation: Whether to run in simulation mode
        """
        # Load configuration
        self.config = DEFAULT_MISSION_CONFIG.copy()
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self.config.update(user_config)
        
        # Initialize state
        self.state = MissionState.INIT
        self.state_lock = Lock()
        self.running = True
        self.simulation = simulation
        
        # ArUco detection variables
        self.aruco_detector = None
        self.target_detections = []  # Recent detections for validation
        self.last_detection_time = 0
        self.target_confirmed = False
        self.search_pattern = None
        
        # Coordinate transformer
        self.transformer = CoordinateTransformer(camera_orientation='down')
        
        # Time tracking
        self.mission_start_time = time.time()
        self.state_start_time = time.time()
        
        logger.info(f"Precision Landing Mission initialized (simulation={simulation})")
        
    def initialize_detector(self):
        """Initialize the ArUco marker detector"""
        if not OAKD_AVAILABLE:
            logger.error("OAK-D camera not available. Cannot initialize detector.")
            return False
            
        try:
            logger.info("Initializing OAK-D ArUco detector")
            
            # Load camera calibration if available
            camera_matrix = None
            dist_coeffs = None
            calibration_file = self.config.get('camera_calibration_file')
            
            if calibration_file and os.path.exists(calibration_file):
                try:
                    calibration = np.load(calibration_file)
                    camera_matrix = calibration['camera_matrix']
                    dist_coeffs = calibration['dist_coeffs']
                    logger.info("Loaded camera calibration from file")
                except Exception as e:
                    logger.warning(f"Failed to load camera calibration: {e}")
            
            # Initialize detector
            resolution = self.config.get('camera_resolution')
            fps = self.config.get('camera_fps')
            
            self.aruco_detector = OAKDArUcoWrapper(
                dictionary=cv2.aruco.DICT_6X6_250,
                marker_size=self.config.get('marker_size'),
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                resolution=resolution,
                fps=fps
            )
            
            # Test detector by running one detection
            self.aruco_detector.start_pipeline()
            time.sleep(1)  # Give the camera time to initialize
            
            test_frame, markers = self.aruco_detector.get_markers()
            if test_frame is not None:
                logger.info("OAK-D ArUco detector initialized successfully")
                return True
            else:
                logger.error("Failed to get frame from OAK-D camera")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize OAK-D ArUco detector: {e}")
            return False
            
    def connect_mavlink(self):
        """Connect to the vehicle via MAVLink"""
        logger.info("Connecting to vehicle via MAVLink")
        
        try:
            # Create MAVLink controller
            connection_string = self.config.get('mavlink_device')
            baudrate = self.config.get('mavlink_baudrate')
            
            logger.info(f"Connecting to {connection_string} at {baudrate} baud")
            
            self.mavlink = MAVLinkController(
                connection_string=connection_string, 
                config=self.config,
                simulation=self.simulation
            )
            
            # Wait for connection
            timeout = self.config.get('mavlink_timeout')
            if self.mavlink.wait_for_connection(timeout):
                logger.info("MAVLink connection established")
                return True
            else:
                logger.error("Failed to establish MAVLink connection")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to MAVLink: {e}")
            return False
            
    def prepare_mission(self):
        """Prepare for the mission"""
        logger.info("Preparing for precision landing mission")
        
        # Initialize search pattern
        area_size = self.config.get('search_area_size')
        spacing = self.config.get('search_pattern_spacing')
        self.search_pattern = SearchPattern(area_size=area_size, spacing=spacing)
        
        # Check if the vehicle is already armed
        vehicle_state = self.mavlink.get_vehicle_state()
        if vehicle_state.get('armed', False):
            logger.info("Vehicle is already armed")
        
        # Check current altitude
        current_alt = vehicle_state.get('relative_altitude', 0)
        logger.info(f"Current relative altitude: {current_alt:.2f}m")
        
        # Reset detection state
        self.target_detections = []
        self.target_confirmed = False
        
        # Initialize mission timers
        self.mission_start_time = time.time()
        self.state_start_time = time.time()
        
        return True
        
    def takeoff(self):
        """Command the vehicle to take off to search altitude"""
        logger.info("Initiating takeoff")
        
        # Check if we're already at a safe altitude
        vehicle_state = self.mavlink.get_vehicle_state()
        current_alt = vehicle_state.get('relative_altitude', 0)
        search_alt = self.config.get('search_altitude')
        
        if current_alt >= search_alt * 0.9:  # If we're already at approximately search altitude
            logger.info(f"Already at {current_alt:.2f}m, which is close to search altitude of {search_alt:.2f}m")
            return True
            
        # Set to GUIDED mode for takeoff
        if not self.mavlink.set_mode("GUIDED"):
            logger.error("Failed to set GUIDED mode for takeoff")
            return False
            
        # Command takeoff
        if not self.mavlink.takeoff(search_alt):
            logger.error(f"Failed to command takeoff to {search_alt}m")
            return False
            
        logger.info(f"Takeoff command accepted, target altitude: {search_alt}m")
        
        # Wait for the vehicle to reach target altitude
        return self.wait_for_altitude(search_alt)
        
    def wait_for_altitude(self, target_alt, tolerance=0.5, timeout=60):
        """Wait for the vehicle to reach a target altitude"""
        logger.info(f"Waiting to reach altitude {target_alt:.2f}m (tolerance: {tolerance:.2f}m)")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            vehicle_state = self.mavlink.get_vehicle_state()
            current_alt = vehicle_state.get('relative_altitude', 0)
            
            if abs(current_alt - target_alt) <= tolerance:
                logger.info(f"Reached target altitude: {current_alt:.2f}m")
                return True
                
            logger.debug(f"Current altitude: {current_alt:.2f}m, target: {target_alt:.2f}m")
            time.sleep(1)
            
        logger.warning(f"Timed out waiting to reach altitude {target_alt:.2f}m")
        return False
        
    def search_for_target(self):
        """Execute the search pattern to find the target marker"""
        logger.info("Beginning search pattern")
        
        # Get the search altitude
        search_alt = self.config.get('search_altitude')
        
        # Get the next waypoint in the search pattern
        waypoint = self.search_pattern.get_next_waypoint()
        
        if waypoint is None:
            logger.warning("No more waypoints in search pattern")
            self.search_pattern.reset()
            waypoint = self.search_pattern.get_next_waypoint()
            
        x, y = waypoint
        logger.info(f"Moving to search waypoint: ({x:.2f}, {y:.2f})")
        
        # Move to the waypoint
        if not self.mavlink.set_position_target(x, y, -search_alt):
            logger.error(f"Failed to set position target to ({x:.2f}, {y:.2f}, {-search_alt:.2f})")
            return False
            
        # Look for the target marker while moving
        detection_result = self.detect_target_marker()
        if detection_result:
            logger.info("Target marker detected during search")
            return True
            
        # Wait for the vehicle to reach the waypoint
        self.wait_for_position(x, y, timeout=30)
        
        return False  # Continue searching
        
    def detect_target_marker(self):
        """Detect the target ArUco marker"""
        if not self.aruco_detector:
            logger.error("ArUco detector not initialized")
            return False
            
        # Get the target marker ID
        target_id = self.config.get('target_marker_id')
        logger.debug(f"Looking for marker ID {target_id}")
        
        try:
            # Get the frame and detected markers
            frame, markers = self.aruco_detector.get_markers()
            
            if frame is None or not markers:
                logger.debug("No markers detected")
                return False
                
            # Check if our target is among the detected markers
            target_marker = None
            for marker in markers:
                marker_id = marker['id']
                if marker_id == target_id:
                    logger.info(f"Detected target marker ID {target_id}")
                    target_marker = marker
                    break
                    
            if not target_marker:
                logger.debug(f"Target marker ID {target_id} not among detected markers")
                return False
                
            # Check if the marker is large enough (reliability check)
            min_size = self.config.get('min_marker_size')
            corners = target_marker['corners']
            
            # Calculate marker size in pixels (average of width and height)
            width = np.linalg.norm(corners[0][0] - corners[0][1])
            height = np.linalg.norm(corners[0][0] - corners[0][3])
            marker_size = (width + height) / 2
            
            if marker_size < min_size:
                logger.debug(f"Marker too small: {marker_size:.2f}px < {min_size}px")
                return False
                
            # Check if marker is within detection range
            if 'tvec' in target_marker:
                tvec = target_marker['tvec'][0]
                distance_mm = np.linalg.norm(tvec)
                max_distance = self.config.get('max_detection_distance')
                
                if distance_mm > max_distance:
                    logger.debug(f"Marker too far: {distance_mm:.2f}mm > {max_distance}mm")
                    return False
                    
                logger.info(f"Target marker distance: {distance_mm:.2f}mm ({distance_mm/1000:.2f}m)")
                
            # Store the detection
            self.target_detections.append({
                'time': time.time(),
                'marker': target_marker,
                'distance': distance_mm if 'tvec' in target_marker else None
            })
            
            # Update last detection time
            self.last_detection_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error detecting target marker: {e}")
            return False
            
    def validate_target(self):
        """Validate the target marker with multiple detections"""
        logger.info("Validating target marker...")
        
        # Check if we have enough recent detections
        now = time.time()
        recent_detections = [d for d in self.target_detections 
                            if now - d['time'] < self.config.get('validation_time')]
        
        # Purge old detections
        self.target_detections = recent_detections
        
        # Check if we have enough detections
        min_detections = self.config.get('confirmation_frames')
        
        if len(recent_detections) < min_detections:
            logger.debug(f"Not enough recent detections: {len(recent_detections)}/{min_detections}")
            return False
            
        # Calculate average position
        positions = []
        for detection in recent_detections:
            if 'tvec' in detection['marker']:
                positions.append(detection['marker']['tvec'][0])
                
        if not positions:
            logger.warning("No position data in recent detections")
            return False
            
        # Convert to numpy array
        positions = np.array(positions)
        avg_position = np.mean(positions, axis=0)
        
        # Calculate position variance
        variance = np.var(positions, axis=0)
        max_variance = np.max(variance)
        
        # Check if variance is acceptable
        max_allowed_variance = self.config.get('position_variance_threshold') * 1000000  # Convert from m² to mm²
        
        if max_variance > max_allowed_variance:
            logger.debug(f"Position variance too high: {max_variance:.2f} > {max_allowed_variance:.2f}")
            return False
            
        logger.info(f"Target validated with {len(recent_detections)} detections")
        logger.info(f"Average position: ({avg_position[0]:.2f}, {avg_position[1]:.2f}, {avg_position[2]:.2f})mm")
        logger.info(f"Position variance: {max_variance:.2f}mm²")
        
        # Mark target as confirmed
        self.target_confirmed = True
        
        return True
        
    def precision_loiter(self):
        """Loiter precisely above the target marker"""
        logger.info("Entering precision loiter mode")
        
        # Switch to PRECISION_LOITER mode if available, otherwise use LOITER
        try:
            if not self.mavlink.set_mode("PRECISION_LOITER"):
                logger.warning("Failed to set PRECISION_LOITER mode, using standard LOITER")
                if not self.mavlink.set_mode("LOITER"):
                    logger.error("Failed to set LOITER mode")
                    return False
        except Exception as e:
            logger.warning(f"Error setting precision loiter mode: {e}")
            if not self.mavlink.set_mode("LOITER"):
                logger.error("Failed to set LOITER mode")
                return False
                
        # Get the latest detection
        if not self.target_detections:
            logger.error("No target detections available for precision loiter")
            return False
            
        latest_detection = max(self.target_detections, key=lambda d: d['time'])
        marker = latest_detection['marker']
        
        if 'tvec' not in marker:
            logger.error("No position data in latest detection")
            return False
            
        # Calculate angles for LANDING_TARGET message
        vehicle_state = self.mavlink.get_vehicle_state()
        if 'attitude' not in vehicle_state:
            logger.error("No attitude data available")
            return False
            
        # Send LANDING_TARGET message to help with position hold
        angle_x, angle_y, distance = self.transformer.calculate_landing_target_angles(
            marker, vehicle_state.get('attitude'))
            
        if angle_x is None or angle_y is None or distance is None:
            logger.error("Failed to calculate landing target angles")
            return False
            
        target_id = self.config.get('target_marker_id')
        if not self.mavlink.send_landing_target(angle_x, angle_y, distance, target_id):
            logger.warning("Failed to send LANDING_TARGET message")
            
        # Loiter for a few seconds to stabilize
        logger.info("Precision loitering for 5 seconds to stabilize")
        time.sleep(5)
        
        return True
        
    def begin_final_approach(self):
        """Begin the final approach phase for landing"""
        logger.info("Beginning final approach")
        
        # Switch to GUIDED mode for controlled descent
        if not self.mavlink.set_mode("GUIDED"):
            logger.error("Failed to set GUIDED mode for final approach")
            return False
            
        # Get current altitude
        vehicle_state = self.mavlink.get_vehicle_state()
        current_alt = vehicle_state.get('relative_altitude', 0)
        
        # Get final approach altitude
        final_approach_alt = self.config.get('final_approach_altitude')
        
        # Calculate descent speed based on current height
        descent_time = 10  # seconds
        descent_rate = (current_alt - final_approach_alt) / descent_time
        
        # Ensure descent rate is reasonable
        max_descent_rate = 1.0  # m/s
        min_descent_rate = 0.1  # m/s
        descent_rate = max(min(descent_rate, max_descent_rate), min_descent_rate)
        
        logger.info(f"Starting controlled descent from {current_alt:.2f}m to {final_approach_alt:.2f}m")
        logger.info(f"Descent rate: {descent_rate:.2f}m/s")
        
        # Descend while maintaining position over the marker
        while current_alt > final_approach_alt + 0.5:  # 0.5m tolerance
            # Detect marker
            self.detect_target_marker()
            
            if not self.target_detections:
                logger.warning("Lost target during descent")
                return False
                
            # Get latest detection
            latest_detection = max(self.target_detections, key=lambda d: d['time'])
            marker = latest_detection['marker']
            
            if 'tvec' not in marker:
                logger.warning("No position data in latest detection")
                continue
                
            # Calculate angles for LANDING_TARGET message
            vehicle_state = self.mavlink.get_vehicle_state()
            if 'attitude' not in vehicle_state:
                logger.warning("No attitude data available")
                continue
                
            angle_x, angle_y, distance = self.transformer.calculate_landing_target_angles(
                marker, vehicle_state.get('attitude'))
                
            if angle_x is None or angle_y is None or distance is None:
                logger.warning("Failed to calculate landing target angles")
                continue
                
            # Send LANDING_TARGET message
            target_id = self.config.get('target_marker_id')
            self.mavlink.send_landing_target(angle_x, angle_y, distance, target_id)
            
            # Calculate position offset
            tvec = marker['tvec'][0]
            x_offset = tvec[0] / 1000.0  # mm to m
            y_offset = tvec[1] / 1000.0  # mm to m
            
            # Calculate new altitude setpoint with controlled descent
            current_alt = vehicle_state.get('relative_altitude', 0)
            new_alt = current_alt - descent_rate * 0.5  # 0.5 seconds per iteration
            new_alt = max(new_alt, final_approach_alt)  # Don't go below final approach altitude
            
            # Apply position correction and altitude change
            # Note: in NED frame, altitude is negative
            self.mavlink.set_position_target(
                y_offset, x_offset, -new_alt,  # x, y, z in NED frame
                0, 0, -descent_rate,  # vx, vy, vz in NED frame
                coordinate_frame=mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED
            )
            
            # Update current altitude
            time.sleep(0.5)
            vehicle_state = self.mavlink.get_vehicle_state()
            current_alt = vehicle_state.get('relative_altitude', 0)
            
            logger.debug(f"Current altitude: {current_alt:.2f}m, target: {final_approach_alt:.2f}m")
        
        logger.info(f"Reached final approach altitude: {current_alt:.2f}m")
        return True
        
    def execute_precision_landing(self):
        """Execute the final precision landing phase"""
        logger.info("Executing precision landing")
        
        # Check if optical flow is available and healthy
        use_flow = self.config.get('use_optical_flow')
        flow_healthy = self.mavlink.is_flow_healthy() if use_flow else False
        
        if use_flow and flow_healthy:
            logger.info("Using optical flow for final descent")
            precision_land_mode = 1  # Use precision landing
        else:
            logger.info("Optical flow not available/healthy, using vision only")
            precision_land_mode = 1  # Still use precision landing with vision
            
        # Command precision landing
        if not self.mavlink.command_precision_land():
            logger.error("Failed to command precision landing")
            return False
            
        logger.info("Precision landing command accepted")
        
        # Monitor landing
        landing_start_time = time.time()
        landing_timeout = self.config.get('landing_timeout')
        
        while time.time() - landing_start_time < landing_timeout:
            # Check if we've landed
            vehicle_state = self.mavlink.get_vehicle_state()
            current_alt = vehicle_state.get('relative_altitude', 0)
            
            if current_alt < 0.2:  # Below 20cm, consider landed
                logger.info(f"Landed at altitude: {current_alt:.2f}m")
                return True
                
            # Continue to track the target
            self.detect_target_marker()
            
            if self.target_detections:
                # Get latest detection
                latest_detection = max(self.target_detections, key=lambda d: d['time'])
                marker = latest_detection['marker']
                
                if 'tvec' in marker:
                    # Send LANDING_TARGET message
                    angle_x, angle_y, distance = self.transformer.calculate_landing_target_angles(
                        marker, vehicle_state.get('attitude', (0, 0, 0)))
                        
                    if angle_x is not None and angle_y is not None and distance is not None:
                        target_id = self.config.get('target_marker_id')
                        self.mavlink.send_landing_target(angle_x, angle_y, distance, target_id)
            
            # Brief pause
            time.sleep(0.5)
            
        logger.warning("Landing timed out")
        return False
        
    def wait_for_position(self, x, y, tolerance=1.0, timeout=30):
        """Wait for the vehicle to reach a position"""
        logger.info(f"Waiting to reach position ({x:.2f}, {y:.2f})")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get current position
            vehicle_state = self.mavlink.get_vehicle_state()
            
            # In a real system, we would convert from GPS coordinates to local NED
            # For simulation or testing, we can assume we have direct position feedback
            current_x = vehicle_state.get('local_x', 0)
            current_y = vehicle_state.get('local_y', 0)
            
            # Calculate distance to target
            dx = current_x - x
            dy = current_y - y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance <= tolerance:
                logger.info(f"Reached position ({current_x:.2f}, {current_y:.2f})")
                return True
                
            logger.debug(f"Current position: ({current_x:.2f}, {current_y:.2f}), distance to target: {distance:.2f}m")
            time.sleep(1)
            
        logger.warning(f"Timed out waiting to reach position ({x:.2f}, {y:.2f})")
        return False
        
    def run_mission(self):
        """Run the full precision landing mission"""
        logger.info("Starting precision landing mission")
        
        try:
            # Initialize
            with self.state_lock:
                self.state = MissionState.INIT
                self.state_start_time = time.time()
                
            # Initialize detector
            if not self.initialize_detector():
                logger.error("Failed to initialize ArUco detector")
                return False
                
            # Connect to vehicle
            with self.state_lock:
                self.state = MissionState.CONNECT
                self.state_start_time = time.time()
                
            if not self.connect_mavlink():
                logger.error("Failed to connect to vehicle")
                return False
                
            # Prepare mission
            if not self.prepare_mission():
                logger.error("Failed to prepare mission")
                return False
                
            # Takeoff
            with self.state_lock:
                self.state = MissionState.TAKEOFF
                self.state_start_time = time.time()
                
            if not self.takeoff():
                logger.error("Failed to takeoff")
                return False
                
            # Search for target
            with self.state_lock:
                self.state = MissionState.SEARCH
                self.state_start_time = time.time()
                
            search_start_time = time.time()
            search_timeout = self.config.get('search_timeout')
            
            while time.time() - search_start_time < search_timeout:
                if self.search_for_target():
                    break
                    
                # Check if we've exceeded max mission time
                if time.time() - self.mission_start_time > self.config.get('max_mission_time'):
                    logger.warning("Mission timeout exceeded")
                    with self.state_lock:
                        self.state = MissionState.ABORT
                    return False
                    
            # Check if we found the target
            if self.state != MissionState.SEARCH:
                logger.error("Exited search loop in unexpected state")
                return False
                
            if not self.target_detections:
                logger.warning("No target found within search timeout")
                with self.state_lock:
                    self.state = MissionState.ABORT
                return False
                
            # Validate target
            with self.state_lock:
                self.state = MissionState.TARGET_VALIDATE
                self.state_start_time = time.time()
                
            validation_start_time = time.time()
            validation_timeout = 10  # seconds
            
            while time.time() - validation_start_time < validation_timeout:
                if self.validate_target():
                    break
                    
                # Keep detecting
                self.detect_target_marker()
                time.sleep(0.5)
                
            if not self.target_confirmed:
                logger.warning("Failed to validate target")
                with self.state_lock:
                    self.state = MissionState.ABORT
                return False
                
            # Precision loiter
            with self.state_lock:
                self.state = MissionState.PRECISION_LOITER
                self.state_start_time = time.time()
                
            if not self.precision_loiter():
                logger.error("Failed to enter precision loiter")
                with self.state_lock:
                    self.state = MissionState.ABORT
                return False
                
            # Final approach
            with self.state_lock:
                self.state = MissionState.FINAL_APPROACH
                self.state_start_time = time.time()
                
            if not self.begin_final_approach():
                logger.error("Failed during final approach")
                with self.state_lock:
                    self.state = MissionState.ABORT
                return False
                
            # Precision landing
            with self.state_lock:
                self.state = MissionState.PRECISION_LAND
                self.state_start_time = time.time()
                
            if not self.execute_precision_landing():
                logger.error("Failed during precision landing")
                with self.state_lock:
                    self.state = MissionState.ABORT
                return False
                
            # Landed successfully
            with self.state_lock:
                self.state = MissionState.LANDED
                self.state_start_time = time.time()
                
            logger.info("Mission completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Exception during mission: {e}")
            with self.state_lock:
                self.state = MissionState.EMERGENCY
            return False
            
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        
        # Close the ArUco detector
        if self.aruco_detector:
            try:
                self.aruco_detector.close()
                logger.info("Closed ArUco detector")
            except Exception as e:
                logger.warning(f"Error closing ArUco detector: {e}")
                
        # Close MAVLink connection
        if hasattr(self, 'mavlink'):
            try:
                self.mavlink.close()
                logger.info("Closed MAVLink connection")
            except Exception as e:
                logger.warning(f"Error closing MAVLink connection: {e}")
                
        self.running = False


def main():
    """Main entry point for the precision landing mission"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autonomous Precision Landing Mission')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--target-id', type=int, help='Target ArUco marker ID')
    parser.add_argument('--search-alt', type=float, help='Search altitude in meters')
    parser.add_argument('--device', type=str, help='MAVLink device')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create mission
    mission = PrecisionLandingMission(
        config_file=args.config,
        simulation=args.simulation
    )
    
    # Update configuration with command line arguments
    if args.target_id is not None:
        mission.config['target_marker_id'] = args.target_id
        
    if args.search_alt is not None:
        mission.config['search_altitude'] = args.search_alt
        
    if args.device is not None:
        mission.config['mavlink_device'] = args.device
    
    # Run the mission
    try:
        result = mission.run_mission()
        if result:
            logger.info("Mission completed successfully")
            return 0
        else:
            logger.error("Mission failed")
            return 1
    except KeyboardInterrupt:
        logger.info("Mission interrupted by user")
        return 1
    finally:
        mission.cleanup()


if __name__ == "__main__":
    sys.exit(main())
