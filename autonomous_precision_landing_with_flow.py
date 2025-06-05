#!/usr/bin/env python3

"""
Enhanced Autonomous Precision Landing System for ArduCopter 4.6

This script uses the OAK-D camera with ArUco detection combined with HereFlow
optical flow sensor to perform a highly stable autonomous precision landing
mission on a specific ArUco marker. It leverages MAVLink commands to control 
the drone while using multi-sensor fusion for robust positioning and landing.

Usage:
  python3 autonomous_precision_landing_with_flow.py --target MARKER_ID [options]

Options:
  --target, -t MARKER_ID     Marker ID to use as landing target (required)
  --mission-area SIZE        Size of search area in meters (default: 27.4 - 30 yards)
  --search-alt METERS        Initial search altitude in meters (default: 18.0)
  --landing-alt METERS       Altitude to start precision landing (default: 10.0)
  --connection CONN_STR      MAVLink connection string (default: 'udp:192.168.2.1:14550')
  --headless                 Run in headless mode (no GUI)
  --stream                   Enable video streaming
  --sim                      Run in simulation mode (no real hardware needed)
  --verbose, -v              Enable verbose output
  --log-file PATH            Path to log file (default: 'mission_log.txt')
  --use-flow, -f             Enable optical flow sensor integration (default: True)
  --final-approach-alt       Altitude to switch to flow-dominated approach (default: 1.0m)

Example:
  python3 autonomous_precision_landing_with_flow.py --target 5 --search-alt 15 --final-approach-alt 1.2
"""

import os
import sys
import time
import argparse
import threading
import queue
import math
import logging
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation as R

# Import PyMAVLink
try:
    from pymavlink import mavutil
    print("PyMAVLink successfully imported")
except ImportError:
    print("Error: PyMAVLink not found. MAVLink functionality will be disabled.")
    print("To enable MAVLink support, install pymavlink:")
    print("  pip install pymavlink")
    sys.exit(1)

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import OAK-D ArUco detector (modify path if needed)
try:
    from oak_d_aruco_6x6_detector import OakDArUcoDetector
    print("OAK-D ArUco detector successfully imported")
except ImportError:
    print("Error: OAK-D ArUco detector not found.")
    print("Make sure oak_d_aruco_6x6_detector.py is in the current directory.")
    sys.exit(1)

# Mission state definitions
class MissionState(Enum):
    """Mission state machine states"""
    INIT = "initializing"
    TAKEOFF = "taking_off"
    SEARCH = "searching"
    TARGET_ACQUIRE = "acquiring_target"
    TARGET_VALIDATE = "validating_target"
    PRECISION_LOITER = "precision_loiter"
    FINAL_APPROACH = "final_approach"     # New state for flow-controlled descent
    PRECISION_LAND = "precision_landing"
    LANDED = "landed"
    ABORT = "aborting"
    EMERGENCY = "emergency"

# Default mission configuration
DEFAULT_MISSION_CONFIG = {
    # Area and altitude settings
    'search_altitude': 18.0,           # meters - safe search height
    'landing_start_altitude': 10.0,    # begin precision landing descent
    'final_approach_altitude': 1.0,    # begin flow-based final approach (m)
    'min_altitude': 2.0,               # minimum safe altitude
    'max_altitude': 30.0,              # maximum mission altitude
    'search_area_size': 27.4,          # 30 yards converted to meters
    'search_pattern_spacing': 5.0,     # meters between search legs
    
    # Target and detection settings  
    'target_marker_id': 5,             # specific landing marker ID
    'confirmation_frames': 5,          # frames to confirm detection
    'validation_time': 3.0,            # seconds to validate target
    'center_tolerance': 0.3,           # acceptable center deviation
    'min_marker_size': 50,             # minimum marker size in pixels
    'max_detection_distance': 12000,   # maximum detection range (mm)
    
    # Optical flow settings
    'use_optical_flow': True,          # Enable optical flow sensor integration
    'final_descent_rate': 0.3,         # Final descent rate (m/s) during flow landing
    'flow_quality_threshold': 50,      # Minimum flow quality (0-255)
    'position_variance_threshold': 0.5,# Maximum acceptable position variance (m²)
    
    # Mission timing and safety
    'max_mission_time': 600,           # 10 minute mission timeout
    'search_timeout': 300,             # 5 minute search timeout  
    'landing_timeout': 120,            # 2 minute landing timeout
    'min_battery_voltage': 22.0,       # abort threshold (adjust for your battery)
    'connection_timeout': 5.0,         # MAVLink connection timeout
    
    # MAVLink settings
    'mavlink_connection': 'udp:192.168.2.1:14550',  # connection string
    'heartbeat_rate': 1.0,             # heartbeat frequency
    'landing_target_rate': 10.0,       # LANDING_TARGET message rate
    
    # Performance settings
    'detection_rate': 20.0,            # ArUco detection frequency (Hz)
    'mission_loop_rate': 10.0,         # mission state machine rate
    'position_update_rate': 10.0,      # position update frequency
    
    # EKF settings to verify
    'ekf_pos_horiz_variance_threshold': 1.0,  # Maximum acceptable horizontal position variance (m²)
    'ekf_pos_vert_variance_threshold': 1.0,   # Maximum acceptable vertical position variance (m²)
}

class CoordinateTransformer:
    """Convert between ArUco camera coordinates and MAVLink NED coordinates"""
    
    def __init__(self, camera_matrix, camera_orientation='down'):
        self.camera_matrix = camera_matrix
        self.camera_orientation = camera_orientation
        
    def aruco_to_ned(self, aruco_position, vehicle_attitude):
        """
        Convert ArUco 3D position to NED coordinates for LANDING_TARGET
        
        Args:
            aruco_position: (x, y, z) in camera frame (mm)
            vehicle_attitude: (roll, pitch, yaw) in radians
            
        Returns:
            (north, east, down) in meters for MAVLink
        """
        if aruco_position is None:
            return None, None, None
            
        # Convert mm to meters
        x_cam, y_cam, z_cam = np.array(aruco_position) / 1000.0
        
        # Camera frame to body frame transformation
        # Assuming camera is pointing down with standard orientation
        if self.camera_orientation == 'down':
            # Camera X -> Body Y (right)
            # Camera Y -> Body X (forward) 
            # Camera Z -> Body -Z (down)
            x_body = y_cam   # forward
            y_body = x_cam   # right  
            z_body = z_cam   # down
        
        # Body frame to NED frame using vehicle attitude
        roll, pitch, yaw = vehicle_attitude
        
        # Rotation matrix from body to NED
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch) 
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        # Combined rotation matrix (Z-Y-X Euler)
        R_body_to_ned = np.array([
            [cos_pitch*cos_yaw, 
             sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw,
             cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw],
            [cos_pitch*sin_yaw,
             sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw, 
             cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw],
            [-sin_pitch,
             sin_roll*cos_pitch,
             cos_roll*cos_pitch]
        ])
        
        # Transform to NED
        body_vector = np.array([x_body, y_body, z_body])
        ned_vector = R_body_to_ned @ body_vector
        
        return ned_vector[0], ned_vector[1], ned_vector[2]  # north, east, down
        
    def calculate_landing_target_message(self, marker_data, vehicle_state):
        """
        Create LANDING_TARGET message from marker detection
        
        Args:
            marker_data: Dictionary with marker position and confidence
            vehicle_state: Current vehicle attitude and position
            
        Returns:
            Dictionary with LANDING_TARGET message parameters
        """
        # Get NED coordinates
        north, east, down = self.aruco_to_ned(
            marker_data['position_3d'], 
            vehicle_state['attitude']
        )
        
        if north is None:
            return None
            
        # Calculate relative position (target relative to vehicle)
        rel_north = north
        rel_east = east  
        rel_down = down
        
        return {
            'time_usec': int(time.time() * 1e6),
            'target_num': marker_data['id'],
            'frame': mavutil.mavlink.MAV_FRAME_BODY_NED,
            'x': rel_north,
            'y': rel_east, 
            'z': rel_down,
            'size_x': marker_data.get('size', 0.3048),  # marker size in meters
            'size_y': marker_data.get('size', 0.3048),
            'quaternion': [0, 0, 0, 0],  # unused for position target
            'type': 1,  # light beacon
            'position_valid': 1
        }

class MAVLinkFlightController:
    """Enhanced MAVLink interface for autonomous flight operations"""
    
    def __init__(self, connection_string, target_system=1, target_component=1, config=None):
        self.connection_string = connection_string
        self.connection = None
        self.target_system = target_system
        self.target_component = target_component
        self.last_heartbeat = 0
        self.vehicle_state = {}
        self.config = config or {}
        self.running = True
        
        # Extended state variables for optical flow and EKF
        self.optical_flow_quality = 0    # Flow quality (0-255)
        self.optical_flow_valid = False  # Flow data validity
        self.ekf_status = {}             # EKF status flags and variances
        self.ekf_flags = 0               # EKF status flags
        self.ekf_velocity_variance = 0.0 # EKF velocity variance
        self.ekf_pos_horiz_variance = 0.0 # EKF horizontal position variance
        self.ekf_pos_vert_variance = 0.0  # EKF vertical position variance
        
        # Initialize logger
        self.logger = logging.getLogger("MAVLink")
        
        # Connect to vehicle
        self.connect()
        
        # Start heartbeat monitoring thread
        self.heartbeat_thread = threading.Thread(target=self._monitor_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        # Start vehicle state monitoring thread
        self.state_thread = threading.Thread(target=self._monitor_vehicle_state)
        self.state_thread.daemon = True
        self.state_thread.start()
        
    def connect(self):
        """Connect to the MAVLink device"""
        try:
            self.logger.info(f"Connecting to MAVLink device at {self.connection_string}")
            self.connection = mavutil.mavlink_connection(self.connection_string)
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to MAVLink: {e}")
            return False
        
    def wait_for_connection(self, timeout=30):
        """Wait for MAVLink connection with timeout"""
        if not self.connection:
            self.logger.error("No MAVLink connection established")
            return False
            
        start_time = time.time()
        self.logger.info("Waiting for MAVLink heartbeat...")
        
        while time.time() - start_time < timeout:
            if self.connection.wait_heartbeat(timeout=1):
                self.logger.info("MAVLink connection established")
                # Set the system and component ID for sending commands
                self.target_system = self.connection.target_system
                self.target_component = self.connection.target_component
                return True
                
        self.logger.error("Timed out waiting for MAVLink heartbeat")
        return False
        
    def _monitor_heartbeat(self):
        """Monitor MAVLink heartbeat in separate thread"""
        while self.running:
            if not self.connection:
                time.sleep(1)
                continue
                
            try:
                msg = self.connection.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
                if msg:
                    self.last_heartbeat = time.time()
            except Exception as e:
                self.logger.warning(f"Error receiving heartbeat: {e}")
                
            time.sleep(0.1)
    
    def _monitor_vehicle_state(self):
        """Monitor vehicle state in separate thread"""
        while self.running:
            if not self.connection:
                time.sleep(1)
                continue
                
            try:
                # Process messages to update vehicle state
                msgs = []
                while True:
                    msg = self.connection.recv_match(blocking=False)
                    if not msg:
                        break
                    msgs.append(msg)
                    
                # Process received messages
                for msg in msgs:
                    msg_type = msg.get_type()
                    
                    # ATTITUDE message (for roll, pitch, yaw)
                    if msg_type == "ATTITUDE":
                        self.vehicle_state['attitude'] = (msg.roll, msg.pitch, msg.yaw)
                        self.vehicle_state['attitude_time'] = time.time()
                        
                    # GLOBAL_POSITION_INT message (for location)
                    elif msg_type == "GLOBAL_POSITION_INT":
                        # Convert to degrees
                        lat = msg.lat / 1.0e7
                        lon = msg.lon / 1.0e7
                        alt = msg.alt / 1000.0  # mm to meters
                        relative_alt = msg.relative_alt / 1000.0  # mm to meters
                        
                        self.vehicle_state['position'] = (lat, lon, alt)
                        self.vehicle_state['relative_altitude'] = relative_alt
                        self.vehicle_state['position_time'] = time.time()
                        
                    # SYS_STATUS message (for battery)
                    elif msg_type == "SYS_STATUS":
                        battery_voltage = msg.voltage_battery / 1000.0  # mV to V
                        battery_current = msg.current_battery / 100.0   # cA to A
                        battery_remaining = msg.battery_remaining       # percentage
                        
                        self.vehicle_state['battery_voltage'] = battery_voltage
                        self.vehicle_state['battery_current'] = battery_current
                        self.vehicle_state['battery_remaining'] = battery_remaining
                        self.vehicle_state['battery_time'] = time.time()
                        
                    # HEARTBEAT message (for mode and armed status)
                    elif msg_type == "HEARTBEAT":
                        armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                        self.vehicle_state['armed'] = armed
                        
                        # Get flight mode
                        flight_mode = self._interpret_mode(msg.custom_mode)
                        self.vehicle_state['mode'] = flight_mode
                        self.vehicle_state['mode_time'] = time.time()
                        
                    # OPTICAL_FLOW message (for optical flow data)
                    elif msg_type == "OPTICAL_FLOW":
                        self.optical_flow_quality = msg.quality
                        self.optical_flow_valid = (msg.quality >= self.config.get('flow_quality_threshold', 50))
                        self.vehicle_state['flow_quality'] = msg.quality
                        self.vehicle_state['flow_x'] = msg.flow_x
                        self.vehicle_state['flow_y'] = msg.flow_y
                        self.vehicle_state['flow_time'] = time.time()
                        
                    # EKF_STATUS_REPORT message (for EKF health monitoring)
                    elif msg_type == "EKF_STATUS_REPORT":
                        self.ekf_flags = msg.flags
                        self.ekf_velocity_variance = msg.velocity_variance
                        self.ekf_pos_horiz_variance = msg.pos_horiz_variance
                        self.ekf_pos_vert_variance = msg.pos_vert_variance
                        
                        # Update EKF health status
                        self.ekf_status = {
                            'velocity_variance': msg.velocity_variance,
                            'pos_horiz_variance': msg.pos_horiz_variance,
                            'pos_vert_variance': msg.pos_vert_variance,
                            'compass_variance': msg.compass_variance,
                            'terrain_alt_variance': msg.terrain_alt_variance,
                            'flags': msg.flags,
                            'healthy': (msg.flags & 0x1F) == 0x1F,  # All bits 0-4 should be set for healthy
                            'time': time.time()
                        }
                        
                        self.vehicle_state['ekf_status'] = self.ekf_status
                
            except Exception as e:
                self.logger.warning(f"Error monitoring vehicle state: {e}")
                
            time.sleep(0.1)
    
    def is_flow_healthy(self):
        """Check if optical flow data is valid and healthy"""
        # Consider flow healthy if quality is above threshold
        flow_health = self.optical_flow_quality >= self.config.get('flow_quality_threshold', 50)
        
        # Also check if the data is recent (last 2 seconds)
        flow_recent = (time.time() - self.vehicle_state.get('flow_time', 0)) < 2.0
        
        return flow_health and flow_recent
    
    def is_ekf_healthy(self):
        """Check if EKF estimates are reliable"""
        # Check if EKF status flags indicate health (bits 0-4 should be set)
        flags_healthy = (self.ekf_flags & 0x1F) == 0x1F
        
        # Check variances are within acceptable limits
        pos_horiz_ok = self.ekf_pos_horiz_variance < self.config.get('ekf_pos_horiz_variance_threshold', 1.0)
        pos_vert_ok = self.ekf_pos_vert_variance < self.config.get('ekf_pos_vert_variance_threshold', 1.0)
        
        return flags_healthy and pos_horiz_ok and pos_vert_ok
    
    def _interpret_mode(self, custom_mode):
        """Convert custom_mode value to human-readable flight mode"""
        # These mode mappings are for ArduCopter
        mode_mapping = {
            0: "STABILIZE",
            1: "ACRO",
            2: "ALT_HOLD",
            3: "AUTO",
            4: "GUIDED",
            5: "LOITER",
            6: "RTL",
            7: "CIRCLE",
            9: "LAND",
            16: "PRECISION_LOITER",
            20: "GUIDED_NOGPS"
        }
        
        return mode_mapping.get(custom_mode, f"UNKNOWN({custom_mode})")
        
    def set_mode(self, mode_name):
        """Set vehicle flight mode"""
        if not self.connection:
            self.logger.error("No MAVLink connection established")
            return False
            
        mode_mapping = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'CIRCLE': 7,
            'LAND': 9, 'PRECISION_LOITER': 16, 'GUIDED_NOGPS': 20
        }
        
        if mode_name not in mode_mapping:
            self.logger.error(f"Unknown mode: {mode_name}")
            return False
            
        mode_id = mode_mapping[mode_name]
        
        self.logger.info(f"Setting flight mode to {mode_name}")
        self.connection.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,  # confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,  # param1
            mode_id,  # param2 - custom mode
            0, 0, 0, 0, 0  # unused parameters
        )
        
        # Wait for mode change confirmation
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        
    def _wait_for_ack(self, command, timeout=3):
        """Wait for command acknowledgment"""
        if not self.connection:
            return False
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.connection.recv_match(type='COMMAND_ACK', blocking=False)
            if msg and msg.command == command:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self.logger.info(f"Command {command} accepted")
                    return True
                else:
                    self.logger.warning(f"Command {command} failed with result {msg.result}")
                    return False
            time.sleep(0.1)
            
        self.logger.warning(f"Timed out waiting for command {command} acknowledgment")
        return False
        
    def send_landing_target(self, target_data):
        """Send LANDING_TARGET message to vehicle"""
        if not self.connection or not target_data:
            return False
            
        try:
            self.connection.mav.landing_target_send(
                target_data['time_usec'],
                target_data['target_num'], 
                target_data['frame'],
                target_data['x'], target_data['y'], target_data['z'],
                target_data['size_x'], target_data['size_y'],
                target_data['quaternion'][0], target_data['quaternion'][1],
                target_data['quaternion'][2], target_data['quaternion'][3],
                target_data['type'], target_data['position_valid']
            )
            return True
        except Exception as e:
            self.logger.error(f"Error sending LANDING_TARGET message: {e}")
            return False
            
    def set_parameter(self, param_id, param_value, param_type='REAL32'):
        """Set a parameter on the flight controller"""
        if not self.connection:
            return False
            
        # Parameter types
        type_mapping = {
            'INT8': 1,
            'INT16': 2,
            'INT32': 3,
            'REAL32': 9
        }
        
        if param_type not in type_mapping:
            self.logger.error(f"Unknown parameter type: {param_type}")
            return False
            
        param_type_id = type_mapping[param_type]
        
        try:
            self.connection.mav.param_set_send(
                self.target_system, self.target_component,
                param_id.encode('utf-8'),
                float(param_value),
                param_type_id
            )
            
            # Wait for parameter to be set
            start_time = time.time()
            while time.time() - start_time < 3:
                msg = self.connection.recv_match(type='PARAM_VALUE', blocking=False)
                if msg and msg.param_id.decode('utf-8') == param_id:
                    self.logger.info(f"Parameter {param_id} set to {param_value}")
                    return True
                time.sleep(0.1)
                
            self.logger.warning(f"Timed out waiting for parameter {param_id} to be set")
            return False
        except Exception as e:
            self.logger.error(f"Error setting parameter {param_id}: {e}")
            return False
        
    def arm(self, arm=True):
        """Arm or disarm the vehicle"""
        if not self.connection:
            self.logger.error("No MAVLink connection established")
            return False
            
        arm_val = 1 if arm else 0
        self.logger.info(f"{'Arming' if arm else 'Disarming'} vehicle")
        
        self.connection.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # confirmation
            arm_val,  # param1 (1=arm, 0=disarm)
            0,  # param2 (force, 0=normal)
            0, 0, 0, 0, 0  # unused parameters
        )
        
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        
    def takeoff(self, altitude):
        """Command the vehicle to takeoff to specified altitude"""
        if not self.connection:
            self.logger.error("No MAVLink connection established")
            return False
            
        # Ensure we're in guided mode
        current_mode = self.get_vehicle_state().get('mode', 'UNKNOWN')
        if current_mode != 'GUIDED':
            self.logger.info("Setting mode to GUIDED for takeoff")
            if not self.set_mode('GUIDED'):
                self.logger.error("Failed to set GUIDED mode for takeoff")
                return False
                
        # Ensure vehicle is armed
        if not self.get_vehicle_state().get('armed', False):
            self.logger.info("Arming vehicle for takeoff")
            if not self.arm(True):
                self.logger.error("Failed to arm vehicle for takeoff")
                return False
                
        # Send takeoff command
        self.logger.info(f"Commanding takeoff to {altitude}m altitude")
        self.connection.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # confirmation
            0, 0, 0, 0, 0, 0,  # unused parameters
            altitude  # param7 - altitude
        )
        
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        
    def command_loiter(self, lat, lon, alt, radius=10):
        """Command vehicle to loiter at specified location"""
        if not self.connection:
            self.logger.error("No MAVLink connection established")
            return False
            
        self.logger.info(f"Commanding loiter at ({lat}, {lon}, {alt}m) with radius {radius}m")
        self.connection.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
            0,  # confirmation
            0,  # param1 (unused)
            0,  # param2 (unused)
            radius,  # param3 - loiter radius
            0,  # param4 (unused) 
            lat, lon, alt  # param5, 6, 7 - position
        )
        
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM)
        
    def command_precision_land(self, lat=0, lon=0, alt=0):
        """Command precision landing"""
        if not self.connection:
            self.logger.error("No MAVLink connection established")
            return False
            
        self.logger.info("Commanding precision landing")
        self.connection.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,  # confirmation
            0,  # param1 - abort altitude (0 = use default)
            1,  # param2 - precision land mode (1 = enabled)
            0, 0,  # param3, 4 - unused
            lat, lon, alt  # param5, 6, 7 - target position (0 = use current)
        )
        
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_NAV_LAND)
        
    def command_rtl(self):
        """Command vehicle to return to launch"""
        if not self.connection:
            self.logger.error("No MAVLink connection established")
            return False
            
        self.logger.info("Commanding return to launch (RTL)")
        return self.set_mode('RTL')
        
    def get_vehicle_state(self):
        """Get current vehicle state from MAVLink messages"""
        return {
            'attitude': self.vehicle_state.get('attitude', (0, 0, 0)),
            'position': self.vehicle_state.get('position', (0, 0, 0)),
            'relative_altitude': self.vehicle_state.get('relative_altitude', 0),
            'battery_voltage': self.vehicle_state.get('battery_voltage', 0),
            'battery_remaining': self.vehicle_state.get('battery_remaining', 0),
            'mode': self.vehicle_state.get('mode', 'UNKNOWN'),
            'armed': self.vehicle_state.get('armed', False),
            'flow_quality': self.vehicle_state.get('flow_quality', 0),
            'ekf_status': self.vehicle_state.get('ekf_status', {})
        }
        
    def set_descent_rate(self, rate_mps):
        """Set descent rate in meters per second"""
        if not self.connection:
            return False
            
        # For precision landing, we use SET_POSITION_TARGET_LOCAL_NED
        # with a velocity setpoint for the Z axis
        # Mask: only use Z velocity, all other fields ignored
        mask = (
            # Don't use position setpoints
            0b0000111111000000 |
            # Don't use X, Y velocity, only Z
            0b0000000000110000 |
            # Don't use acceleration or yaw
            0b0000000000000111
        )
        
        # Convert rate to NED frame (positive down)
        ned_down_velocity = rate_mps
        
        self.logger.info(f"Setting descent rate to {rate_mps} m/s")
        
        try:
            self.connection.mav.set_position_target_local_ned_send(
                0,  # timestamp (0 = use system time)
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mask,
                0, 0, 0,  # position x, y, z (not used)
                0, 0, ned_down_velocity,  # velocity x, y, z (only z used)
                0, 0, 0,  # acceleration x, y, z (not used)
                0, 0  # yaw, yaw_rate (not used)
            )
            return True
        except Exception as e:
            self.logger.error(f"Error setting descent rate: {e}")
            return False
            
    def configure_optical_flow(self):
        """Configure ArduCopter parameters for optimal optical flow use"""
        if not self.connection:
            return False
            
        # Key parameters for HereFlow optical flow sensor
        parameters = {
            'FLOW_TYPE': 6,           # HereFlow
            'FLOW_ORIENT_YAW': 0,     # Flow sensor orientation
            'FLOW_POS_X': 0.0,        # X position on vehicle
            'FLOW_POS_Y': 0.0,        # Y position on vehicle
            'FLOW_POS_Z': 0.0,        # Z position on vehicle
            'EK3_SRC1_VELXY': 5,      # Optical flow for horizontal velocity
            'PLND_ENABLED': 1,        # Enable precision landing
            'PLND_TYPE': 1,           # MAVLink target type
            'PLND_EST_TYPE': 0,       # Use vehicle position estimate
            'LAND_SPEED': 30,         # Slower landing speed (cm/s)
        }
        
        success = True
        for param_id, param_value in parameters.items():
            if not self.set_parameter(param_id, param_value):
                self.logger.warning(f"Failed to set parameter {param_id}={param_value}")
                success = False
                
        return success
        
    def close(self):
        """Close the MAVLink connection and stop threads"""
        self.logger.info("Closing MAVLink connection")
        self.running = False
        
        if self.connection:
            self.connection.close()
            self.connection = None

class EnhancedOakDArUcoDetector(OakDArUcoDetector):
    """Enhanced detector for autonomous missions with flow integration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_detections = {}
        self.detection_queue = queue.Queue(maxsize=10)
        self.detection_thread = None
        self.stop_detection = False
        
        # Initialize logger
        self.logger = logging.getLogger("ArUcoDetector")
        
    def start_autonomous_detection(self):
        """Start detection in separate thread for mission use"""
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        self.logger.info("Started autonomous detection thread")
        
    def _detection_loop(self):
        """Detection loop for autonomous operation"""
        while not self.stop_detection:
            # Get frames
            rgb_frame = self.get_rgb_frame()
            if rgb_frame is not None:
                markers_frame, corners, ids = self.detect_aruco_markers(rgb_frame)
                
                # Package detection results
                detection_data = {
                    'timestamp': time.time(),
                    'markers': {},
                    'frame': markers_frame
                }
                
                if ids is not None:
                    for i, marker_id in enumerate(ids):
                        marker_id_val = marker_id[0]
                        
                        # Get 3D position using spatial data
                        spatial_data = self.get_spatial_data()
                        if i < len(spatial_data):
                            coords = spatial_data[i].spatialCoordinates
                            position_3d = (coords.x, coords.y, coords.z)
                        else:
                            position_3d = None
                            
                        detection_data['markers'][marker_id_val] = {
                            'id': marker_id_val,
                            'corners': corners[i][0],
                            'position_3d': position_3d,
                            'confidence': self._calculate_confidence(corners[i])
                        }
                
                # Update latest detections
                self.latest_detections = detection_data
                
                # Add to queue for mission controller
                try:
                    self.detection_queue.put_nowait(detection_data)
                except queue.Full:
                    # Remove oldest detection
                    self.detection_queue.get_nowait()
                    self.detection_queue.put_nowait(detection_data)
                    
            time.sleep(0.05)  # 20Hz detection rate
            
    def _calculate_confidence(self, corners):
        """Calculate confidence score for marker detection (0-1)"""
        # Implement confidence calculation based on:
        # - Marker size (larger is better)
        # - Corner sharpness
        # - Detection stability
        
        # For now, use a simple size-based confidence
        if corners is None or len(corners) == 0:
            return 0.0
            
        # Calculate marker perimeter
        perimeter = 0
        for i in range(4):
            p1 = corners[0][i]
            p2 = corners[0][(i + 1) % 4]
            perimeter += np.linalg.norm(p1 - p2)
            
        # Normalize confidence based on perimeter
        # 0-200px: 0.1-0.3
        # 200-500px: 0.3-0.7
        # 500+px: 0.7-1.0
        if perimeter < 200:
            confidence = 0.1 + (perimeter / 200) * 0.2
        elif perimeter < 500:
            confidence = 0.3 + ((perimeter - 200) / 300) * 0.4
        else:
            confidence = min(0.7 + ((perimeter - 500) / 1000) * 0.3, 1.0)
            
        return confidence
        
    def get_latest_detections(self):
        """Get most recent detection results"""
        return self.latest_detections.get('markers', {})
        
    def stop(self):
        """Stop the detection thread"""
        self.stop_detection = True
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        super().stop()

class FlowIntegrationManager:
    """Manages the integration of optical flow with vision-based positioning"""
    
    def __init__(self, mavlink_controller, config=None):
        self.mavlink = mavlink_controller
        self.config = config or {}
        self.flow_enabled = self.config.get('use_optical_flow', True)
        self.flow_quality_history = []  # Track flow quality over time
        self.current_descent_rate = 0.0
        
        # Initialize logger
        self.logger = logging.getLogger("FlowManager")
        
        # Configure optical flow if enabled
        if self.flow_enabled:
            self.mavlink.configure_optical_flow()
            
    def is_flow_available(self):
        """Check if optical flow is available and providing valid data"""
        if not self.flow_enabled:
            return False
            
        return self.mavlink.is_flow_healthy()
        
    def get_flow_quality(self):
        """Get current optical flow quality (0-100%)"""
        if not self.flow_enabled:
            return 0.0
            
        quality = self.mavlink.vehicle_state.get('flow_quality', 0)
        return min(100.0, quality / 255.0 * 100.0)  # Convert to percentage
        
    def adapt_descent_for_flow(self, altitude, has_valid_marker):
        """
        Adapt descent rate based on flow quality and altitude
        
        Args:
            altitude: Current altitude in meters
            has_valid_marker: Whether we have a valid marker detection
            
        Returns:
            optimal_descent_rate: The recommended descent rate in m/s
        """
        flow_quality = self.get_flow_quality() / 100.0  # 0.0-1.0
        
        # Add quality to history (max 10 samples)
        self.flow_quality_history.append(flow_quality)
        if len(self.flow_quality_history) > 10:
            self.flow_quality_history.pop(0)
            
        # Get average quality to smooth out fluctuations
        avg_quality = sum(self.flow_quality_history) / len(self.flow_quality_history)
        
        # Base descent rate from config
        base_rate = self.config.get('final_descent_rate', 0.3)
        
        # Calculate descent rate
        if altitude > self.config.get('final_approach_altitude', 1.0):
            # Above flow-dominated zone - use normal rate
            if has_valid_marker:
                # Good ArUco visibility - descend at normal rate
                optimal_rate = base_rate
            else:
                # Poor ArUco visibility - slow down
                optimal_rate = base_rate * 0.5
        else:
            # In flow-dominated zone
            if avg_quality > 0.7:
                # Good flow quality - can maintain normal rate
                optimal_rate = base_rate
            elif avg_quality > 0.4:
                # Medium flow quality - reduce rate
                optimal_rate = base_rate * 0.7
            else:
                # Poor flow quality - very slow descent
                optimal_rate = base_rate * 0.3
                
        # Limit rate of change
        max_change = 0.05  # m/s per update
        if abs(optimal_rate - self.current_descent_rate) > max_change:
            if optimal_rate > self.current_descent_rate:
                self.current_descent_rate += max_change
            else:
                self.current_descent_rate -= max_change
        else:
            self.current_descent_rate = optimal_rate
            
        return self.current_descent_rate
        
    def update_landing_strategy(self, altitude, has_valid_marker):
        """
        Update landing strategy based on sensor health
        
        Args:
            altitude: Current altitude in meters
            has_valid_marker: Whether we have a valid marker detection
            
        Returns:
            descent_rate: Recommended descent rate
            use_precision_land: Whether to use precision land mode
        """
        # Check flow health
        flow_healthy = self.is_flow_available()
        
        # Check EKF health
        ekf_healthy = self.mavlink.is_ekf_healthy()
        
        # Default strategy
        descent_rate = self.config.get('final_descent_rate', 0.3)
        use_precision_land = True
        
        # Adjust strategy based on sensor health
        if altitude <= self.config.get('final_approach_altitude', 1.0):
            # In final approach zone where flow is critical
            if flow_healthy and ekf_healthy:
                # Optimal conditions - use adaptive descent rate
                descent_rate = self.adapt_descent_for_flow(altitude, has_valid_marker)
                use_precision_land = True
                self.logger.info(f"Flow-optimized landing: {descent_rate:.2f} m/s")
            elif has_valid_marker and ekf_healthy:
                # No flow, but good ArUco and EKF - slow but continue
                descent_rate = self.config.get('final_descent_rate', 0.3) * 0.5
                use_precision_land = True
                self.logger.warning("Flow unavailable, using slower vision-based landing")
            else:
                # Poor conditions - pause descent and hold
                descent_rate = 0.0
                use_precision_land = False
                self.logger.warning("Unsafe landing conditions, holding position")
        else:
            # Higher altitude - primarily vision-based
            if has_valid_marker and ekf_healthy:
                # Good conditions for standard approach
                descent_rate = self.config.get('final_descent_rate', 0.3)
                use_precision_land = True
            else:
                # Poor conditions - slow descent
                descent_rate = self.config.get('final_descent_rate', 0.3) * 0.5
                use_precision_land = has_valid_marker
                
        return descent_rate, use_precision_land

class SafetyManager:
    """Comprehensive safety monitoring and emergency handling"""
    
    def __init__(self, config, mavlink_controller):
        self.config = config
        self.mavlink = mavlink_controller
        self.emergency_conditions = []
        self.warning_conditions = []
        self.mission_start_time = time.time()
        
        # Initialize logger
        self.logger = logging.getLogger("SafetyManager")
        
    def check_safety_conditions(self):
        """Check all safety conditions, return False if unsafe"""
        conditions_ok = True
        
        # Clear previous conditions
        self.emergency_conditions = []
        self.warning_conditions = []
        
        # Get current vehicle state
        vehicle_state = self.mavlink.get_vehicle_state()
        
        # Battery level check
        battery_voltage = vehicle_state['battery_voltage']
        if battery_voltage < self.config['min_battery_voltage']:
            self.emergency_conditions.append("LOW_BATTERY")
            self.logger.warning(f"Low battery: {battery_voltage}V")
            conditions_ok = False
            
        # Mission timeout check  
        mission_time = time.time() - self.mission_start_time
        if mission_time > self.config['max_mission_time']:
            self.emergency_conditions.append("MISSION_TIMEOUT")
            self.logger.warning(f"Mission timeout: {mission_time:.1f}s")
            conditions_ok = False
            
        # Altitude bounds check
        current_alt = vehicle_state['relative_altitude']
        if current_alt > self.config['max_altitude']:
            self.emergency_conditions.append("ALTITUDE_EXCEEDED")
            self.logger.warning(f"Maximum altitude exceeded: {current_alt}m")
            conditions_ok = False
            
        # Connection health check
        if time.time() - self.mavlink.last_heartbeat > 5:
            self.emergency_conditions.append("LOST_CONNECTION")
            self.logger.warning("Lost MAVLink connection")
            conditions_ok = False
            
        # EKF health check
        ekf_status = vehicle_state.get('ekf_status', {})
        if ekf_status and 'healthy' in ekf_status and not ekf_status['healthy']:
            self.warning_conditions.append("EKF_UNHEALTHY")
            self.logger.warning("EKF estimates degraded")
            
            # Only abort mission if in critical landing phase and EKF is bad
            if current_alt < self.config.get('final_approach_altitude', 1.0):
                self.emergency_conditions.append("EKF_CRITICAL")
                self.logger.error("Critical EKF failure during landing approach")
                conditions_ok = False
            
        return conditions_ok
        
    def execute_emergency_protocol(self):
        """Execute emergency landing/RTL procedure"""
        self.logger.error("EMERGENCY: Executing safety protocol")
        
        # Try precision RTL first if GPS is good
        if "LOST_CONNECTION" not in self.emergency_conditions and "EKF_CRITICAL" not in self.emergency_conditions:
            try:
                self.mavlink.set_mode('RTL')
                self.logger.info("Emergency RTL activated")
                return True
            except Exception as e:
                self.logger.error(f"RTL activation failed: {e}")
                
        # If RTL fails, try emergency land
        try:
            self.mavlink.set_mode('LAND') 
            self.logger.info("Emergency LAND activated")
            return True
        except Exception as e:
            self.logger.error(f"LAND activation failed: {e}")
            
        self.logger.critical("CRITICAL: Unable to activate emergency modes")
        return False

class EnhancedMissionStateMachine:
    """Handles the mission state transitions and logic with flow integration"""
    
    def __init__(self, detector, mavlink_controller, safety_manager, config):
        self.detector = detector
        self.mavlink = mavlink_controller
        self.safety = safety_manager
        self.config = config
        self.state = MissionState.INIT
        self.state_entry_time = time.time()
        self.target_confirmations = 0
        self.required_confirmations = config.get('confirmation_frames', 5)
        self.target_centered_time = 0
        self.running = True
        self.target_marker_id = config.get('target_marker_id', 5)
        
        # Create coordinate transformer
        self.transformer = CoordinateTransformer(
            detector.camera_matrix,
            camera_orientation='down'
        )
        
        # Create flow integration manager
        self.flow_manager = FlowIntegrationManager(mavlink_controller, config)
        
        # Initialize logger
        self.logger = logging.getLogger("Mission")
        
        # Initialize search pattern
        self.search_waypoints = self._generate_search_pattern()
        self.current_waypoint_index = 0
        
    def _generate_search_pattern(self):
        """Generate waypoints for a search pattern over the target area"""
        # Get search area dimensions
        area_size = self.config.get('search_area_size', 27.4)  # 30 yards in meters
        spacing = self.config.get('search_pattern_spacing', 5.0)
        altitude = self.config.get('search_altitude', 18.0)
        
        # Calculate number of legs
        num_legs = max(2, int(area_size / spacing))
        
        # Get home position as center of search area
        home_pos = self.mavlink.get_vehicle_state().get('position', (0, 0, 0))
        center_lat, center_lon, _ = home_pos
        
        # Generate a grid pattern
        waypoints = []
        for i in range(num_legs):
            # Calculate offset from center
            north_offset = (i - num_legs/2) * spacing
            
            # Add waypoints for this leg (alternating directions)
            if i % 2 == 0:
                # Left to right
                waypoints.append((north_offset, -area_size/2, altitude))
                waypoints.append((north_offset, area_size/2, altitude))
            else:
                # Right to left
                waypoints.append((north_offset, area_size/2, altitude))
                waypoints.append((north_offset, -area_size/2, altitude))
                
        return waypoints
        
    def transition_to(self, new_state):
        """Transition to a new mission state"""
        old_state = self.state
        self.state = new_state
        self.state_entry_time = time.time()
        self.logger.info(f"Mission state transition: {old_state.value} -> {new_state.value}")
        
        # Reset state-specific variables
        if new_state == MissionState.TARGET_ACQUIRE:
            self.target_confirmations = 0
            
        elif new_state == MissionState.PRECISION_LOITER:
            self.target_centered_time = 0
            
    def run_mission(self):
        """Main mission loop"""
        self.logger.info("Starting autonomous mission")
        
        while self.running and self.state != MissionState.LANDED and self.state != MissionState.ABORT:
            # Safety check every iteration
            if not self.safety.check_safety_conditions():
                self.transition_to(MissionState.EMERGENCY)
                
            # Execute current state
            if self.state == MissionState.INIT:
                self._handle_init()
            elif self.state == MissionState.TAKEOFF:
                self._handle_takeoff()
            elif self.state == MissionState.SEARCH:
                self._handle_search()
            elif self.state == MissionState.TARGET_ACQUIRE:
                self._handle_target_acquire()
            elif self.state == MissionState.TARGET_VALIDATE:
                self._handle_target_validate()
            elif self.state == MissionState.PRECISION_LOITER:
                self._handle_precision_loiter()
            elif self.state == MissionState.FINAL_APPROACH:
                self._handle_final_approach()
            elif self.state == MissionState.PRECISION_LAND:
                self._handle_precision_land()
            elif self.state == MissionState.EMERGENCY:
                self._handle_emergency()
                
            time.sleep(1.0 / self.config.get('mission_loop_rate', 10.0))
            
        self.logger.info("Mission completed")
        return self.state
        
    def _handle_init(self):
        """Initialize the mission"""
        self.logger.info("Initializing mission")
        
        # Check MAVLink connection
        if not self.mavlink.wait_for_connection(timeout=10):
            self.logger.error("Failed to establish MAVLink connection")
            self.transition_to(MissionState.ABORT)
            return
            
        # Transition to takeoff
        self.transition_to(MissionState.TAKEOFF)
        
    def _handle_takeoff(self):
        """Handle the takeoff sequence"""
        vehicle_state = self.mavlink.get_vehicle_state()
        
        # Check if already in air
        if vehicle_state['relative_altitude'] > 3.0:
            self.logger.info("Vehicle already in air, skipping takeoff")
            self.transition_to(MissionState.SEARCH)
            return
            
        # Command takeoff if not already done
        if time.time() - self.state_entry_time < 2.0:
            self.logger.info(f"Commanding takeoff to {self.config['search_altitude']}m")
            self.mavlink.takeoff(self.config['search_altitude'])
            
        # Wait for target altitude
        current_alt = vehicle_state['relative_altitude']
        target_alt = self.config['search_altitude']
        
        # Consider takeoff complete when within 90% of target altitude
        if current_alt >= target_alt * 0.9:
            self.logger.info(f"Takeoff complete: {current_alt:.1f}m")
            self.transition_to(MissionState.SEARCH)
            
        # Check for timeout
        elif time.time() - self.state_entry_time > 60:
            self.logger.warning("Takeoff timeout, proceeding with search")
            self.transition_to(MissionState.SEARCH)
            
    def _handle_search(self):
        """Execute search pattern and look for target"""
        # Get latest detection results
        detected_markers = self.detector.get_latest_detections()
        
        # Check if target marker is detected
        if self.target_marker_id in detected_markers:
            self.logger.info(f"Target marker {self.target_marker_id} detected during search")
            self.transition_to(MissionState.TARGET_ACQUIRE)
            return
            
        # Continue search pattern
        if len(self.search_waypoints) > 0 and self.current_waypoint_index < len(self.search_waypoints):
            # Get current waypoint
            north, east, alt = self.search_waypoints[self.current_waypoint_index]
            
            # Convert to lat/lon
            home_pos = self.mavlink.get_vehicle_state().get('position', (0, 0, 0))
            target_lat, target_lon = self._offset_position(home_pos[0], home_pos[1], north, east)
            
            # Move to waypoint if not already commanded
            if time.time() - self.state_entry_time < 2.0 or self.current_waypoint_index == 0:
                self.logger.info(f"Moving to search waypoint {self.current_waypoint_index+1}/{len(self.search_waypoints)}")
                self.mavlink.command_loiter(target_lat, target_lon, alt, radius=5)
                
            # Check if we've reached the waypoint
            current_pos = self.mavlink.get_vehicle_state().get('position', (0, 0, 0))
            distance = self._calculate_distance(current_pos[0], current_pos[1], target_lat, target_lon)
            
            if distance < 5.0:  # Within 5m
                self.logger.info(f"Reached waypoint {self.current_waypoint_index+1}")
                self.current_waypoint_index += 1
                
        else:
            # Search pattern complete, restart
            self.logger.info("Search pattern complete, restarting")
            self.current_waypoint_index = 0
            
        # Check search timeout
        if time.time() - self.state_entry_time > self.config.get('search_timeout', 300):
            self.logger.warning("Search timeout, aborting mission")
            self.transition_to(MissionState.ABORT)
            
    def _handle_target_acquire(self):
        """Acquire and validate target marker"""
        detected_markers = self.detector.get_latest_detections()
        
        if self.target_marker_id in detected_markers:
            self.target_confirmations += 1
            
            # Show progress
            if self.target_confirmations % 10 == 0:
                self.logger.info(f"Target confirmations: {self.target_confirmations}/{self.required_confirmations}")
                
            # Check marker quality
            marker_data = detected_markers[self.target_marker_id]
            confidence = marker_data.get('confidence', 0)
            
            if confidence < 0.5:
                self.logger.debug(f"Low confidence detection: {confidence:.2f}")
                self.target_confirmations = max(0, self.target_confirmations - 1)
                
            # Check if we have enough confirmations
            if self.target_confirmations >= self.required_confirmations:
                self.logger.info("Target acquisition complete")
                self.transition_to(MissionState.TARGET_VALIDATE)
        else:
            # Lost target, reduce confirmation count
            self.target_confirmations = max(0, self.target_confirmations - 2)
            
            # If we've lost it for too many frames, go back to search
            if self.target_confirmations == 0 and time.time() - self.state_entry_time > 5.0:
                self.logger.warning("Lost target during acquisition, returning to search")
                self.transition_to(MissionState.SEARCH)
                
    def _handle_target_validate(self):
        """Validate target marker before precision approach"""
        detected_markers = self.detector.get_latest_detections()
        
        if self.target_marker_id in detected_markers:
            marker_data = detected_markers[self.target_marker_id]
            
            # Check distance to marker
            position_3d = marker_data.get('position_3d')
            if position_3d is not None:
                distance_mm = position_3d[2]  # Z coordinate is distance
                
                # Convert to meters
                distance_m = distance_mm / 1000.0
                
                self.logger.info(f"Target distance: {distance_m:.2f}m")
                
                # Check if we're at a good distance to start precision approach
                if distance_m < 15.0:
                    # Validation complete, transition to precision loiter
                    self.logger.info("Target validation complete, starting precision approach")
                    self.transition_to(MissionState.PRECISION_LOITER)
                else:
                    # Move closer to target before precision approach
                    self.logger.info("Moving closer to target")
                    vehicle_state = self.mavlink.get_vehicle_state()
                    
                    # Calculate target position in NED frame
                    landing_target = self.transformer.calculate_landing_target_message(
                        marker_data, vehicle_state
                    )
                    
                    if landing_target:
                        self.mavlink.send_landing_target(landing_target)
        else:
            # Lost target during validation
            self.logger.warning("Lost target during validation")
            
            # If we've lost it for too long, go back to acquisition
            if time.time() - self.state_entry_time > 5.0:
                self.transition_to(MissionState.TARGET_ACQUIRE)
                
    def _handle_precision_loiter(self):
        """Handle precision loiter over the target"""
        # Ensure we're in PRECISION_LOITER mode
        vehicle_state = self.mavlink.get_vehicle_state()
        current_mode = vehicle_state['mode']
        
        if current_mode != 'PRECISION_LOITER' and time.time() - self.state_entry_time < 2.0:
            self.logger.info("Setting mode to PRECISION_LOITER")
            self.mavlink.set_mode('PRECISION_LOITER')
            
        # Get latest marker detections
        detected_markers = self.detector.get_latest_detections()
        
        if self.target_marker_id in detected_markers:
            marker_data = detected_markers[self.target_marker_id]
            
            # Calculate landing target message
            landing_target = self.transformer.calculate_landing_target_message(
                marker_data, vehicle_state
            )
            
            # Send landing target updates to vehicle
            if landing_target:
                self.mavlink.send_landing_target(landing_target)
                
                # Check if marker is centered
                x_offset = abs(landing_target['x'])
                y_offset = abs(landing_target['y'])
                
                # Get current altitude
                current_alt = vehicle_state['relative_altitude']
                
                # Check if we should transition to final approach (flow-based)
                if current_alt <= self.config.get('landing_start_altitude', 10.0):
                    if x_offset < self.config.get('center_tolerance', 0.3) and y_offset < self.config.get('center_tolerance', 0.3):
                        # Marker is centered, update time
                        if self.target_centered_time == 0:
                            self.target_centered_time = time.time()
                            self.logger.info("Target centered, maintaining position")
                            
                        # Check if we've maintained center position long enough
                        if time.time() - self.target_centered_time > self.config.get('validation_time', 3.0):
                            # We're ready for final approach
                            self.logger.info("Target centered and stable, beginning final approach")
                            self.transition_to(MissionState.FINAL_APPROACH)
                    else:
                        # Not centered, reset timer
                        self.target_centered_time = 0
        else:
            # Lost target during precision loiter
            self.logger.warning("Lost target during precision loiter")
            
            # If we've lost it for too long, go back to validation
            if time.time() - self.state_entry_time > 10.0:
                self.transition_to(MissionState.TARGET_VALIDATE)
                
    def _handle_final_approach(self):
        """Handle flow-optimized final approach phase"""
        # This is a new state that uses optical flow for increased stability
        vehicle_state = self.mavlink.get_vehicle_state()
        current_alt = vehicle_state['relative_altitude']
        
        # Ensure we're in GUIDED mode for controlled descent
        if vehicle_state['mode'] != 'GUIDED' and time.time() - self.state_entry_time < 2.0:
            self.logger.info("Setting mode to GUIDED for flow-optimized descent")
            self.mavlink.set_mode('GUIDED')
            
        # Get latest marker detections
        detected_markers = self.detector.get_latest_detections()
        has_valid_marker = self.target_marker_id in detected_markers
        
        # Use flow manager to determine optimal descent strategy
        descent_rate, use_precision_land = self.flow_manager.update_landing_strategy(
            current_alt, has_valid_marker
        )
        
        # Check if we've reached the altitude for precision landing
        if current_alt <= 0.5:  # Final 0.5m
            self.logger.info("Reached precision landing altitude, transitioning to landing")
            self.transition_to(MissionState.PRECISION_LAND)
            return
            
        # If we have valid marker, send landing target update
        if has_valid_marker:
            marker_data = detected_markers[self.target_marker_id]
            landing_target = self.transformer.calculate_landing_target_message(
                marker_data, vehicle_state
            )
            
            if landing_target:
                self.mavlink.send_landing_target(landing_target)
                
                # Check if marker is centered enough to continue descent
                x_offset = abs(landing_target['x'])
                y_offset = abs(landing_target['y'])
                
                if x_offset > self.config.get('center_tolerance', 0.3) * 2 or y_offset > self.config.get('center_tolerance', 0.3) * 2:
                    # Marker not centered enough - reduce descent rate
                    descent_rate = max(0.0, descent_rate * 0.5)
                    self.logger.info(f"Marker offset ({x_offset:.2f}, {y_offset:.2f}) - reducing descent rate")
                    
            # Log flow and position quality
            self.logger.info(
                f"Alt: {current_alt:.2f}m, Flow: {self.flow_manager.get_flow_quality():.0f}%, " +
                f"Descent: {descent_rate:.2f}m/s"
            )
                    
            # Set the descent rate
            self.mavlink.set_descent_rate(descent_rate)
        else:
            # Lost marker during final approach
            self.logger.warning("Lost marker during final approach")
            
            # Decide what to do based on altitude and flow quality
            if self.flow_manager.is_flow_available() and current_alt < self.config.get('final_approach_altitude', 1.0):
                # We have flow data and we're close to ground - continue slower descent
                reduced_rate = self.config.get('final_descent_rate', 0.3) * 0.3
                self.logger.info(f"Using flow for positioning, slow descent at {reduced_rate:.2f}m/s")
                self.mavlink.set_descent_rate(reduced_rate)
            else:
                # No marker and insufficient flow - pause descent and try to reacquire
                self.logger.warning("Insufficient sensor data, pausing descent")
                self.mavlink.set_descent_rate(0.0)
                
                # If we've lost the marker for too long, go back to precision loiter
                if time.time() - self.state_entry_time > 5.0:
                    self.logger.warning("Lost marker for too long, returning to precision loiter")
                    self.transition_to(MissionState.PRECISION_LOITER)
                
    def _handle_precision_land(self):
        """Handle final precision landing phase"""
        # Ensure we're in LAND mode with precision landing enabled
        vehicle_state = self.mavlink.get_vehicle_state()
        current_mode = vehicle_state['mode']
        
        if current_mode != 'LAND' and time.time() - self.state_entry_time < 2.0:
            self.logger.info("Commanding precision landing")
            self.mavlink.command_precision_land()
            
        # Continue to send landing target updates during descent
        detected_markers = self.detector.get_latest_detections()
        
        if self.target_marker_id in detected_markers:
            marker_data = detected_markers[self.target_marker_id]
            
            # Calculate landing target message
            landing_target = self.transformer.calculate_landing_target_message(
                marker_data, vehicle_state
            )
            
            # Send landing target updates to vehicle
            if landing_target:
                self.mavlink.send_landing_target(landing_target)
                
        # Check if we've landed
        if vehicle_state['relative_altitude'] < 0.1 and not vehicle_state['armed']:
            self.logger.info("Landing complete")
            self.transition_to(MissionState.LANDED)
            
        # Check for landing timeout
        if time.time() - self.state_entry_time > self.config.get('landing_timeout', 120):
            self.logger.warning("Landing timeout, executing emergency protocol")
            self.transition_to(MissionState.EMERGENCY)
            
    def _handle_emergency(self):
        """Handle emergency situations"""
        self.logger.error("Executing emergency protocol")
        
        # Execute emergency landing protocol
        if self.safety.execute_emergency_protocol():
            self.logger.info("Emergency protocol activated")
            
        # Wait for vehicle to land or timeout
        vehicle_state = self.mavlink.get_vehicle_state()
        if vehicle_state['relative_altitude'] < 0.5 and not vehicle_state['armed']:
            self.logger.info("Emergency landing complete")
            self.transition_to(MissionState.LANDED)
            
        # Emergency timeout
        if time.time() - self.state_entry_time > 60:
            self.logger.error("Emergency timeout, mission aborted")
            self.transition_to(MissionState.ABORT)
            
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two lat/lon points in meters"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
        
    def _offset_position(self, lat, lon, north_offset, east_offset):
        """Calculate new lat/lon from a position with north/east offsets in meters"""
        # Earth's radius in meters
        earth_radius = 6378137.0
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        # New latitude
        new_lat_rad = lat_rad + (north_offset / earth_radius)
        
        # New longitude
        new_lon_rad = lon_rad + (east_offset / (earth_radius * math.cos(lat_rad)))
        
        # Convert back to degrees
        new_lat = math.degrees(new_lat_rad)
        new_lon = math.degrees(new_lon_rad)
        
        return new_lat, new_lon
        
    def stop(self):
        """Stop the mission"""
        self.logger.info("Stopping mission")
        self.running = False

def setup_logging(log_file=None, verbose=False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to root logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Autonomous Precision Landing System with Optical Flow')
    
    parser.add_argument('--target', '-t', type=int, required=True,
                      help='Marker ID to use as landing target')
    parser.add_argument('--mission-area', type=float, default=DEFAULT_MISSION_CONFIG['search_area_size'],
                      help='Size of search area in meters')
    parser.add_argument('--search-alt', type=float, default=DEFAULT_MISSION_CONFIG['search_altitude'],
                      help='Initial search altitude in meters')
    parser.add_argument('--landing-alt', type=float, default=DEFAULT_MISSION_CONFIG['landing_start_altitude'],
                      help='Altitude to start precision landing in meters')
    parser.add_argument('--final-approach-alt', type=float, default=DEFAULT_MISSION_CONFIG['final_approach_altitude'],
                      help='Altitude to switch to flow-dominated approach')
    parser.add_argument('--connection', type=str, default=DEFAULT_MISSION_CONFIG['mavlink_connection'],
                      help='MAVLink connection string')
    parser.add_argument('--headless', action='store_true',
                      help='Run in headless mode (no GUI)')
    parser.add_argument('--stream', action='store_true',
                      help='Enable video streaming')
    parser.add_argument('--sim', action='store_true',
                      help='Run in simulation mode')
    parser.add_argument('--use-flow', '-f', action='store_true', default=True,
                      help='Enable optical flow sensor integration')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--log-file', type=str, default='flow_mission_log.txt',
                      help='Path to log file')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.verbose)
    logger.info("Starting Enhanced Autonomous Precision Landing System with Flow Integration")
    
    # Create configuration based on arguments
    config = DEFAULT_MISSION_CONFIG.copy()
    config.update({
        'target_marker_id': args.target,
        'search_area_size': args.mission_area,
        'search_altitude': args.search_alt,
        'landing_start_altitude': args.landing_alt,
        'final_approach_altitude': args.final_approach_alt,
        'mavlink_connection': args.connection,
        'use_optical_flow': args.use_flow
    })
    
    # Initialize MAVLink controller
    mavlink = MAVLinkFlightController(
        connection_string=config['mavlink_connection'],
        config=config
    )
    
    # Initialize ArUco detector
    detector = EnhancedOakDArUcoDetector(
        target_id=config['target_marker_id'],
        resolution="adaptive",
        use_cuda=True,
        high_performance=True,
        mavlink_connection=config['mavlink_connection'],
        enable_servo_control=False,
        enable_streaming=args.stream,
        stream_ip="192.168.2.1",
        stream_port=5600,
        headless=args.headless,
        quiet=not args.verbose
    )
    
    # Initialize safety manager
    safety = SafetyManager(config, mavlink)
    
    # Initialize mission state machine
    mission = EnhancedMissionStateMachine(detector, mavlink, safety, config)
    
    try:
        # Start detector
        detector.start()
        detector.start_autonomous_detection()
        
        # Run mission
        final_state = mission.run_mission()
        
        # Report mission result
        if final_state == MissionState.LANDED:
            logger.info("Mission completed successfully")
        elif final_state == MissionState.ABORT:
            logger.warning("Mission aborted")
        else:
            logger.error(f"Mission ended in unexpected state: {final_state.value}")
            
    except KeyboardInterrupt:
        logger.info("Mission interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up
        logger.info("Cleaning up...")
        mission.stop()
        detector.stop()
        mavlink.close()
        
    logger.info("Mission terminated")

if __name__ == "__main__":
    main()
