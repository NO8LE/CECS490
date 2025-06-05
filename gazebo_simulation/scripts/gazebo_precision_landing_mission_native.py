#!/usr/bin/env python3

"""
Native Gazebo Precision Landing Mission Controller (No ROS)

This script implements a mission controller for autonomous precision landing
simulations in Gazebo without requiring ROS. It:
1. Controls the simulated drone via MAVLink
2. Receives ArUco marker detections from the detector
3. Implements the mission state machine for search, loiter, and landing
4. Integrates optical flow data for enhanced landing precision

Usage:
  python3 gazebo_precision_landing_mission_native.py [options]

Options:
  --target-id, -t MARKER_ID  Marker ID to use as landing target (default: 5)
  --search-alt METERS        Initial search altitude in meters (default: 15.0)
  --final-approach-alt ALT   Altitude to start flow-based approach (default: 1.0)
  --config CONFIG_FILE       Path to mission configuration file
  --mavlink-connection CONN  MAVLink connection string (default: udp:localhost:14550)
  --verbose, -v              Enable verbose output
  --headless                 Run in headless mode (no visualization)
  --external-detector        Use external ArUco detector instead of integrated one
"""

import os
import sys
import time
import math
import yaml
import argparse
import threading
import numpy as np
from enum import Enum
from pymavlink import mavutil
from threading import Thread, Lock

# Try to import the native ArUco detector class
try:
    from gazebo_aruco_detector_native import ArUcoDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    print("Warning: gazebo_aruco_detector_native.py not found in Python path")
    print("Running without integrated detector. Detector must be started separately.")

# Mission state definitions
class MissionState(Enum):
    """Mission state machine states"""
    INIT = "initializing"
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
    
    # EKF settings
    'ekf_pos_horiz_variance_threshold': 1.0,  # Maximum acceptable horizontal position variance
    'ekf_pos_vert_variance_threshold': 1.0,   # Maximum acceptable vertical position variance
}

class CoordinateTransformer:
    """Convert between ArUco camera coordinates and MAVLink NED coordinates"""
    
    def __init__(self, camera_orientation='down'):
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
        
        # Create rotation matrix from body to NED frame
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
        if 'tvec' not in marker_data:
            return None
            
        # Get marker position from tvec
        tvec = marker_data['tvec']
        position_3d = (tvec[0], tvec[1], tvec[2])
        
        # Get NED coordinates
        north, east, down = self.aruco_to_ned(
            position_3d, 
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
            'target_num': marker_data['id'] if 'id' in marker_data else 0,
            'frame': mavutil.mavlink.MAV_FRAME_BODY_NED,
            'x': rel_north,
            'y': rel_east, 
            'z': rel_down,
            'size_x': 0.3048,  # marker size in meters
            'size_y': 0.3048,
            'type': 1,  # LANDING_TARGET_TYPE_VISION_FIDUCIAL
            'position_valid': 1
        }

class MAVLinkController:
    """MAVLink interface for controlling the simulated drone"""
    
    def __init__(self, connection_string, config=None):
        self.connection_string = connection_string
        self.connection = None
        self.target_system = 1
        self.target_component = 1
        self.last_heartbeat = 0
        self.vehicle_state = {}
        self.config = config or {}
        self.running = True
        
        # Optical flow data
        self.optical_flow_quality = 0
        self.optical_flow_valid = False
        
        # EKF status
        self.ekf_status = {
            'velocity_variance': 0.1,
            'pos_horiz_variance': 0.2,
            'pos_vert_variance': 0.2,
            'compass_variance': 0.1,
            'terrain_alt_variance': 0.0,
            'flags': 0x1F,  # All healthy by default
            'healthy': True
        }
        
        # Initialize state monitoring
        self.state_lock = Lock()
        
        # Connect to vehicle
        self.connect()
        
        # Start heartbeat monitoring thread
        self.heartbeat_thread = Thread(target=self._monitor_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        # Start vehicle state monitoring thread
        self.state_thread = Thread(target=self._monitor_vehicle_state)
        self.state_thread.daemon = True
        self.state_thread.start()
        
    def connect(self):
        """Connect to the MAVLink device"""
        try:
            print(f"Connecting to MAVLink device at {self.connection_string}")
            self.connection = mavutil.mavlink_connection(self.connection_string)
            return True
        except Exception as e:
            print(f"Error connecting to MAVLink: {e}")
            return False
        
    def wait_for_connection(self, timeout=30):
        """Wait for MAVLink connection with timeout"""
        if not self.connection:
            print("No MAVLink connection established")
            return False
            
        start_time = time.time()
        print("Waiting for MAVLink heartbeat...")
        
        while time.time() - start_time < timeout:
            if self.connection.wait_heartbeat(timeout=1):
                print("MAVLink connection established")
                # Set the system and component ID for sending commands
                self.target_system = self.connection.target_system
                self.target_component = self.connection.target_component
                return True
                
        print("Timed out waiting for MAVLink heartbeat")
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
                print(f"Error receiving heartbeat: {e}")
                
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
                with self.state_lock:
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
                print(f"Error monitoring vehicle state: {e}")
                
            time.sleep(0.1)
    
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
            print("No MAVLink connection established")
            return False
            
        mode_mapping = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'CIRCLE': 7,
            'LAND': 9, 'PRECISION_LOITER': 16, 'GUIDED_NOGPS': 20
        }
        
        if mode_name not in mode_mapping:
            print(f"Unknown mode: {mode_name}")
            return False
            
        mode_id = mode_mapping[mode_name]
        
        print(f"Setting flight mode to {mode_name}")
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
                    print(f"Command {command} accepted")
                    return True
                else:
                    print(f"Command {command} failed with result {msg.result}")
                    return False
            time.sleep(0.1)
            
        print(f"Timed out waiting for command {command} acknowledgment")
        return False
        
    def send_landing_target(self, target_data):
        """Send LANDING_TARGET message to vehicle"""
        if not self.connection or not target_data:
            return False
            
        try:
            self.connection.mav.landing_target_send(
                target_data['time_usec'],
                target_data.get('target_num', 0), 
                target_data.get('frame', mavutil.mavlink.MAV_FRAME_BODY_NED),
                target_data.get('x', 0), 
                target_data.get('y', 0), 
                target_data.get('z', 0),
                target_data.get('size_x', 0), 
                target_data.get('size_y', 0),
                0, 0, 0, 0,  # quaternion (not used)
                target_data.get('type', 1), 
                target_data.get('position_valid', 1)
            )
            return True
        except Exception as e:
            print(f"Error sending LANDING_TARGET message: {e}")
            return False
            
    def set_position_target(self, x, y, z, vx=0, vy=0, vz=0, yaw=0, yaw_rate=0):
        """Send SET_POSITION_TARGET_LOCAL_NED message to vehicle"""
        if not self.connection:
            return False
            
        # Calculate appropriate type_mask
        type_mask = (
            # Position and velocity control
            0b0000_0000_0111_1000 |  # Use position
            0b0000_0000_0000_0111    # Ignore acceleration
        )
        
        if vx != 0 or vy != 0 or vz != 0:
            # Also include velocity
            type_mask &= ~(0b0000_0000_0111_0000)  # Use velocity
            
        if yaw != 0:
            # Include yaw
            type_mask &= ~(0b0000_0100_0000_0000)  # Use yaw
        
        if yaw_rate != 0:
            # Include yaw rate
            type_mask &= ~(0b0000_1000_0000_0000)  # Use yaw rate
            
        try:
            self.connection.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms - use system time
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
                type_mask,  # type_mask
                x, y, z,  # position
                vx, vy, vz,  # velocity
                0, 0, 0,  # acceleration
                yaw, yaw_rate  # yaw, yaw_rate
            )
            return True
        except Exception as e:
            print(f"Error sending SET_POSITION_TARGET_LOCAL_NED: {e}")
            return False
            
    def arm(self, arm=True):
        """Arm or disarm the vehicle"""
        if not self.connection:
            print("No MAVLink connection established")
            return False
            
        arm_val = 1 if arm else 0
        print(f"{'Arming' if arm else 'Disarming'} vehicle")
        
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
            print("No MAVLink connection established")
            return False
            
        # Ensure we're in guided mode
        with self.state_lock:
            current_mode = self.get_vehicle_state().get('mode', 'UNKNOWN')
        
        if current_mode != 'GUIDED':
            print("Setting mode to GUIDED for takeoff")
            if not self.set_mode('GUIDED'):
                print("Failed to set GUIDED mode for takeoff")
                return False
                
        # Ensure vehicle is armed
        with self.state_lock:
            if not self.get_vehicle_state().get('armed', False):
                print("Arming vehicle for takeoff")
                if not self.arm(True):
                    print("Failed to arm vehicle for takeoff")
                    return False
                
        # Send takeoff command
        print(f"Commanding takeoff to {altitude}m altitude")
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
            print("No MAVLink connection established")
            return False
            
        print(f"Commanding loiter at ({lat}, {lon}, {alt}m) with radius {radius}m")
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
            print("No MAVLink connection established")
            return False
            
        print("Commanding precision landing")
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
            print("No MAVLink connection established")
            return False
            
        print("Commanding return to launch (RTL)")
        return self.set_mode('RTL')
        
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
        
        print(f"Setting descent rate to {rate_mps} m/s")
        
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
            print(f"Error setting descent rate: {e}")
            return False
        
    def get_vehicle_state(self):
        """Get current vehicle state from MAVLink messages"""
        with self.state_lock:
            return {
                'attitude': self.vehicle_state.get('attitude', (0, 0, 0)),
                'position': self.vehicle_state.get('position', (0, 0, 0)),
                'relative_altitude': self.vehicle_state.get('relative_altitude', 0),
                'battery_voltage': self.vehicle_state.get('battery_voltage', 0),
                'battery_remaining': self.vehicle_state.get('battery_remaining', 0),
                'mode': self.vehicle_state.get('mode', 'UNKNOWN'),
                'armed': self.vehicle_state.get('armed', False),
                'flow_quality': self.vehicle_state.get('flow_quality', 0),
                'ekf_status': self.vehicle_state.get('ekf_status', self.ekf_status)
            }
    
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
        flags_healthy = (self.ekf_status['flags'] & 0x1F) == 0x1F
        
        # Check variances are within acceptable limits
        pos_horiz_ok = self.ekf_status['pos_horiz_variance'] < self.config.get('ekf_pos_horiz_variance_threshold', 1.0)
        pos_vert_ok = self.ekf_status['pos_vert_variance'] < self.config.get('ekf_pos_vert_variance_threshold', 1.0)
        
        return flags_healthy and pos_horiz_ok and pos_vert_ok
        
    def close(self):
        """Close the MAVLink connection and stop threads"""
        print("Closing MAVLink connection")
        self.running = False
        
        if self.connection:
            self.connection.close()
            self.connection = None

class FlowIntegrationManager:
    """Manages the integration of optical flow with vision-based positioning"""
    
    def __init__(self, mavlink_controller, config=None):
        self.mavlink = mavlink_controller
        self.config = config or {}
        self.flow_enabled = self.config.get('use_optical_flow', True)
        self.flow_quality_history = []  # Track flow quality over time
        self.current_descent_rate = 0.0
        
        # Configure flow settings
        self.flow_quality_threshold = self.config.get('flow_quality_threshold', 50)
        self.position_variance_threshold = self.config.get('position_variance_threshold', 0.5)
        self.flow_descent_rate = self.config.get('final_descent_rate', 0.3)
    
    def update_flow_quality(self, quality):
        """Update flow quality history"""
        self.flow_quality_history.append(quality)
        
        # Keep only the last 10 samples
        if len(self.flow_quality_history) > 10:
            self.flow_quality_history.pop(0)
    
    def get_flow_quality_average(self):
        """Get average flow quality from recent history"""
        if not self.flow_quality_history:
            return 0
            
        return sum(self.flow_quality_history) / len(self.flow_quality_history)
    
    def is_flow_reliable(self):
        """Check if optical flow is reliable for landing"""
        if not self.flow_enabled or not self.mavlink:
            return False
            
        # Get flow quality from MAVLink
        flow_quality = self.mavlink.get_vehicle_state().get('flow_quality', 0)
        
        # Update history
        self.update_flow_quality(flow_quality)
        
        # Check if average quality is above threshold
        avg_quality = self.get_flow_quality_average()
        return avg_quality >= self.flow_quality_threshold
    
    def calculate_descent_rate(self, altitude):
        """Calculate appropriate descent rate based on altitude and flow quality"""
        if not self.is_flow_reliable():
            # If flow is not reliable, use a very slow descent rate
            return 0.1
            
        # Base descent rate from config
        base_rate = self.flow_descent_rate
        
        # Get current flow quality
        flow_quality = self.get_flow_quality_average()
        quality_factor = min(flow_quality / 100.0, 1.0)  # 0.0-1.0 based on quality
        
        # Calculate rate based on altitude:
        # - Higher altitude: faster descent
        # - Lower altitude: slower, more careful descent
        altitude_factor = min(altitude / 5.0, 1.0)  # Scale with altitude up to 5m
        
        # Combine factors to get final rate
        rate = base_rate * quality_factor * altitude_factor
        
        # Ensure minimum and maximum rates
        rate = max(0.1, min(rate, base_rate))
        
        # For very low altitudes, always use slowest rate
        if altitude < 0.5:
            rate = 0.1
            
        self.current_descent_rate = rate
        return rate


class PrecisionLandingMissionController:
    """Main mission controller for autonomous precision landing"""
    
    def __init__(self, args):
        # Store arguments
        self.args = args
        self.verbose = args.verbose
        self.headless = args.headless
        
        # Load configuration
        self.config = DEFAULT_MISSION_CONFIG.copy()
        if args.config:
            self.load_config(args.config)
            
        # Override config with command line arguments
        if args.target_id:
            self.config['target_marker_id'] = args.target_id
        if args.search_alt:
            self.config['search_altitude'] = args.search_alt
        if args.final_approach_alt:
            self.config['final_approach_altitude'] = args.final_approach_alt
            
        # Initialize state
        self.mission_state = MissionState.INIT
        self.mission_start_time = None
        self.state_entry_time = time.time()
        self.target_marker_id = self.config['target_marker_id']
        self.target_detections = []  # List of recent target detections
        self.target_validated = False
        self.target_position = None
        
        # Connect to MAVLink
        self.mavlink = MAVLinkController(
            args.mavlink_connection, 
            config=self.config
        )
        
        # Initialize coordinate transformer
        self.transformer = CoordinateTransformer('down')
        
        # Initialize flow manager
        self.flow_manager = FlowIntegrationManager(self.mavlink, self.config)
        
        # Initialize detector if available
        self.detector = None
        if DETECTOR_AVAILABLE and not args.external_detector:
            try:
                # Create detector with appropriate settings
                detector_args = argparse.Namespace(
                    target_id=self.target_marker_id,
                    camera_topic=args.camera_topic,
                    marker_size=self.config.get('marker_size', 0.3048),
                    headless=args.headless,
                    mavlink_connection=None,  # We'll handle MAVLink ourselves
                    verbose=args.verbose
                )
                self.detector = ArUcoDetector(detector_args)
                print("Created internal ArUco detector")
            except Exception as e:
                print(f"Failed to create internal detector: {e}")
                self.detector = None
        
        # State machine handlers
        self.state_handlers = {
            MissionState.INIT: self._handle_init,
            MissionState.TAKEOFF: self._handle_takeoff,
            MissionState.SEARCH: self._handle_search,
            MissionState.TARGET_ACQUIRE: self._handle_target_acquire,
            MissionState.TARGET_VALIDATE: self._handle_target_validate,
            MissionState.PRECISION_LOITER: self._handle_precision_loiter,
            MissionState.FINAL_APPROACH: self._handle_final_approach,
            MissionState.PRECISION_LAND: self._handle_precision_land,
            MissionState.LANDED: self._handle_landed,
            MissionState.ABORT: self._handle_abort,
            MissionState.EMERGENCY: self._handle_emergency
        }
        
        # Search pattern variables
        self.search_waypoints = []
        self.current_search_waypoint = 0
        self.search_complete = False
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            # Update config with loaded values
            for key, value in yaml_config.items():
                self.config[key] = value
                
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading configuration from {config_file}: {e}")
    
    def start(self):
        """Start the mission controller"""
        print("Starting precision landing mission controller")
        
        # Connect to MAVLink
        if not self.mavlink.wait_for_connection():
            print("Failed to connect to MAVLink. Exiting.")
            return False
            
        # Start detector if available
        if self.detector:
            self.detector.start()
            
        # Start mission
        self.mission_start_time = time.time()
        self._enter_state(MissionState.INIT)
        
        return True
    
    def update(self):
        """Update mission state machine"""
        # Check for timeout
        if self.mission_start_time and time.time() - self.mission_start_time > self.config['max_mission_time']:
            print("Mission timeout reached")
            self._enter_state(MissionState.ABORT)
            
        # Check battery voltage
        battery_voltage = self.mavlink.get_vehicle_state().get('battery_voltage', 0)
        if battery_voltage < self.config['min_battery_voltage'] and battery_voltage > 1.0:
            print(f"Low battery voltage: {battery_voltage}V")
            self._enter_state(MissionState.ABORT)
            
        # Process marker detections if we have a detector
        if self.detector:
            # Get latest detections
            detections = self.detector.get_latest_detections()
            
            # Check if target marker is detected
            target_id = self.config['target_marker_id']
            if target_id in detections:
                # We have a target detection
                target_data = detections[target_id]
                self._process_target_detection(target_data)
        
        # Run state handler
        handler = self.state_handlers.get(self.mission_state)
        if handler:
            handler()
    
    def _process_target_detection(self, detection_data):
        """Process target marker detection"""
        # Add timestamp if not present
        if 'timestamp' not in detection_data:
            detection_data['timestamp'] = time.time()
            
        # Add to recent detections
        self.target_detections.append(detection_data)
        
        # Keep only the most recent detections
        max_detections = self.config['confirmation_frames']
        if len(self.target_detections) > max_detections:
            self.target_detections = self.target_detections[-max_detections:]
            
        # Get vehicle state for coordinate transformation
        vehicle_state = self.mavlink.get_vehicle_state()
        
        # Calculate landing target message
        target_message = self.transformer.calculate_landing_target_message(
            detection_data, vehicle_state
        )
        
        # Send to MAVLink if in appropriate state
        if self.mission_state in [
            MissionState.TARGET_ACQUIRE,
            MissionState.TARGET_VALIDATE,
            MissionState.PRECISION_LOITER,
            MissionState.FINAL_APPROACH,
            MissionState.PRECISION_LAND
        ]:
            if target_message:
                self.mavlink.send_landing_target(target_message)
    
    def _enter_state(self, new_state):
        """Enter a new mission state"""
        old_state = self.mission_state
        if new_state != old_state:
            print(f"Mission state change: {old_state.value} -> {new_state.value}")
            self.mission_state = new_state
            self.state_entry_time = time.time()
    
    def _handle_init(self):
        """Handle INIT state"""
        # Check if MAVLink is connected
        if not self.mavlink.connection:
            return
            
        # Proceed to takeoff
        self._enter_state(MissionState.TAKEOFF)
    
    def _handle_takeoff(self):
        """Handle TAKEOFF state"""
        # Get current altitude
        altitude = self.mavlink.get_vehicle_state().get('relative_altitude', 0)
        
        if altitude < 0.5:  # Still on the ground
            # Command takeoff if not already in progress
            state_time = time.time() - self.state_entry_time
            if state_time > 2.0:  # Wait a moment before sending takeoff command
                self.mavlink.takeoff(self.config['search_altitude'])
        elif altitude >= 0.9 * self.config['search_altitude']:
            # Reached search altitude, switch to search mode
            print(f"Reached search altitude: {altitude}m")
            self._enter_state(MissionState.SEARCH)
    
    def _handle_search(self):
        """Handle SEARCH state"""
        # Check for timeout
        state_time = time.time() - self.state_entry_time
        if state_time > self.config['search_timeout']:
            print("Search timeout reached")
            self._enter_state(MissionState.ABORT)
            return
            
        # Check if we have any target detections
        if self.target_detections:
            # Check how many recent detections we have
            recent_detections = [d for d in self.target_detections 
                              if time.time() - d['timestamp'] < 2.0]
                              
            if len(recent_detections) >= self.config['confirmation_frames']:
                print(f"Target marker {self.target_marker_id} detected during search")
                self._enter_state(MissionState.TARGET_ACQUIRE)
                return
        
        # If no detections, continue search pattern
        self._execute_search_pattern()
    
    def _handle_target_acquire(self):
        """Handle TARGET_ACQUIRE state"""
        # Check if we still have recent detections
        recent_detections = [d for d in self.target_detections 
                          if time.time() - d['timestamp'] < 1.0]
                          
        if not recent_detections:
            print("Lost target during acquisition")
            self._enter_state(MissionState.SEARCH)
            return
            
        # Set LOITER mode to hold position while we validate the target
        vehicle_state = self.mavlink.get_vehicle_state()
        if vehicle_state.get('mode') != 'LOITER':
            self.mavlink.set_mode('LOITER')
            
        # Move to validation state
        self._enter_state(MissionState.TARGET_VALIDATE)
    
    def _handle_target_validate(self):
        """Handle TARGET_VALIDATE state"""
        # Check how long we've been validating
        validation_time = time.time() - self.state_entry_time
        
        # Check if we have enough recent detections
        recent_detections = [d for d in self.target_detections 
                          if time.time() - d['timestamp'] < 1.0]
        
        # For validation, we need consistent detections
        if len(recent_detections) < self.config['confirmation_frames']:
            print("Lost consistent target detection during validation")
            if validation_time > 5.0:  # Give some time to reacquire
                self._enter_state(MissionState.SEARCH)
            return
            
        # Check if we've validated long enough
        if validation_time >= self.config['validation_time']:
            print(f"Target validated after {validation_time:.1f} seconds")
            self.target_validated = True
            self._enter_state(MissionState.PRECISION_LOITER)
    
    def _handle_precision_loiter(self):
        """Handle PRECISION_LOITER state"""
        # Check if we still have recent detections
        recent_detections = [d for d in self.target_detections 
                          if time.time() - d['timestamp'] < 1.0]
                          
        if not recent_detections:
            print("Lost target during precision loiter")
            self._enter_state(MissionState.TARGET_ACQUIRE)
            return
            
        # Set PRECISION_LOITER mode if needed
        vehicle_state = self.mavlink.get_vehicle_state()
        current_mode = vehicle_state.get('mode', 'UNKNOWN')
        
        if current_mode != 'PRECISION_LOITER' and current_mode != 'LOITER':
            if 'PRECISION_LOITER' in self.mavlink._interpret_mode(16):  # Check if available
                self.mavlink.set_mode('PRECISION_LOITER')
            else:
                self.mavlink.set_mode('LOITER')
                
        # Check if we're stable in loiter mode
        loiter_time = time.time() - self.state_entry_time
        
        # Get current altitude
        altitude = vehicle_state.get('relative_altitude', 0)
        
        # If we've been loitering long enough and have a good target lock,
        # start final approach
        if loiter_time > 5.0 and len(recent_detections) >= self.config['confirmation_frames']:
            # If we're already at or below final approach altitude, go directly to precision land
            if altitude <= self.config['final_approach_altitude']:
                print(f"Already at final approach altitude ({altitude}m), proceeding to precision land")
                self._enter_state(MissionState.PRECISION_LAND)
            else:
                print(f"Starting final approach from {altitude}m")
                self._enter_state(MissionState.FINAL_APPROACH)
    
    def _handle_final_approach(self):
        """Handle FINAL_APPROACH state"""
        # Check if we still have recent detections
        recent_detections = [d for d in self.target_detections 
                          if time.time() - d['timestamp'] < 1.0]
                          
        if not recent_detections:
            print("Lost target during final approach")
            self._enter_state(MissionState.PRECISION_LOITER)
            return
            
        # Get current altitude
        vehicle_state = self.mavlink.get_vehicle_state()
        altitude = vehicle_state.get('relative_altitude', 0)
        
        # If using optical flow, check its health
        use_flow = self.config.get('use_optical_flow', True)
        flow_healthy = self.flow_manager.is_flow_reliable() if use_flow else False
        
        # Set mode to GUIDED for controlled descent
        if vehicle_state.get('mode') != 'GUIDED':
            self.mavlink.set_mode('GUIDED')
            
        # Calculate appropriate descent rate
        if use_flow and flow_healthy:
            # Use flow-based descent rate
            descent_rate = self.flow_manager.calculate_descent_rate(altitude)
            self.mavlink.set_descent_rate(descent_rate)
            
            if self.verbose:
                print(f"Flow-guided descent at {descent_rate:.2f} m/s (alt: {altitude:.2f}m)")
        else:
            # Use standard descent rate
            descent_rate = 0.5 * (altitude / self.config['landing_start_altitude'])
            descent_rate = max(0.2, min(descent_rate, 0.5))  # Keep between 0.2-0.5 m/s
            self.mavlink.set_descent_rate(descent_rate)
            
            if self.verbose:
                print(f"Standard descent at {descent_rate:.2f} m/s (alt: {altitude:.2f}m)")
        
        # Check if we've reached final approach altitude
        if altitude <= self.config['final_approach_altitude']:
            print(f"Reached final approach altitude: {altitude:.2f}m")
            self._enter_state(MissionState.PRECISION_LAND)
    
    def _handle_precision_land(self):
        """Handle PRECISION_LAND state"""
        # Check if we still have recent detections
        recent_detections = [d for d in self.target_detections 
                          if time.time() - d['timestamp'] < 1.0]
                          
        # Even without detections, continue landing if we're low enough
        vehicle_state = self.mavlink.get_vehicle_state()
        altitude = vehicle_state.get('relative_altitude', 0)
        
        # If we're very low, continue landing even without detections
        if not recent_detections and altitude > 0.5:
            print("Lost target during precision landing")
            if altitude > 1.0:
                self._enter_state(MissionState.PRECISION_LOITER)
                return
        
        # Check landing timeout
        landing_time = time.time() - self.state_entry_time
        if landing_time > self.config['landing_timeout']:
            print("Landing timeout reached")
            self._enter_state(MissionState.ABORT)
            return
            
        # If not already in LAND mode, command precision landing
        current_mode = vehicle_state.get('mode', 'UNKNOWN')
        if current_mode != 'LAND':
            self.mavlink.command_precision_land()
            
        # Check if we've landed
        if altitude < 0.1 or not vehicle_state.get('armed', True):
            print("Landing complete")
            self._enter_state(MissionState.LANDED)
    
    def _handle_landed(self):
        """Handle LANDED state"""
        # Ensure the vehicle is disarmed
        vehicle_state = self.mavlink.get_vehicle_state()
        if vehicle_state.get('armed', False):
            self.mavlink.arm(False)  # Disarm
        
        # Nothing else to do in landed state
        pass
    
    def _handle_abort(self):
        """Handle ABORT state"""
        # Command RTL for safe return
        self.mavlink.command_rtl()
        
        # Nothing else to do in abort state
        pass
    
    def _handle_emergency(self):
        """Handle EMERGENCY state"""
        # In emergency, try to land immediately wherever we are
        self.mavlink.set_mode('LAND')
        
        # Nothing else to do in emergency state
        pass
    
    def _execute_search_pattern(self):
        """Execute search pattern logic"""
        # Generate search pattern if we don't have one
        if not self.search_waypoints:
            self._generate_search_pattern()
            
        # If we've completed the pattern, restart
        if self.search_complete:
            print("Restarting search pattern")
            self.current_search_waypoint = 0
            self.search_complete = False
            
        # Check if we have waypoints
        if not self.search_waypoints:
            print("No search waypoints available")
            return
            
        # Get current waypoint
        if self.current_search_waypoint < len(self.search_waypoints):
            waypoint = self.search_waypoints[self.current_search_waypoint]
            
            # Get current position and check if we've reached the waypoint
            vehicle_state = self.mavlink.get_vehicle_state()
            current_pos = vehicle_state.get('position', (0, 0, 0))
            current_alt = vehicle_state.get('relative_altitude', 0)
            
            # Set GUIDED mode if needed
            if vehicle_state.get('mode') != 'GUIDED':
                self.mavlink.set_mode('GUIDED')
                
            # Command movement to waypoint
            self.mavlink.set_position_target(
                waypoint[0], waypoint[1], -self.config['search_altitude']
            )
            
            # Check if we've reached the waypoint
            # Simple 2D distance check for now
            dx = waypoint[0] - current_pos[0]
            dy = waypoint[1] - current_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 1.0:  # Within 1m of waypoint
                print(f"Reached search waypoint {self.current_search_waypoint+1}/{len(self.search_waypoints)}")
                self.current_search_waypoint += 1
                
                # Check if we've completed the pattern
                if self.current_search_waypoint >= len(self.search_waypoints):
                    print("Search pattern complete")
                    self.search_complete = True
        else:
            # If no more waypoints, just loiter
            if not self.search_complete:
                print("Search pattern complete")
                self.search_complete = True
                
            # Set LOITER mode
            self.mavlink.set_mode('LOITER')
    
    def _generate_search_pattern(self):
        """Generate a search pattern over the target area"""
        # Get search area size and spacing
        area_size = self.config['search_area_size']
        spacing = self.config['search_pattern_spacing']
        
        # Calculate number of legs
        num_legs = int(area_size / spacing) + 1
        
        # Generate a lawn-mower pattern
        waypoints = []
        for i in range(num_legs):
            # Alternate direction on each leg
            if i % 2 == 0:
                waypoints.append((-area_size/2, -area_size/2 + i*spacing))
                waypoints.append((area_size/2, -area_size/2 + i*spacing))
            else:
                waypoints.append((area_size/2, -area_size/2 + i*spacing))
                waypoints.append((-area_size/2, -area_size/2 + i*spacing))
        
        # Add final center point
        waypoints.append((0, 0))
        
        self.search_waypoints = waypoints
        print(f"Generated search pattern with {len(waypoints)} waypoints")
    
    def stop(self):
        """Stop the mission controller"""
        print("Stopping precision landing mission controller")
        
        # Stop detector if available
        if self.detector:
            self.detector.stop()
            
        # Close MAVLink connection
        self.mavlink.close()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Native Gazebo Precision Landing Mission Controller')
    
    parser.add_argument('--target-id', '-t', type=int, default=None,
                        help='Target marker ID (default: from config)')
    
    parser.add_argument('--search-alt', type=float, default=None,
                        help='Search altitude in meters (default: from config)')
    
    parser.add_argument('--final-approach-alt', type=float, default=None,
                        help='Final approach altitude in meters (default: from config)')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to mission configuration file')
    
    parser.add_argument('--mavlink-connection', type=str, 
                        default='udp:localhost:14550',
                        help='MAVLink connection string (default: udp:localhost:14550)')
    
    parser.add_argument('--camera-topic', type=str, 
                        default='/drone/camera/link/camera/image',
                        help='Gazebo camera topic for integrated detector')
    
    parser.add_argument('--external-detector', action='store_true',
                        help='Use external ArUco detector instead of integrated one')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no visualization)')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create mission controller
    controller = PrecisionLandingMissionController(args)
    
    try:
        # Start controller
        if not controller.start():
            print("Failed to start mission controller")
            return 1
            
        # Main loop
        print("Mission controller running. Press Ctrl+C to exit.")
        while True:
            controller.update()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        controller.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
