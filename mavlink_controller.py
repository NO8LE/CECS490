#!/usr/bin/env python3

"""
MAVLink Controller for Precision Landing

This module provides a MAVLink interface for controlling a drone with a focus on
precision landing operations.
"""

import time
import math
import logging
import numpy as np
from threading import Thread, Lock
from pymavlink import mavutil

# Set up logging
logger = logging.getLogger("MAVLinkController")

class MAVLinkController:
    """MAVLink interface for controlling the drone"""
    
    def __init__(self, connection_string, config=None, simulation=False):
        self.connection_string = connection_string
        self.connection = None
        self.target_system = 1
        self.target_component = 1
        self.last_heartbeat = 0
        self.vehicle_state = {}
        self.config = config or {}
        self.running = True
        self.simulation = simulation
        
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
        if not simulation:
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
            logger.info(f"Connecting to MAVLink device at {self.connection_string}")
            # For serial connections
            if self.connection_string.startswith('/dev/'):
                baudrate = self.config.get('mavlink_baudrate', 921600)
                self.connection = mavutil.mavlink_connection(
                    self.connection_string, 
                    baud=baudrate
                )
            # For UDP/TCP connections
            else:
                self.connection = mavutil.mavlink_connection(self.connection_string)
                
            return True
        except Exception as e:
            logger.error(f"Error connecting to MAVLink: {e}")
            return False
        
    def wait_for_connection(self, timeout=30):
        """Wait for MAVLink connection with timeout"""
        if self.simulation:
            logger.info("Simulation mode: Skipping MAVLink connection wait")
            return True
            
        if not self.connection:
            logger.error("No MAVLink connection established")
            return False
            
        start_time = time.time()
        logger.info("Waiting for MAVLink heartbeat...")
        
        while time.time() - start_time < timeout:
            if self.connection.wait_heartbeat(timeout=1):
                logger.info("MAVLink connection established")
                # Set the system and component ID for sending commands
                self.target_system = self.connection.target_system
                self.target_component = self.connection.target_component
                return True
                
        logger.error("Timed out waiting for MAVLink heartbeat")
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
                logger.warning(f"Error receiving heartbeat: {e}")
                
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
                            
                        # OPTICAL_FLOW_RAD message (for optical flow data)
                        elif msg_type == "OPTICAL_FLOW_RAD":
                            self.optical_flow_quality = msg.quality
                            self.optical_flow_valid = (msg.quality >= self.config.get('flow_quality_threshold', 50))
                            self.vehicle_state['flow_quality'] = msg.quality
                            self.vehicle_state['flow_comp_m_x'] = msg.flow_comp_m_x
                            self.vehicle_state['flow_comp_m_y'] = msg.flow_comp_m_y
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
                logger.warning(f"Error monitoring vehicle state: {e}")
                
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
        if self.simulation:
            logger.info(f"Simulation: Setting mode to {mode_name}")
            return True
            
        if not self.connection:
            logger.error("No MAVLink connection established")
            return False
            
        mode_mapping = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'CIRCLE': 7,
            'LAND': 9, 'PRECISION_LOITER': 16, 'GUIDED_NOGPS': 20
        }
        
        if mode_name not in mode_mapping:
            logger.error(f"Unknown mode: {mode_name}")
            return False
            
        mode_id = mode_mapping[mode_name]
        
        logger.info(f"Setting flight mode to {mode_name}")
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
        if self.simulation:
            return True
            
        if not self.connection:
            return False
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.connection.recv_match(type='COMMAND_ACK', blocking=False)
            if msg and msg.command == command:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logger.info(f"Command {command} accepted")
                    return True
                else:
                    logger.warning(f"Command {command} failed with result {msg.result}")
                    return False
            time.sleep(0.1)
            
        logger.warning(f"Timed out waiting for command {command} acknowledgment")
        return False
        
    def send_landing_target(self, angle_x, angle_y, distance, target_id=0):
        """Send LANDING_TARGET message for precision landing"""
        if self.simulation:
            return True
            
        if not self.connection:
            return False
            
        try:
            # Send LANDING_TARGET message
            self.connection.mav.landing_target_send(
                int(time.time() * 1e6),  # time_usec (microseconds)
                target_id,                # target_num
                mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
                angle_x,                  # angle_x (rad)
                angle_y,                  # angle_y (rad)
                distance,                 # distance (m)
                0.0,                      # size_x (m) - optional
                0.0,                      # size_y (m) - optional
                0.0, 0.0, 0.0,            # x, y, z (m) - optional 3D position
                (0, 0, 0, 0),             # q - optional orientation quaternion
                mavutil.mavlink.LANDING_TARGET_TYPE_VISION_FIDUCIAL,  # type (1=fiducial marker)
                1                         # position_valid (1=valid)
            )
            return True
        except Exception as e:
            logger.error(f"Error sending LANDING_TARGET message: {e}")
            return False
            
    def arm(self, arm=True):
        """Arm or disarm the vehicle"""
        if self.simulation:
            logger.info(f"Simulation: {'Arming' if arm else 'Disarming'} vehicle")
            return True
            
        if not self.connection:
            logger.error("No MAVLink connection established")
            return False
            
        arm_val = 1 if arm else 0
        logger.info(f"{'Arming' if arm else 'Disarming'} vehicle")
        
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
        if self.simulation:
            logger.info(f"Simulation: Taking off to {altitude}m")
            return True
            
        if not self.connection:
            logger.error("No MAVLink connection established")
            return False
            
        # Ensure we're in guided mode
        with self.state_lock:
            current_mode = self.get_vehicle_state().get('mode', 'UNKNOWN')
        
        if current_mode != 'GUIDED':
            logger.info("Setting mode to GUIDED for takeoff")
            if not self.set_mode('GUIDED'):
                logger.error("Failed to set GUIDED mode for takeoff")
                return False
                
        # Ensure vehicle is armed
        with self.state_lock:
            if not self.get_vehicle_state().get('armed', False):
                logger.info("Arming vehicle for takeoff")
                if not self.arm(True):
                    logger.error("Failed to arm vehicle for takeoff")
                    return False
                
        # Send takeoff command
        logger.info(f"Commanding takeoff to {altitude}m altitude")
        self.connection.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # confirmation
            0, 0, 0, 0, 0, 0,  # unused parameters
            altitude  # param7 - altitude
        )
        
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        
    def command_precision_land(self, lat=0, lon=0, alt=0):
        """Command precision landing"""
        if self.simulation:
            logger.info("Simulation: Commanding precision landing")
            return True
            
        if not self.connection:
            logger.error("No MAVLink connection established")
            return False
            
        logger.info("Commanding precision landing")
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
        if self.simulation:
            logger.info("Simulation: Commanding RTL")
            return True
            
        if not self.connection:
            logger.error("No MAVLink connection established")
            return False
            
        logger.info("Commanding return to launch (RTL)")
        return self.set_mode('RTL')
        
    def set_position_target(self, x, y, z, vx=0, vy=0, vz=0, yaw=0, yaw_rate=0, coordinate_frame=None):
        """Send SET_POSITION_TARGET_LOCAL_NED message to move the vehicle"""
        if self.simulation:
            logger.info(f"Simulation: Setting position target to ({x}, {y}, {z})")
            return True
            
        if not self.connection:
            return False
            
        # Default to MAV_FRAME_LOCAL_NED if not specified
        if coordinate_frame is None:
            coordinate_frame = mavutil.mavlink.MAV_FRAME_LOCAL_NED
            
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
                coordinate_frame,  # frame
                type_mask,  # type_mask
                x, y, z,  # position
                vx, vy, vz,  # velocity
                0, 0, 0,  # acceleration
                yaw, yaw_rate  # yaw, yaw_rate
            )
            return True
        except Exception as e:
            logger.error(f"Error sending SET_POSITION_TARGET_LOCAL_NED: {e}")
            return False
            
    def set_attitude_target(self, roll, pitch, yaw=None, thrust=0.5):
        """Send SET_ATTITUDE_TARGET message to control vehicle attitude"""
        if self.simulation:
            logger.info(f"Simulation: Setting attitude target to ({roll}, {pitch}, {yaw if yaw else 'None'})")
            return True
            
        if not self.connection:
            return False
            
        # Convert Euler angles to quaternion
        if yaw is None:
            # If yaw is not provided, get current yaw from vehicle state
            with self.state_lock:
                current_attitude = self.vehicle_state.get('attitude', (0, 0, 0))
                yaw = current_attitude[2]  # Current yaw
                
        # Create quaternion from Euler angles
        q = self._euler_to_quaternion(roll, pitch, yaw)
        
        # Calculate appropriate type_mask
        type_mask = 0b00000000  # Use all fields
        
        if yaw is None:
            # Ignore yaw rate
            type_mask |= 0b00010000
            
        try:
            self.connection.mav.set_attitude_target_send(
                0,  # time_boot_ms (0 = use system time)
                self.target_system, self.target_component,
                type_mask,
                q,  # quaternion
                0, 0, 0,  # roll, pitch, yaw rates (ignored with mask)
                thrust  # thrust (0-1)
            )
            return True
        except Exception as e:
            logger.error(f"Error sending SET_ATTITUDE_TARGET: {e}")
            return False
    
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        # This is a simplified conversion for small angles
        # For a full implementation, consider using scipy.spatial.transform
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        q = [0, 0, 0, 0]
        q[0] = cy * cp * cr + sy * sp * sr  # w
        q[1] = cy * cp * sr - sy * sp * cr  # x
        q[2] = cy * sp * cr + sy * cp * sr  # y
        q[3] = sy * cp * cr - cy * sp * sr  # z
        
        return q
    
    def get_vehicle_state(self):
        """Get current vehicle state"""
        with self.state_lock:
            return self.vehicle_state.copy()
    
    def is_heartbeat_healthy(self):
        """Check if we've received a heartbeat recently"""
        heartbeat_timeout = self.config.get('heartbeat_timeout', 3)
        return (time.time() - self.last_heartbeat) < heartbeat_timeout
    
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
        logger.info("Closing MAVLink connection")
        self.running = False
        
        if self.connection:
            self.connection.close()
            self.connection = None


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
        
    def calculate_landing_target_angles(self, marker_data, vehicle_state):
        """
        Calculate the LANDING_TARGET message angles from marker detection
        
        Args:
            marker_data: Dictionary with marker position (tvec)
            vehicle_state: Current vehicle attitude
            
        Returns:
            tuple of (angle_x, angle_y, distance) for LANDING_TARGET message
        """
        if 'tvec' not in marker_data:
            return None, None, None
            
        # Get marker position from tvec
        tvec = marker_data['tvec'][0]  # First (and only) marker
        x, y, z = tvec[0], tvec[1], tvec[2]  # In camera frame (mm)
        
        # Calculate angles (in radians)
        # X angle (positive right in camera frame)
        angle_x = math.atan2(x, z)
        
        # Y angle (positive down in camera frame)
        angle_y = math.atan2(y, z)
        
        # Distance in meters
        distance = math.sqrt(x*x + y*y + z*z) / 1000.0  # mm to meters
        
        return angle_x, angle_y, distance
