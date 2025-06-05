#!/usr/bin/env python3

"""
Enhanced Simulation Environment for Autonomous Precision Landing with Optical Flow

This script provides a simulation environment for testing the flow-enhanced autonomous 
precision landing system without requiring real hardware. It simulates:
1. MAVLink communication
2. Drone flight dynamics
3. ArUco marker detection
4. Optical flow sensor data
5. Visualization of the mission with sensor health indicators

Usage:
  python3 simulate_precision_landing_with_flow.py [options]

Options:
  --target, -t MARKER_ID     Marker ID to use as landing target (default: 5)
  --mission-area SIZE        Size of search area in meters (default: 27.4)
  --search-alt METERS        Initial search altitude in meters (default: 18.0)
  --display-scale SCALE      Display scale factor (default: 15)
  --wind-speed SPEED         Simulated wind speed in m/s (default: 0.5)
  --marker-size SIZE         Marker size in meters (default: 0.3048)
  --flow-quality QUALITY     Base optical flow quality (0-255, default: 180)
  --flow-noise NOISE         Noise in optical flow measurements (default: 0.2)
  --flow-max-alt ALTITUDE    Maximum altitude for reliable flow data (default: 5.0)
  --verbose, -v              Enable verbose output

Example:
  python3 simulate_precision_landing_with_flow.py --target 5 --flow-quality 150 --verbose
"""

import os
import sys
import time
import math
import argparse
import threading
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import mission state definitions from flow-enhanced autonomous landing script
try:
    from autonomous_precision_landing_with_flow import MissionState, DEFAULT_MISSION_CONFIG
    print("Successfully imported flow-enhanced autonomous_precision_landing_with_flow module")
except ImportError:
    print("Error: autonomous_precision_landing_with_flow.py not found.")
    print("Make sure the file is in the current directory.")
    sys.exit(1)

# Mock MAVLink mavutil for simulation
class MockMavUtil:
    class mavlink:
        MAV_FRAME_BODY_NED = 8
        MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
        MAV_CMD_DO_SET_MODE = 176
        MAV_CMD_COMPONENT_ARM_DISARM = 400
        MAV_CMD_NAV_TAKEOFF = 22
        MAV_CMD_NAV_LOITER_UNLIM = 17
        MAV_CMD_NAV_LAND = 21
        MAV_RESULT_ACCEPTED = 0

# Simulated drone state with optical flow
class SimulatedDroneWithFlow:
    """Simulates a drone with basic flight dynamics and optical flow sensor"""
    
    def __init__(self, config):
        self.config = config
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters (NED)
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz in m/s
        self.velocity_prev = np.array([0.0, 0.0, 0.0])  # Previous velocity for optical flow
        self.attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw in radians
        self.attitude_rate = np.array([0.0, 0.0, 0.0])  # roll_rate, pitch_rate, yaw_rate in rad/s
        
        self.target_position = np.array([0.0, 0.0, 0.0])  # Target position for GUIDED mode
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # Target velocity
        self.descent_rate = 0.0  # Controlled descent rate for final approach
        self.using_flow_control = False  # Whether we're using flow for control
        
        self.mode = "STABILIZE"
        self.armed = False
        self.battery_voltage = 25.0  # Full battery
        self.battery_remaining = 100.0  # Percent
        
        # Control parameters
        self.max_velocity = 5.0  # m/s
        self.max_acceleration = 2.0  # m/s^2
        self.position_p_gain = 0.5  # Position proportional gain
        self.velocity_p_gain = 0.7  # Velocity proportional gain
        self.attitude_p_gain = 3.0  # Attitude proportional gain
        
        # Landing parameters
        self.precision_landing_target = None
        self.landing_target_updates = 0
        
        # Environmental factors
        self.wind_speed = config.get('wind_speed', 0.5)  # m/s
        self.wind_direction = np.array([1.0, 1.0, 0.0])  # Normalized vector
        self.wind_direction = self.wind_direction / np.linalg.norm(self.wind_direction)
        
        # Optical flow parameters
        self.flow_quality_base = config.get('flow_quality', 180)  # Base quality (0-255)
        self.flow_noise = config.get('flow_noise', 0.2)  # Noise factor
        self.flow_max_altitude = config.get('flow_max_altitude', 5.0)  # Max reliable altitude
        self.flow_data = {
            'flow_x': 0.0,
            'flow_y': 0.0,
            'quality': 0
        }
        
        # EKF parameters
        self.ekf_status = {
            'velocity_variance': 0.1,
            'pos_horiz_variance': 0.2,
            'pos_vert_variance': 0.2,
            'compass_variance': 0.1,
            'terrain_alt_variance': 0.0,
            'flags': 0x1F,  # All healthy by default
            'healthy': True
        }
        
        # History for plotting
        self.position_history = []
        self.time_history = []
        self.flow_quality_history = []
        self.start_time = time.time()
        
        # Markers in the environment
        self.markers = self.generate_markers()
        
        # Initialize logger
        self.logger = logging.getLogger("SimDrone")
        
        # Start simulation loop
        self.running = True
        self.simulation_thread = threading.Thread(target=self.simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
    def generate_markers(self):
        """Generate random markers in the search area"""
        area_size = self.config.get('search_area_size', 27.4)
        num_markers = 10
        markers = {}
        
        # Target marker in center of area with slight offset
        target_id = self.config.get('target_marker_id', 5)
        target_x = random.uniform(-area_size/4, area_size/4)
        target_y = random.uniform(-area_size/4, area_size/4)
        markers[target_id] = {
            'id': target_id,
            'position': np.array([target_x, target_y, 0.0]),
            'size': self.config.get('marker_size', 0.3048),
            'is_target': True
        }
        
        # Random markers
        for i in range(num_markers - 1):
            marker_id = i
            if marker_id == target_id:
                marker_id = num_markers
                
            # Random position within search area
            x = random.uniform(-area_size/2, area_size/2)
            y = random.uniform(-area_size/2, area_size/2)
            
            markers[marker_id] = {
                'id': marker_id,
                'position': np.array([x, y, 0.0]),
                'size': self.config.get('marker_size', 0.3048),
                'is_target': False
            }
            
        return markers
        
    def simulation_loop(self):
        """Main simulation loop for drone dynamics"""
        last_time = time.time()
        
        while self.running:
            # Calculate time delta
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Skip if dt is too large (e.g., debugger pause)
            if dt > 0.1:
                dt = 0.1
                
            # Run simulation step
            self.update(dt)
            
            # Store history for plotting
            self.position_history.append(self.position.copy())
            self.time_history.append(current_time - self.start_time)
            self.flow_quality_history.append(self.flow_data['quality'])
            
            # Simulate at approximately 50Hz
            time.sleep(0.02)
            
    def update(self, dt):
        """Update drone state based on mode and dynamics"""
        # Store previous velocity for optical flow calculation
        self.velocity_prev = self.velocity.copy()
        
        # Handle different flight modes
        if self.mode == "GUIDED" and self.armed:
            if self.using_flow_control:
                self.update_flow_guided_mode(dt)
            else:
                self.update_guided_mode(dt)
        elif self.mode == "LOITER" and self.armed:
            self.update_loiter_mode(dt)
        elif self.mode == "PRECISION_LOITER" and self.armed:
            self.update_precision_loiter_mode(dt)
        elif self.mode == "LAND" and self.armed:
            self.update_land_mode(dt)
        elif self.mode == "RTL" and self.armed:
            self.update_rtl_mode(dt)
            
        # Update optical flow data
        self.update_optical_flow(dt)
        
        # Update EKF status
        self.update_ekf_status(dt)
            
        # Simulate battery drain
        self.update_battery(dt)
        
        # Add random walk to attitude to simulate disturbances
        self.add_disturbances(dt)
        
    def update_guided_mode(self, dt):
        """Update drone state in GUIDED mode"""
        # Calculate position error
        pos_error = self.target_position - self.position
        
        # Calculate desired velocity (with P controller)
        desired_velocity = pos_error * self.position_p_gain
        
        # Limit velocity
        velocity_magnitude = np.linalg.norm(desired_velocity)
        if velocity_magnitude > self.max_velocity:
            desired_velocity = desired_velocity * (self.max_velocity / velocity_magnitude)
            
        # Calculate velocity error
        vel_error = desired_velocity - self.velocity
        
        # Calculate acceleration (with P controller)
        acceleration = vel_error * self.velocity_p_gain
        
        # Limit acceleration
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > self.max_acceleration:
            acceleration = acceleration * (self.max_acceleration / accel_magnitude)
            
        # Add wind effects
        wind_effect = self.wind_direction * self.wind_speed
        acceleration[0:2] += wind_effect[0:2] * 0.1  # Wind affects x,y only
        
        # Update velocity
        self.velocity += acceleration * dt
        
        # Update position
        self.position += self.velocity * dt
        
        # Generate attitude from velocity
        if np.linalg.norm(self.velocity[0:2]) > 0.1:
            # Calculate yaw from velocity
            desired_yaw = math.atan2(self.velocity[1], self.velocity[0])
            
            # Calculate pitch and roll from acceleration
            desired_pitch = -math.asin(acceleration[0] / self.max_acceleration) * 0.5
            desired_roll = math.asin(acceleration[1] / self.max_acceleration) * 0.5
            
            # Limit attitudes
            desired_pitch = max(-math.radians(20), min(math.radians(20), desired_pitch))
            desired_roll = max(-math.radians(20), min(math.radians(20), desired_roll))
            
            # Update attitude with simple smoothing
            self.attitude[0] += (desired_roll - self.attitude[0]) * 2.0 * dt
            self.attitude[1] += (desired_pitch - self.attitude[1]) * 2.0 * dt
            self.attitude[2] += self.normalize_angle(desired_yaw - self.attitude[2]) * 2.0 * dt
            
    def update_flow_guided_mode(self, dt):
        """Update drone state in GUIDED mode with flow-based descent control"""
        # Similar to standard guided mode but with controlled descent rate
        
        # Calculate position error for horizontal position
        pos_error = self.target_position.copy()
        pos_error[2] = self.position[2]  # Ignore altitude component
        pos_error = pos_error - self.position
        
        # Calculate desired velocity (with P controller)
        desired_velocity = pos_error * self.position_p_gain
        
        # Set vertical velocity based on descent rate
        desired_velocity[2] = self.descent_rate
        
        # Limit velocity
        velocity_magnitude = np.linalg.norm(desired_velocity)
        if velocity_magnitude > self.max_velocity:
            desired_velocity = desired_velocity * (self.max_velocity / velocity_magnitude)
            
        # Calculate velocity error
        vel_error = desired_velocity - self.velocity
        
        # Calculate acceleration (with P controller)
        acceleration = vel_error * self.velocity_p_gain
        
        # Limit acceleration
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > self.max_acceleration:
            acceleration = acceleration * (self.max_acceleration / accel_magnitude)
            
        # Add wind effects
        wind_effect = self.wind_direction * self.wind_speed
        acceleration[0:2] += wind_effect[0:2] * 0.1  # Wind affects x,y only
        
        # Update velocity
        self.velocity += acceleration * dt
        
        # Update position
        self.position += self.velocity * dt
        
        # Generate attitude from velocity (same as guided mode)
        if np.linalg.norm(self.velocity[0:2]) > 0.1:
            # Calculate yaw from velocity
            desired_yaw = math.atan2(self.velocity[1], self.velocity[0])
            
            # Calculate pitch and roll from acceleration
            desired_pitch = -math.asin(acceleration[0] / self.max_acceleration) * 0.5
            desired_roll = math.asin(acceleration[1] / self.max_acceleration) * 0.5
            
            # Limit attitudes
            desired_pitch = max(-math.radians(20), min(math.radians(20), desired_pitch))
            desired_roll = max(-math.radians(20), min(math.radians(20), desired_roll))
            
            # Update attitude with simple smoothing
            self.attitude[0] += (desired_roll - self.attitude[0]) * 2.0 * dt
            self.attitude[1] += (desired_pitch - self.attitude[1]) * 2.0 * dt
            self.attitude[2] += self.normalize_angle(desired_yaw - self.attitude[2]) * 2.0 * dt
        
    def update_loiter_mode(self, dt):
        """Update drone state in LOITER mode"""
        # Similar to GUIDED but tries to maintain position
        self.update_guided_mode(dt)
        
    def update_precision_loiter_mode(self, dt):
        """Update drone state in PRECISION_LOITER mode"""
        # Use landing target for guidance if available
        if self.precision_landing_target is not None and self.landing_target_updates > 0:
            # Get landing target offsets
            x_offset = self.precision_landing_target['x']
            y_offset = self.precision_landing_target['y']
            
            # Calculate target position based on current position and offset
            # Need to transform from body-relative to NED
            yaw = self.attitude[2]
            rot_matrix = np.array([
                [math.cos(yaw), -math.sin(yaw)],
                [math.sin(yaw), math.cos(yaw)]
            ])
            
            offset_ned = rot_matrix @ np.array([x_offset, y_offset])
            
            # Set target position (same altitude, adjusted x,y)
            self.target_position[0] = self.position[0] - offset_ned[0]
            self.target_position[1] = self.position[1] - offset_ned[1]
            
            # Decay landing target updates (simulate sensor updates)
            self.landing_target_updates -= 1
        
        # Call guided mode update
        self.update_guided_mode(dt)
        
    def update_land_mode(self, dt):
        """Update drone state in LAND mode"""
        # Calculate horizontal position similar to PRECISION_LOITER
        if self.precision_landing_target is not None and self.landing_target_updates > 0:
            # Get landing target offsets
            x_offset = self.precision_landing_target['x']
            y_offset = self.precision_landing_target['y']
            
            # Calculate target position based on current position and offset
            # Need to transform from body-relative to NED
            yaw = self.attitude[2]
            rot_matrix = np.array([
                [math.cos(yaw), -math.sin(yaw)],
                [math.sin(yaw), math.cos(yaw)]
            ])
            
            offset_ned = rot_matrix @ np.array([x_offset, y_offset])
            
            # Set target position (descending altitude, adjusted x,y)
            self.target_position[0] = self.position[0] - offset_ned[0]
            self.target_position[1] = self.position[1] - offset_ned[1]
            
            # Decay landing target updates (simulate sensor updates)
            self.landing_target_updates -= 1
        
        # Set descending altitude
        self.target_position[2] = min(self.position[2] + 0.5, 0.0)  # Descend at 0.5 m/s max
        
        # Apply vertical landing logic
        if self.position[2] > -0.1:  # Close to ground
            # Slow down more as we get closer to ground
            descent_rate = max(0.1, self.position[2] + 0.1) * 0.5
            self.velocity[2] = descent_rate
            
            # If on the ground, consider landed
            if self.position[2] >= -0.05:
                self.position[2] = 0.0
                self.velocity = np.zeros(3)
                self.attitude = np.zeros(3)
                self.armed = False
                self.mode = "STABILIZE"
                self.logger.info("Landed")
        else:
            # Call guided mode update for horizontal control
            self.update_guided_mode(dt)
        
    def update_rtl_mode(self, dt):
        """Update drone state in RTL mode"""
        # Set target to home position
        self.target_position = np.array([0.0, 0.0, -self.config.get('search_altitude', 18.0)])
        
        # If close to home, start landing
        distance_to_home = np.linalg.norm(self.position[0:2])
        if distance_to_home < 1.0:
            self.mode = "LAND"
            self.logger.info("Reached home position, transitioning to LAND mode")
        else:
            # Otherwise use guided mode to return home
            self.update_guided_mode(dt)
            
    def update_optical_flow(self, dt):
        """Update simulated optical flow sensor data"""
        # Calculate the optical flow based on velocity changes
        altitude = -self.position[2]  # Altitude above ground
        
        # Optical flow diminishes with altitude and degrades rapidly beyond max_altitude
        if altitude <= 0.1:  # Almost on ground
            quality = 0  # No flow when on ground
        else:
            # Quality decreases with altitude
            altitude_factor = max(0, 1.0 - (altitude / self.flow_max_altitude))
            
            # Base quality affected by altitude
            quality = int(self.flow_quality_base * altitude_factor)
            
            # Add random noise
            noise = np.random.normal(0, self.flow_noise * (1 + altitude / 2))
            quality = int(max(0, min(255, quality + noise * 50)))
        
        # Calculate flow
        if altitude > 0.1:
            # Flow is proportional to lateral velocity and inversely proportional to height
            flow_x = -self.velocity[1] / altitude * 1000  # Scaled for typical flow units
            flow_y = -self.velocity[0] / altitude * 1000
            
            # Add noise proportional to altitude
            noise_scale = self.flow_noise * (1 + altitude / 5)
            flow_x += np.random.normal(0, noise_scale)
            flow_y += np.random.normal(0, noise_scale)
        else:
            flow_x = 0
            flow_y = 0
        
        # Update flow data
        self.flow_data = {
            'flow_x': flow_x,
            'flow_y': flow_y,
            'quality': quality
        }
        
    def update_ekf_status(self, dt):
        """Update EKF status based on drone state and sensors"""
        # Base variances that increase with altitude
        altitude = max(0.1, -self.position[2])
        base_variance = 0.1 + (altitude / 20.0)
        
        # Variances affected by velocity
        velocity_magnitude = np.linalg.norm(self.velocity)
        velocity_factor = 1.0 + (velocity_magnitude / 5.0)
        
        # Calculate variances
        velocity_variance = base_variance * velocity_factor
        pos_horiz_variance = base_variance * velocity_factor * 1.5
        
        # Position vertical variance affected more by altitude
        pos_vert_variance = base_variance * (1.0 + altitude / 10.0)
        
        # EKF flags - all healthy by default
        ekf_flags = 0x1F  # Bits 0-4 set for healthy
        
        # If altitude is very low and we have good optical flow, improve position estimates
        if altitude < self.flow_max_altitude and self.flow_data['quality'] > 100:
            pos_horiz_variance *= 0.5
            velocity_variance *= 0.7
        
        # Update EKF status
        self.ekf_status = {
            'velocity_variance': velocity_variance,
            'pos_horiz_variance': pos_horiz_variance,
            'pos_vert_variance': pos_vert_variance,
            'compass_variance': 0.1,  # Fixed for simplicity
            'terrain_alt_variance': 0.0,  # Assuming perfect terrain data
            'flags': ekf_flags,
            'healthy': (ekf_flags & 0x1F) == 0x1F,
            'time': time.time()
        }
        
    def update_battery(self, dt):
        """Simulate battery drain"""
        # Drain faster when moving
        velocity_magnitude = np.linalg.norm(self.velocity)
        power_consumption = 0.001 + 0.0005 * velocity_magnitude
        
        # Higher altitude means lower air density and more power
        altitude_factor = 1.0 + max(0, -self.position[2]) * 0.01
        power_consumption *= altitude_factor
        
        # Drain battery
        self.battery_voltage -= power_consumption * dt
        self.battery_remaining = (self.battery_voltage - 20.0) / (25.0 - 20.0) * 100.0
        self.battery_remaining = max(0.0, min(100.0, self.battery_remaining))
        
    def add_disturbances(self, dt):
        """Add random disturbances to simulate real conditions"""
        # Random walk for wind
        self.wind_direction += np.random.normal(0, 0.1, 3) * dt
        self.wind_direction = self.wind_direction / np.linalg.norm(self.wind_direction)
        
        # Random attitude disturbances
        attitude_disturbance = np.random.normal(0, 0.01, 3) * dt
        self.attitude += attitude_disturbance
        
    def set_mode(self, mode):
        """Set drone flight mode"""
        if mode == self.mode:
            return True
            
        self.logger.info(f"Changing mode from {self.mode} to {mode}")
        self.mode = mode
        
        # Reset appropriate state when changing modes
        if mode == "GUIDED":
            # Start at current position
            self.target_position = self.position.copy()
            # Reset flow control flag
            self.using_flow_control = False
        elif mode == "RTL":
            # Set target to home
            self.target_position = np.array([0.0, 0.0, -self.config.get('search_altitude', 18.0)])
            
        return True
        
    def set_descent_rate(self, rate_mps):
        """Set controlled descent rate for flow-guided mode"""
        self.descent_rate = rate_mps
        self.using_flow_control = True
        return True
        
    def arm(self, arm_state):
        """Arm or disarm the drone"""
        if self.armed == arm_state:
            return True
            
        self.logger.info(f"{'Arming' if arm_state else 'Disarming'} drone")
        self.armed = arm_state
        return True
        
    def takeoff(self, altitude):
        """Simulate takeoff to specified altitude"""
        if not self.armed:
            self.logger.warning("Cannot takeoff: drone not armed")
            return False
            
        self.logger.info(f"Taking off to {altitude}m")
        self.target_position = np.array([self.position[0], self.position[1], -altitude])
        self.mode = "GUIDED"
        return True
        
    def get_state(self):
        """Get current drone state"""
        return {
            'attitude': tuple(self.attitude),
            'position': (0.0, 0.0, 0.0),  # Not used in simulation
            'relative_altitude': -self.position[2],
            'battery_voltage': self.battery_voltage,
            'battery_remaining': self.battery_remaining,
            'mode': self.mode,
            'armed': self.armed,
            'flow_quality': self.flow_data['quality'],
            'ekf_status': self.ekf_status
        }
        
    def get_visible_markers(self):
        """Get markers visible from current drone position"""
        visible_markers = {}
        
        for marker_id, marker in self.markers.items():
            # Calculate relative position (NED frame)
            rel_position = marker['position'] - self.position
            
            # Calculate distance
            distance = np.linalg.norm(rel_position)
            
            # Check if marker is below drone (visible to downward camera)
            if rel_position[2] > 0:
                # Calculate marker size in pixels (simulated)
                # Using a simplified pinhole camera model
                marker_size_pixels = (marker['size'] / distance) * 1000.0  # Arbitrary scale
                
                # Check if marker is large enough to be detected
                if marker_size_pixels > 10.0 and distance < 20.0:
                    # Transform to body frame
                    yaw = self.attitude[2]
                    rot_matrix = np.array([
                        [math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]
                    ])
                    
                    rel_xy = rot_matrix @ rel_position[0:2]
                    
                    # Check if in camera FOV (simulated 90-degree FOV)
                    max_angle = math.radians(45)  # Half of FOV
                    x_angle = math.atan2(rel_xy[0], rel_position[2])
                    y_angle = math.atan2(rel_xy[1], rel_position[2])
                    
                    if abs(x_angle) < max_angle and abs(y_angle) < max_angle:
                        # Calculate position in camera frame
                        # Forward is Z, right is X, down is Y
                        position_3d = (rel_xy[0] * 1000.0, rel_xy[1] * 1000.0, rel_position[2] * 1000.0)
                        
                        # Calculate confidence based on size and distance
                        confidence = max(0.1, min(1.0, marker_size_pixels / 100.0))
                        
                        # Add noise to detection
                        noise_scale = max(0.01, 0.1 - confidence * 0.1)
                        position_noise = np.random.normal(0, noise_scale * distance, 3)
                        noisy_position = (
                            position_3d[0] + position_noise[0],
                            position_3d[1] + position_noise[1],
                            position_3d[2] + position_noise[2]
                        )
                        
                        # Create marker data
                        visible_markers[marker_id] = {
                            'id': marker_id,
                            'position_3d': noisy_position,
                            'confidence': confidence,
                            'is_target': marker['is_target'],
                            'size': marker['size']
                        }
        
        return visible_markers
        
    def send_landing_target(self, target_data):
        """Process landing target message"""
        if target_data is None:
            return False
            
        self.precision_landing_target = target_data
        self.landing_target_updates = 5  # Will be used for 5 simulation steps
        return True
        
    def normalize_angle(self, angle):
        """Normalize angle to -pi..pi range"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
        
    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)

# Enhanced MAVLink controller with flow support
class EnhancedMAVLinkController:
    """Simulated MAVLink controller with flow support for testing"""
    
    def __init__(self, simulated_drone, config=None):
        self.drone = simulated_drone
        self.config = config or {}
        self.last_heartbeat = time.time()
        self.running = True
        
        # Mavutil reference
        self.mavutil = MockMavUtil()
        
        # Initialize logger
        self.logger = logging.getLogger("SimMAVLink")
        
    def connect(self):
        """Simulate connecting to vehicle"""
        self.logger.info("Simulated MAVLink connection established")
        return True
        
    def wait_for_connection(self, timeout=30):
        """Wait for MAVLink connection"""
        self.logger.info("Simulated MAVLink heartbeat received")
        self.last_heartbeat = time.time()
        return True
        
    def set_mode(self, mode_name):
        """Set vehicle flight mode"""
        self.logger.info(f"Setting flight mode to {mode_name}")
        return self.drone.set_mode(mode_name)
        
    def arm(self, arm=True):
        """Arm or disarm the vehicle"""
        return self.drone.arm(arm)
        
    def takeoff(self, altitude):
        """Command takeoff"""
        return self.drone.takeoff(altitude)
        
    def command_loiter(self, lat, lon, alt, radius=10):
        """Command loiter at position"""
        self.logger.info(f"Commanding loiter at altitude {alt}m")
        # Convert from lat/lon to local NED
        x, y = self._latlon_to_ned(lat, lon)
        self.drone.target_position = np.array([x, y, -alt])
        self.drone.set_mode("LOITER")
        return True
        
    def command_precision_land(self, lat=0, lon=0, alt=0):
        """Command precision landing"""
        self.logger.info("Commanding precision landing")
        self.drone.set_mode("LAND")
        return True
        
    def command_rtl(self):
        """Command return to launch"""
        self.logger.info("Commanding RTL")
        self.drone.set_mode("RTL")
        return True
        
    def send_landing_target(self, target_data):
        """Send landing target update"""
        return self.drone.send_landing_target(target_data)
        
    def set_descent_rate(self, rate_mps):
        """Set controlled descent rate for flow-guided approach"""
        self.logger.info(f"Setting descent rate to {rate_mps} m/s")
        return self.drone.set_descent_rate(rate_mps)
        
    def is_flow_healthy(self):
        """Check if flow sensor is providing reliable data"""
        flow_quality = self.drone.flow_data.get('quality', 0)
        return flow_quality >= self.config.get('flow_quality_threshold', 50)
        
    def is_ekf_healthy(self):
        """Check if EKF status is healthy"""
        return self.drone.ekf_status.get('healthy', False)
        
    def get_vehicle_state(self):
        """Get current vehicle state"""
        return self.drone.get_state()
        
    def close(self):
        """Close the MAVLink connection"""
        self.logger.info("Closing simulated MAVLink connection")
        self.running = False
        
    def _latlon_to_ned(self, lat, lon):
        """Convert lat/lon to local NED coordinates (simulated)"""
        # In simulation, we just use x,y directly
        return lon * 111000.0, lat * 111000.0

# Enhanced ArUco detector
class EnhancedArUcoDetector:
    """Simulated ArUco detector with flow integration for testing"""
    
    def __init__(self, simulated_drone, target_id=None, **kwargs):
        self.drone = simulated_drone
        self.target_id = target_id
        self.latest_detections = {}
        self.detection_thread = None
        self.stop_detection = False
        
        # Camera matrix (unused in simulation)
        self.camera_matrix = np.array([
            [860.0, 0, 640.0],
            [0, 860.0, 360.0],
            [0, 0, 1]
        ])
        
        # Initialize logger
        self.logger = logging.getLogger("SimDetector")
        
    def start(self):
        """Start the detector"""
        self.logger.info("Starting simulated ArUco detector")
        return True
        
    def start_autonomous_detection(self):
        """Start detection in separate thread"""
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        self.logger.info("Started simulated detection thread")
        
    def _detection_loop(self):
        """Detection loop"""
        while not self.stop_detection:
            # Get visible markers from drone
            visible_markers = self.drone.get_visible_markers()
            
            # Update latest detections
            self.latest_detections = visible_markers
            
            # Simulate detection rate
            time.sleep(0.05)  # 20Hz
            
    def get_latest_detections(self):
        """Get most recent detection results"""
        return self.latest_detections
        
    def stop(self):
        """Stop the detection thread"""
        self.stop_detection = True
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)

# Enhanced Visualization for simulation
class EnhancedMissionVisualizer:
    """Visualize the simulated mission with flow sensor data"""
    
    def __init__(self, simulated_drone, mission_state_machine, config):
        self.drone = simulated_drone
        self.mission = mission_state_machine
        self.config = config
        
        # Display scale
        self.display_scale = config.get('display_scale', 15)
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(15, 10))
        self.ax1 = self.fig.add_subplot(221)  # 2D top-down view
        self.ax2 = self.fig.add_subplot(222, projection='3d')  # 3D view
        self.ax3 = self.fig.add_subplot(223)  # Flow quality plot
        self.ax4 = self.fig.add_subplot(224)  # EKF variance plot
        
        # Search area
        area_size = config.get('search_area_size', 27.4)
        self.search_area = [-area_size/2, area_size/2, -area_size/2, area_size/2]
        
        # Plot objects
        self.drone_marker = None
        self.target_markers = []
        self.drone_trail = None
        self.status_text = None
        self.flow_quality_line = None
        self.ekf_variance_lines = {}
        
        # Initialize plots
        self.initialize_plots()
        
        # Update timer
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update_plots)
        self.timer.start()
        
    def initialize_plots(self):
        """Initialize the plot elements"""
        # Setup 2D plot
        self.ax1.set_xlim(self.search_area[0], self.search_area[1])
        self.ax1.set_ylim(self.search_area[2], self.search_area[3])
        self.ax1.set_xlabel('East (m)')
        self.ax1.set_ylabel('North (m)')
        self.ax1.set_title('Top-Down View')
        self.ax1.grid(True)
        
        # Draw search area
        area_rect = Rectangle(
            (self.search_area[0], self.search_area[2]),
            self.search_area[1] - self.search_area[0],
            self.search_area[3] - self.search_area[2],
            fill=False, edgecolor='gray', linestyle='--'
        )
        self.ax1.add_patch(area_rect)
        
        # Draw origin/home
        home_marker = Circle((0, 0), 1.0, fill=True, color='blue', alpha=0.5, label='Home')
        self.ax1.add_patch(home_marker)
        
        # Draw drone
        drone_pos = self.drone.position
        self.drone_marker = self.ax1.scatter(
            drone_pos[1], drone_pos[0], s=100, color='red', marker='o', label='Drone'
        )
        
        # Draw drone trail
        self.drone_trail, = self.ax1.plot([], [], 'r-', alpha=0.5)
        
        # Draw markers
        for marker_id, marker in self.drone.markers.items():
            pos = marker['position']
            if marker['is_target']:
                self.ax1.scatter(pos[1], pos[0], s=80, color='green', marker='s', label=f'Target {marker_id}')
            else:
                self.ax1.scatter(pos[1], pos[0], s=50, color='orange', marker='s', label=f'Marker {marker_id}')
                
        # Setup 3D plot
        self.ax2.set_xlim(self.search_area[0], self.search_area[1])
        self.ax2.set_ylim(self.search_area[2], self.search_area[3])
        self.ax2.set_zlim(0, self.config.get('search_altitude', 18.0) * 1.2)
        self.ax2.set_xlabel('East (m)')
        self.ax2.set_ylabel('North (m)')
        self.ax2.set_zlabel('Up (m)')
        self.ax2.set_title('3D View')
        
        # Setup flow quality plot
        self.ax3.set_xlim(0, 60)  # 60 seconds of data
        self.ax3.set_ylim(0, 255)  # Flow quality is 0-255
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Flow Quality')
        self.ax3.set_title('Optical Flow Quality')
        self.ax3.grid(True)
        self.ax3.axhspan(0, 50, alpha=0.2, color='red', label='Poor')
        self.ax3.axhspan(50, 150, alpha=0.2, color='yellow', label='Fair')
        self.ax3.axhspan(150, 255, alpha=0.2, color='green', label='Good')
        self.flow_quality_line, = self.ax3.plot([], [], 'b-', linewidth=2, label='Flow Quality')
        self.ax3.legend(loc='upper right')
        
        # Setup EKF variance plot
        self.ax4.set_xlim(0, 60)  # 60 seconds of data
        self.ax4.set_ylim(0, 1.5)  # Variance scale
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('EKF Variance')
        self.ax4.set_title('EKF Variances')
        self.ax4.grid(True)
        
        # Add lines for different variances
        self.ekf_variance_lines['pos_horiz'] = self.ax4.plot([], [], 'r-', linewidth=2, label='Position Horiz')[0]
        self.ekf_variance_lines['pos_vert'] = self.ax4.plot([], [], 'g-', linewidth=2, label='Position Vert')[0]
        self.ekf_variance_lines['velocity'] = self.ax4.plot([], [], 'b-', linewidth=2, label='Velocity')[0]
        
        # Add threshold line
        self.ax4.axhline(y=self.config.get('position_variance_threshold', 0.5), color='k', linestyle='--', alpha=0.7, label='Threshold')
        self.ax4.legend(loc='upper right')
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, "", fontsize=10)
        
        # Add legend to top-down view
        self.ax1.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
    def update_plots(self):
        """Update plots with current simulation state"""
        # Update drone marker
        drone_pos = self.drone.position
        self.drone_marker.set_offsets(np.array([[drone_pos[1], drone_pos[0]]]))
        
        # Update drone trail
        if len(self.drone.position_history) > 1:
            trail_x = [pos[1] for pos in self.drone.position_history[-100:]]
            trail_y = [pos[0] for pos in self.drone.position_history[-100:]]
            self.drone_trail.set_data(trail_x, trail_y)
            
        # Update 3D view
        self.ax2.clear()
        self.ax2.set_xlim(self.search_area[0], self.search_area[1])
        self.ax2.set_ylim(self.search_area[2], self.search_area[3])
        self.ax2.set_zlim(0, self.config.get('search_altitude', 18.0) * 1.2)
        self.ax2.set_xlabel('East (m)')
        self.ax2.set_ylabel('North (m)')
        self.ax2.set_zlabel('Up (m)')
        self.ax2.set_title('3D View')
        
        # Plot drone in 3D
        self.ax2.scatter(
            drone_pos[1], drone_pos[0], -drone_pos[2],
            s=100, color='red', marker='o'
        )
        
        # Plot markers in 3D
        for marker_id, marker in self.drone.markers.items():
            pos = marker['position']
            color = 'green' if marker['is_target'] else 'orange'
            self.ax2.scatter(
                pos[1], pos[0], -pos[2],
                s=50, color=color, marker='s'
            )
            
        # Plot trajectory in 3D if we have enough points
        if len(self.drone.position_history) > 10:
            trail_x = [pos[1] for pos in self.drone.position_history[-100:]]
            trail_y = [pos[0] for pos in self.drone.position_history[-100:]]
            trail_z = [-pos[2] for pos in self.drone.position_history[-100:]]
            self.ax2.plot(trail_x, trail_y, trail_z, 'r-', alpha=0.5)
            
        # Update flow quality plot
        if len(self.drone.time_history) > 1 and len(self.drone.flow_quality_history) > 1:
            # Get the last 60 seconds of data
            start_idx = max(0, len(self.drone.time_history) - 600)  # Last 600 points (60 seconds at 10 Hz)
            times = self.drone.time_history[start_idx:]
            quality = self.drone.flow_quality_history[start_idx:]
            
            # Adjust time to show last 60 seconds
            if len(times) > 0:
                current_time = times[-1]
                plot_times = [t - current_time + 60 for t in times]
                
                # Update flow quality line
                self.flow_quality_line.set_data(plot_times, quality)
                
                # Update EKF variance lines if data is available
                if hasattr(self.drone, 'ekf_status') and self.drone.ekf_status is not None:
                    # Mock up some variance data for visualization
                    pos_horiz_var = [self.drone.ekf_status['pos_horiz_variance']] * len(plot_times)
                    pos_vert_var = [self.drone.ekf_status['pos_vert_variance']] * len(plot_times)
                    vel_var = [self.drone.ekf_status['velocity_variance']] * len(plot_times)
                    
                    self.ekf_variance_lines['pos_horiz'].set_data(plot_times, pos_horiz_var)
                    self.ekf_variance_lines['pos_vert'].set_data(plot_times, pos_vert_var)
                    self.ekf_variance_lines['velocity'].set_data(plot_times, vel_var)
            
        # Update status text
        status = f"Mode: {self.drone.mode}\n"
        status += f"Armed: {self.drone.armed}\n"
        status += f"Altitude: {-self.drone.position[2]:.1f}m\n"
        status += f"Battery: {self.drone.battery_remaining:.1f}%\n"
        status += f"Flow Quality: {self.drone.flow_data['quality']}\n"
        status += f"Flow Control: {'Active' if self.drone.using_flow_control else 'Inactive'}\n"
        status += f"Mission State: {self.mission.state.value if hasattr(self.mission, 'state') else 'N/A'}\n"
        
        self.status_text.set_text(status)
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def show(self):
        """Show the visualization"""
        plt.show()

# Enhanced Simulation coordinator
class EnhancedSimulationCoordinator:
    """Coordinates the simulation of the autonomous landing mission with flow"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize logger
        self.logger = logging.getLogger("Simulation")
        self.logger.info("Initializing enhanced simulation with optical flow")
        
        # Create simulated drone with flow
        self.drone = SimulatedDroneWithFlow(config)
        
        # Create enhanced MAVLink controller
        self.mavlink = EnhancedMAVLinkController(self.drone, config)
        
        # Create enhanced detector
        self.detector = EnhancedArUcoDetector(
            self.drone,
            target_id=config.get('target_marker_id', 5)
        )
        
        # Create flow integration manager
        from autonomous_precision_landing_with_flow import FlowIntegrationManager
        self.flow_manager = FlowIntegrationManager(self.mavlink, config)
        
        # Create safety manager
        from autonomous_precision_landing_with_flow import SafetyManager
        self.safety = SafetyManager(config, self.mavlink)
        
        # Create mission state machine
        from autonomous_precision_landing_with_flow import EnhancedMissionStateMachine
        self.mission = EnhancedMissionStateMachine(
            self.detector,
            self.mavlink,
            self.safety,
            config
        )
        
        # Override flow manager reference to use our simulation's instance
        self.mission.flow_manager = self.flow_manager
        
        # Create enhanced visualizer
        self.visualizer = EnhancedMissionVisualizer(self.drone, self.mission, config)
        
    def run(self):
        """Run the simulation"""
        self.logger.info("Starting enhanced simulation with optical flow")
        
        # Start detector
        self.detector.start()
        self.detector.start_autonomous_detection()
        
        # Start mission in separate thread
        mission_thread = threading.Thread(target=self._run_mission)
        mission_thread.daemon = True
        mission_thread.start()
        
        # Show visualization (blocking call)
        self.visualizer.show()
        
    def _run_mission(self):
        """Run the mission in a separate thread"""
        try:
            # Run mission
            self.mission.run_mission()
            
            # Report mission result
            self.logger.info(f"Mission ended in state: {self.mission.state.value}")
        except Exception as e:
            self.logger.error(f"Error in mission: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
    def stop(self):
        """Stop the simulation"""
        self.logger.info("Stopping simulation")
        self.mission.stop()
        self.detector.stop()
        self.drone.stop()

def setup_logging(verbose=False):
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
    
    return logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Simulation Environment for Autonomous Precision Landing with Optical Flow')
    
    parser.add_argument('--target', '-t', type=int, default=5,
                      help='Marker ID to use as landing target')
    parser.add_argument('--mission-area', type=float, default=DEFAULT_MISSION_CONFIG['search_area_size'],
                      help='Size of search area in meters')
    parser.add_argument('--search-alt', type=float, default=DEFAULT_MISSION_CONFIG['search_altitude'],
                      help='Initial search altitude in meters')
    parser.add_argument('--display-scale', type=float, default=15,
                      help='Display scale factor')
    parser.add_argument('--wind-speed', type=float, default=0.5,
                      help='Simulated wind speed in m/s')
    parser.add_argument('--marker-size', type=float, default=0.3048,
                      help='Marker size in meters')
    parser.add_argument('--flow-quality', type=int, default=180,
                      help='Base optical flow quality (0-255)')
    parser.add_argument('--flow-noise', type=float, default=0.2,
                      help='Noise in optical flow measurements')
    parser.add_argument('--flow-max-alt', type=float, default=5.0,
                      help='Maximum altitude for reliable flow data in meters')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting Enhanced Autonomous Precision Landing Simulation with Optical Flow")
    
    # Create configuration based on arguments
    config = DEFAULT_MISSION_CONFIG.copy()
    config.update({
        'target_marker_id': args.target,
        'search_area_size': args.mission_area,
        'search_altitude': args.search_alt,
        'display_scale': args.display_scale,
        'wind_speed': args.wind_speed,
        'marker_size': args.marker_size,
        'flow_quality': args.flow_quality,
        'flow_noise': args.flow_noise,
        'flow_max_altitude': args.flow_max_alt,
        'use_optical_flow': True
    })
    
    # Create simulation coordinator
    sim = EnhancedSimulationCoordinator(config)
    
    try:
        # Run simulation
        sim.run()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up
        logger.info("Cleaning up...")
        sim.stop()
        
    logger.info("Simulation terminated")

if __name__ == "__main__":
    main()
