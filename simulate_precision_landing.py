#!/usr/bin/env python3

"""
Simulation Environment for Autonomous Precision Landing

This script provides a simulation environment for testing the autonomous precision
landing system without requiring a real drone or OAK-D camera. It simulates:
1. MAVLink communication
2. Drone flight dynamics
3. ArUco marker detection
4. Visualization of the mission

Usage:
  python3 simulate_precision_landing.py [options]

Options:
  --target, -t MARKER_ID     Marker ID to use as landing target (default: 5)
  --mission-area SIZE        Size of search area in meters (default: 27.4)
  --search-alt METERS        Initial search altitude in meters (default: 18.0)
  --display-scale SCALE      Display scale factor (default: 15)
  --wind-speed SPEED         Simulated wind speed in m/s (default: 0.5)
  --marker-size SIZE         Marker size in meters (default: 0.3048)
  --verbose, -v              Enable verbose output

Example:
  python3 simulate_precision_landing.py --target 5 --wind-speed 1.0 --verbose
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

# Import mission state definitions from autonomous landing script
try:
    from autonomous_precision_landing import MissionState, DEFAULT_MISSION_CONFIG
    print("Successfully imported autonomous_precision_landing module")
except ImportError:
    print("Error: autonomous_precision_landing.py not found.")
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

# Simulated drone state
class SimulatedDrone:
    """Simulates a drone with basic flight dynamics"""
    
    def __init__(self, config):
        self.config = config
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters (NED)
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz in m/s
        self.attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw in radians
        self.attitude_rate = np.array([0.0, 0.0, 0.0])  # roll_rate, pitch_rate, yaw_rate in rad/s
        
        self.target_position = np.array([0.0, 0.0, 0.0])  # Target position for GUIDED mode
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # Target velocity
        
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
        
        # History for plotting
        self.position_history = []
        self.time_history = []
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
            
            # Simulate at approximately 50Hz
            time.sleep(0.02)
            
    def update(self, dt):
        """Update drone state based on mode and dynamics"""
        # Handle different flight modes
        if self.mode == "GUIDED" and self.armed:
            self.update_guided_mode(dt)
        elif self.mode == "LOITER" and self.armed:
            self.update_loiter_mode(dt)
        elif self.mode == "PRECISION_LOITER" and self.armed:
            self.update_precision_loiter_mode(dt)
        elif self.mode == "LAND" and self.armed:
            self.update_land_mode(dt)
        elif self.mode == "RTL" and self.armed:
            self.update_rtl_mode(dt)
            
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
        elif mode == "RTL":
            # Set target to home
            self.target_position = np.array([0.0, 0.0, -self.config.get('search_altitude', 18.0)])
            
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
            'armed': self.armed
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

# Simulated MAVLink controller
class SimulatedMAVLinkController:
    """Simulated MAVLink controller for testing"""
    
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

# Simulated ArUco detector
class SimulatedArUcoDetector:
    """Simulated ArUco detector for testing"""
    
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

# Visualization for simulation
class MissionVisualizer:
    """Visualize the simulated mission"""
    
    def __init__(self, simulated_drone, mission_state_machine, config):
        self.drone = simulated_drone
        self.mission = mission_state_machine
        self.config = config
        
        # Display scale
        self.display_scale = config.get('display_scale', 15)
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax1 = self.fig.add_subplot(121)  # 2D top-down view
        self.ax2 = self.fig.add_subplot(122, projection='3d')  # 3D view
        
        # Search area
        area_size = config.get('search_area_size', 27.4)
        self.search_area = [-area_size/2, area_size/2, -area_size/2, area_size/2]
        
        # Plot objects
        self.drone_marker = None
        self.target_markers = []
        self.drone_trail = None
        self.status_text = None
        
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
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, "", fontsize=10)
        
        # Add legend
        self.ax1.legend(loc='upper right')
        
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
            
        # Update status text
        status = f"Mode: {self.drone.mode}\n"
        status += f"Armed: {self.drone.armed}\n"
        status += f"Altitude: {-self.drone.position[2]:.1f}m\n"
        status += f"Battery: {self.drone.battery_remaining:.1f}%\n"
        status += f"Mission State: {self.mission.state.value if hasattr(self.mission, 'state') else 'N/A'}\n"
        
        self.status_text.set_text(status)
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def show(self):
        """Show the visualization"""
        plt.show()

# Simulation coordinator
class SimulationCoordinator:
    """Coordinates the simulation of the autonomous landing mission"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize logger
        self.logger = logging.getLogger("Simulation")
        self.logger.info("Initializing simulation")
        
        # Create simulated drone
        self.drone = SimulatedDrone(config)
        
        # Create simulated MAVLink controller
        self.mavlink = SimulatedMAVLinkController(self.drone, config)
        
        # Create simulated detector
        self.detector = SimulatedArUcoDetector(
            self.drone,
            target_id=config.get('target_marker_id', 5)
        )
        
        # Create safety manager
        from autonomous_precision_landing import SafetyManager
        self.safety = SafetyManager(config, self.mavlink)
        
        # Create mission state machine
        from autonomous_precision_landing import MissionStateMachine
        self.mission = MissionStateMachine(
            self.detector,
            self.mavlink,
            self.safety,
            config
        )
        
        # Create visualizer
        self.visualizer = MissionVisualizer(self.drone, self.mission, config)
        
    def run(self):
        """Run the simulation"""
        self.logger.info("Starting simulation")
        
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
    parser = argparse.ArgumentParser(description='Simulation Environment for Autonomous Precision Landing')
    
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
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting Autonomous Precision Landing Simulation")
    
    # Create configuration based on arguments
    config = DEFAULT_MISSION_CONFIG.copy()
    config.update({
        'target_marker_id': args.target,
        'search_area_size': args.mission_area,
        'search_altitude': args.search_alt,
        'display_scale': args.display_scale,
        'wind_speed': args.wind_speed,
        'marker_size': args.marker_size
    })
    
    # Create simulation coordinator
    sim = SimulationCoordinator(config)
    
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
