#!/usr/bin/env python3

"""
QGroundControl Mission Monitor for Precision Landing Integration

This module provides functionality to monitor and interact with QGroundControl
missions for integration with precision landing capabilities.
"""

import os
import time
import math
import logging
import threading
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Callable

# Import our utilities
from qgc_mission_integration.utils.plan_parser import PlanParser

# Set up logging
logger = logging.getLogger("MissionMonitor")

class MissionStatus(Enum):
    """Mission status enumeration"""
    UNKNOWN = "unknown"              # Mission status unknown
    INACTIVE = "inactive"            # No mission active
    ACTIVE = "active"                # Mission is active
    PAUSED = "paused"                # Mission is paused
    COMPLETE = "complete"            # Mission completed
    INTERRUPTED = "interrupted"      # Mission was interrupted
    ERROR = "error"                  # Error in mission

class MissionMonitor:
    """
    Monitors and interacts with QGroundControl missions
    
    This class provides functionality to monitor mission progress,
    parse mission files, and safely interrupt missions for precision landing.
    """
    
    def __init__(self, config: Dict[str, Any], mavlink_controller):
        """
        Initialize the mission monitor
        
        Args:
            config: Configuration dictionary
            mavlink_controller: MAVLink controller for vehicle communication
        """
        self.config = config
        self.mavlink = mavlink_controller
        self.mission_status = MissionStatus.UNKNOWN
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.callbacks = []
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_interrupted_at = None
        self.mission_file_path = None
        self.home_position = None
        self.last_check_time = time.time()
        self.check_interval = config.get('mission_check_interval', 1.0)  # seconds
        self.mission_item_count = 0
        self.mission_distance = 0.0
        self.read_only_mode = config.get('mission_monitor_read_only', True)
        
        # Create mission parser
        self.parser = PlanParser()
        
    def load_mission_file(self, file_path: str) -> bool:
        """
        Load a mission from a QGC .plan file
        
        Args:
            file_path: Path to the .plan file
            
        Returns:
            True if mission loaded successfully
        """
        logger.info(f"Loading mission file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Mission file not found: {file_path}")
            return False
            
        if not self.parser.parse_file(file_path):
            logger.error(f"Failed to parse mission file: {file_path}")
            return False
            
        # Extract mission information
        self.waypoints = self.parser.get_waypoints()
        self.home_position = self.parser.get_home_position()
        self.mission_item_count = self.parser.get_mission_item_count()
        self.mission_distance = self.parser.calculate_mission_distance()
        self.mission_file_path = file_path
        
        logger.info(f"Loaded mission with {len(self.waypoints)} waypoints, {self.mission_distance:.1f}m total distance")
        
        # Reset mission state
        self.current_waypoint_index = 0
        self.mission_status = MissionStatus.INACTIVE
        self.mission_interrupted_at = None
        
        # Notify callbacks
        self._notify_callbacks('mission_loaded', {
            'waypoints': self.waypoints,
            'home_position': self.home_position,
            'item_count': self.mission_item_count,
            'distance': self.mission_distance,
            'file_path': self.mission_file_path
        })
        
        return True
        
    def start_monitoring(self) -> None:
        """Start the mission monitoring thread"""
        logger.info("Starting mission monitoring")
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop the mission monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Mission monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main mission monitoring loop"""
        logger.info("Mission monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get current vehicle state
                vehicle_state = self.mavlink.get_vehicle_state()
                
                # Update mission status
                self._update_mission_status(vehicle_state)
                
                # Update current waypoint
                self._update_current_waypoint(vehicle_state)
                
                # Log mission status periodically
                current_time = time.time()
                if current_time - self.last_check_time > 10.0:  # Log every 10 seconds
                    self._log_mission_status()
                    self.last_check_time = current_time
                    
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in mission monitoring loop: {e}")
                time.sleep(1.0)
                
        logger.info("Mission monitoring loop stopped")
        
    def _update_mission_status(self, vehicle_state: Dict[str, Any]) -> None:
        """
        Update mission status based on vehicle state
        
        Args:
            vehicle_state: Current vehicle state from MAVLink
        """
        # Get previous status for change detection
        previous_status = self.mission_status
        
        # Get current flight mode
        mode = vehicle_state.get('mode', 'UNKNOWN')
        
        # Update status based on mode
        if mode == 'AUTO':
            # In AUTO mode with waypoints = active mission
            if self.waypoints:
                self.mission_status = MissionStatus.ACTIVE
            else:
                # No waypoints loaded but in AUTO - error
                self.mission_status = MissionStatus.ERROR
                
        elif mode == 'GUIDED' and self.mission_interrupted_at is not None:
            # In GUIDED mode with previous interruption = interrupted mission
            self.mission_status = MissionStatus.INTERRUPTED
            
        elif mode in ['LOITER', 'GUIDED'] and previous_status == MissionStatus.ACTIVE:
            # Switched from AUTO to LOITER/GUIDED = paused mission
            self.mission_status = MissionStatus.PAUSED
            
        elif mode == 'RTL' and previous_status in [MissionStatus.ACTIVE, MissionStatus.PAUSED, MissionStatus.INTERRUPTED]:
            # Return to launch after mission = completed mission
            self.mission_status = MissionStatus.COMPLETE
            
        elif not self.waypoints:
            # No waypoints loaded
            self.mission_status = MissionStatus.INACTIVE
            
        # Notify if status changed
        if self.mission_status != previous_status:
            logger.info(f"Mission status changed: {previous_status.value} -> {self.mission_status.value}")
            self._notify_callbacks('mission_status_changed', {
                'previous_status': previous_status,
                'current_status': self.mission_status,
                'mode': mode
            })
            
    def _update_current_waypoint(self, vehicle_state: Dict[str, Any]) -> None:
        """
        Update current waypoint index based on vehicle state
        
        Args:
            vehicle_state: Current vehicle state from MAVLink
        """
        # Check if we have mission messages
        if 'mission' not in vehicle_state:
            return
            
        # Get current waypoint from MAVLink
        mission_data = vehicle_state['mission']
        if 'current_seq' in mission_data:
            new_index = mission_data['current_seq']
            
            # Update if changed
            if new_index != self.current_waypoint_index:
                old_index = self.current_waypoint_index
                self.current_waypoint_index = new_index
                
                logger.info(f"Current waypoint changed: {old_index} -> {new_index}")
                
                # Notify callbacks
                self._notify_callbacks('waypoint_changed', {
                    'previous_index': old_index,
                    'current_index': new_index,
                    'waypoint': self._get_waypoint_by_index(new_index)
                })
                
    def _get_waypoint_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get waypoint information by index
        
        Args:
            index: Waypoint index
            
        Returns:
            Waypoint information or None if not found
        """
        if not self.waypoints or index < 0 or index >= len(self.waypoints):
            return None
            
        return self.waypoints[index]
        
    def _log_mission_status(self) -> None:
        """Log current mission status and progress"""
        if self.mission_status == MissionStatus.ACTIVE:
            # Calculate progress
            if self.waypoints:
                progress = (self.current_waypoint_index / len(self.waypoints)) * 100.0
                logger.info(f"Mission active: Waypoint {self.current_waypoint_index}/{len(self.waypoints)} ({progress:.1f}%)")
            else:
                logger.info("Mission active but no waypoints loaded")
                
        elif self.mission_status == MissionStatus.INTERRUPTED:
            logger.info(f"Mission interrupted at waypoint {self.mission_interrupted_at}")
            
        else:
            logger.info(f"Mission status: {self.mission_status.value}")
            
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for mission events
        
        Args:
            event: Event type ('mission_loaded', 'mission_status_changed', 'waypoint_changed')
            callback: Function to call when event occurs
        """
        self.callbacks.append((event, callback))
        logger.debug(f"Registered callback for event: {event}")
        
    def _notify_callbacks(self, event: str, data: Any) -> None:
        """Notify registered callbacks for a specific event"""
        for evt, callback in self.callbacks:
            if evt == event:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for event {event}: {e}")
                    
    def get_mission_status(self) -> MissionStatus:
        """Get the current mission status"""
        return self.mission_status
        
    def get_current_waypoint(self) -> Optional[Dict[str, Any]]:
        """Get the current waypoint information"""
        return self._get_waypoint_by_index(self.current_waypoint_index)
        
    def get_waypoint_count(self) -> int:
        """Get total number of waypoints"""
        return len(self.waypoints)
        
    def get_mission_progress(self) -> float:
        """
        Get mission progress as percentage
        
        Returns:
            Percentage of mission completed (0-100)
        """
        if not self.waypoints:
            return 0.0
            
        return min(100.0, (self.current_waypoint_index / len(self.waypoints)) * 100.0)
        
    def interrupt_mission(self) -> bool:
        """
        Safely interrupt the current mission
        
        This method will interrupt the mission at the current waypoint
        so it can be resumed later.
        
        Returns:
            True if mission interrupted successfully
        """
        if self.read_only_mode:
            logger.warning("Cannot interrupt mission in read-only mode")
            return False
            
        if self.mission_status != MissionStatus.ACTIVE:
            logger.warning(f"Cannot interrupt mission in {self.mission_status.value} state")
            return False
            
        logger.info(f"Interrupting mission at waypoint {self.current_waypoint_index}")
        
        try:
            # Store interruption point
            self.mission_interrupted_at = self.current_waypoint_index
            
            # Set GUIDED mode to pause mission
            if not self.mavlink.set_mode('GUIDED'):
                logger.error("Failed to set GUIDED mode for mission interruption")
                return False
                
            # Wait for mode change confirmation
            start_time = time.time()
            while time.time() - start_time < 3.0:
                vehicle_state = self.mavlink.get_vehicle_state()
                if vehicle_state.get('mode', '') == 'GUIDED':
                    logger.info("Mission interrupted successfully")
                    
                    # Notify callbacks
                    self._notify_callbacks('mission_interrupted', {
                        'waypoint_index': self.mission_interrupted_at,
                        'waypoint': self._get_waypoint_by_index(self.mission_interrupted_at)
                    })
                    
                    return True
                    
                time.sleep(0.2)
                
            logger.error("Timed out waiting for GUIDED mode confirmation")
            return False
            
        except Exception as e:
            logger.error(f"Error interrupting mission: {e}")
            return False
            
    def resume_mission(self) -> bool:
        """
        Resume interrupted mission
        
        Returns:
            True if mission resumed successfully
        """
        if self.read_only_mode:
            logger.warning("Cannot resume mission in read-only mode")
            return False
            
        if self.mission_status != MissionStatus.INTERRUPTED or self.mission_interrupted_at is None:
            logger.warning("No interrupted mission to resume")
            return False
            
        logger.info(f"Resuming mission from waypoint {self.mission_interrupted_at}")
        
        try:
            # Set AUTO mode to resume mission
            if not self.mavlink.set_mode('AUTO'):
                logger.error("Failed to set AUTO mode for mission resumption")
                return False
                
            # Clear interruption point
            interrupted_at = self.mission_interrupted_at
            self.mission_interrupted_at = None
            
            # Notify callbacks
            self._notify_callbacks('mission_resumed', {
                'waypoint_index': interrupted_at,
                'waypoint': self._get_waypoint_by_index(interrupted_at)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error resuming mission: {e}")
            return False
            
    def command_rtl(self) -> bool:
        """
        Command Return-to-Launch to end mission
        
        Returns:
            True if RTL commanded successfully
        """
        if self.read_only_mode:
            logger.warning("Cannot command RTL in read-only mode")
            return False
            
        logger.info("Commanding RTL to end mission")
        
        try:
            if not self.mavlink.command_rtl():
                logger.error("Failed to command RTL")
                return False
                
            # Notify callbacks
            self._notify_callbacks('mission_rtl', {
                'status': self.mission_status,
                'waypoint_index': self.current_waypoint_index
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error commanding RTL: {e}")
            return False
            
    def switch_to_read_write_mode(self) -> None:
        """Switch from read-only to read-write mode"""
        if not self.read_only_mode:
            logger.info("Already in read-write mode")
            return
            
        logger.warning("Switching from read-only to read-write mode - mission control enabled")
        self.read_only_mode = False
        
    def switch_to_read_only_mode(self) -> None:
        """Switch from read-write to read-only mode"""
        if self.read_only_mode:
            logger.info("Already in read-only mode")
            return
            
        logger.info("Switching to read-only mode - mission control disabled")
        self.read_only_mode = True
        
    def get_nearest_waypoint(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Find the nearest waypoint to a given position
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with waypoint info and distance
        """
        if not self.waypoints:
            return {'waypoint': None, 'index': -1, 'distance': float('inf')}
            
        min_distance = float('inf')
        nearest_index = -1
        
        for i, waypoint in enumerate(self.waypoints):
            waypoint_lat = waypoint.get('lat')
            waypoint_lon = waypoint.get('lon')
            
            if waypoint_lat is not None and waypoint_lon is not None:
                distance = self._calculate_distance(lat, lon, waypoint_lat, waypoint_lon)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_index = i
                    
        return {
            'waypoint': self._get_waypoint_by_index(nearest_index),
            'index': nearest_index,
            'distance': min_distance
        }
        
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two lat/lon points in meters
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mock MAVLink controller for testing
    class MockMAVLinkController:
        def __init__(self):
            self.mode = 'GUIDED'
            
        def get_vehicle_state(self):
            return {
                'mode': self.mode,
                'mission': {
                    'current_seq': 2
                }
            }
            
        def set_mode(self, mode):
            print(f"Setting mode to {mode}")
            self.mode = mode
            return True
            
        def command_rtl(self):
            print("Commanding RTL")
            self.mode = 'RTL'
            return True
    
    # Mission callback
    def on_mission_event(data):
        print(f"Mission event: {data}")
    
    # Configuration
    config = {
        'mission_check_interval': 1.0,
        'mission_monitor_read_only': False
    }
    
    # Create mission monitor
    mock_mavlink = MockMAVLinkController()
    monitor = MissionMonitor(config, mock_mavlink)
    
    # Register callbacks
    monitor.register_callback('mission_loaded', on_mission_event)
    monitor.register_callback('mission_status_changed', on_mission_event)
    monitor.register_callback('waypoint_changed', on_mission_event)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Test loading a mission file
        mission_file = "aruco_finder_test_flight_long_beach.plan"
        if os.path.exists(mission_file):
            print(f"Loading mission file: {mission_file}")
            monitor.load_mission_file(mission_file)
        else:
            print(f"Mission file not found: {mission_file}")
            
        # Simulate mission changes
        print("\nSimulating mission progression...")
        time.sleep(2)
        mock_mavlink.mode = 'AUTO'
        time.sleep(2)
        
        # Test mission interruption
        print("\nInterrupting mission...")
        monitor.interrupt_mission()
        time.sleep(2)
        
        # Test mission resumption
        print("\nResuming mission...")
        monitor.resume_mission()
        time.sleep(2)
        
        # Test RTL
        print("\nCommanding RTL...")
        monitor.command_rtl()
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        monitor.stop_monitoring()
