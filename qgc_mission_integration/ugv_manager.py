#!/usr/bin/env python3

"""
UGV Manager for QGC Mission Integration

This module provides functionality to communicate with and command a ground
vehicle (UGV) to converge on RTK GPS coordinates acquired during UAV precision landing.
"""

import time
import socket
import logging
import threading
from typing import Dict, Optional, Tuple, Any

# Set up logging
logger = logging.getLogger("UGVManager")

class UGVManager:
    """
    Manages communication with ground vehicle via MAVLink over IP network
    
    This class handles sending position targets to a UGV, monitoring
    heartbeat messages, and managing command threads for continuous
    communication with the UGV.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the UGV Manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ugv_ip = config.get('ugv_ip', '192.168.2.6')
        self.ugv_port = config.get('ugv_port', 14550)
        self.ugv_system_id = config.get('ugv_system_id', 2)
        self.ugv_component_id = config.get('ugv_component_id', 1)
        self.message_rate = config.get('ugv_message_rate', 2.0)  # Hz
        
        self.socket = None
        self.connected = False
        self.last_heartbeat_time = 0
        self.command_thread = None
        self.stop_command_thread_event = threading.Event()
        self.target_coordinates = None
        
        # Import pymavlink
        try:
            from pymavlink import mavutil
            from pymavlink.dialects.v20 import common as mavlink
            self.mavutil = mavutil
            self.mavlink = mavlink
            self.import_success = True
        except ImportError:
            logger.error("Failed to import pymavlink. UGV Manager will be disabled.")
            self.import_success = False
            
    def connect(self) -> bool:
        """
        Connect to the UGV via MAVLink over UDP
        
        Returns:
            True if connection was successful, False otherwise
        """
        if not self.import_success:
            logger.error("Cannot connect: pymavlink not imported")
            return False
            
        if self.connected:
            logger.info("Already connected to UGV")
            return True
            
        try:
            # Create MAVLink connection
            connection_string = f'udpout:{self.ugv_ip}:{self.ugv_port}'
            logger.info(f"Connecting to UGV at {connection_string}")
            
            self.mavlink_connection = self.mavutil.mavlink_connection(
                connection_string,
                source_system=1,  # UAV system ID
                source_component=1,  # UAV component ID
                autoreconnect=True,
                force_connected=True  # Allow sending even without receiving heartbeats
            )
            
            # Wait for heartbeat
            if self._wait_for_heartbeat(timeout=5.0):
                self.connected = True
                logger.info(f"Connected to UGV (system: {self.ugv_system_id}, component: {self.ugv_component_id})")
                return True
            else:
                logger.warning("No heartbeat received from UGV, but continuing with connection")
                self.connected = True  # Consider connected even without heartbeat for outbound-only comms
                return True
                
        except Exception as e:
            logger.error(f"Error connecting to UGV: {e}")
            self.connected = False
            return False
            
    def disconnect(self) -> None:
        """Disconnect from the UGV"""
        if self.command_thread and self.command_thread.is_alive():
            self.stop_command_thread_event.set()
            self.command_thread.join(timeout=2.0)
            
        if hasattr(self, 'mavlink_connection') and self.mavlink_connection:
            try:
                self.mavlink_connection.close()
            except Exception as e:
                logger.error(f"Error closing MAVLink connection: {e}")
                
        self.connected = False
        logger.info("Disconnected from UGV")
        
    def _wait_for_heartbeat(self, timeout: float = 5.0) -> bool:
        """
        Wait for heartbeat from UGV
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if heartbeat received, False otherwise
        """
        if not hasattr(self, 'mavlink_connection'):
            return False
            
        start_time = time.time()
        logger.info(f"Waiting for heartbeat from UGV (timeout: {timeout}s)")
        
        while time.time() - start_time < timeout:
            msg = self.mavlink_connection.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                # Check if it's from the UGV
                if msg.get_srcSystem() == self.ugv_system_id:
                    self.last_heartbeat_time = time.time()
                    logger.info(f"Received heartbeat from UGV (system: {msg.get_srcSystem()})")
                    return True
            
            # Sleep briefly to prevent CPU overuse
            time.sleep(0.1)
            
        logger.warning("No heartbeat received from UGV")
        return False
        
    def send_position_target_global_int(self, lat: float, lon: float, alt: float) -> bool:
        """
        Send a SET_POSITION_TARGET_GLOBAL_INT message to the UGV
        
        Args:
            lat: Target latitude
            lon: Target longitude
            alt: Target altitude (usually 0 for ground vehicles)
            
        Returns:
            True if message sent successfully
        """
        if not self.connected:
            logger.warning("Cannot send position target: not connected to UGV")
            return False
            
        try:
            # Convert latitude and longitude to int (1e7 precision)
            lat_int = int(lat * 1e7)
            lon_int = int(lon * 1e7)
            alt_float = float(alt)  # Altitude in meters above sea level
            
            # Create message
            # Type mask: ignore velocity, acceleration, yaw, yaw rate
            # Only use position
            type_mask = (
                0b0000111111000111  # Ignore velocity, acceleration, yaw, yaw rate
            )
            
            # Log the message content
            logger.info(f"Sending position target to UGV: lat={lat}, lon={lon}, alt={alt}")
            
            # Send the message
            self.mavlink_connection.mav.set_position_target_global_int_send(
                0,  # Timestamp (0 for immediate)
                self.ugv_system_id,  # Target system
                self.ugv_component_id,  # Target component
                self.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # Frame
                type_mask,  # Type mask
                lat_int,  # Latitude (degrees * 1e7)
                lon_int,  # Longitude (degrees * 1e7)
                alt_float,  # Altitude (meters)
                0, 0, 0,  # Velocity (not used)
                0, 0, 0,  # Acceleration (not used)
                0, 0  # Yaw and yaw rate (not used)
            )
            
            # Store target coordinates
            self.target_coordinates = (lat, lon, alt)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending position target to UGV: {e}")
            return False
            
    def start_command_thread(self, lat: float, lon: float, alt: float) -> bool:
        """
        Start a thread to send position commands continuously
        
        Args:
            lat: Target latitude
            lon: Target longitude
            alt: Target altitude
            
        Returns:
            True if thread started successfully
        """
        if self.command_thread and self.command_thread.is_alive():
            logger.warning("Command thread already running")
            return False
            
        if not self.connected:
            if not self.connect():
                logger.error("Cannot start command thread: failed to connect to UGV")
                return False
                
        # Store target coordinates
        self.target_coordinates = (lat, lon, alt)
        
        # Reset stop event
        self.stop_command_thread_event.clear()
        
        # Start command thread
        self.command_thread = threading.Thread(target=self._command_loop)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        logger.info("UGV command thread started")
        return True
        
    def stop_command_thread(self) -> None:
        """Stop the command thread"""
        if self.command_thread and self.command_thread.is_alive():
            self.stop_command_thread_event.set()
            self.command_thread.join(timeout=2.0)
            logger.info("UGV command thread stopped")
        else:
            logger.info("No active command thread to stop")
            
    def _command_loop(self) -> None:
        """Main loop for sending commands to UGV"""
        logger.info("UGV command loop started")
        
        while not self.stop_command_thread_event.is_set():
            try:
                if self.target_coordinates:
                    lat, lon, alt = self.target_coordinates
                    self.send_position_target_global_int(lat, lon, alt)
                    
                # Sleep according to message rate
                sleep_time = 1.0 / self.message_rate
                self.stop_command_thread_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in UGV command loop: {e}")
                time.sleep(1.0)
                
        logger.info("UGV command loop stopped")
        
    def is_heartbeat_healthy(self) -> bool:
        """
        Check if UGV heartbeat is healthy
        
        Returns:
            True if heartbeat received recently
        """
        # If we haven't received any heartbeats, consider it healthy
        # This is for cases where the UGV doesn't send heartbeats but can receive commands
        if self.last_heartbeat_time == 0:
            return True
            
        # Check if we've received a heartbeat recently
        heartbeat_timeout = self.config.get('ugv_heartbeat_timeout', 5.0)
        return (time.time() - self.last_heartbeat_time) < heartbeat_timeout
        
    def get_target_coordinates(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the current target coordinates
        
        Returns:
            Tuple of (latitude, longitude, altitude) or None if not set
        """
        return self.target_coordinates
        
    def send_command_long(self, command: int, param1: float = 0, param2: float = 0,
                        param3: float = 0, param4: float = 0, param5: float = 0,
                        param6: float = 0, param7: float = 0) -> bool:
        """
        Send a COMMAND_LONG message to the UGV
        
        Args:
            command: Command ID
            param1-param7: Command parameters
            
        Returns:
            True if message sent successfully
        """
        if not self.connected:
            logger.warning("Cannot send command: not connected to UGV")
            return False
            
        try:
            # Send the command
            self.mavlink_connection.mav.command_long_send(
                self.ugv_system_id,  # Target system
                self.ugv_component_id,  # Target component
                command,  # Command ID
                0,  # Confirmation
                param1, param2, param3, param4, param5, param6, param7  # Parameters
            )
            
            logger.info(f"Sent COMMAND_LONG {command} to UGV")
            return True
            
        except Exception as e:
            logger.error(f"Error sending command to UGV: {e}")
            return False
            
    def send_set_mode(self, custom_mode: int) -> bool:
        """
        Send a SET_MODE command to the UGV
        
        Args:
            custom_mode: Custom mode to set
            
        Returns:
            True if message sent successfully
        """
        return self.send_command_long(
            self.mavlink.MAV_CMD_DO_SET_MODE,  # Command ID
            1,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            custom_mode,  # Custom mode
            0, 0, 0, 0, 0  # Other parameters (not used)
        )
        
# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configuration
    config = {
        'ugv_ip': '192.168.2.6',
        'ugv_port': 14550,
        'ugv_system_id': 2,
        'ugv_component_id': 1,
        'ugv_message_rate': 2.0,
        'ugv_heartbeat_timeout': 5.0
    }
    
    # Create UGV manager
    ugv_manager = UGVManager(config)
    
    try:
        # Connect to UGV
        if ugv_manager.connect():
            print("Connected to UGV")
            
            # Send position target
            lat = 34.1234567
            lon = -118.1234567
            alt = 0.0
            
            if ugv_manager.send_position_target_global_int(lat, lon, alt):
                print(f"Sent position target: {lat}, {lon}, {alt}")
            else:
                print("Failed to send position target")
                
            # Start command thread
            if ugv_manager.start_command_thread(lat, lon, alt):
                print("Command thread started")
                
                # Run for 10 seconds
                print("Running command thread for 10 seconds...")
                time.sleep(10)
                
                # Stop command thread
                ugv_manager.stop_command_thread()
                print("Command thread stopped")
            else:
                print("Failed to start command thread")
        else:
            print("Failed to connect to UGV")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        ugv_manager.disconnect()
        print("Disconnected from UGV")
