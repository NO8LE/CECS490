#!/usr/bin/env python3

"""
RTK GPS Interface for Precision Landing Integration

This module provides an interface to the RTK GPS server for recording precise
GPS coordinates at landing locations.
"""

import os
import json
import time
import logging
import requests
from threading import Thread, Event
from typing import Dict, Optional, Tuple, Any, Callable

# Optional import for UGV management
try:
    from qgc_mission_integration.ugv_manager import UGVManager
    UGV_MANAGER_AVAILABLE = True
except ImportError:
    UGV_MANAGER_AVAILABLE = False

# Set up logging
logger = logging.getLogger("RTKGPSInterface")

class RTKGPSInterface:
    """Interface to RTK GPS server for precision coordinate capture"""
    
    def __init__(self, server_url: str = "http://localhost:8000", 
                 check_interval: float = 1.0,
                 auto_connect: bool = True,
                 config: Dict[str, Any] = None):
        """
        Initialize RTK GPS Interface
        
        Args:
            server_url: URL of the RTK GPS server
            check_interval: Interval for checking GPS data in seconds
            auto_connect: Whether to connect automatically
            config: Configuration dictionary for additional settings
        """
        self.server_url = server_url
        self.check_interval = check_interval
        self.connected = False
        self.latest_gps_data = None
        self.monitoring_thread = None
        self.stop_event = Event()
        self.callbacks = []
        self.config = config or {}
        
        # UGV coordination
        self.ugv_manager = None
        self.ugv_coordination_active = False
        
        # Automatically connect if requested
        if auto_connect:
            self.connect()
            
    def connect(self) -> bool:
        """
        Connect to the RTK GPS server
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Test connection to server
            response = requests.get(f"{self.server_url}/gps_location.json", timeout=5)
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to RTK GPS server at {self.server_url}")
                self.connected = True
                
                # Start monitoring thread
                self.stop_event.clear()
                self.monitoring_thread = Thread(target=self._monitor_gps_data, daemon=True)
                self.monitoring_thread.start()
                
                return True
            else:
                logger.warning(f"RTK GPS server returned status code {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to RTK GPS server: {e}")
            self.connected = False
            return False
            
    def disconnect(self) -> None:
        """Disconnect from the RTK GPS server"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=2.0)
            
        self.connected = False
        logger.info("Disconnected from RTK GPS server")
        
    def _monitor_gps_data(self) -> None:
        """Monitor GPS data from the server in a separate thread"""
        while not self.stop_event.is_set():
            try:
                # Request latest GPS data
                response = requests.get(f"{self.server_url}/gps_location.json", timeout=2)
                
                if response.status_code == 200:
                    # Parse JSON data
                    gps_data = response.json()
                    
                    # Check if data has changed
                    if self._has_data_changed(gps_data):
                        self.latest_gps_data = gps_data
                        logger.info(f"New GPS data received: {gps_data.get('latitude', 0)}, {gps_data.get('longitude', 0)}")
                        
                        # Notify callbacks
                        self._notify_callbacks(gps_data)
                
            except requests.exceptions.RequestException as e:
                logger.debug(f"Error fetching GPS data: {e}")
                # Don't set connected to False here, just log and continue trying
                
            # Wait for next check interval
            self.stop_event.wait(self.check_interval)
            
    def _has_data_changed(self, new_data: Dict[str, Any]) -> bool:
        """Check if GPS data has changed"""
        if not self.latest_gps_data:
            return True
            
        # Check critical fields
        for field in ['latitude', 'longitude', 'altitude_m', 'timestamp']:
            if field not in new_data or field not in self.latest_gps_data:
                return True
                
            if new_data[field] != self.latest_gps_data[field]:
                return True
                
        return False
        
    def _notify_callbacks(self, gps_data: Dict[str, Any]) -> None:
        """Notify all registered callbacks with new GPS data"""
        for callback in self.callbacks:
            try:
                callback(gps_data)
            except Exception as e:
                logger.error(f"Error in GPS data callback: {e}")
                
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to be called when new GPS data is received
        
        Args:
            callback: Function to call with new GPS data
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            
    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister a previously registered callback
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def get_latest_coordinates(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the latest GPS coordinates
        
        Returns:
            Tuple of (latitude, longitude, altitude) or None if no data available
        """
        if not self.latest_gps_data:
            return None
            
        try:
            lat = self.latest_gps_data.get('latitude', 0.0)
            lon = self.latest_gps_data.get('longitude', 0.0)
            alt = self.latest_gps_data.get('altitude_m', 0.0)
            
            return (lat, lon, alt)
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing GPS coordinates: {e}")
            return None
            
    def get_latest_fix_quality(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest GPS fix quality information
        
        Returns:
            Dictionary with fix_type, eph, satellites, accuracy_status or None if no data
        """
        if not self.latest_gps_data:
            return None
            
        try:
            return {
                'fix_type': self.latest_gps_data.get('fix_type', 0),
                'eph': self.latest_gps_data.get('eph', 99.0),
                'satellites': self.latest_gps_data.get('satellites', 0),
                'accuracy_status': self.latest_gps_data.get('accuracy_status', 'unknown')
            }
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing GPS fix quality: {e}")
            return None
            
    def is_rtk_fixed(self) -> bool:
        """
        Check if RTK is in fixed mode (highest precision)
        
        Returns:
            True if RTK is fixed, False otherwise
        """
        if not self.latest_gps_data:
            return False
            
        # Check for RTK fixed (fix_type 5) or cm-level accuracy
        fix_type = self.latest_gps_data.get('fix_type', 0)
        accuracy = self.latest_gps_data.get('accuracy_status', '')
        
        return fix_type >= 5 or accuracy == 'cm-level'
        
    def save_coordinates_to_file(self, file_path: str = "landing_coordinates.json") -> bool:
        """
        Save the latest coordinates to a file
        
        Args:
            file_path: Path to save the coordinates
            
        Returns:
            True if save was successful, False otherwise
        """
        if not self.latest_gps_data:
            logger.warning("No GPS data available to save")
            return False
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.latest_gps_data, f, indent=2)
                
            logger.info(f"Saved GPS coordinates to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving GPS coordinates: {e}")
            return False
            
    def get_coordinates_and_validate(self) -> Optional[Dict[str, Any]]:
        """
        Get coordinates and validate RTK quality
        
        Returns:
            Dictionary with GPS data if quality is sufficient, None otherwise
        """
        if not self.latest_gps_data:
            logger.warning("No GPS data available")
            return None
            
        # Check for minimum quality requirements
        fix_type = self.latest_gps_data.get('fix_type', 0)
        eph = self.latest_gps_data.get('eph', 99.0)
        satellites = self.latest_gps_data.get('satellites', 0)
        
        if fix_type < 3:
            logger.warning(f"Insufficient GPS fix type: {fix_type} (needs 3 or higher)")
            return None
            
        if satellites < 8:
            logger.warning(f"Insufficient satellites: {satellites} (needs 8 or more)")
            return None
            
        if eph > 2.0:  # 2 meters horizontal precision
            logger.warning(f"Insufficient horizontal precision: {eph}m (needs 2.0m or better)")
            return None
            
        # Add quality assessment to data
        data = self.latest_gps_data.copy()
        
        # Classify quality
        if fix_type >= 5 and eph <= 0.02:  # 2cm precision
            data['quality'] = 'excellent'
        elif fix_type >= 4:
            data['quality'] = 'good'
        else:
            data['quality'] = 'acceptable'
            
            return data
            
    def initialize_ugv_manager(self, config: Dict[str, Any] = None) -> bool:
        """
        Initialize UGV Manager for coordinate transmission
        
        Args:
            config: Configuration dictionary for UGV Manager
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not UGV_MANAGER_AVAILABLE:
            logger.error("UGV Manager not available - make sure ugv_manager.py is in the path")
            return False
            
        # Use provided config or the one stored during initialization
        cfg = config or self.config
        
        if not cfg.get('ugv_enabled', False):
            logger.info("UGV coordination is disabled in configuration")
            return False
            
        # Create UGV Manager
        try:
            logger.info("Initializing UGV Manager")
            self.ugv_manager = UGVManager(cfg)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize UGV Manager: {e}")
            return False
            
    def connect_to_ugv(self) -> bool:
        """
        Connect to UGV using the UGV Manager
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.ugv_manager:
            if not self.initialize_ugv_manager():
                return False
                
        try:
            return self.ugv_manager.connect()
        except Exception as e:
            logger.error(f"Error connecting to UGV: {e}")
            return False
            
    def send_coordinates_to_ugv(self, alt: float = 0.0) -> bool:
        """
        Send the latest coordinates to UGV
        
        Args:
            alt: Altitude for UGV (usually 0 for ground vehicles)
            
        Returns:
            True if coordinates sent successfully, False otherwise
        """
        if not self.latest_gps_data:
            logger.warning("No GPS data available to send to UGV")
            return False
            
        if not self.ugv_manager:
            if not self.connect_to_ugv():
                return False
                
        # Extract coordinates
        try:
            lat = self.latest_gps_data.get('latitude', 0.0)
            lon = self.latest_gps_data.get('longitude', 0.0)
            
            logger.info(f"Sending coordinates to UGV: {lat}, {lon}, {alt}")
            
            # Start command thread in UGV manager
            return self.ugv_manager.start_command_thread(lat, lon, alt)
        except Exception as e:
            logger.error(f"Error sending coordinates to UGV: {e}")
            return False
            
    def stop_ugv_coordination(self) -> None:
        """Stop UGV coordination and command sending"""
        if self.ugv_manager:
            logger.info("Stopping UGV coordination")
            self.ugv_manager.stop_command_thread()
            self.ugv_manager.disconnect()
            self.ugv_coordination_active = False
            
    def is_ugv_coordination_active(self) -> bool:
        """
        Check if UGV coordination is active
        
        Returns:
            True if coordination is active, False otherwise
        """
        if not self.ugv_manager:
            return False
            
        status = self.ugv_manager.get_status()
        return status.get('command_active', False)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Simple callback function
    def on_gps_update(data):
        print(f"New GPS data: {data['latitude']}, {data['longitude']}, {data['altitude_m']}m")
        print(f"Fix: {data['fix_type']}, Satellites: {data['satellites']}, Accuracy: {data['accuracy_status']}")
    
    # Create interface
    rtk = RTKGPSInterface(server_url="http://localhost:8000")
    
    # Register callback
    rtk.register_callback(on_gps_update)
    
    try:
        # Run for 60 seconds
        print("Monitoring RTK GPS data for 60 seconds...")
        time.sleep(60)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        rtk.disconnect()
