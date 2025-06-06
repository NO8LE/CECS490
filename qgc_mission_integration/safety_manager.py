#!/usr/bin/env python3

"""
Safety Manager for QGC Mission Integration

This module provides comprehensive safety monitoring and emergency handling
to ensure safe drone operations during precision landing missions.
"""

import time
import logging
import threading
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Callable

# Set up logging
logger = logging.getLogger("SafetyManager")

class SafetyLevel(Enum):
    """Safety level enumeration"""
    NORMAL = "normal"        # Normal operation, all systems nominal
    WARNING = "warning"      # Warning condition, continue with caution
    CRITICAL = "critical"    # Critical safety issue, intervention needed
    EMERGENCY = "emergency"  # Emergency condition, immediate action required

class SafetyManager:
    """
    Comprehensive safety monitoring and emergency handling
    
    This class monitors various safety parameters and can trigger
    emergency protocols if safety conditions are violated.
    """
    
    def __init__(self, config: Dict[str, Any], mavlink_controller):
        """
        Initialize the safety manager
        
        Args:
            config: Configuration dictionary with safety parameters
            mavlink_controller: MAVLink controller for commanding vehicle
        """
        self.config = config
        self.mavlink = mavlink_controller
        self.safety_level = SafetyLevel.NORMAL
        self.emergency_conditions = []
        self.warning_conditions = []
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.callbacks = []
        self.mission_start_time = time.time()
        self.last_check_time = 0
        self.check_interval = config.get('safety_check_interval', 0.5)  # seconds
        
        # If true, override mission controller in emergency
        self.emergency_override = config.get('safety_emergency_override', True)
        
    def start_monitoring(self) -> None:
        """Start the safety monitoring thread"""
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Safety monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop the safety monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Safety monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Main safety monitoring loop"""
        logger.info("Safety monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Perform safety checks
                current_level = self.check_safety_conditions()
                
                # If safety level changed, notify callbacks
                if current_level != self.safety_level:
                    self._notify_safety_level_changed(current_level)
                    self.safety_level = current_level
                    
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                time.sleep(1.0)
                
        logger.info("Safety monitoring loop stopped")
        
    def check_safety_conditions(self) -> SafetyLevel:
        """
        Check all safety conditions and return current safety level
        
        Returns:
            SafetyLevel indicating the current safety status
        """
        # Clear previous conditions
        self.emergency_conditions = []
        self.warning_conditions = []
        
        # Get current vehicle state
        vehicle_state = self.mavlink.get_vehicle_state()
        
        # Update last check time
        self.last_check_time = time.time()
        
        # Perform various safety checks
        
        # 1. Battery level check
        battery_voltage = vehicle_state.get('battery_voltage', 0)
        battery_remaining = vehicle_state.get('battery_remaining', 0)
        
        if battery_voltage < self.config.get('min_battery_voltage', 22.0):
            self.emergency_conditions.append("LOW_BATTERY_VOLTAGE")
            logger.warning(f"Low battery voltage: {battery_voltage}V")
        elif battery_voltage < self.config.get('warning_battery_voltage', 23.0):
            self.warning_conditions.append("BATTERY_VOLTAGE_WARNING")
            logger.info(f"Battery voltage warning: {battery_voltage}V")
            
        if battery_remaining < self.config.get('min_battery_remaining', 15):
            self.emergency_conditions.append("LOW_BATTERY_REMAINING")
            logger.warning(f"Low battery remaining: {battery_remaining}%")
        elif battery_remaining < self.config.get('warning_battery_remaining', 30):
            self.warning_conditions.append("BATTERY_REMAINING_WARNING")
            logger.info(f"Battery remaining warning: {battery_remaining}%")
            
        # 2. Mission timeout check
        mission_time = time.time() - self.mission_start_time
        if mission_time > self.config.get('max_mission_time', 600):
            self.emergency_conditions.append("MISSION_TIMEOUT")
            logger.warning(f"Mission timeout: {mission_time:.1f}s")
            
        # 3. Altitude bounds check
        current_alt = vehicle_state.get('relative_altitude', 0)
        if current_alt > self.config.get('max_altitude', 30.0):
            self.emergency_conditions.append("ALTITUDE_EXCEEDED")
            logger.warning(f"Maximum altitude exceeded: {current_alt}m")
        elif current_alt < self.config.get('min_altitude', 2.0) and current_alt > 0.5:
            # Only warn if not in landing phase (above 0.5m)
            self.warning_conditions.append("ALTITUDE_LOW")
            logger.info(f"Altitude low: {current_alt}m")
            
        # 4. Connection health check
        if time.time() - self.mavlink.last_heartbeat > self.config.get('heartbeat_timeout', 3):
            self.emergency_conditions.append("LOST_CONNECTION")
            logger.warning("Lost MAVLink connection")
            
        # 5. EKF health check
        ekf_status = vehicle_state.get('ekf_status', {})
        if ekf_status:
            if 'healthy' in ekf_status and not ekf_status['healthy']:
                self.warning_conditions.append("EKF_UNHEALTHY")
                logger.warning("EKF estimates degraded")
                
                # Only abort mission if in critical landing phase and EKF is bad
                if current_alt < self.config.get('final_approach_altitude', 1.0):
                    self.emergency_conditions.append("EKF_CRITICAL")
                    logger.error("Critical EKF failure during landing approach")
            
            # Check position variance
            pos_horiz_variance = ekf_status.get('pos_horiz_variance', 0)
            pos_vert_variance = ekf_status.get('pos_vert_variance', 0)
            
            if pos_horiz_variance > self.config.get('ekf_pos_horiz_variance_threshold', 1.0):
                self.warning_conditions.append("EKF_HORIZ_VARIANCE")
                logger.info(f"EKF horizontal variance high: {pos_horiz_variance}")
                
            if pos_vert_variance > self.config.get('ekf_pos_vert_variance_threshold', 1.0):
                self.warning_conditions.append("EKF_VERT_VARIANCE")
                logger.info(f"EKF vertical variance high: {pos_vert_variance}")
                
        # 6. GPS health check
        gps_data = vehicle_state.get('gps', {})
        if gps_data:
            fix_type = gps_data.get('fix_type', 0)
            num_sats = gps_data.get('satellites_visible', 0)
            
            if fix_type < 3:
                self.warning_conditions.append("GPS_NO_FIX")
                logger.warning(f"GPS no fix: {fix_type}")
                
            if num_sats < self.config.get('min_satellites', 6):
                self.warning_conditions.append("GPS_LOW_SATELLITES")
                logger.warning(f"GPS low satellites: {num_sats}")
                
        # Determine overall safety level
        if self.emergency_conditions:
            return SafetyLevel.EMERGENCY
        elif "EKF_UNHEALTHY" in self.warning_conditions:
            # EKF issues are critical for precision operation
            return SafetyLevel.CRITICAL
        elif self.warning_conditions:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.NORMAL
            
    def _notify_safety_level_changed(self, new_level: SafetyLevel) -> None:
        """Notify callbacks about safety level change"""
        for callback in self.callbacks:
            try:
                callback(new_level, self.emergency_conditions, self.warning_conditions)
            except Exception as e:
                logger.error(f"Error in safety callback: {e}")
                
    def register_callback(self, callback: Callable) -> None:
        """
        Register a callback for safety level changes
        
        Args:
            callback: Function to call with (safety_level, emergency_conditions, warning_conditions)
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            
    def unregister_callback(self, callback: Callable) -> None:
        """
        Unregister a callback
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def reset_mission_timer(self) -> None:
        """Reset the mission timer"""
        self.mission_start_time = time.time()
        logger.info("Mission timer reset")
        
    def is_safe_to_continue(self) -> bool:
        """
        Check if it's safe to continue mission
        
        Returns:
            True if safety level is NORMAL or WARNING, False otherwise
        """
        return self.safety_level in [SafetyLevel.NORMAL, SafetyLevel.WARNING]
        
    def is_emergency(self) -> bool:
        """
        Check if we're in an emergency condition
        
        Returns:
            True if safety level is EMERGENCY
        """
        return self.safety_level == SafetyLevel.EMERGENCY
        
    def validate_flight_state(self) -> Tuple[bool, str]:
        """
        Comprehensive validation of flight state for safety-critical operations
        
        Returns:
            Tuple of (is_valid, reason)
        """
        vehicle_state = self.mavlink.get_vehicle_state()
        
        # Check if armed
        if not vehicle_state.get('armed', False):
            return False, "Vehicle not armed"
            
        # Check flight mode
        mode = vehicle_state.get('mode', 'UNKNOWN')
        if mode not in ['GUIDED', 'AUTO', 'LOITER', 'PRECISION_LOITER']:
            return False, f"Unsafe flight mode: {mode}"
            
        # Check altitude
        altitude = vehicle_state.get('relative_altitude', 0)
        if altitude < 0.5:
            return False, f"Too low altitude: {altitude}m"
            
        # Check EKF
        ekf_status = vehicle_state.get('ekf_status', {})
        if ekf_status and not ekf_status.get('healthy', False):
            return False, "EKF unhealthy"
            
        # Check connection
        if time.time() - self.mavlink.last_heartbeat > 2.0:
            return False, "MAVLink connection unstable"
            
        # Check if in emergency
        if self.is_emergency():
            return False, f"Emergency condition: {', '.join(self.emergency_conditions)}"
            
        # All checks passed
        return True, "Flight state valid"
        
    def execute_emergency_protocol(self) -> bool:
        """
        Execute emergency protocol based on conditions
        
        Returns:
            True if protocol was executed successfully
        """
        logger.error("EMERGENCY: Executing safety protocol")
        
        # Determine appropriate emergency action
        if "LOST_CONNECTION" in self.emergency_conditions:
            # Connection loss is critical - can't rely on commands
            logger.critical("Connection lost - emergency protocol limited")
            return False
            
        if "LOW_BATTERY_VOLTAGE" in self.emergency_conditions or "LOW_BATTERY_REMAINING" in self.emergency_conditions:
            # Low battery - RTL is the safest option
            logger.warning("Low battery - commanding RTL")
            return self._execute_rtl()
            
        if "ALTITUDE_EXCEEDED" in self.emergency_conditions:
            # Altitude exceeded - descend to safe altitude
            logger.warning("Altitude exceeded - commanding altitude reduction")
            return self._execute_altitude_reduction()
            
        if "EKF_CRITICAL" in self.emergency_conditions:
            # EKF critical during landing - land immediately
            logger.warning("EKF critical - commanding immediate land")
            return self._execute_land()
            
        # Generic emergency - RTL is the safest option
        logger.warning("Generic emergency - commanding RTL")
        return self._execute_rtl()
        
    def _execute_rtl(self) -> bool:
        """Execute Return to Launch command"""
        try:
            if self.mavlink.command_rtl():
                logger.info("RTL activated successfully")
                return True
            else:
                logger.error("Failed to activate RTL")
                return self._execute_land()  # Fallback to land
        except Exception as e:
            logger.error(f"Error executing RTL: {e}")
            return self._execute_land()  # Fallback to land
            
    def _execute_land(self) -> bool:
        """Execute Land command"""
        try:
            if self.mavlink.set_mode('LAND'):
                logger.info("LAND mode activated successfully")
                return True
            else:
                logger.error("Failed to activate LAND mode")
                return False
        except Exception as e:
            logger.error(f"Error executing LAND: {e}")
            return False
            
    def _execute_altitude_reduction(self) -> bool:
        """Execute altitude reduction"""
        try:
            # Get current position
            vehicle_state = self.mavlink.get_vehicle_state()
            current_pos = vehicle_state.get('position', (0, 0, 0))
            
            # Set lower altitude
            safe_alt = self.config.get('emergency_altitude', 15.0)
            if self.mavlink.set_position_target(0, 0, -safe_alt):
                logger.info(f"Altitude reduction to {safe_alt}m commanded")
                return True
            else:
                logger.error("Failed to command altitude reduction")
                return self._execute_rtl()  # Fallback to RTL
        except Exception as e:
            logger.error(f"Error executing altitude reduction: {e}")
            return self._execute_rtl()  # Fallback to RTL
            
    def validate_transition(self, from_mode: str, to_mode: str) -> Tuple[bool, str]:
        """
        Validate flight mode transition for safety
        
        Args:
            from_mode: Current flight mode
            to_mode: Target flight mode
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check overall flight state
        valid, reason = self.validate_flight_state()
        if not valid:
            return False, f"Invalid flight state: {reason}"
            
        # Some mode transitions require special validation
        if to_mode == 'LAND' or to_mode == 'RTL':
            # Always allow landing or RTL
            return True, "Safety transition to landing mode"
            
        if from_mode == 'AUTO' and to_mode == 'GUIDED':
            # Mission interruption for precision landing
            return True, "Mission interruption for precision landing"
            
        if to_mode == 'AUTO' and not self.is_safe_to_continue():
            # Don't allow AUTO mode if not safe
            return False, f"Unsafe to enter AUTO mode: {self.safety_level.value}"
            
        # Generic validation
        if self.safety_level == SafetyLevel.EMERGENCY:
            return False, "Cannot transition during emergency"
            
        if self.safety_level == SafetyLevel.CRITICAL:
            # Only allow transitions to safer modes
            if to_mode not in ['LAND', 'RTL', 'LOITER']:
                return False, f"Unsafe mode {to_mode} during CRITICAL safety condition"
                
        # All checks passed
        return True, "Mode transition allowed"

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mock MAVLink controller for testing
    class MockMAVLinkController:
        def __init__(self):
            self.last_heartbeat = time.time()
            
        def get_vehicle_state(self):
            return {
                'armed': True,
                'mode': 'GUIDED',
                'relative_altitude': 10.0,
                'battery_voltage': 23.5,
                'battery_remaining': 75,
                'ekf_status': {
                    'healthy': True,
                    'pos_horiz_variance': 0.2,
                    'pos_vert_variance': 0.3
                }
            }
            
        def command_rtl(self):
            print("RTL commanded")
            return True
            
        def set_mode(self, mode):
            print(f"Mode set to {mode}")
            return True
            
        def set_position_target(self, x, y, z):
            print(f"Position target set to ({x}, {y}, {z})")
            return True
    
    # Safety callback
    def on_safety_change(level, emergency, warning):
        print(f"Safety level changed to {level.value}")
        if emergency:
            print(f"Emergency conditions: {emergency}")
        if warning:
            print(f"Warning conditions: {warning}")
    
    # Configuration
    config = {
        'min_battery_voltage': 22.0,
        'warning_battery_voltage': 23.0,
        'min_battery_remaining': 15,
        'warning_battery_remaining': 30,
        'max_mission_time': 600,
        'max_altitude': 30.0,
        'min_altitude': 2.0,
        'heartbeat_timeout': 3.0,
        'ekf_pos_horiz_variance_threshold': 1.0,
        'ekf_pos_vert_variance_threshold': 1.0,
        'min_satellites': 6,
        'safety_check_interval': 0.5
    }
    
    # Create safety manager
    mock_mavlink = MockMAVLinkController()
    safety = SafetyManager(config, mock_mavlink)
    
    # Register callback
    safety.register_callback(on_safety_change)
    
    try:
        # Start monitoring
        safety.start_monitoring()
        
        # Run for 10 seconds
        print("Running safety monitoring for 10 seconds...")
        time.sleep(10)
        
        # Test emergency protocol
        print("\nTesting emergency protocol...")
        safety.emergency_conditions.append("LOW_BATTERY_VOLTAGE")
        safety.execute_emergency_protocol()
        
        # Test mode transition validation
        print("\nTesting mode transition validation...")
        valid, reason = safety.validate_transition('GUIDED', 'AUTO')
        print(f"Transition valid: {valid}, reason: {reason}")
        
        # Run for another 5 seconds
        print("\nRunning for 5 more seconds...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        safety.stop_monitoring()
