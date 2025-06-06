#!/usr/bin/env python3

"""
QGroundControl Mission Integrator for Precision Landing

This module provides the main control class for integrating QGroundControl missions
with ArUco marker detection and precision landing capabilities.
"""

import os
import sys
import time
import yaml
import json
import logging
import threading
import argparse
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Callable

# Add the project root to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import MAVLink controller
try:
    from mavlink_controller import MAVLinkController
except ImportError:
    print("Error: Could not import MAVLinkController.")
    print("Make sure mavlink_controller.py is in the project root directory.")
    sys.exit(1)

# Import QGC mission integration modules
from qgc_mission_integration.mission_monitor import MissionMonitor, MissionStatus
from qgc_mission_integration.aruco_detection_manager import ArUcoDetectionManager
from qgc_mission_integration.safety_manager import SafetyManager, SafetyLevel
from qgc_mission_integration.rtk_gps_interface import RTKGPSInterface
from qgc_mission_integration.mission_event_logger import MissionEventLogger

# Optional import for UGV Manager
try:
    from qgc_mission_integration.ugv_manager import UGVManager
    UGV_MANAGER_AVAILABLE = True
except ImportError:
    UGV_MANAGER_AVAILABLE = False
    logger.warning("UGV Manager not available - UGV coordination will be disabled")

# Set up logging
logger = logging.getLogger("QGCMissionIntegrator")

class PrecisionLandingState(Enum):
    """Precision landing state machine states"""
    IDLE = "idle"                        # Idle state, waiting for mission
    MONITORING = "monitoring"            # Monitoring mission for target detection
    TARGET_DETECTION = "target_detection"  # Target detection in progress
    TARGET_VALIDATION = "target_validation"  # Validating detected target
    MISSION_INTERRUPTION = "mission_interruption"  # Interrupting mission
    PRECISION_LOITER = "precision_loiter"  # Loitering over target
    PRECISION_LANDING = "precision_landing"  # Executing precision landing
    RTK_ACQUISITION = "rtk_acquisition"  # Acquiring RTK coordinates
    UGV_COORDINATION = "ugv_coordination"  # Coordinating with UGV
    POST_LANDING_LOITER = "post_landing_loiter"  # Loitering after landing while commanding UGV
    UGV_COMMAND_ACTIVE = "ugv_command_active"  # Actively sending position commands to UGV
    MISSION_RESUMPTION = "mission_resumption"  # Resuming mission
    COMPLETE = "complete"                # Precision landing completed
    ERROR = "error"                      # Error state

class QGCMissionIntegrator:
    """
    Main controller for QGroundControl mission integration with precision landing
    
    This class coordinates the various components (mission monitor, ArUco detection,
    safety management, RTK GPS) to implement autonomous precision landing during
    QGroundControl missions.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the QGC Mission Integrator
        
        Args:
            config_file: Path to YAML configuration file
        """
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Initialize state
        self.state = PrecisionLandingState.IDLE
        self.state_entry_time = time.time()
        self.running = False
        self.target_marker_id = self.config.get('target_marker_id', 5)
        self.state_thread = None
        self.stop_event = threading.Event()
        self.target_validation_count = 0
        self.target_position = None
        self.callbacks = []
        self.target_detected = False
        self.rtk_coordinates = None
        self.precision_landing_start_time = None
        self.precision_landing_complete = False
        
        # Safety flags and counters
        self.precision_landing_completed = False  # Flag for mission completion
        self.precision_landing_attempts = 0  # Track number of attempts
        self.max_precision_landing_attempts = self.config.get('precision_landing_max_attempts', 1)
        self.rtl_protection_altitude = self.config.get('rtl_protection_altitude', 15.0)
        
        # UGV coordination state
        self.ugv_enabled = self.config.get('ugv_enabled', False)
        self.ugv_manager = None
        self.ugv_coordination_active = False
        self.ugv_command_start_time = None
        self.post_landing_behavior = self.config.get('post_landing_behavior', 'loiter')
        
        # Event logging
        self.event_logging_enabled = self.config.get('event_logging_enabled', True)
        self.event_logger = None
        if self.event_logging_enabled:
            event_log_dir = self.config.get('event_log_dir', 'mission_logs')
            event_console_output = self.config.get('event_console_output', True)
            self.event_logger = MissionEventLogger(
                log_dir=event_log_dir, 
                enable_console=event_console_output
            )
            logger.info("Mission event logger initialized")
        
        # Initialize components
        self._initialize_components()
        
        # Register callbacks for component events
        self._register_component_callbacks()
        
    def _load_configuration(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            # General settings
            'target_marker_id': 5,
            'log_level': 'INFO',
            'log_file': 'qgc_mission_integrator.log',
            
            # MAVLink settings
            'mavlink_connection': 'udp:192.168.2.1:14550',
            'mavlink_baudrate': 921600,
            'mavlink_timeout': 10,
            
            # Mission settings
            'mission_file': 'mission.plan',
            'mission_monitor_read_only': False,
            'mission_check_interval': 1.0,
            
            # Detection settings
            'detection_mode': 'continuous',  # 'continuous' or 'periodic'
            'detection_interval': 5.0,  # seconds, for periodic mode
            'detection_confidence_threshold': 0.7,
            'detection_required_confirmations': 5,
            'detection_resolution': 'adaptive',
            'detection_use_cuda': True,
            
            # Precision landing settings
            'landing_start_altitude': 10.0,  # meters
            'landing_final_approach_altitude': 1.0,  # meters
            'landing_descent_rate': 0.3,  # meters/second
            'landing_center_tolerance': 0.3,  # meters
            'landing_timeout': 120,  # seconds
            
            # Safety settings
            'safety_check_interval': 0.5,  # seconds
            'safety_min_battery_voltage': 22.0,  # volts
            'safety_min_battery_remaining': 15,  # percentage
            'safety_max_mission_time': 600,  # seconds
            'safety_max_altitude': 30.0,  # meters
            'safety_min_altitude': 2.0,  # meters
            
            # RTK GPS settings
            'rtk_server_url': 'http://localhost:8000',
            'rtk_check_interval': 1.0,  # seconds
            'rtk_auto_connect': True,
            
            # Video streaming settings
            'enable_streaming': False,
            'stream_ip': '192.168.2.1',
            'stream_port': 5600
        }
        
        # Load configuration from file if provided
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        # Update default config with file config
                        default_config.update(file_config)
                        logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.info("Using default configuration")
                
        return default_config
        
    def _initialize_components(self) -> None:
        """Initialize all components"""
        logger.info("Initializing components")
        
        try:
            # Initialize MAVLink controller
            logger.info(f"Initializing MAVLink controller: {self.config['mavlink_connection']}")
            self.mavlink = MAVLinkController(
                connection_string=self.config['mavlink_connection'],
                config=self.config
            )
            
            # Initialize mission monitor
            logger.info("Initializing mission monitor")
            self.mission_monitor = MissionMonitor(self.config, self.mavlink)
            
            # Initialize safety manager
            logger.info("Initializing safety manager")
            self.safety_manager = SafetyManager(self.config, self.mavlink)
            
            # Initialize ArUco detection manager
            logger.info(f"Initializing ArUco detection manager (target ID: {self.target_marker_id})")
            self.detection_manager = ArUcoDetectionManager(
                self.config,
                headless=self.config.get('headless', False),
                enable_streaming=self.config.get('enable_streaming', False)
            )
            
            # Initialize RTK GPS interface
            logger.info(f"Initializing RTK GPS interface: {self.config['rtk_server_url']}")
            self.rtk_interface = RTKGPSInterface(
                server_url=self.config['rtk_server_url'],
                check_interval=self.config.get('rtk_check_interval', 1.0),
                auto_connect=self.config.get('rtk_auto_connect', True),
                config=self.config  # Pass config for UGV coordination
            )
            
            # Initialize UGV Manager if enabled
            if self.ugv_enabled and UGV_MANAGER_AVAILABLE:
                logger.info("Initializing UGV Manager")
                self.ugv_manager = UGVManager(self.config)
                logger.info("UGV Manager initialized successfully")
            elif self.ugv_enabled and not UGV_MANAGER_AVAILABLE:
                logger.warning("UGV coordination is enabled but UGV Manager is not available")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
            
    def _register_component_callbacks(self) -> None:
        """Register callbacks for component events"""
        # Register mission monitor callbacks
        self.mission_monitor.register_callback('mission_loaded', self._on_mission_loaded)
        self.mission_monitor.register_callback('mission_status_changed', self._on_mission_status_changed)
        self.mission_monitor.register_callback('waypoint_changed', self._on_waypoint_changed)
        self.mission_monitor.register_callback('mission_interrupted', self._on_mission_interrupted)
        self.mission_monitor.register_callback('mission_resumed', self._on_mission_resumed)
        
        # Register ArUco detection callbacks
        self.detection_manager.register_callback('target_confirmed', self._on_target_confirmed)
        
        # Register safety manager callbacks
        self.safety_manager.register_callback(self._on_safety_level_changed)
        
        # Register RTK GPS callbacks
        self.rtk_interface.register_callback(self._on_rtk_gps_update)
        
    def start(self) -> None:
        """Start the QGC mission integrator"""
        if self.running:
            logger.warning("QGC mission integrator already running")
            return
            
        logger.info("Starting QGC mission integrator")
        
        try:
            # Connect to vehicle
            if not self.mavlink.wait_for_connection(timeout=self.config.get('mavlink_timeout', 10)):
                logger.error("Failed to connect to vehicle")
                return
                
            # Start safety monitoring
            self.safety_manager.start_monitoring()
            
            # Start mission monitoring
            self.mission_monitor.start_monitoring()
            
            # Start ArUco detection
            self.detection_manager.start()
            
            # Start state machine
            self.running = True
            self.stop_event.clear()
            self.state_thread = threading.Thread(target=self._state_machine_loop)
            self.state_thread.daemon = True
            self.state_thread.start()
            
            # Load mission file if specified
            mission_file = self.config.get('mission_file')
            if mission_file and os.path.exists(mission_file):
                logger.info(f"Loading mission file: {mission_file}")
                self.mission_monitor.load_mission_file(mission_file)
                
            logger.info("QGC mission integrator started successfully")
            
            # Log UAV start event
            if self.event_logging_enabled and self.event_logger:
                vehicle_state = self.mavlink.get_vehicle_state()
                self.event_logger.log_uav_start({
                    "mode": vehicle_state.get('mode', 'UNKNOWN'),
                    "battery": f"{vehicle_state.get('battery_remaining', 0)}%",
                    "position": vehicle_state.get('position', (0, 0, 0))
                })
                logger.info("Logged UAV start event")
            
            # Transition to monitoring state
            self._transition_to(PrecisionLandingState.MONITORING)
            
        except Exception as e:
            logger.error(f"Error starting QGC mission integrator: {e}")
            self.stop()
            
    def stop(self) -> None:
        """Stop the QGC mission integrator"""
        if not self.running:
            return
            
        logger.info("Stopping QGC mission integrator")
        
        # Stop state machine
        self.running = False
        self.stop_event.set()
        if self.state_thread and self.state_thread.is_alive():
            self.state_thread.join(timeout=2.0)
            
        # Stop components
        self.detection_manager.stop()
        self.mission_monitor.stop_monitoring()
        self.safety_manager.stop_monitoring()
        self.rtk_interface.disconnect()
        
        # Stop UGV manager if active
        if self.ugv_enabled and hasattr(self, 'ugv_manager') and self.ugv_manager:
            logger.info("Stopping UGV manager")
            self.ugv_manager.stop_command_thread()
            self.ugv_manager.disconnect()
        
        # Log UAV end event
        if self.event_logging_enabled and self.event_logger:
            vehicle_state = self.mavlink.get_vehicle_state()
            self.event_logger.log_uav_end({
                "mode": vehicle_state.get('mode', 'UNKNOWN'),
                "battery": f"{vehicle_state.get('battery_remaining', 0)}%",
                "final_state": self.state.value
            })
            logger.info("Logged UAV end event")
        
        logger.info("QGC mission integrator stopped")
        
    def _state_machine_loop(self) -> None:
        """Main state machine loop"""
        logger.info("State machine loop started")
        
        while self.running and not self.stop_event.is_set():
            try:
                # Get current state for logging
                current_state = self.state
                
                # Execute state handler
                if current_state == PrecisionLandingState.IDLE:
                    self._handle_idle_state()
                elif current_state == PrecisionLandingState.MONITORING:
                    self._handle_monitoring_state()
                elif current_state == PrecisionLandingState.TARGET_DETECTION:
                    self._handle_target_detection_state()
                elif current_state == PrecisionLandingState.TARGET_VALIDATION:
                    self._handle_target_validation_state()
                elif current_state == PrecisionLandingState.MISSION_INTERRUPTION:
                    self._handle_mission_interruption_state()
                elif current_state == PrecisionLandingState.PRECISION_LOITER:
                    self._handle_precision_loiter_state()
                elif current_state == PrecisionLandingState.PRECISION_LANDING:
                    self._handle_precision_landing_state()
                elif current_state == PrecisionLandingState.RTK_ACQUISITION:
                    self._handle_rtk_acquisition_state()
                elif current_state == PrecisionLandingState.UGV_COORDINATION:
                    self._handle_ugv_coordination_state()
                elif current_state == PrecisionLandingState.POST_LANDING_LOITER:
                    self._handle_post_landing_loiter_state()
                elif current_state == PrecisionLandingState.UGV_COMMAND_ACTIVE:
                    self._handle_ugv_command_active_state()
                elif current_state == PrecisionLandingState.MISSION_RESUMPTION:
                    self._handle_mission_resumption_state()
                elif current_state == PrecisionLandingState.COMPLETE:
                    self._handle_complete_state()
                elif current_state == PrecisionLandingState.ERROR:
                    self._handle_error_state()
                    
                # Check for safety issues
                self._check_safety()
                
                # Sleep to prevent CPU overuse
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in state machine loop: {e}")
                self._transition_to(PrecisionLandingState.ERROR)
                time.sleep(1.0)
                
        logger.info("State machine loop stopped")
        
    def _transition_to(self, new_state: PrecisionLandingState) -> None:
        """
        Transition to a new state
        
        Args:
            new_state: The new state to transition to
        """
        old_state = self.state
        self.state = new_state
        self.state_entry_time = time.time()
        
        logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        
        # Perform state entry actions
        if new_state == PrecisionLandingState.TARGET_DETECTION:
            # Reset validation count on entering detection state
            self.target_validation_count = 0
            self.target_detected = False
            
        elif new_state == PrecisionLandingState.PRECISION_LANDING:
            # Record start time for precision landing
            self.precision_landing_start_time = time.time()
            self.precision_landing_complete = False
            # Increment precision landing attempts counter
            self.precision_landing_attempts += 1
            logger.info(f"Precision landing attempt #{self.precision_landing_attempts} started")
            
        elif new_state == PrecisionLandingState.COMPLETE:
            # Set the completion flag to prevent further landing attempts
            self.precision_landing_completed = True
            logger.info("Precision landing mission completed - no further attempts will be processed")
            
            # Stop ArUco detection to save resources and prevent re-detection
            if hasattr(self, 'detection_manager') and self.detection_manager:
                logger.info("Stopping ArUco detection after mission completion")
                self.detection_manager.stop()
            
        # Notify callbacks
        self._notify_callbacks('state_changed', {
            'previous_state': old_state,
            'current_state': new_state,
            'timestamp': time.time()
        })
        
        # Log state transition event
        if self.event_logging_enabled and self.event_logger:
            self.event_logger.log_state_transition(
                previous_state=old_state.value,
                current_state=new_state.value
            )
        
    def _check_safety(self) -> None:
        """Check safety conditions and handle emergencies"""
        if self.safety_manager.is_emergency():
            logger.warning("Safety emergency detected, aborting precision landing")
            
            # Only transition to error if we're in an active landing state
            active_states = [
                PrecisionLandingState.TARGET_DETECTION,
                PrecisionLandingState.TARGET_VALIDATION,
                PrecisionLandingState.MISSION_INTERRUPTION,
                PrecisionLandingState.PRECISION_LOITER,
                PrecisionLandingState.PRECISION_LANDING
            ]
            
            if self.state in active_states:
                self._transition_to(PrecisionLandingState.ERROR)
                
    def _handle_idle_state(self) -> None:
        """Handle IDLE state"""
        # Wait for mission to be loaded
        if self.mission_monitor.get_mission_status() != MissionStatus.INACTIVE:
            self._transition_to(PrecisionLandingState.MONITORING)
            
    def _handle_monitoring_state(self) -> None:
        """Handle MONITORING state"""
        # Monitor mission status for active mission
        mission_status = self.mission_monitor.get_mission_status()
        
        if mission_status == MissionStatus.ACTIVE:
            # Check if we should look for the target
            self._transition_to(PrecisionLandingState.TARGET_DETECTION)
            
    def _handle_target_detection_state(self) -> None:
        """Handle TARGET_DETECTION state"""
        # Check if target is detected
        if self.detection_manager.is_target_detected():
            logger.info(f"Target marker {self.target_marker_id} detected")
            self._transition_to(PrecisionLandingState.TARGET_VALIDATION)
            
        # Check for timeout
        if time.time() - self.state_entry_time > self.config.get('detection_timeout', 300):
            logger.warning("Target detection timeout")
            self._transition_to(PrecisionLandingState.MONITORING)
            
    def _handle_target_validation_state(self) -> None:
        """Handle TARGET_VALIDATION state"""
        # Validate target detection
        is_valid, validation_info = self.detection_manager.validate_target_detection()
        
        if is_valid:
            # Store target position
            self.target_position = validation_info.get('position')
            logger.info(f"Target validated at distance: {validation_info.get('distance', 0):.2f}m")
            
            # Transition to mission interruption
            self._transition_to(PrecisionLandingState.MISSION_INTERRUPTION)
        else:
            # Log validation failure
            reason = validation_info.get('reason', 'unknown')
            logger.debug(f"Target validation failed: {reason}")
            
            # Check if we should keep trying or give up
            if time.time() - self.state_entry_time > self.config.get('validation_timeout', 30):
                logger.warning("Target validation timeout")
                self._transition_to(PrecisionLandingState.MONITORING)
                
    def _handle_mission_interruption_state(self) -> None:
        """Handle MISSION_INTERRUPTION state"""
        # Interrupt mission to perform precision landing
        if self.mission_monitor.interrupt_mission():
            logger.info("Mission interrupted for precision landing")
            self._transition_to(PrecisionLandingState.PRECISION_LOITER)
        else:
            logger.error("Failed to interrupt mission")
            self._transition_to(PrecisionLandingState.ERROR)
            
    def _handle_precision_loiter_state(self) -> None:
        """Handle PRECISION_LOITER state"""
        # Check if drone is in position over target
        vehicle_state = self.mavlink.get_vehicle_state()
        target_info = self.detection_manager.get_target_info()
        
        if target_info and 'position_3d' in target_info:
            # Get target position in 3D space
            x, y, z = target_info['position_3d']
            
            # Calculate offset from center
            center_tolerance = self.config.get('landing_center_tolerance', 0.3)
            x_offset_m = abs(x) / 1000.0  # mm to m
            y_offset_m = abs(y) / 1000.0  # mm to m
            
            # Log positioning information
            logger.debug(f"Target offset: ({x_offset_m:.2f}m, {y_offset_m:.2f}m)")
            
            # Check if centered for precision landing
            if x_offset_m < center_tolerance and y_offset_m < center_tolerance:
                # Check altitude for landing
                current_alt = vehicle_state.get('relative_altitude', 0)
                landing_alt = self.config.get('landing_start_altitude', 10.0)
                
                # If we're at or below landing altitude, begin precision landing
                if current_alt <= landing_alt:
                    logger.info(f"Target centered at altitude {current_alt:.2f}m, starting precision landing")
                    self._transition_to(PrecisionLandingState.PRECISION_LANDING)
                else:
                    # Still descending to landing altitude
                    logger.debug(f"Descending to landing altitude: {current_alt:.2f}m / {landing_alt:.2f}m")
            
            # Send landing target updates to vehicle
            # This requires implementing a method to calculate landing target message
            # which would be similar to the calculate_landing_target_message in autonomous_precision_landing_with_flow.py
        else:
            # Lost target during loiter
            logger.warning("Lost target during precision loiter")
            
            # Check for timeout
            if time.time() - self.state_entry_time > self.config.get('loiter_timeout', 60):
                logger.error("Precision loiter timeout")
                self._transition_to(PrecisionLandingState.ERROR)
                
    def _handle_precision_landing_state(self) -> None:
        """Handle PRECISION_LANDING state"""
        # Execute precision landing
        vehicle_state = self.mavlink.get_vehicle_state()
        current_alt = vehicle_state.get('relative_altitude', 0)
        
        # Log current altitude
        logger.debug(f"Precision landing altitude: {current_alt:.2f}m")
        
        # Check if we've landed
        if current_alt < 0.2 and not vehicle_state.get('armed', True):
            logger.info("Precision landing complete")
            self.precision_landing_complete = True
            self._transition_to(PrecisionLandingState.RTK_ACQUISITION)
            return
            
        # Continue sending landing target updates
        target_info = self.detection_manager.get_target_info()
        if target_info and 'position_3d' in target_info:
            # Send landing target updates to vehicle
            # Similar to sending updates in precision_loiter_state
            pass
            
        # Check for landing timeout
        landing_timeout = self.config.get('landing_timeout', 120)
        if time.time() - self.precision_landing_start_time > landing_timeout:
            logger.error("Precision landing timeout")
            self._transition_to(PrecisionLandingState.ERROR)
            
    def _handle_rtk_acquisition_state(self) -> None:
        """Handle RTK_ACQUISITION state"""
        # Acquire RTK GPS coordinates
        if self.rtk_coordinates:
            logger.info(f"RTK coordinates acquired: {self.rtk_coordinates}")
            
            # Log ArUco location with RTK coordinates
            if self.event_logging_enabled and self.event_logger:
                self.event_logger.log_aruco_location(
                    lat=self.rtk_coordinates.get('latitude', 0.0),
                    lon=self.rtk_coordinates.get('longitude', 0.0),
                    alt=self.rtk_coordinates.get('altitude_m', 0.0),
                    accuracy=self.rtk_coordinates.get('quality', 'unknown')
                )
                logger.info("Logged ArUco location with RTK coordinates")
            
            # Save coordinates to file
            self._save_rtk_coordinates()
            
            # If UGV coordination is enabled, transition to UGV coordination
            if self.ugv_enabled and UGV_MANAGER_AVAILABLE:
                logger.info("UGV coordination enabled, transitioning to UGV coordination")
                self._transition_to(PrecisionLandingState.UGV_COORDINATION)
            else:
                # Otherwise, resume mission as before
                logger.info("UGV coordination not enabled, resuming mission")
                self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
        else:
            # Request coordinates from RTK GPS server
            coords = self.rtk_interface.get_coordinates_and_validate()
            if coords:
                self.rtk_coordinates = coords
                logger.info(f"RTK coordinates received: {coords['latitude']}, {coords['longitude']}")
            else:
                logger.debug("Waiting for valid RTK coordinates")
                
            # Check for timeout
            if time.time() - self.state_entry_time > self.config.get('rtk_timeout', 60):
                logger.warning("RTK acquisition timeout, using non-RTK GPS")
                
                # Use current GPS coordinates as fallback
                vehicle_state = self.mavlink.get_vehicle_state()
                if 'position' in vehicle_state:
                    lat, lon, alt = vehicle_state['position']
                    self.rtk_coordinates = {
                        'latitude': lat,
                        'longitude': lon,
                        'altitude_m': alt,
                        'quality': 'non_rtk'
                    }
                    
                    # Save coordinates and continue
                    self._save_rtk_coordinates()
                    self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
                else:
                    logger.error("No GPS coordinates available")
                    self._transition_to(PrecisionLandingState.ERROR)
                    
    def _handle_ugv_coordination_state(self) -> None:
        """Handle UGV_COORDINATION state"""
        # Make decisions about UGV coordination based on configuration
        logger.info("Determining UGV coordination approach")
        
        # Check what post-landing behavior is configured
        if self.post_landing_behavior == 'loiter':
            # Loiter and command UGV
            logger.info("Post-landing behavior: loiter and command UGV")
            
            # First take off to loiter altitude if we're on the ground
            vehicle_state = self.mavlink.get_vehicle_state()
            current_alt = vehicle_state.get('relative_altitude', 0)
            loiter_alt = self.config.get('ugv_loiter_altitude', 10.0)
            
            if current_alt < 1.0:
                # We're on the ground, need to take off
                logger.info(f"Taking off to loiter altitude: {loiter_alt}m")
                if self.mavlink.takeoff(loiter_alt):
                    logger.info("Takeoff command sent successfully")
                    self._transition_to(PrecisionLandingState.POST_LANDING_LOITER)
                else:
                    logger.error("Failed to send takeoff command")
                    self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
            else:
                # Already in the air, transition to loiter
                logger.info("Already airborne, transitioning to loiter")
                if self.mavlink.set_mode('LOITER'):
                    logger.info("LOITER mode set successfully")
                    self._transition_to(PrecisionLandingState.POST_LANDING_LOITER)
                else:
                    logger.error("Failed to set LOITER mode")
                    self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
        else:
            # RTL after landing (default fallback)
            logger.info("Post-landing behavior: RTL")
            
            # Initialize UGV communication and send target before returning
            if self.ugv_manager and self.rtk_coordinates:
                # Connect to UGV if not already connected
                if not self.ugv_manager.connect():
                    logger.warning("Failed to connect to UGV, continuing with RTL")
                else:
                    # Log UGV start event
                    if self.event_logging_enabled and self.event_logger:
                        self.event_logger.log_ugv_start({
                            "ip": self.ugv_manager.ugv_ip,
                            "port": self.ugv_manager.ugv_port,
                            "mode": "one_time_command"
                        })
                        logger.info("Logged UGV start event")
                    
                    # Send target coordinates to UGV
                    lat = self.rtk_coordinates.get('latitude', 0.0)
                    lon = self.rtk_coordinates.get('longitude', 0.0)
                    alt = 0.0  # Ground vehicle altitude is 0
                    
                    logger.info(f"Sending target coordinates to UGV: {lat}, {lon}, {alt}")
                    if self.ugv_manager.send_position_target_global_int(lat, lon, alt):
                        logger.info("Target coordinates sent to UGV successfully")
                        
                        # Log UGV command sent event
                        if self.event_logging_enabled and self.event_logger:
                            self.event_logger.log_ugv_command_sent(lat, lon, alt, self.ugv_manager.ugv_ip)
                            logger.info("Logged UGV command sent event")
                    else:
                        logger.warning("Failed to send target coordinates to UGV")
                        
            # Continue with RTL/mission resumption
            self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
    
    def _handle_post_landing_loiter_state(self) -> None:
        """Handle POST_LANDING_LOITER state"""
        # Check if we've reached loiter altitude
        vehicle_state = self.mavlink.get_vehicle_state()
        current_alt = vehicle_state.get('relative_altitude', 0)
        loiter_alt = self.config.get('ugv_loiter_altitude', 10.0)
        
        # Allow a tolerance of 1m for altitude
        if abs(current_alt - loiter_alt) < 1.0:
            logger.info(f"Reached loiter altitude: {current_alt:.1f}m")
            
            # If we have UGV manager and RTK coordinates, start UGV command
            if self.ugv_manager and self.rtk_coordinates:
                # Connect to UGV if not already connected
                if not self.ugv_manager.connect():
                    logger.error("Failed to connect to UGV")
                    self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
                    return
                    
                # Send target coordinates to UGV
                lat = self.rtk_coordinates.get('latitude', 0.0)
                lon = self.rtk_coordinates.get('longitude', 0.0)
                alt = 0.0  # Ground vehicle altitude is 0
                
                logger.info(f"Starting UGV command thread with coordinates: {lat}, {lon}, {alt}")
                if self.ugv_manager.start_command_thread(lat, lon, alt):
                    logger.info("UGV command started successfully")
                    self.ugv_command_start_time = time.time()
                    self.ugv_coordination_active = True
                    
                    # Log UGV start event
                    if self.event_logging_enabled and self.event_logger:
                        self.event_logger.log_ugv_start({
                            "ip": self.ugv_manager.ugv_ip,
                            "port": self.ugv_manager.ugv_port,
                            "mode": "continuous_command"
                        })
                        # Log command sent event
                        self.event_logger.log_ugv_command_sent(lat, lon, alt, self.ugv_manager.ugv_ip)
                        logger.info("Logged UGV start and command events")
                    
                    self._transition_to(PrecisionLandingState.UGV_COMMAND_ACTIVE)
                else:
                    logger.error("Failed to start UGV command")
                    self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
            else:
                logger.warning("UGV manager or RTK coordinates not available")
                self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
        else:
            # Still climbing to loiter altitude
            logger.debug(f"Climbing to loiter altitude: {current_alt:.1f}m / {loiter_alt:.1f}m")
            
            # Check for timeout
            if time.time() - self.state_entry_time > 30.0:  # 30 seconds timeout
                logger.warning("Timeout reaching loiter altitude")
                self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
    
    def _handle_ugv_command_active_state(self) -> None:
        """Handle UGV_COMMAND_ACTIVE state"""
        # Check if UGV command is still active
        if self.ugv_manager and self.ugv_manager.is_heartbeat_healthy():
            # Check for command timeout
            ugv_coordination_timeout = self.config.get('ugv_coordination_timeout', 60)
            elapsed = time.time() - self.ugv_command_start_time
            
            # Variables to track UGV state
            static_variables = getattr(self, '_ugv_state_vars', None)
            if static_variables is None:
                # Initialize state tracking variables if not already present
                self._ugv_state_vars = {
                    'receipt_confirmed': False,
                    'delivery_confirmed': False,
                    'delivery_start_time': None,
                    'last_status_check': 0
                }
                static_variables = self._ugv_state_vars
            
            # Check for heartbeat receipt confirmation (only log once)
            if not static_variables['receipt_confirmed']:
                if self.event_logging_enabled and self.event_logger:
                    self.event_logger.log_ugv_receipt_confirmed({
                        "status": "acknowledged",
                        "heartbeat_healthy": True
                    })
                    logger.info("Logged UGV receipt confirmation")
                static_variables['receipt_confirmed'] = True
            
            # Only check delivery status every few seconds to avoid excessive calculations
            current_time = time.time()
            if current_time - static_variables['last_status_check'] > 2.0:
                static_variables['last_status_check'] = current_time
                
                # Check if UGV has reached target
                target_reached = False
                if hasattr(self.ugv_manager, 'get_current_position'):
                    # If UGV manager can report position
                    ugv_position = self.ugv_manager.get_current_position()
                    if ugv_position and self.rtk_coordinates:
                        # Calculate distance to target
                        from math import radians, cos, sin, asin, sqrt
                        
                        def haversine(lat1, lon1, lat2, lon2):
                            """Calculate distance between two lat/lon points in meters"""
                            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                            dlon = lon2 - lon1
                            dlat = lat2 - lat1
                            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                            c = 2 * asin(sqrt(a))
                            r = 6371000  # Earth radius in meters
                            return c * r
                        
                        # Get positions
                        target_lat = self.rtk_coordinates.get('latitude', 0.0)
                        target_lon = self.rtk_coordinates.get('longitude', 0.0)
                        ugv_lat = ugv_position[0]
                        ugv_lon = ugv_position[1]
                        
                        # Calculate distance
                        distance = haversine(target_lat, target_lon, ugv_lat, ugv_lon)
                        
                        # Check if within delivery threshold
                        delivery_threshold = self.config.get('ugv_delivery_distance_threshold', 1.0)
                        if distance <= delivery_threshold:
                            target_reached = True
                            
                            # If we've been at target for enough time, consider delivery confirmed
                            if not static_variables['delivery_start_time']:
                                static_variables['delivery_start_time'] = current_time
                                logger.info(f"UGV reached target, distance: {distance:.2f}m")
                            elif current_time - static_variables['delivery_start_time'] >= self.config.get('ugv_delivery_time_threshold', 5.0):
                                # UGV has been at target for required time
                                if not static_variables['delivery_confirmed']:
                                    logger.info("UGV delivery confirmed")
                                    # Log delivery event
                                    if self.event_logging_enabled and self.event_logger:
                                        self.event_logger.log_ugv_delivery({
                                            "distance": distance,
                                            "position": ugv_position,
                                            "time_at_target": current_time - static_variables['delivery_start_time']
                                        })
                                        logger.info("Logged UGV delivery event")
                                    static_variables['delivery_confirmed'] = True
                        else:
                            # Reset delivery timer if UGV moves away from target
                            if static_variables['delivery_start_time'] and not static_variables['delivery_confirmed']:
                                logger.info(f"UGV moved away from target, distance: {distance:.2f}m")
                                static_variables['delivery_start_time'] = None
                
            # Check for timeout or if delivery is complete
            if elapsed > ugv_coordination_timeout or static_variables['delivery_confirmed']:
                if static_variables['delivery_confirmed']:
                    logger.info("UGV delivery completed successfully")
                else:
                    logger.info(f"UGV coordination timeout after {elapsed:.1f} seconds")
                
                # Log UGV end event
                if self.event_logging_enabled and self.event_logger:
                    self.event_logger.log_ugv_end({
                        "status": "mission_complete" if static_variables['delivery_confirmed'] else "timeout",
                        "duration": elapsed
                    })
                    logger.info("Logged UGV end event")
                
                # Stop UGV command
                self.ugv_manager.stop_command_thread()
                self.ugv_coordination_active = False
                
                # Resume mission
                self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
            else:
                # Still commanding UGV
                remaining = ugv_coordination_timeout - elapsed
                if int(remaining) % 10 == 0 and remaining > 0:  # Log every 10 seconds
                    logger.info(f"UGV command active, {remaining:.0f} seconds remaining")
        else:
            # Lost connection to UGV
            logger.warning("Lost connection to UGV")
            
            # Log UGV end event if not already logged
            if self.event_logging_enabled and self.event_logger:
                self.event_logger.log_ugv_end({
                    "status": "connection_lost",
                    "duration": time.time() - self.ugv_command_start_time if self.ugv_command_start_time else 0
                })
                logger.info("Logged UGV disconnection event")
            
            # Stop UGV command if manager exists
            if self.ugv_manager:
                self.ugv_manager.stop_command_thread()
            
            self.ugv_coordination_active = False
            
            # Resume mission
            self._transition_to(PrecisionLandingState.MISSION_RESUMPTION)
            
    def _handle_mission_resumption_state(self) -> None:
        """Handle MISSION_RESUMPTION state"""
        # Resume mission after precision landing
        if self.mission_monitor.resume_mission():
            logger.info("Mission resumed after precision landing")
            self._transition_to(PrecisionLandingState.COMPLETE)
        else:
            # If mission resumption fails, try RTL
            logger.warning("Failed to resume mission, commanding RTL")
            if self.mission_monitor.command_rtl():
                self._transition_to(PrecisionLandingState.COMPLETE)
            else:
                logger.error("Failed to command RTL")
                self._transition_to(PrecisionLandingState.ERROR)
                
    def _handle_complete_state(self) -> None:
        """Handle COMPLETE state"""
        # Mission complete - nothing to do here
        # This is a terminal state
        pass
        
    def _handle_error_state(self) -> None:
        """Handle ERROR state"""
        # Try to recover from error
        # This could involve RTL or emergency landing
        if not hasattr(self, 'error_recovery_attempted') or not self.error_recovery_attempted:
            logger.info("Attempting error recovery with RTL")
            if self.mission_monitor.command_rtl():
                logger.info("RTL commanded for error recovery")
            else:
                logger.error("Failed to command RTL for error recovery")
                
            self.error_recovery_attempted = True
            
    def _save_rtk_coordinates(self) -> None:
        """Save RTK coordinates to file"""
        if not self.rtk_coordinates:
            return
            
        # Save to file
        try:
            save_file = self.config.get('rtk_save_file', 'rtk_coordinates.json')
            with open(save_file, 'w') as f:
                json.dump(self.rtk_coordinates, f, indent=2)
                
            logger.info(f"Saved RTK coordinates to {save_file}")
        except Exception as e:
            logger.error(f"Error saving RTK coordinates: {e}")
            
    def _on_mission_loaded(self, data: Dict[str, Any]) -> None:
        """Callback for mission loaded event"""
        logger.info(f"Mission loaded: {data.get('file_path')}")
        logger.info(f"Waypoints: {len(data.get('waypoints', []))}, Distance: {data.get('distance', 0):.1f}m")
        
    def _on_mission_status_changed(self, data: Dict[str, Any]) -> None:
        """Callback for mission status changed event"""
        previous = data.get('previous_status')
        current = data.get('current_status')
        
        logger.info(f"Mission status changed: {previous.value} -> {current.value}")
        
        # If mission becomes active, transition to monitoring state
        if current == MissionStatus.ACTIVE:
            # Only transition to MONITORING if we haven't completed a precision landing
            # and we're not already in a terminal state
            if (not self.precision_landing_completed and 
                self.state not in [PrecisionLandingState.COMPLETE, PrecisionLandingState.ERROR]):
                
                logger.info("Mission active, transitioning to monitoring state")
                self._transition_to(PrecisionLandingState.MONITORING)
            else:
                logger.info("Mission active but precision landing already completed - not transitioning to monitoring")
            
    def _on_waypoint_changed(self, data: Dict[str, Any]) -> None:
        """Callback for waypoint changed event"""
        logger.debug(f"Waypoint changed: {data.get('previous_index')} -> {data.get('current_index')}")
        
    def _on_mission_interrupted(self, data: Dict[str, Any]) -> None:
        """Callback for mission interrupted event"""
        logger.info(f"Mission interrupted at waypoint {data.get('waypoint_index')}")
        
    def _on_mission_resumed(self, data: Dict[str, Any]) -> None:
        """Callback for mission resumed event"""
        logger.info(f"Mission resumed from waypoint {data.get('waypoint_index')}")
        
    def _on_target_confirmed(self, data: Dict[str, Any]) -> None:
        """Callback for target confirmed event"""
        logger.info(f"Target confirmed: ID {data.get('id')}")
        
        # Log ArUco discovery event
        if self.event_logging_enabled and self.event_logger:
            position = None
            if 'position_3d' in data:
                position = data['position_3d']
            
            self.event_logger.log_aruco_discovery(
                marker_id=data.get('id', self.target_marker_id),
                position=position,
                confidence=data.get('confidence', 0.0)
            )
            logger.info("Logged ArUco discovery event")
        
        # Check if we can attempt precision landing
        if not self._can_attempt_precision_landing():
            return
            
        # If in monitoring state, transition to detection
        if self.state == PrecisionLandingState.MONITORING:
            self._transition_to(PrecisionLandingState.TARGET_DETECTION)
            
        # If in detection state, transition to validation
        elif self.state == PrecisionLandingState.TARGET_DETECTION:
            self._transition_to(PrecisionLandingState.TARGET_VALIDATION)
            
    def _can_attempt_precision_landing(self) -> bool:
        """
        Check if we can attempt precision landing based on safety rules
        
        Returns:
            True if safe to attempt precision landing, False otherwise
        """
        # Check if we've already completed a precision landing mission
        if self.precision_landing_completed:
            logger.info("Ignoring target detection - precision landing already completed")
            return False
            
        # Check if we've reached the maximum number of attempts
        if self.precision_landing_attempts >= self.max_precision_landing_attempts:
            logger.info(f"Ignoring target detection - max attempts ({self.max_precision_landing_attempts}) reached")
            return False
            
        # Check if we're in a terminal state
        if self.state in [PrecisionLandingState.COMPLETE, PrecisionLandingState.ERROR]:
            logger.info(f"Ignoring target detection - in terminal state: {self.state.value}")
            return False
            
        # Check current altitude - don't try to land if too high (like during RTL)
        vehicle_state = self.mavlink.get_vehicle_state()
        current_alt = vehicle_state.get('relative_altitude', 0)
        
        if current_alt > self.rtl_protection_altitude:
            logger.info(f"Ignoring target detection - altitude ({current_alt:.1f}m) above protection threshold ({self.rtl_protection_altitude:.1f}m)")
            return False
            
        # Check current flight mode - avoid landing during RTL
        mode = vehicle_state.get('mode', '')
        if mode == 'RTL':
            logger.info(f"Ignoring target detection - vehicle in RTL mode")
            return False
            
        # All checks passed
        return True
            
    def _on_safety_level_changed(self, level: SafetyLevel, emergency_conditions: List[str], warning_conditions: List[str]) -> None:
        """Callback for safety level changed event"""
        logger.info(f"Safety level changed to {level.value}")
        
        if emergency_conditions:
            logger.warning(f"Emergency conditions: {', '.join(emergency_conditions)}")
            
        if warning_conditions:
            logger.info(f"Warning conditions: {', '.join(warning_conditions)}")
            
    def _on_rtk_gps_update(self, data: Dict[str, Any]) -> None:
        """Callback for RTK GPS update event"""
        logger.debug(f"RTK GPS update: {data.get('latitude', 0)}, {data.get('longitude', 0)}")
        
        # Store RTK coordinates
        self.rtk_coordinates = data
        
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for integrator events
        
        Args:
            event: Event type ('state_changed', 'precision_landing_complete', etc.)
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
                    
    def get_state(self) -> PrecisionLandingState:
        """Get current precision landing state"""
        return self.state
        
    def get_mission_status(self) -> MissionStatus:
        """Get current mission status"""
        return self.mission_monitor.get_mission_status()
        
    def get_rtk_coordinates(self) -> Optional[Dict[str, Any]]:
        """Get acquired RTK coordinates"""
        return self.rtk_coordinates
        
    def load_mission(self, mission_file: str) -> bool:
        """
        Load a mission from a QGC .plan file
        
        Args:
            mission_file: Path to the .plan file
            
        Returns:
            True if mission loaded successfully
        """
        return self.mission_monitor.load_mission_file(mission_file)
        
# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='QGC Mission Integrator for Precision Landing')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mission', type=str, help='Path to mission file')
    parser.add_argument('--connection', type=str, help='MAVLink connection string')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    args = parser.parse_args()
    
    # Override configuration with command line arguments
    config_overrides = {}
    if args.connection:
        config_overrides['mavlink_connection'] = args.connection
    if args.headless:
        config_overrides['headless'] = True
        
    try:
        # Create integrator
        integrator = QGCMissionIntegrator(config_file=args.config)
        
        # Apply command line overrides
        for key, value in config_overrides.items():
            integrator.config[key] = value
        
        # Start integrator
        integrator.start()
        
        # Load mission if specified
        if args.mission:
            if integrator.load_mission(args.mission):
                print(f"Mission loaded: {args.mission}")
            else:
                print(f"Failed to load mission: {args.mission}")
        
        # Wait for completion
        try:
            print("QGC Mission Integrator running. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'integrator' in locals():
            integrator.stop()
