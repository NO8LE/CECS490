#!/usr/bin/env python3

"""
QGroundControl Mission Integrator - Main Entry Point

This script provides the main entry point for the QGC Mission Integrator system,
which enables precision landing and RTK GPS coordinate acquisition during
QGroundControl missions.
"""

import os
import sys
import time
import logging
import argparse
import signal
from typing import Dict, Any

# Add the project root to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import QGC mission integrator
from qgc_mission_integration.qgc_mission_integrator import QGCMissionIntegrator, PrecisionLandingState

def setup_logging(log_file=None, log_level="INFO"):
    """
    Set up logging configuration
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
        
    return root_logger

def signal_handler(signum, frame):
    """
    Handle system signals for graceful shutdown
    """
    print("\nReceived signal to terminate. Shutting down...")
    if 'integrator' in globals():
        print("Stopping QGC mission integrator...")
        integrator.stop()
    sys.exit(0)

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='QGroundControl Mission Integrator for Precision Landing')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--mission', '-m', type=str, help='Path to mission file')
    parser.add_argument('--connection', type=str, help='MAVLink connection string')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--log-level', '-l', type=str, default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    parser.add_argument('--target-id', '-t', type=int, help='Target ArUco marker ID')
    parser.add_argument('--rtk-server', type=str, help='RTK GPS server URL')
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set up logging
    log_file = args.log_file
    logger = setup_logging(log_file, args.log_level)
    logger.info("Starting QGC Mission Integrator")
    
    # Create configuration overrides from command line arguments
    config_overrides = {}
    if args.connection:
        config_overrides['mavlink_connection'] = args.connection
    if args.headless:
        config_overrides['headless'] = True
    if args.target_id is not None:
        config_overrides['target_marker_id'] = args.target_id
    if args.rtk_server:
        config_overrides['rtk_server_url'] = args.rtk_server
        
    # Create and start integrator
    global integrator
    try:
        # Create integrator
        integrator = QGCMissionIntegrator(config_file=args.config)
        
        # Apply command line overrides
        for key, value in config_overrides.items():
            logger.info(f"Overriding config {key} = {value}")
            integrator.config[key] = value
        
        # Start integrator
        integrator.start()
        
        # Load mission if specified
        if args.mission:
            if integrator.load_mission(args.mission):
                logger.info(f"Mission loaded: {args.mission}")
            else:
                logger.error(f"Failed to load mission: {args.mission}")
        
        # Register state change callback for logging
        def on_state_change(data):
            current_state = data.get('current_state')
            previous_state = data.get('previous_state')
            logger.info(f"State changed: {previous_state.value} -> {current_state.value}")
            
            # Special handling for certain states
            if current_state == PrecisionLandingState.PRECISION_LANDING:
                logger.info("Starting precision landing sequence")
            elif current_state == PrecisionLandingState.RTK_ACQUISITION:
                logger.info("Acquiring RTK GPS coordinates")
            elif current_state == PrecisionLandingState.COMPLETE:
                logger.info("Mission completed successfully")
                
        integrator.register_callback('state_changed', on_state_change)
        
        # Main loop
        while True:
            # Check if mission is complete
            if integrator.get_state() == PrecisionLandingState.COMPLETE:
                # Get RTK coordinates if available
                rtk_coords = integrator.get_rtk_coordinates()
                if rtk_coords:
                    logger.info(f"Final RTK coordinates: {rtk_coords.get('latitude')}, {rtk_coords.get('longitude')}")
                    
                logger.info("Mission complete, press Ctrl+C to exit")
                
            # Sleep to prevent CPU overuse
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Clean up
        if 'integrator' in locals():
            logger.info("Stopping QGC mission integrator")
            integrator.stop()
            
        logger.info("QGC mission integrator exited")

if __name__ == "__main__":
    main()
