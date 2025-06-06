#!/usr/bin/env python3

"""
QGC Mission Integration Test Script

This script tests the core components of the QGC Mission Integration system
to verify that the setup is working correctly.
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any

# Add the project root to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import components to test
try:
    from mavlink_controller import MAVLinkController
except ImportError:
    print("Error: Could not import MAVLinkController.")
    print("Make sure mavlink_controller.py is in the project root directory.")
    sys.exit(1)

from qgc_mission_integration.mission_monitor import MissionMonitor
from qgc_mission_integration.aruco_detection_manager import ArUcoDetectionManager
from qgc_mission_integration.rtk_gps_interface import RTKGPSInterface
from qgc_mission_integration.safety_manager import SafetyManager
from qgc_mission_integration.utils.plan_parser import PlanParser

# Set up logging
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Setup Test")

def test_mavlink_connection(connection_string: str) -> bool:
    """Test MAVLink connection"""
    logger.info(f"Testing MAVLink connection to {connection_string}")
    try:
        mavlink = MAVLinkController(connection_string=connection_string)
        if mavlink.wait_for_connection(timeout=10):
            logger.info("✅ MAVLink connection successful")
            vehicle_state = mavlink.get_vehicle_state()
            logger.info(f"    Vehicle mode: {vehicle_state.get('mode', 'Unknown')}")
            logger.info(f"    Armed: {vehicle_state.get('armed', False)}")
            logger.info(f"    GPS: {vehicle_state.get('position', (0,0,0))}")
            logger.info(f"    Battery: {vehicle_state.get('battery_voltage', 0)}V")
            mavlink.close()
            return True
        else:
            logger.error("❌ Failed to establish MAVLink connection")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to MAVLink: {e}")
        return False

def test_aruco_detection(headless: bool = False) -> bool:
    """Test ArUco detection system"""
    logger.info("Testing ArUco detection")
    try:
        # Create minimal config for detection
        config = {
            'detection_resolution': 'low',  # Use low resolution for faster startup
            'detection_use_cuda': False,    # Disable CUDA for compatibility
            'target_marker_id': 5           # Default target ID
        }
        
        # Initialize detector
        detector = ArUcoDetectionManager(
            config=config,
            headless=headless,
            enable_streaming=False,
            test_mode=True  # Special flag for testing - doesn't start full detection
        )
        
        # Test initialization
        if detector.test_camera_connection():
            logger.info("✅ Camera connection successful")
            detector.close()
            return True
        else:
            logger.error("❌ Failed to connect to camera")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing ArUco detection: {e}")
        return False

def test_rtk_server(server_url: str) -> bool:
    """Test RTK GPS server connection"""
    logger.info(f"Testing RTK GPS server connection to {server_url}")
    try:
        rtk = RTKGPSInterface(server_url=server_url)
        if rtk.test_connection():
            logger.info("✅ RTK GPS server connection successful")
            return True
        else:
            logger.error("❌ Failed to connect to RTK GPS server")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to RTK GPS server: {e}")
        return False

def test_mission_parser(mission_file: str = None) -> bool:
    """Test mission plan parser"""
    logger.info("Testing mission plan parser")
    
    if mission_file and os.path.exists(mission_file):
        test_file = mission_file
    else:
        # Look for any .plan files in the current directory
        plan_files = [f for f in os.listdir('.') if f.endswith('.plan')]
        if plan_files:
            test_file = plan_files[0]
            logger.info(f"Found mission file: {test_file}")
        else:
            logger.warning("No mission file provided or found, using built-in test data")
            test_file = None
    
    try:
        parser = PlanParser()
        
        if test_file:
            # Test with actual file
            mission_data = parser.parse_plan_file(test_file)
        else:
            # Test with sample data
            sample_data = {
                "fileType": "Plan",
                "geoFence": {
                    "circles": [],
                    "polygons": [],
                    "version": 2
                },
                "groundStation": "QGroundControl",
                "mission": {
                    "cruiseSpeed": 15,
                    "firmwareType": 3,
                    "hoverSpeed": 5,
                    "items": [
                        {
                            "AMSLAltAboveTerrain": None,
                            "Altitude": 50,
                            "AltitudeMode": 1,
                            "autoContinue": True,
                            "command": 22,
                            "doJumpId": 1,
                            "frame": 3,
                            "params": [
                                0,
                                0,
                                0,
                                None,
                                47.397742,
                                8.545594,
                                50
                            ],
                            "type": "SimpleItem"
                        }
                    ],
                    "plannedHomePosition": [
                        47.397742,
                        8.545594,
                        488
                    ],
                    "vehicleType": 2,
                    "version": 2
                },
                "rallyPoints": {
                    "points": [],
                    "version": 2
                },
                "version": 1
            }
            mission_data = parser.parse_plan_data(sample_data)
        
        if mission_data and 'waypoints' in mission_data:
            logger.info("✅ Mission parser working correctly")
            logger.info(f"    Parsed {len(mission_data['waypoints'])} waypoints")
            return True
        else:
            logger.error("❌ Failed to parse mission data")
            return False
    except Exception as e:
        logger.error(f"❌ Error parsing mission: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test QGC Mission Integration Setup')
    parser.add_argument('--connection', type=str, default='udp:127.0.0.1:14550',
                      help='MAVLink connection string')
    parser.add_argument('--rtk-server', type=str, default='http://localhost:8000',
                      help='RTK GPS server URL')
    parser.add_argument('--mission', type=str, help='Path to mission file for testing')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--skip-mavlink', action='store_true', help='Skip MAVLink test')
    parser.add_argument('--skip-camera', action='store_true', help='Skip camera test')
    parser.add_argument('--skip-rtk', action='store_true', help='Skip RTK server test')
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("QGC Mission Integration Setup Test".center(70))
    print("=" * 70 + "\n")
    
    # Run tests
    results = {}
    
    # Test MAVLink connection
    if not args.skip_mavlink:
        results['mavlink'] = test_mavlink_connection(args.connection)
    else:
        logger.info("Skipping MAVLink test")
        
    # Test ArUco detection
    if not args.skip_camera:
        results['aruco'] = test_aruco_detection(args.headless)
    else:
        logger.info("Skipping camera test")
        
    # Test RTK server
    if not args.skip_rtk:
        results['rtk'] = test_rtk_server(args.rtk_server)
    else:
        logger.info("Skipping RTK server test")
        
    # Test mission parser
    results['mission'] = test_mission_parser(args.mission)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Results Summary".center(70))
    print("=" * 70)
    
    all_passed = True
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        status_formatted = f"[ {status} ]"
        if result:
            print(f"{status_formatted.ljust(10)} {test.upper()}")
        else:
            print(f"{status_formatted.ljust(10)} {test.upper()} - See errors above")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed! Your system is ready to run QGC Mission Integration.".center(70))
    else:
        print("Some tests failed. Please fix the issues before proceeding.".center(70))
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
