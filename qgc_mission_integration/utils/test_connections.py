#!/usr/bin/env python3

"""
Connection Tester for QGC Mission Integration

This utility script tests the critical connections needed for the QGC Mission Integration
system to function properly. It helps diagnose common connection issues.
"""

import os
import sys
import time
import yaml
import argparse
import logging
import requests
from typing import Dict, Any, Tuple

# Add the project root to the path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConnectionTester")

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError("Empty configuration file")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def test_mavlink_connection(connection_string: str, baudrate: int = None) -> Tuple[bool, str]:
    """
    Test MAVLink connection to autopilot
    
    Args:
        connection_string: MAVLink connection string
        baudrate: Optional baudrate for serial connections
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Import here to avoid dependency for users who don't have pymavlink
        from pymavlink import mavutil
        
        logger.info(f"Testing MAVLink connection to {connection_string}")
        
        # Create connection arguments
        conn_kwargs = {}
        if baudrate and "dev/tty" in connection_string:
            conn_kwargs['baud'] = baudrate
            
        # Create connection
        start_time = time.time()
        mav_conn = mavutil.mavlink_connection(connection_string, **conn_kwargs)
        
        # Wait for heartbeat (max 10 seconds)
        logger.info("Waiting for heartbeat (max 10 seconds)...")
        heartbeat = mav_conn.wait_heartbeat(timeout=10)
        
        if heartbeat:
            elapsed = time.time() - start_time
            # Get system information
            mode = mav_conn.flightmode
            sys_id = mav_conn.target_system
            comp_id = mav_conn.target_component
            
            return True, (
                f"✅ MAVLink connection successful in {elapsed:.1f}s\n"
                f"   System ID: {sys_id}, Component ID: {comp_id}\n"
                f"   Mode: {mode}\n"
                f"   Autopilot type: {heartbeat.autopilot}\n"
                f"   MAVLink version: {heartbeat.mavlink_version}"
            )
        else:
            return False, "❌ Timed out waiting for heartbeat"
            
    except ImportError:
        return False, "❌ Failed to import pymavlink. Install with: pip install pymavlink"
    except Exception as e:
        return False, f"❌ Error connecting to MAVLink: {e}"
    finally:
        if 'mav_conn' in locals():
            mav_conn.close()

def test_rtk_server(server_url: str) -> Tuple[bool, str]:
    """
    Test connection to RTK GPS server
    
    Args:
        server_url: URL of the RTK GPS server
        
    Returns:
        Tuple of (success, message)
    """
    try:
        logger.info(f"Testing RTK GPS server connection to {server_url}")
        
        # Make request to the GPS location endpoint
        endpoint = f"{server_url}/gps_location.json"
        response = requests.get(endpoint, timeout=5)
        
        if response.status_code == 200:
            # Try to parse the JSON response
            try:
                gps_data = response.json()
                lat = gps_data.get('latitude', 'N/A')
                lon = gps_data.get('longitude', 'N/A')
                alt = gps_data.get('altitude_m', 'N/A')
                fix_type = gps_data.get('fix_type', 'N/A')
                satellites = gps_data.get('satellites', 'N/A')
                
                return True, (
                    f"✅ RTK GPS server connection successful\n"
                    f"   Latitude: {lat}\n"
                    f"   Longitude: {lon}\n"
                    f"   Altitude: {alt}m\n"
                    f"   Fix type: {fix_type}\n"
                    f"   Satellites: {satellites}"
                )
            except ValueError:
                return True, "✅ RTK GPS server connection successful (response not JSON)"
        else:
            return False, f"❌ RTK GPS server returned status code {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, f"❌ Connection error: Could not connect to {server_url}"
    except requests.exceptions.Timeout:
        return False, f"❌ Connection timeout: {server_url} did not respond in time"
    except Exception as e:
        return False, f"❌ Error connecting to RTK GPS server: {e}"

def test_usb_connection(device_path: str) -> Tuple[bool, str]:
    """
    Test if a USB device exists and is accessible
    
    Args:
        device_path: Path to USB device (e.g., /dev/ttyACM0)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        logger.info(f"Testing USB device access: {device_path}")
        
        if os.path.exists(device_path):
            # Check if we can open it
            try:
                with open(device_path, 'rb') as f:
                    pass
                return True, f"✅ USB device {device_path} exists and is accessible"
            except PermissionError:
                return False, f"❌ Permission denied: Cannot access {device_path}. Try: sudo chmod a+rw {device_path}"
            except Exception as e:
                return False, f"❌ USB device exists but cannot be opened: {e}"
        else:
            return False, f"❌ USB device {device_path} does not exist"
            
    except Exception as e:
        return False, f"❌ Error checking USB device: {e}"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test connections for QGC Mission Integration')
    parser.add_argument('--config', '-c', type=str, default='../config_template.yaml',
                      help='Path to configuration file (default: ../config_template.yaml)')
    parser.add_argument('--mavlink-only', action='store_true', help='Test only MAVLink connection')
    parser.add_argument('--rtk-only', action='store_true', help='Test only RTK GPS server connection')
    parser.add_argument('--usb-only', action='store_true', help='Test only USB device access')
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), args.config))
    config = load_config(config_path)
    
    if not config:
        logger.error(f"Failed to load configuration from {config_path}")
        sys.exit(1)
    
    # Print header
    print("\n" + "=" * 70)
    print("QGC Mission Integration Connection Test".center(70))
    print("=" * 70 + "\n")
    
    # Track overall success
    all_tests_passed = True
    
    # Test USB connection if it's a serial connection
    connection_string = config.get('mavlink_connection', '')
    if (not args.rtk_only and not args.mavlink_only) or args.usb_only:
        if '/dev/tty' in connection_string:
            success, message = test_usb_connection(connection_string)
            print(f"\n{message}")
            all_tests_passed = all_tests_passed and success
    
    # Test MAVLink connection
    if (not args.rtk_only and not args.usb_only) or args.mavlink_only:
        baudrate = config.get('mavlink_baudrate')
        success, message = test_mavlink_connection(connection_string, baudrate)
        print(f"\n{message}")
        all_tests_passed = all_tests_passed and success
    
    # Test RTK server connection
    if (not args.mavlink_only and not args.usb_only) or args.rtk_only:
        rtk_server_url = config.get('rtk_server_url', 'http://localhost:8000')
        success, message = test_rtk_server(rtk_server_url)
        print(f"\n{message}")
        all_tests_passed = all_tests_passed and success
    
    # Print summary
    print("\n" + "=" * 70)
    if all_tests_passed:
        print("✅ ALL CONNECTIONS SUCCESSFUL!".center(70))
    else:
        print("❌ SOME CONNECTIONS FAILED - See details above".center(70))
    print("=" * 70 + "\n")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
