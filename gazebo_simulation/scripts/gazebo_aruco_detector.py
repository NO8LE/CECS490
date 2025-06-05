#!/usr/bin/env python3

"""
Gazebo ArUco Marker Detector for Precision Landing Simulation (Non-ROS)

This script detects ArUco markers in camera images from a Gazebo simulation.
It is designed for autonomous precision landing simulations on Ubuntu:
1. Connects directly to Gazebo's camera sensor
2. Processes images to detect ArUco markers
3. Sends commands to the simulated drone via MAVLink
4. Visualizes detections (when not in headless mode)

Usage:
  python3 gazebo_aruco_detector.py [options]

Options:
  --target, -t MARKER_ID     Marker ID to use as landing target (default: 5)
  --camera-topic TOPIC       Gazebo camera topic (default: /gazebo/default/camera/link/camera/image)
  --marker-size SIZE         Size of marker in meters (default: 0.3048)
  --headless                 Run in headless mode (no visualization)
  --mavlink-connection CONN  MAVLink connection string (default: udp:localhost:14560)
  --verbose, -v              Enable verbose output
"""

import os
import sys
import time
import cv2
import math
import numpy as np
import argparse
import threading
import socket
import struct
from threading import Thread
from scipy.spatial.transform import Rotation as R
from pymavlink import mavutil

# Import ArUco OpenCV 4.10 fixes if available
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../aruco'))
    from opencv410_aruco_fix import OpenCV410ArUcoFix
    USE_ARUCO_FIX = True
    print("Using OpenCV 4.10 ArUco fixes")
except ImportError:
    USE_ARUCO_FIX = False
    print("OpenCV 4.10 ArUco fixes not found, using standard OpenCV ArUco")

class GazeboConnection:
    """Direct connection to Gazebo's transport system"""
    
    def __init__(self, camera_topic="/gazebo/default/camera/link/camera/image"):
        self.camera_topic = camera_topic
        self.running = True
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.connection_thread = None
        
        # Gazebo connection settings
        self.gazebo_ip = "127.0.0.1"
        self.gazebo_port = 11345  # Default Gazebo transport port
        
    def start(self):
        """Start connection to Gazebo"""
        self.connection_thread = Thread(target=self._connection_loop)
        self.connection_thread.daemon = True
        self.connection_thread.start()
        print(f"Started Gazebo connection on topic {self.camera_topic}")
        
    def _connection_loop(self):
        """Main connection loop"""
        try:
            # Create UDP socket for Gazebo transport
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", 0))
            
            # Subscribe to camera topic
            subscribe_msg = f"sub:{self.camera_topic}"
            sock.sendto(subscribe_msg.encode(), (self.gazebo_ip, self.gazebo_port))
            
            # Set timeout
            sock.settimeout(1.0)
            
            # Main reception loop
            while self.running:
                try:
                    # Receive data
                    data, addr = sock.recvfrom(1024*1024)  # Increase buffer for image data
                    
                    # Parse image data (simplified - in real implementation this would
                    # need to match Gazebo's image format)
                    if len(data) > 100:  # Basic check for a valid image message
                        # Extract image dimensions and format
                        header_size = 16  # This would need to match Gazebo's format
                        width, height = struct.unpack('II', data[0:8])
                        
                        # Convert to OpenCV format (assuming RGB)
                        img_data = data[header_size:]
                        img_array = np.frombuffer(img_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        # Update latest image
                        with self.image_lock:
                            self.latest_image = img
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error receiving Gazebo data: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"Gazebo connection error: {e}")
        finally:
            if sock:
                sock.close()
    
    def get_latest_image(self):
        """Get the latest camera image"""
        with self.image_lock:
            return self.latest_image.copy() if self.latest_image is not None else None
            
    def stop(self):
        """Stop the connection"""
        self.running = False
        if self.connection_thread:
            self.connection_thread.join(timeout=1.0)

class ArUcoDetector:
    """ArUco marker detector for Gazebo simulation"""
    
    def __init__(self, args):
        # Store arguments
        self.args = args
        self.target_id = args.target_id
        self.marker_size = args.marker_size
        self.headless = args.headless
        self.verbose = args.verbose
        
        # MAVLink connection
        self.mavlink = None
        if args.mavlink_connection:
            try:
                self.mavlink = mavutil.mavlink_connection(args.mavlink_connection)
                print(f"Connected to MAVLink at {args.mavlink_connection}")
            except Exception as e:
                print(f"Failed to connect to MAVLink: {e}")
        
        # Gazebo connection
        self.gazebo = GazeboConnection(args.camera_topic)
        
        # Camera calibration (default values - will be updated from camera info)
        self.camera_matrix = np.array([
            [860.0, 0, 640.0],
            [0, 860.0, 360.0],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((1, 5))
        
        # ArUco detector setup
        self.initialize_detector()
        
        # Latest detection results
        self.latest_detections = {}
        self.detection_thread = None
        self.running = True
        
        # Visualization window
        if not self.headless:
            cv2.namedWindow("Gazebo ArUco Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Gazebo ArUco Detector", 800, 600)
    
    def initialize_detector(self):
        """Initialize ArUco detector with OpenCV 4.10 compatibility"""
        # Define ArUco dictionary
        self.aruco_dict_type = cv2.aruco.DICT_6X6_250
        
        # Initialize detector
        if USE_ARUCO_FIX:
            # Use OpenCV 4.10 fix
            self.aruco_fix = OpenCV410ArUcoFix()
            self.aruco_dict = self.aruco_fix.create_dictionary(self.aruco_dict_type)
            self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            # Standard OpenCV approach
            try:
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
            except AttributeError:
                self.aruco_dict = cv2.aruco.Dictionary_get(self.aruco_dict_type)
            
            try:
                self.aruco_params = cv2.aruco.DetectorParameters()
            except AttributeError:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
                
        # Configure detector parameters
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.01
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
        
        print("ArUco detector initialized")
    
    def start(self):
        """Start the detector"""
        # Start Gazebo connection
        self.gazebo.start()
        
        # Start detection thread
        self.detection_thread = Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        print("ArUco detector started")
    
    def _detection_loop(self):
        """Main detection loop"""
        while self.running:
            # Get latest image
            frame = self.gazebo.get_latest_image()
            if frame is None:
                time.sleep(0.01)
                continue
                
            # Detect markers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            try:
                if USE_ARUCO_FIX:
                    # Use OpenCV 4.10 fix
                    corners, ids, rejected = self.aruco_fix.detect_markers(
                        gray, self.aruco_dict, parameters=self.aruco_params
                    )
                else:
                    # Standard OpenCV approach
                    corners, ids, rejected = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict, parameters=self.aruco_params
                    )
            except Exception as e:
                print(f"Error detecting markers: {e}")
                corners, ids = [], None
            
            # Process detections
            if ids is not None and len(ids) > 0:
                # Draw detected markers
                markers_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
                
                # Check for target marker
                target_idx = None
                for i, marker_id in enumerate(ids):
                    if marker_id[0] == self.target_id:
                        target_idx = i
                        break
                
                # If target found, estimate pose and send to MAVLink
                if target_idx is not None:
                    try:
                        # Estimate pose
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            [corners[target_idx]], self.marker_size, 
                            self.camera_matrix, self.dist_coeffs
                        )
                        
                        # Draw axis
                        cv2.aruco.drawAxis(
                            markers_frame, self.camera_matrix, self.dist_coeffs,
                            rvec[0], tvec[0], self.marker_size/2
                        )
                        
                        # Send to MAVLink if connected
                        if self.mavlink:
                            self._send_landing_target(rvec[0], tvec[0])
                        
                        # Store in latest detections
                        self.latest_detections[self.target_id] = {
                            'corners': corners[target_idx],
                            'rvec': rvec[0],
                            'tvec': tvec[0],
                            'timestamp': time.time()
                        }
                        
                        if self.verbose:
                            # Print pose information
                            x, y, z = tvec[0][0], tvec[0][1], tvec[0][2]
                            distance = np.sqrt(x*x + y*y + z*z)
                            print(f"Target marker {self.target_id} at position [{x:.2f}, {y:.2f}, {z:.2f}], distance: {distance:.2f}m")
                    except Exception as e:
                        print(f"Error processing target marker: {e}")
                
                # Display result if not headless
                if not self.headless:
                    cv2.imshow("Gazebo ArUco Detector", markers_frame)
                    cv2.waitKey(1)
            else:
                # No markers detected
                if not self.headless:
                    cv2.imshow("Gazebo ArUco Detector", frame)
                    cv2.waitKey(1)
            
            # Sleep to maintain frame rate
            time.sleep(0.03)  # ~30 FPS
    
    def _send_landing_target(self, rvec, tvec):
        """Send landing target message to MAVLink"""
        if self.mavlink is None:
            return
            
        try:
            # Convert tvec to meters
            x, y, z = tvec[0]
            
            # Calculate angle to target
            angle_x = math.atan2(x, z)
            angle_y = math.atan2(y, z)
            
            # Calculate distance to target
            distance = math.sqrt(x*x + y*y + z*z)
            
            # Send LANDING_TARGET message
            self.mavlink.mav.landing_target_send(
                int(time.time() * 1000000),  # time_usec
                0,                            # target_num
                8,                            # frame (MAV_FRAME_BODY_NED)
                angle_x,                      # angle_x
                angle_y,                      # angle_y
                distance,                     # distance
                0,                            # size_x (not used)
                0,                            # size_y (not used)
                0,                            # x (not used in this mode)
                0,                            # y (not used in this mode)
                0,                            # z (not used in this mode)
                (0, 0, 0, 0),                 # quaternion (not used)
                1,                            # type (1=LANDING_TARGET_TYPE_VISION_FIDUCIAL)
                1                             # position_valid
            )
            
            if self.verbose:
                print(f"Sent LANDING_TARGET: angle_x={angle_x:.2f}, angle_y={angle_y:.2f}, distance={distance:.2f}m")
        except Exception as e:
            print(f"Error sending LANDING_TARGET: {e}")
    
    def get_latest_detections(self):
        """Get latest marker detections"""
        return self.latest_detections
    
    def stop(self):
        """Stop the detector"""
        self.running = False
        self.gazebo.stop()
        
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
            
        if not self.headless:
            cv2.destroyAllWindows()
            
        print("ArUco detector stopped")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Gazebo ArUco Marker Detector')
    
    parser.add_argument('--target', '-t', type=int, default=5,
                        help='Marker ID to use as landing target (default: 5)')
    
    parser.add_argument('--camera-topic', type=str, 
                        default='/gazebo/default/camera/link/camera/image',
                        help='Gazebo camera topic')
    
    parser.add_argument('--marker-size', type=float, default=0.3048,
                        help='Marker size in meters (default: 0.3048m / 12 inches)')
    
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no visualization)')
    
    parser.add_argument('--mavlink-connection', type=str, 
                        default='udp:localhost:14560',
                        help='MAVLink connection string (default: udp:localhost:14560)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    detector = ArUcoDetector(args)
    try:
        detector.start()
        
        # Run until interrupted
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        detector.stop()

if __name__ == "__main__":
    main()
