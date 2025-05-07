#!/usr/bin/env python3

"""
Wrapper script to integrate the enhanced ArUco detector for OpenCV 4.10.0
with the main OAK-D ArUco detector.

This script bridges the original OAK-D ArUco detector with the enhanced
detector implementation that properly handles OpenCV 4.10.0 API changes
and adds additional validation to prevent false positives.
"""

import cv2
import numpy as np
import sys
import os
import importlib
import types

# Import our enhanced detector
try:
    from fix_aruco_opencv410 import detect_aruco_markers
    print("Enhanced ArUco detector for OpenCV 4.10.0 loaded successfully")
except ImportError:
    print("Error: Could not import enhanced ArUco detector.")
    print("Make sure fix_aruco_opencv410.py is in the same directory.")
    sys.exit(1)

def run_detector():
    """
    Run the OAK-D ArUco detector with OpenCV 4.10.0 compatibility
    
    This function executes the original detector script with the enhanced
    ArUco detection implementation.
    """
    # Import the main detector script
    script_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create backup of original script
    original_script = os.path.join(script_path, "oak_d_aruco_6x6_detector.py")
    backup_script = os.path.join(script_path, "oak_d_aruco_6x6_detector.bak.py")
    
    if not os.path.exists(backup_script):
        print("Creating backup of original detector script...")
        try:
            with open(original_script, 'r') as f_in:
                with open(backup_script, 'w') as f_out:
                    f_out.write(f_in.read())
            print(f"Backup created at {backup_script}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    print("Running OAK-D ArUco detector with OpenCV 4.10.0 compatibility...")
    print("=" * 50)
    
    try:
        # Import the original detector
        import oak_d_aruco_6x6_detector
        
        # Monkey-patch the detect_aruco_markers method
        def monkey_patched_detect_aruco_markers(self, frame, simple_detection=False):
            """
            Patched version of detect_aruco_markers function
            
            This function replaces the original implementation with our enhanced
            version that properly handles OpenCV 4.10.0 API changes.
            """
            # Preprocess image
            if simple_detection:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.preprocess_image(frame)
            
            # For OpenCV 4.10.0, use our enhanced detector
            if cv2.__version__.startswith("4.10"):
                print("Using enhanced ArUco detector for OpenCV 4.10.0")
                
                # Use our enhanced detector
                markers_frame, corners, ids = detect_aruco_markers(
                    frame, 
                    self.aruco_dict, 
                    self.aruco_params,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                return markers_frame, corners, ids
            else:
                # For other versions, call the original method
                print(f"Using original detector for OpenCV {cv2.__version__}")
                
                # Get the original class to access the original method
                original_class = oak_d_aruco_6x6_detector.OakDArUcoDetector
                
                # Get the unbound original method
                original_method = original_class.detect_aruco_markers.__func__
                
                # Call the original method (we need to pass self explicitly for unbound methods)
                return original_method(self, frame, simple_detection)
        
        # Apply the monkey patch to the class
        oak_d_aruco_6x6_detector.OakDArUcoDetector.detect_aruco_markers = monkey_patched_detect_aruco_markers
        
        # Run the original script with command line arguments
        print("Starting detector with enhanced ArUco detection...")
        oak_d_aruco_6x6_detector.main()
        
    except Exception as e:
        print(f"Error patching and running detector: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if running the correct version of OpenCV
    print(f"OpenCV version: {cv2.__version__}")
    
    if not cv2.__version__.startswith("4.10"):
        print("Warning: This wrapper is designed for OpenCV 4.10.x")
        print(f"Current version: {cv2.__version__}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Run the detector with our enhanced implementation
    run_detector()
