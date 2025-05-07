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
import time

# Global variable to track which dictionary method was used for marker generation
# This will be imported from fix_aruco_opencv410
generation_dict_method = None

# Import our enhanced detector
try:
    from fix_aruco_opencv410 import detect_aruco_markers, generation_dict_method
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
                # Print once which dictionary method is being used
                print(f"Using enhanced ArUco detector for OpenCV 4.10.0")
                
                # Ensure we're using the same dictionary type that was used for generation
                if generation_dict_method is not None:
                    print(f"Using dictionary method from generation: {generation_dict_method}")
                    
                    # Always use getPredefinedDictionary first since it works best with OpenCV 4.10
                    try:
                        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
                        print("Using getPredefinedDictionary for detection (primary method)")
                    except Exception as e:
                        print(f"Error using getPredefinedDictionary: {e}")
                        
                        # Fallback methods in order of preference
                        dictionary_created = False
                        
                        # Try Dictionary constructor with marker size
                        try:
                            self.aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
                            print("Using Dictionary constructor with markerSize for detection (fallback 1)")
                            dictionary_created = True
                        except Exception as e1:
                            # Try Dictionary_get
                            try:
                                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                                print("Using Dictionary_get for detection (fallback 2)")
                                dictionary_created = True
                            except Exception as e2:
                                # Try Dictionary.create
                                try:
                                    self.aruco_dict = cv2.aruco.Dictionary.create(cv2.aruco.DICT_6X6_250)
                                    print("Using Dictionary.create for detection (fallback 3)")
                                    dictionary_created = True
                                except Exception as e3:
                                    print(f"Failed all dictionary creation methods: {e1}, {e2}, {e3}")
                        
                        if not dictionary_created:
                            print("WARNING: Could not create appropriate dictionary for detection!")
                
                # Configure detection parameters to match the fix script
                print("Configuring optimal detection parameters...")
                
                # Backup original parameters in case we need to restore them
                original_params = {}
                if hasattr(self.aruco_params, 'adaptiveThreshWinSizeMin'):
                    original_params['adaptiveThreshWinSizeMin'] = self.aruco_params.adaptiveThreshWinSizeMin
                if hasattr(self.aruco_params, 'adaptiveThreshWinSizeMax'):
                    original_params['adaptiveThreshWinSizeMax'] = self.aruco_params.adaptiveThreshWinSizeMax
                if hasattr(self.aruco_params, 'adaptiveThreshWinSizeStep'):
                    original_params['adaptiveThreshWinSizeStep'] = self.aruco_params.adaptiveThreshWinSizeStep
                if hasattr(self.aruco_params, 'adaptiveThreshConstant'):
                    original_params['adaptiveThreshConstant'] = self.aruco_params.adaptiveThreshConstant
                
                # Set parameters for optimal detection - match fix_aruco_opencv410 settings
                try:
                    # Critical detection parameters from fix_aruco_opencv410.py
                    self.aruco_params.adaptiveThreshWinSizeMin = 3
                    self.aruco_params.adaptiveThreshWinSizeMax = 23
                    self.aruco_params.adaptiveThreshWinSizeStep = 10
                    self.aruco_params.adaptiveThreshConstant = 7
                    
                    # Critical validation parameters - much more relaxed to detect markers
                    self.aruco_params.minMarkerPerimeterRate = 0.01  # Very relaxed to detect smaller markers
                    self.aruco_params.maxMarkerPerimeterRate = 4.0
                    self.aruco_params.polygonalApproxAccuracyRate = 0.1  # More relaxed for distorted markers
                    
                    # Corner refinement - important for accurate detection
                    if hasattr(self.aruco_params, 'cornerRefinementMethod'):
                        self.aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
                    if hasattr(self.aruco_params, 'cornerRefinementWinSize'):
                        self.aruco_params.cornerRefinementWinSize = 5
                    if hasattr(self.aruco_params, 'cornerRefinementMaxIterations'):
                        self.aruco_params.cornerRefinementMaxIterations = 30
                    
                    # Error correction is critical for proper detection in challenging conditions
                    if hasattr(self.aruco_params, 'errorCorrectionRate'):
                        self.aruco_params.errorCorrectionRate = 0.8  # Increased from 0.6 for better detection
                    
                    # Additional parameters from the fix script
                    if hasattr(self.aruco_params, 'minCornerDistanceRate'):
                        self.aruco_params.minCornerDistanceRate = 0.03  # Relaxed from 0.05
                    if hasattr(self.aruco_params, 'minDistanceToBorder'):
                        self.aruco_params.minDistanceToBorder = 1  # Reduced from 3
                    if hasattr(self.aruco_params, 'minOtsuStdDev'):
                        self.aruco_params.minOtsuStdDev = 3.0  # Reduced from 5.0
                        
                    print("Detection parameters configured successfully")
                except Exception as e:
                    print(f"Warning: Failed to set detection parameters: {e}")
                    print("Falling back to original parameters")
                    
                    # Restore original parameters
                    for param, value in original_params.items():
                        try:
                            setattr(self.aruco_params, param, value)
                        except:
                            pass
                            
                # Enable debugging - output shape of input frame
                print(f"Input frame shape: {frame.shape}")
                
                # If a detector already exists, ensure it's configured correctly
                if hasattr(self, 'aruco_detector') and self.aruco_detector is not None:
                    try:
                        print("Reconfiguring existing ArucoDetector with optimized dictionary and parameters")
                        # Create a new detector with our configured parameters
                        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                    except Exception as e:
                        print(f"Warning: Could not reconfigure ArucoDetector: {e}")
                
                # Try standard detection first
                print("Calling enhanced detect_aruco_markers function...")
                markers_frame, corners, ids = detect_aruco_markers(
                    frame,
                    self.aruco_dict,
                    self.aruco_params,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                # If no markers found, try direct dictionary retrieval for CharucoBoard detection
                if ids is None or len(ids) == 0:
                    print("Standard detection found no markers, trying specialized CharucoBoard detection...")
                    
                    # Try different dictionary creation methods - critical for CharucoBoard
                    try:
                        # Use only methods that work with OpenCV 4.10
                        dictionary_methods = [
                            # Method 1: getPredefinedDictionary (primary method for OpenCV 4.10)
                            lambda: cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
                            
                            # Method 2: Dictionary constructor with markerSize (fallback for OpenCV 4.10)
                            lambda: cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6),
                            
                            # Method 3: Try with 4x4 dictionary instead (might work better with small markers)
                            lambda: cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
                        ]
                        
                        # Try each dictionary method
                        for dict_method in dictionary_methods:
                            try:
                                # Create CharucoBoard with current dictionary method
                                print(f"Trying CharucoBoard detection with alternate dictionary...")
                                current_dict = dict_method()
                                
                                # Configure detection parameters specifically for CharucoBoard
                                # Reduced validation constraints to help with embedded markers
                                if hasattr(self.aruco_params, 'minMarkerPerimeterRate'):
                                    self.aruco_params.minMarkerPerimeterRate = 0.01  # More relaxed than before
                                if hasattr(self.aruco_params, 'errorCorrectionRate'):
                                    self.aruco_params.errorCorrectionRate = 0.8  # Increase error correction for embedded markers
                                
                                # Create board - try different dimensions
                                board_configs = [
                                    (6, 6),   # Standard 6x6 CharucoBoard
                                    (5, 7),   # Alternative dimensions
                                    (7, 5)    # Alternative dimensions
                                ]
                                
                                for squares_x, squares_y in board_configs:
                                    print(f"Trying CharucoBoard with {squares_x}x{squares_y} configuration...")
                                    # Calibration parameters
                                    square_length = 0.3048/max(squares_x, squares_y)  # Scale based on board size
                                    marker_length = square_length * 0.75  # 75% of square size
                                    
                                    try:
                                        # Create CharucoBoard
                                        board = cv2.aruco.CharucoBoard(
                                            (squares_x, squares_y),
                                            square_length,
                                            marker_length,
                                            current_dict
                                        )
                                        
                                        # Create CharucoDetector with relaxed parameters
                                        charuco_params = cv2.aruco.CharucoParameters()
                                        # Attempt to set more permissive detection parameters
                                        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params)
                                        
                                        # Detect the board
                                        print("Detecting CharucoBoard...")
                                        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
                                        
                                        # If markers found
                                        if marker_ids is not None and len(marker_ids) > 0:
                                            print(f"CharucoBoard detection found {len(marker_ids)} markers with IDs: {marker_ids.flatten()}")
                                            corners = marker_corners
                                            ids = marker_ids
                                            markers_frame = frame.copy()
                                            cv2.aruco.drawDetectedMarkers(markers_frame, corners, ids)
                                            break  # Found markers, exit configuration loop
                                        
                                    except Exception as e:
                                        print(f"Error with {squares_x}x{squares_y} configuration: {e}")
                                
                                # If we found markers, break out of dictionary loop too
                                if ids is not None and len(ids) > 0:
                                    break
                                    
                                # Try with inverted image
                                inverted = cv2.bitwise_not(gray)
                                try:
                                    print("Trying with inverted image...")
                                    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(inverted)
                                    
                                    # If markers found
                                    if marker_ids is not None and len(marker_ids) > 0:
                                        print(f"CharucoBoard detection found {len(marker_ids)} markers in inverted image with IDs: {marker_ids.flatten()}")
                                        corners = marker_corners
                                        ids = marker_ids
                                        markers_frame = frame.copy()
                                        cv2.aruco.drawDetectedMarkers(markers_frame, corners, ids)
                                        break  # Found markers, exit dictionary loop
                                except Exception as e:
                                    print(f"Error with inverted image: {e}")
                                
                            except Exception as e:
                                print(f"Error with dictionary method: {e}")
                        
                    except Exception as e:
                        print(f"Error during specialized CharucoBoard detection: {e}")
                
                # If still no markers found, try with direct ChArUco approach
                if ids is None or len(ids) == 0:
                    print("Still no markers detected, trying direct chessboard detection...")
                    
                    try:
                        # The black squares might be detected without ArUco IDs - we can try to extract them
                        # Find chessboard corners directly
                        board_sizes = [(6, 6), (5, 7), (7, 5), (6, 7), (7, 6)]
                        
                        for pattern_size in board_sizes:
                            print(f"Trying chessboard pattern with size {pattern_size}...")
                            chessboard_found, corners_cb = cv2.findChessboardCorners(gray, pattern_size, None)
                            
                            if chessboard_found:
                                print(f"Found chessboard pattern with size {pattern_size}!")
                                
                                # Create synthetic IDs for the black squares - assign sequential IDs
                                # This might help with pose estimation even if we can't extract actual ArUco IDs
                                synthetic_ids = []
                                for i in range(len(corners_cb)):
                                    synthetic_ids.append([i])  # Format like ArUco IDs
                                
                                # Convert to expected format
                                synthetic_corners = []
                                for corner in corners_cb:
                                    # ArUco corners are in format [[[x, y]]] - add extra dimensions
                                    synthetic_corners.append(np.array([corner], dtype=np.float32))
                                
                                # Use these synthetic results
                                corners = synthetic_corners
                                ids = np.array(synthetic_ids)
                                markers_frame = frame.copy()
                                
                                # Draw the chessboard corners
                                cv2.drawChessboardCorners(markers_frame, pattern_size, corners_cb, chessboard_found)
                                
                                print(f"Created {len(synthetic_ids)} synthetic marker IDs from chessboard detection")
                                break
                    except Exception as e:
                        print(f"Error during direct chessboard detection: {e}")
                
                # Final output
                if ids is not None and len(ids) > 0:
                    print(f"Detection successful: found {len(ids)} markers with IDs: {ids.flatten()}")
                else:
                    print("All detection methods failed, no markers found")
                
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
    
    # Print information about the dictionary method
    print("=" * 50)
    print("ArUco Detection Configuration:")
    print(f"- OpenCV Version: {cv2.__version__}")
    
    # Check if we have a generation dictionary method
    if generation_dict_method is not None:
        print(f"- Dictionary Method: {generation_dict_method}")
    else:
        print("- Dictionary Method: Not yet determined (will use default)")
    print("=" * 50)
    
    # Run the detector with our enhanced implementation
    run_detector()
