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
                    
                    # Create the appropriate dictionary based on the generation method
                    if generation_dict_method == "getPredefinedDictionary":
                        try:
                            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
                            print("Using getPredefinedDictionary for detection")
                        except Exception as e:
                            print(f"Error using getPredefinedDictionary: {e}")
                    elif generation_dict_method == "Dictionary_get":
                        try:
                            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                            print("Using Dictionary_get for detection")
                        except Exception as e:
                            print(f"Error using Dictionary_get: {e}")
                    elif generation_dict_method == "Dictionary" or generation_dict_method == "constructor":
                        try:
                            self.aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
                            print("Using Dictionary/constructor for detection")
                        except Exception as e:
                            print(f"Error using Dictionary constructor: {e}")
                    elif generation_dict_method == "create" or generation_dict_method == "Dictionary.create":
                        try:
                            self.aruco_dict = cv2.aruco.Dictionary.create(cv2.aruco.DICT_6X6_250)
                            print("Using Dictionary.create for detection")
                        except Exception as e:
                            print(f"Error using Dictionary.create: {e}")
                    elif generation_dict_method == "manual" or generation_dict_method == "text_fallback":
                        # For manual or fallback methods, try multiple approaches
                        print("Attempting multiple dictionary creation methods for manual/fallback markers")
                        dictionary_created = False
                        
                        # Try each method in sequence
                        try:
                            self.aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
                            print("Using Dictionary constructor for manual markers")
                            dictionary_created = True
                        except Exception as e1:
                            try:
                                self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
                                print("Using getPredefinedDictionary for manual markers")
                                dictionary_created = True
                            except Exception as e2:
                                try:
                                    self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                                    print("Using Dictionary_get for manual markers")
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
                    
                    # Critical validation parameters - more relaxed than original
                    self.aruco_params.minMarkerPerimeterRate = 0.03  # More relaxed to detect smaller markers (fix script uses 0.01)
                    self.aruco_params.maxMarkerPerimeterRate = 4.0
                    self.aruco_params.polygonalApproxAccuracyRate = 0.05
                    
                    # Corner refinement - important for accurate detection
                    if hasattr(self.aruco_params, 'cornerRefinementMethod'):
                        self.aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
                    if hasattr(self.aruco_params, 'cornerRefinementWinSize'):
                        self.aruco_params.cornerRefinementWinSize = 5
                    if hasattr(self.aruco_params, 'cornerRefinementMaxIterations'):
                        self.aruco_params.cornerRefinementMaxIterations = 30
                    
                    # Error correction is critical for proper detection in challenging conditions
                    if hasattr(self.aruco_params, 'errorCorrectionRate'):
                        self.aruco_params.errorCorrectionRate = 0.6  # Default value from fix script
                    
                    # Additional parameters from the fix script
                    if hasattr(self.aruco_params, 'minCornerDistanceRate'):
                        self.aruco_params.minCornerDistanceRate = 0.05
                    if hasattr(self.aruco_params, 'minDistanceToBorder'):
                        self.aruco_params.minDistanceToBorder = 3
                    if hasattr(self.aruco_params, 'minOtsuStdDev'):
                        self.aruco_params.minOtsuStdDev = 5.0
                        
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
                
                # Save a copy of the input frame for diagnostics if needed
                debug_frame = frame.copy()
                debug_dir = "aruco_debug"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
                
                # Try standard detection first
                print("Calling enhanced detect_aruco_markers function...")
                markers_frame, corners, ids = detect_aruco_markers(
                    frame,
                    self.aruco_dict,
                    self.aruco_params,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                # If no markers found, try with CharucoBoard detection
                if ids is None or len(ids) == 0:
                    print("Standard detection found no markers, trying CharucoBoard detection...")
                    try:
                        # Create CharucoBoard for detection (same parameters as calibration)
                        squares_x, squares_y = 6, 6  # Common CharucoBoard dimensions
                        square_length = 0.3048/6  # Default to 12 inches (0.3048m) / 6 squares
                        marker_length = square_length * 0.75  # 75% of square size
                        
                        # Create CharucoBoard
                        board = cv2.aruco.CharucoBoard(
                            (squares_x, squares_y),  # (squaresX, squaresY) tuple
                            square_length,  # squareLength
                            marker_length,  # markerLength 
                            self.aruco_dict  # Dictionary
                        )
                        
                        # Create CharucoDetector
                        charuco_params = cv2.aruco.CharucoParameters()
                        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params)
                        
                        # Save a copy of the grayscale image
                        gray_copy = gray.copy()
                        cv2.imwrite(f"{debug_dir}/gray_{timestamp}.jpg", gray_copy)
                        
                        print("Attempting CharucoBoard detection...")
                        # Detect the board
                        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
                        
                        # If markers found
                        if marker_ids is not None and len(marker_ids) > 0:
                            print(f"CharucoBoard detection found {len(marker_ids)} markers!")
                            # Use these results
                            corners = marker_corners
                            ids = marker_ids
                            
                            # Create a visualization of the results
                            markers_frame = frame.copy()
                            cv2.aruco.drawDetectedMarkers(markers_frame, corners, ids)
                            cv2.imwrite(f"{debug_dir}/charuco_detection_{timestamp}.jpg", markers_frame)
                    except Exception as e:
                        print(f"Error during CharucoBoard detection: {e}")
                
                # If still no markers found, try with inverted image (black/white flipped)
                if ids is None or len(ids) == 0:
                    print("CharucoBoard detection found no markers, trying with inverted image...")
                    try:
                        # Invert the image (flip black and white)
                        inverted = cv2.bitwise_not(gray)
                        cv2.imwrite(f"{debug_dir}/inverted_{timestamp}.jpg", inverted)
                        
                        # Try with inverted image for regular detection
                        inv_frame = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
                        inv_markers_frame, inv_corners, inv_ids = detect_aruco_markers(
                            inv_frame,
                            self.aruco_dict,
                            self.aruco_params,
                            self.camera_matrix,
                            self.dist_coeffs
                        )
                        
                        # If found markers in inverted image
                        if inv_ids is not None and len(inv_ids) > 0:
                            print(f"Found {len(inv_ids)} markers in inverted image!")
                            corners = inv_corners
                            ids = inv_ids
                            markers_frame = inv_markers_frame
                            cv2.imwrite(f"{debug_dir}/inverted_markers_{timestamp}.jpg", markers_frame)
                    except Exception as e:
                        print(f"Error during inverted image detection: {e}")
                
                # If still no markers found, try with different thresholding approaches
                if ids is None or len(ids) == 0:
                    print("Trying alternative thresholding approaches...")
                    
                    # Try different thresholding methods
                    thresholding_methods = [
                        # Method 1: Adaptive Gaussian thresholding
                        ("adaptive_gaussian", lambda img: cv2.adaptiveThreshold(
                            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                        
                        # Method 2: Adaptive Mean thresholding
                        ("adaptive_mean", lambda img: cv2.adaptiveThreshold(
                            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
                        
                        # Method 3: Otsu's thresholding
                        ("otsu", lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                        
                        # Method 4: Simple global thresholding (mid-range)
                        ("global_128", lambda img: cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1])
                    ]
                    
                    # Try each method
                    for method_name, threshold_func in thresholding_methods:
                        try:
                            print(f"Trying {method_name} thresholding...")
                            
                            # Apply thresholding
                            binary = threshold_func(gray)
                            cv2.imwrite(f"{debug_dir}/{method_name}_{timestamp}.jpg", binary)
                            
                            # Convert back to color for detection
                            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                            
                            # Try detection on thresholded image
                            try:
                                thresh_markers_frame, thresh_corners, thresh_ids = detect_aruco_markers(
                                    binary_color,
                                    self.aruco_dict,
                                    self.aruco_params,
                                    self.camera_matrix,
                                    self.dist_coeffs
                                )
                                
                                # If found markers
                                if thresh_ids is not None and len(thresh_ids) > 0:
                                    print(f"Found {len(thresh_ids)} markers with {method_name} thresholding!")
                                    corners = thresh_corners
                                    ids = thresh_ids
                                    markers_frame = thresh_markers_frame
                                    cv2.imwrite(f"{debug_dir}/{method_name}_markers_{timestamp}.jpg", markers_frame)
                                    break  # Exit loop if found markers
                            except Exception as e:
                                print(f"Error during {method_name} detection: {e}")
                                
                            # Also try inverted version
                            binary_inv = cv2.bitwise_not(binary)
                            cv2.imwrite(f"{debug_dir}/{method_name}_inv_{timestamp}.jpg", binary_inv)
                            binary_inv_color = cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR)
                            
                            try:
                                inv_thresh_markers_frame, inv_thresh_corners, inv_thresh_ids = detect_aruco_markers(
                                    binary_inv_color,
                                    self.aruco_dict,
                                    self.aruco_params,
                                    self.camera_matrix,
                                    self.dist_coeffs
                                )
                                
                                # If found markers
                                if inv_thresh_ids is not None and len(inv_thresh_ids) > 0:
                                    print(f"Found {len(inv_thresh_ids)} markers with inverted {method_name} thresholding!")
                                    corners = inv_thresh_corners
                                    ids = inv_thresh_ids
                                    markers_frame = inv_thresh_markers_frame
                                    cv2.imwrite(f"{debug_dir}/{method_name}_inv_markers_{timestamp}.jpg", markers_frame)
                                    break  # Exit loop if found markers
                            except Exception as e:
                                print(f"Error during inverted {method_name} detection: {e}")
                                
                        except Exception as e:
                            print(f"Error applying {method_name} thresholding: {e}")
                
                # Debug detection results
                if ids is not None and len(ids) > 0:
                    print(f"Detection successful: found {len(ids)} markers: {ids.flatten()}")
                    try:
                        # Save final annotated image
                        result_path = os.path.join(debug_dir, f"final_detection_{timestamp}.jpg")
                        cv2.imwrite(result_path, markers_frame)
                        print(f"Saved final detection image to {result_path}")
                    except Exception as e:
                        print(f"Warning: Could not save final detection image: {e}")
                else:
                    print("All detection methods failed, no markers found")
                    # Save original frame for reference
                    try:
                        cv2.imwrite(f"{debug_dir}/failed_detection_{timestamp}.jpg", debug_frame)
                    except Exception as e:
                        print(f"Warning: Could not save failed detection frame: {e}")
                
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
