#!/usr/bin/env python3

"""
Test script for ArUco marker detection with OpenCV 4.12.0-dev
This script tests different approaches to detect ArUco markers with the new API
"""

import cv2
import numpy as np
import argparse
import os
import sys

def print_opencv_info():
    """Print OpenCV version and build information"""
    print(f"OpenCV version: {cv2.__version__}")
    print("Testing ArUco detection with OpenCV 4.12.0-dev")
    
    # Check if we're using the expected version
    if not cv2.__version__.startswith("4.12"):
        print(f"WARNING: This script is designed for OpenCV 4.12.0-dev, but you're using {cv2.__version__}")

def create_test_marker():
    """Create a test ArUco marker image for detection testing"""
    print("Creating test ArUco marker...")
    
    # Create a 6x6 dictionary with marker size 6
    dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    
    # Create a test marker (ID 0)
    marker_size = 200
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    
    try:
        # Try dictionary's drawMarker method
        marker_image = dictionary.drawMarker(0, marker_size, marker_image, 1)
        print("Created marker using dictionary.drawMarker")
    except Exception as e:
        print(f"Error using dictionary.drawMarker: {e}")
        try:
            # Try alternative approach with ArucoDetector
            detector = cv2.aruco.ArucoDetector(dictionary)
            marker_image = detector.generateImageMarker(0, marker_size, marker_image, 1)
            print("Created marker using detector.generateImageMarker")
        except Exception as e2:
            print(f"Error generating marker with ArucoDetector: {e2}")
            # Create a blank marker with text
            marker_image.fill(255)
            cv2.putText(marker_image, "ArUco ID: 0", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
            print("Created fallback marker with text")
    
    # Save the marker
    cv2.imwrite("test_marker.png", marker_image)
    print("Test marker saved as test_marker.png")
    return marker_image

def test_detection(image, verbose=True):
    """Test ArUco marker detection with different approaches"""
    if verbose:
        print("\nTesting ArUco marker detection...")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create a 6x6 dictionary with marker size 6
    dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    
    # Create detector parameters
    parameters = cv2.aruco.DetectorParameters()
    
    # Configure parameters for better detection
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
    
    # Try different detection approaches
    results = []
    
    # Approach 1: ArucoDetector
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 1 (ArucoDetector): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("ArucoDetector", corners, ids))
        else:
            if verbose:
                print("Approach 1 (ArucoDetector): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 1 (ArucoDetector) error: {e}")
    
    # Approach 2: Direct detectMarkers
    try:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 2 (detectMarkers): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("detectMarkers", corners, ids))
        else:
            if verbose:
                print("Approach 2 (detectMarkers): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 2 (detectMarkers) error: {e}")
    
    # Approach 3: Try with different dictionary
    try:
        # Try with a different dictionary type
        alt_dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50, 4)
        detector = cv2.aruco.ArucoDetector(alt_dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 3 (4x4 Dictionary): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("4x4 Dictionary", corners, ids))
        else:
            if verbose:
                print("Approach 3 (4x4 Dictionary): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 3 (4x4 Dictionary) error: {e}")
    
    # Approach 4: Try with custom dictionary
    try:
        # Create a custom dictionary
        custom_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
        # Modify parameters
        custom_params = cv2.aruco.DetectorParameters()
        custom_params.adaptiveThreshConstant = 15
        custom_params.minMarkerPerimeterRate = 0.01
        custom_params.maxMarkerPerimeterRate = 10.0
        
        detector = cv2.aruco.ArucoDetector(custom_dict, custom_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 4 (Custom Parameters): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("Custom Parameters", corners, ids))
        else:
            if verbose:
                print("Approach 4 (Custom Parameters): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 4 (Custom Parameters) error: {e}")
    
    return results

def visualize_results(image, results):
    """Visualize detection results"""
    print("\nVisualizing detection results...")
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Draw detected markers for each approach
    for i, (approach, corners, ids) in enumerate(results):
        # Use a different color for each approach
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        color = colors[i % len(colors)]
        
        # Draw markers
        cv2.aruco.drawDetectedMarkers(vis_image, corners, ids, color)
        
        # Add approach name
        cv2.putText(vis_image, approach, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save and display the result
    cv2.imwrite("detection_result.png", vis_image)
    print("Detection result saved as detection_result.png")
    
    try:
        cv2.imshow("Detection Result", vis_image)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (headless mode?)")

def main():
    parser = argparse.ArgumentParser(description='Test ArUco marker detection with OpenCV 4.12.0-dev')
    parser.add_argument('--image', type=str, help='Path to input image (if not provided, a test marker will be created)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    print_opencv_info()
    
    # Get input image
    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            sys.exit(1)
    else:
        # Create a test marker
        image = create_test_marker()
    
    # Test detection
    results = test_detection(image, args.verbose)
    
    # Visualize results if any markers were detected
    if results:
        visualize_results(image, results)
        print("\nDetection test completed successfully!")
    else:
        print("\nNo markers detected with any approach.")
        print("Try adjusting detection parameters or using a different image.")

if __name__ == "__main__":
    main()