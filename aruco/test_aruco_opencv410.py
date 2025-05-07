#!/usr/bin/env python3

"""
Test script for ArUco marker detection with OpenCV 4.10.0
This script tests different approaches to detect ArUco markers with OpenCV 4.10.0 API
"""

import cv2
import numpy as np
import argparse
import os
import sys

def print_opencv_info():
    """Print OpenCV version and build information"""
    print(f"OpenCV version: {cv2.__version__}")
    print("Testing ArUco detection with OpenCV 4.10.0")
    
    # Check if we're using the expected version
    if not cv2.__version__.startswith("4.10"):
        print(f"WARNING: This script is designed for OpenCV 4.10.0, but you're using {cv2.__version__}")

def create_test_marker():
    """Create a test ArUco marker image for detection testing"""
    print("Creating test ArUco marker...")
    
    try:
        # Try getPredefinedDictionary first (for OpenCV 4.10)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        print("Created dictionary using getPredefinedDictionary")
    except Exception as e:
        print(f"Error using getPredefinedDictionary: {e}")
        try:
            # Second try Dictionary constructor with marker size
            dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
            print("Created dictionary using Dictionary constructor with marker size")
        except Exception as e2:
            print(f"Error using Dictionary constructor: {e2}")
            try:
                # Third try Dictionary_get method
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                print("Created dictionary using Dictionary_get")
            except Exception as e3:
                print(f"Error using Dictionary_get: {e3}")
                print("Could not create ArUco dictionary with any method.")
                return None
    
    # Create a test marker (ID 0)
    marker_size = 200
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    
    try:
        # Try generateImageMarker method (OpenCV 4.10)
        marker_image = cv2.aruco.generateImageMarker(dictionary, 0, marker_size)
        print("Created marker using cv2.aruco.generateImageMarker")
    except Exception as e:
        print(f"Error using generateImageMarker: {e}")
        try:
            # Try alternative drawMarker approach 
            marker_image = cv2.aruco.drawMarker(dictionary, 0, marker_size, marker_image, 1)
            print("Created marker using cv2.aruco.drawMarker")
        except Exception as e2:
            print(f"Error using drawMarker: {e2}")
            try:
                # Try dictionary's drawMarker method if available
                marker_image = dictionary.drawMarker(0, marker_size, marker_image, 1)
                print("Created marker using dictionary.drawMarker")
            except Exception as e3:
                print(f"Error using dictionary.drawMarker: {e3}")
                # Create a blank marker with text as fallback
                marker_image.fill(255)
                cv2.putText(marker_image, "ArUco ID: 0", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
                print("Created fallback marker with text")
    
    # Save the marker
    cv2.imwrite("test_marker_410.png", marker_image)
    print("Test marker saved as test_marker_410.png")
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
    
    # Results storage
    results = []
    
    # Approach 1: Using Dictionary and ArucoDetector
    try:
        # Create dictionary with marker size parameter (4.10 style)
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
        
        # Create and use detector
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 1 (Dictionary + ArucoDetector): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("Dictionary + ArucoDetector", corners, ids))
        else:
            if verbose:
                print("Approach 1 (Dictionary + ArucoDetector): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 1 (Dictionary + ArucoDetector) error: {e}")
    
    # Approach 2: Using getPredefinedDictionary
    try:
        # Create dictionary using getPredefinedDictionary
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        # Create detector parameters
        parameters = cv2.aruco.DetectorParameters()
        
        # Create and use detector
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 2 (getPredefinedDictionary): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("getPredefinedDictionary", corners, ids))
        else:
            if verbose:
                print("Approach 2 (getPredefinedDictionary): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 2 (getPredefinedDictionary) error: {e}")
    
    # Approach 3: Using Dictionary_get (older method)
    try:
        # Create dictionary using Dictionary_get
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        
        # Create detector parameters (try different methods)
        try:
            parameters = cv2.aruco.DetectorParameters()
        except:
            try:
                parameters = cv2.aruco.DetectorParameters.create()
            except:
                parameters = cv2.aruco.DetectorParameters_create()
        
        # Try direct detectMarkers function
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 3 (Dictionary_get + detectMarkers): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("Dictionary_get", corners, ids))
        else:
            if verbose:
                print("Approach 3 (Dictionary_get + detectMarkers): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 3 (Dictionary_get + detectMarkers) error: {e}")
    
    # Approach 4: Try with relaxed parameters
    try:
        # Create dictionary using Dictionary constructor with marker size
        dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
        
        # Create detector with relaxed parameters
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 15  # More tolerance in thresholding
        parameters.minMarkerPerimeterRate = 0.01  # Detect smaller markers
        parameters.maxMarkerPerimeterRate = 10.0  # Detect larger markers
        parameters.polygonalApproxAccuracyRate = 0.1  # More tolerance in contour approximation
        
        # Create and use detector
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 4 (Relaxed Parameters): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("Relaxed Parameters", corners, ids))
        else:
            if verbose:
                print("Approach 4 (Relaxed Parameters): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 4 (Relaxed Parameters) error: {e}")
    
    # Approach 5: Try with image inversion (in case of contrast issues)
    try:
        # Invert the image
        gray_inv = cv2.bitwise_not(gray)
        
        # Create dictionary and parameters
        dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
        parameters = cv2.aruco.DetectorParameters()
        
        # Create and use detector on inverted image
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray_inv)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 5 (Inverted Image): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            results.append(("Inverted Image", corners, ids))
        else:
            if verbose:
                print("Approach 5 (Inverted Image): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 5 (Inverted Image) error: {e}")
    
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
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]
        
        # Draw markers
        cv2.aruco.drawDetectedMarkers(vis_image, corners, ids, color)
        
        # Add approach name
        cv2.putText(vis_image, approach, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save and display the result
    cv2.imwrite("detection_result_410.png", vis_image)
    print("Detection result saved as detection_result_410.png")
    
    try:
        cv2.imshow("Detection Result", vis_image)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (headless mode?)")

def main():
    parser = argparse.ArgumentParser(description='Test ArUco marker detection with OpenCV 4.10.0')
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
        if image is None:
            print("Error: Could not create test marker with any method.")
            sys.exit(1)
    
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