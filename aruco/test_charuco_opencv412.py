#!/usr/bin/env python3

"""
Test script for CharucoBoard detection with OpenCV 4.12.0-dev
This script tests different approaches to detect CharucoBoard with the new API
"""

import cv2
import numpy as np
import argparse
import os
import sys

def print_opencv_info():
    """Print OpenCV version and build information"""
    print(f"OpenCV version: {cv2.__version__}")
    print("Testing CharucoBoard detection with OpenCV 4.12.0-dev")
    
    # Check if we're using the expected version
    if not cv2.__version__.startswith("4.12"):
        print(f"WARNING: This script is designed for OpenCV 4.12.0-dev, but you're using {cv2.__version__}")

def create_test_charuco_board():
    """Create a test CharucoBoard image for detection testing"""
    print("Creating test CharucoBoard...")
    
    # Create a 6x6 dictionary with marker size 6
    dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    
    # Create CharucoBoard
    try:
        # For OpenCV 4.12+
        board = cv2.aruco.CharucoBoard(
            (6, 6),  # (squaresX, squaresY) as a tuple
            0.04,    # squareLength (in arbitrary units)
            0.03,    # markerLength (in arbitrary units)
            dictionary
        )
        print("Created CharucoBoard using CharucoBoard constructor")
    except Exception as e:
        print(f"Error creating CharucoBoard: {e}")
        return None
    
    # Generate board image
    board_size = (600, 600)
    board_img = np.zeros((board_size[1], board_size[0]), dtype=np.uint8)
    
    try:
        # Try to generate the board image
        board_img = board.generateImage(board_size, board_img, marginSize=10)
        print("Generated board image using generateImage method")
    except Exception as e:
        print(f"Error generating board image: {e}")
        # Create a blank image with text
        board_img.fill(255)
        cv2.putText(board_img, "CharucoBoard generation failed", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        print("Created fallback board with error message")
    
    # Save the board
    cv2.imwrite("test_charuco_board.png", board_img)
    print("Test CharucoBoard saved as test_charuco_board.png")
    return board_img

def test_charuco_detection(image, verbose=True):
    """Test CharucoBoard detection with different approaches"""
    if verbose:
        print("\nTesting CharucoBoard detection...")
    
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
    
    # Create CharucoBoard
    board = cv2.aruco.CharucoBoard(
        (6, 6),  # (squaresX, squaresY) as a tuple
        0.04,    # squareLength (in arbitrary units)
        0.03,    # markerLength (in arbitrary units)
        dictionary
    )
    
    # Try different detection approaches
    results = []
    
    # Approach 1: CharucoDetector
    try:
        # Create CharucoDetector
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params)
        
        # Detect the board
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
        if charuco_corners is not None and len(charuco_corners) > 0:
            if verbose:
                print(f"Approach 1 (CharucoDetector): Detected {len(charuco_corners)} corners")
                if charuco_ids is not None:
                    print(f"  Corner IDs: {charuco_ids.flatten()}")
            results.append(("CharucoDetector", charuco_corners, charuco_ids, marker_corners, marker_ids))
        else:
            if verbose:
                print("Approach 1 (CharucoDetector): No corners detected")
    except Exception as e:
        if verbose:
            print(f"Approach 1 (CharucoDetector) error: {e}")
    
    # Approach 2: Two-step process (detect markers, then interpolate)
    try:
        # First detect ArUco markers
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 2 (Two-step): Detected {len(ids)} markers")
            
            # Then interpolate CharucoBoard corners
            try:
                # Try direct interpolation
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                
                if ret and charuco_corners is not None and len(charuco_corners) > 0:
                    if verbose:
                        print(f"  Interpolated {len(charuco_corners)} corners")
                    results.append(("Two-step", charuco_corners, charuco_ids, corners, ids))
                else:
                    if verbose:
                        print("  Interpolation failed")
            except Exception as e:
                if verbose:
                    print(f"  Interpolation error: {e}")
        else:
            if verbose:
                print("Approach 2 (Two-step): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 2 (Two-step) error: {e}")
    
    # Approach 3: Custom parameters
    try:
        # Create detector with custom parameters
        custom_params = cv2.aruco.DetectorParameters()
        custom_params.adaptiveThreshConstant = 15
        custom_params.minMarkerPerimeterRate = 0.01
        custom_params.maxMarkerPerimeterRate = 10.0
        
        detector = cv2.aruco.ArucoDetector(dictionary, custom_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 3 (Custom Parameters): Detected {len(ids)} markers")
            
            # Create CharucoDetector with custom parameters
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector)
            
            # Detect the board
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 0:
                if verbose:
                    print(f"  Detected {len(charuco_corners)} corners")
                results.append(("Custom Parameters", charuco_corners, charuco_ids, marker_corners, marker_ids))
            else:
                if verbose:
                    print("  No corners detected")
        else:
            if verbose:
                print("Approach 3 (Custom Parameters): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 3 (Custom Parameters) error: {e}")
    
    return results

def visualize_results(image, results):
    """Visualize detection results"""
    print("\nVisualizing detection results...")
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Draw detected markers and corners for each approach
    for i, (approach, charuco_corners, charuco_ids, marker_corners, marker_ids) in enumerate(results):
        # Use a different color for each approach
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        color = colors[i % len(colors)]
        
        # Draw markers
        if marker_corners is not None and marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_image, marker_corners, marker_ids, color)
        
        # Draw CharucoBoard corners
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_image, charuco_corners, charuco_ids, color)
        
        # Add approach name
        cv2.putText(vis_image, approach, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save and display the result
    cv2.imwrite("charuco_detection_result.png", vis_image)
    print("Detection result saved as charuco_detection_result.png")
    
    try:
        cv2.imshow("CharucoBoard Detection Result", vis_image)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (headless mode?)")

def main():
    parser = argparse.ArgumentParser(description='Test CharucoBoard detection with OpenCV 4.12.0-dev')
    parser.add_argument('--image', type=str, help='Path to input image (if not provided, a test board will be created)')
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
        # Create a test CharucoBoard
        image = create_test_charuco_board()
        if image is None:
            print("Error: Could not create test CharucoBoard")
            sys.exit(1)
    
    # Test detection
    results = test_charuco_detection(image, args.verbose)
    
    # Visualize results if any corners were detected
    if results:
        visualize_results(image, results)
        print("\nDetection test completed successfully!")
    else:
        print("\nNo CharucoBoard corners detected with any approach.")
        print("Try adjusting detection parameters or using a different image.")

if __name__ == "__main__":
    main()