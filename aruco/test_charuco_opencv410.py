#!/usr/bin/env python3

"""
Test script for CharucoBoard detection with OpenCV 4.10.0
This script tests different approaches to detect CharucoBoard with OpenCV 4.10.0 API
"""

import cv2
import numpy as np
import argparse
import os
import sys

def print_opencv_info():
    """Print OpenCV version and build information"""
    print(f"OpenCV version: {cv2.__version__}")
    print("Testing CharucoBoard detection with OpenCV 4.10.0")
    
    # Check if we're using the expected version
    if not cv2.__version__.startswith("4.10"):
        print(f"WARNING: This script is designed for OpenCV 4.10.0, but you're using {cv2.__version__}")

def create_test_charuco_board():
    """Create a test CharucoBoard image for detection testing"""
    print("Creating test CharucoBoard...")
    
    # Create dictionary - try multiple methods for OpenCV 4.10 compatibility
    try:
        # Try getPredefinedDictionary first (may work in 4.10)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        print("Created dictionary using getPredefinedDictionary")
    except Exception as e:
        print(f"Error using getPredefinedDictionary: {e}")
        try:
            # Try Dictionary constructor with marker size
            dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
            print("Created dictionary using Dictionary constructor with marker size")
        except Exception as e2:
            print(f"Error using Dictionary constructor: {e2}")
            try:
                # Last resort: Try Dictionary_get
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                print("Created dictionary using Dictionary_get")
            except Exception as e3:
                print(f"Error using Dictionary_get: {e3}")
                print("Could not create ArUco dictionary with any method.")
                return None
    
    # Create CharucoBoard - try multiple methods for OpenCV 4.10 compatibility
    board = None
    
    # Method 1: Try CharucoBoard constructor first
    try:
        board = cv2.aruco.CharucoBoard(
            (6, 6),  # (squaresX, squaresY) as a tuple
            0.04,    # squareLength (in arbitrary units)
            0.03,    # markerLength (in arbitrary units)
            dictionary
        )
        print("Created CharucoBoard using CharucoBoard constructor")
    except Exception as e:
        print(f"Error creating CharucoBoard with constructor: {e}")
        
        # Method 2: Try CharucoBoard.create
        try:
            board = cv2.aruco.CharucoBoard.create(
                squaresX=6,
                squaresY=6,
                squareLength=0.04,
                markerLength=0.03,
                dictionary=dictionary
            )
            print("Created CharucoBoard using CharucoBoard.create")
        except Exception as e2:
            print(f"Error using CharucoBoard.create: {e2}")
            
            # Method 3: Try CharucoBoard_create
            try:
                board = cv2.aruco.CharucoBoard_create(
                    squaresX=6,
                    squaresY=6,
                    squareLength=0.04,
                    markerLength=0.03,
                    dictionary=dictionary
                )
                print("Created CharucoBoard using CharucoBoard_create")
            except Exception as e3:
                print(f"Error using CharucoBoard_create: {e3}")
                print("Could not create CharucoBoard with any method.")
                return None
    
    # Generate board image
    board_size = (600, 600)
    board_img = np.zeros((board_size[1], board_size[0]), dtype=np.uint8)
    
    # Try multiple methods to generate the board image
    try:
        # Method 1: Try draw method
        board_img = board.draw(board_size)
        print("Generated board image using draw method")
    except Exception as e:
        print(f"Error using board.draw: {e}")
        try:
            # Method 2: Try generateImage method
            board_img = board.generateImage(board_size, board_img, marginSize=10)
            print("Generated board image using generateImage method")
        except Exception as e2:
            print(f"Error using board.generateImage: {e2}")
            try:
                # Method 3: Try drawBoard method if available
                board_img = cv2.aruco.drawCharucoBoard(board, board_size, board_img, marginSize=10)
                print("Generated board image using cv2.aruco.drawCharucoBoard")
            except Exception as e3:
                print(f"Error using drawCharucoBoard: {e3}")
                # Create a blank image with text as fallback
                board_img.fill(255)
                cv2.putText(board_img, "CharucoBoard generation failed", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
                print("Created fallback board with error message")
    
    # Save the board
    cv2.imwrite("test_charuco_board_410.png", board_img)
    print("Test CharucoBoard saved as test_charuco_board_410.png")
    return board_img, board

def test_charuco_detection(image, board=None, verbose=True):
    """Test CharucoBoard detection with different approaches"""
    if verbose:
        print("\nTesting CharucoBoard detection...")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create dictionary - try multiple methods for OpenCV 4.10
    dictionary = None
    try:
        # First try Dictionary with marker size
        dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    except:
        try:
            # Then try getPredefinedDictionary
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        except:
            try:
                # Last resort: Dictionary_get
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            except:
                print("Could not create dictionary with any method.")
                return []
    
    # Create detection parameters
    parameters = None
    try:
        parameters = cv2.aruco.DetectorParameters()
    except:
        try:
            parameters = cv2.aruco.DetectorParameters.create()
        except:
            try:
                parameters = cv2.aruco.DetectorParameters_create()
            except:
                print("Could not create detector parameters with any method.")
                return []
    
    # Configure parameters for better detection
    try:
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.adaptiveThreshConstant = 7
        parameters.minMarkerPerimeterRate = 0.03
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.polygonalApproxAccuracyRate = 0.05
        parameters.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
    except:
        print("Warning: Failed to configure some detector parameters")
    
    # Create or use CharucoBoard
    if board is None:
        try:
            # Try to create a board
            board = cv2.aruco.CharucoBoard(
                (6, 6),  # (squaresX, squaresY)
                0.04,    # squareLength
                0.03,    # markerLength
                dictionary
            )
        except Exception as e:
            try:
                # Try alternative method
                board = cv2.aruco.CharucoBoard.create(
                    squaresX=6,
                    squaresY=6,
                    squareLength=0.04,
                    markerLength=0.03,
                    dictionary=dictionary
                )
            except Exception as e2:
                try:
                    # Last resort
                    board = cv2.aruco.CharucoBoard_create(
                        squaresX=6,
                        squaresY=6,
                        squareLength=0.04,
                        markerLength=0.03,
                        dictionary=dictionary
                    )
                except Exception as e3:
                    print("Error: Could not create CharucoBoard. Cannot proceed with detection.")
                    return []
    
    # Try different detection approaches
    results = []
    
    # Approach 1: Two-step process (detect markers, then interpolate) - most reliable for 4.10
    try:
        # First detect ArUco markers
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 1 (Two-step): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            
            # Then interpolate CharucoBoard corners
            try:
                # Try interpolateCornersCharuco
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                
                if ret and charuco_corners is not None and len(charuco_corners) > 0:
                    if verbose:
                        print(f"  Interpolated {len(charuco_corners)} corners")
                        if charuco_ids is not None:
                            print(f"  Corner IDs: {charuco_ids.flatten()}")
                    results.append(("Two-step", charuco_corners, charuco_ids, corners, ids))
                else:
                    if verbose:
                        print("  Interpolation failed (no corners)")
            except Exception as e:
                if verbose:
                    print(f"  Interpolation error: {e}")
        else:
            if verbose:
                print("Approach 1 (Two-step): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 1 (Two-step) error: {e}")
    
    # Approach 2: Using relaxed parameters
    try:
        # Create detector with relaxed parameters
        custom_params = cv2.aruco.DetectorParameters()
        custom_params.adaptiveThreshConstant = 15
        custom_params.minMarkerPerimeterRate = 0.01
        custom_params.maxMarkerPerimeterRate = 10.0
        custom_params.cornerRefinementMethod = 1
        
        detector = cv2.aruco.ArucoDetector(dictionary, custom_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 2 (Relaxed Parameters): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            
            # Then interpolate CharucoBoard corners
            try:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                
                if ret and charuco_corners is not None and len(charuco_corners) > 0:
                    if verbose:
                        print(f"  Interpolated {len(charuco_corners)} corners")
                    results.append(("Relaxed Parameters", charuco_corners, charuco_ids, corners, ids))
                else:
                    if verbose:
                        print("  Interpolation failed")
            except Exception as e:
                if verbose:
                    print(f"  Interpolation error: {e}")
        else:
            if verbose:
                print("Approach 2 (Relaxed Parameters): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 2 (Relaxed Parameters) error: {e}")
    
    # Approach 3: Try with image inversion (in case of contrast issues)
    try:
        # Invert the image
        gray_inv = cv2.bitwise_not(gray)
        
        # Use standard detector
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray_inv)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 3 (Inverted Image): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            
            # Then interpolate CharucoBoard corners on inverted image
            try:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray_inv, board
                )
                
                if ret and charuco_corners is not None and len(charuco_corners) > 0:
                    if verbose:
                        print(f"  Interpolated {len(charuco_corners)} corners")
                    results.append(("Inverted Image", charuco_corners, charuco_ids, corners, ids))
                else:
                    if verbose:
                        print("  Interpolation failed")
            except Exception as e:
                if verbose:
                    print(f"  Interpolation error: {e}")
        else:
            if verbose:
                print("Approach 3 (Inverted Image): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 3 (Inverted Image) error: {e}")
    
    # Approach 4: Try with Dictionary_get and Adaptive Thresholding
    try:
        # Create dictionary using Dictionary_get method (older API)
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        
        # Apply adaptive thresholding to the image
        threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Try direct detectMarkers function
        corners, ids, rejected = cv2.aruco.detectMarkers(threshold, dictionary, parameters=parameters)
        
        if ids is not None and len(ids) > 0:
            if verbose:
                print(f"Approach 4 (Dictionary_get + Thresholding): Detected {len(ids)} markers")
                print(f"  IDs: {ids.flatten()}")
            
            # Then interpolate CharucoBoard corners
            try:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, board
                )
                
                if ret and charuco_corners is not None and len(charuco_corners) > 0:
                    if verbose:
                        print(f"  Interpolated {len(charuco_corners)} corners")
                    results.append(("Dictionary_get + Thresholding", charuco_corners, charuco_ids, corners, ids))
                else:
                    if verbose:
                        print("  Interpolation failed")
            except Exception as e:
                if verbose:
                    print(f"  Interpolation error: {e}")
        else:
            if verbose:
                print("Approach 4 (Dictionary_get + Thresholding): No markers detected")
    except Exception as e:
        if verbose:
            print(f"Approach 4 (Dictionary_get + Thresholding) error: {e}")
    
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
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]
        
        # Draw markers
        if marker_corners is not None and marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_image, marker_corners, marker_ids, color)
        
        # Draw CharucoBoard corners
        if charuco_corners is not None and charuco_ids is not None:
            try:
                # Try direct function
                cv2.aruco.drawDetectedCornersCharuco(vis_image, charuco_corners, charuco_ids, color)
            except Exception as e:
                print(f"Error drawing CharucoBoard corners: {e}")
                # Fallback: Draw circles at corner positions
                if charuco_corners is not None:
                    for corner in charuco_corners:
                        cv2.circle(vis_image, tuple(corner[0].astype(int)), 5, color, 2)
        
        # Add approach name
        cv2.putText(vis_image, approach, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save and display the result
    cv2.imwrite("charuco_detection_result_410.png", vis_image)
    print("Detection result saved as charuco_detection_result_410.png")
    
    try:
        cv2.imshow("CharucoBoard Detection Result", vis_image)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (headless mode?)")

def main():
    parser = argparse.ArgumentParser(description='Test CharucoBoard detection with OpenCV 4.10.0')
    parser.add_argument('--image', type=str, help='Path to input image (if not provided, a test board will be created)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    print_opencv_info()
    
    # Get input image and board
    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            sys.exit(1)
        board = None  # Will create during detection
    else:
        # Create a test CharucoBoard
        image, board = create_test_charuco_board()
        if image is None:
            print("Error: Could not create test CharucoBoard")
            sys.exit(1)
    
    # Test detection
    results = test_charuco_detection(image, board, args.verbose)
    
    # Visualize results if any corners were detected
    if results:
        visualize_results(image, results)
        print("\nDetection test completed successfully!")
    else:
        print("\nNo CharucoBoard corners detected with any approach.")
        print("Try adjusting detection parameters or using a different image.")

if __name__ == "__main__":
    main()