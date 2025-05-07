#!/usr/bin/env python3

import cv2
import numpy as np
import sys

# Print OpenCV version
print(f"OpenCV version: {cv2.__version__}")

# Try to access ArUco module
try:
    # Check if aruco is directly accessible
    if hasattr(cv2, 'aruco'):
        print("ArUco module is available via cv2.aruco")
        aruco = cv2.aruco
    else:
        # Try to import as a separate module
        try:
            from cv2 import aruco
            print("ArUco module is available via 'from cv2 import aruco'")
            # Make it available as cv2.aruco for consistency
            cv2.aruco = aruco
        except ImportError:
            print("Error: OpenCV ArUco module not found.")
            sys.exit(1)
except Exception as e:
    print(f"Error accessing ArUco module: {str(e)}")
    sys.exit(1)

# Try to create an ArUco dictionary
print("\nTesting ArUco dictionary creation:")

# First method (old API)
try:
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
    print("Successfully created dictionary using cv2.aruco.Dictionary_get()")
except Exception as e:
    print(f"Failed with cv2.aruco.Dictionary_get(): {str(e)}")

# Second method (newer API)
try:
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.Dictionary.get(ARUCO_DICT_TYPE)
    print("Successfully created dictionary using cv2.aruco.Dictionary.get()")
except Exception as e:
    print(f"Failed with cv2.aruco.Dictionary.get(): {str(e)}")

# Third method (OpenCV 4.x approach)
try:
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.Dictionary.create(ARUCO_DICT_TYPE)
    print("Successfully created dictionary using cv2.aruco.Dictionary.create()")
except Exception as e:
    print(f"Failed with cv2.aruco.Dictionary.create(): {str(e)}")

# Fourth method (constructor approach)
try:
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE)
    print("Successfully created dictionary using cv2.aruco.Dictionary()")
except Exception as e:
    print(f"Failed with cv2.aruco.Dictionary(): {str(e)}")

# Fifth method (with marker size parameter)
try:
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE, 6)
    print("Successfully created dictionary using cv2.aruco.Dictionary() with marker size")
except Exception as e:
    print(f"Failed with cv2.aruco.Dictionary() with marker size: {str(e)}")

# Test detector parameters creation
print("\nTesting detector parameters creation:")

# First method (old API)
try:
    detector_params = cv2.aruco.DetectorParameters_create()
    print("Successfully created detector parameters using cv2.aruco.DetectorParameters_create()")
except Exception as e:
    print(f"Failed with cv2.aruco.DetectorParameters_create(): {str(e)}")

# Second method (newer API)
try:
    detector_params = cv2.aruco.DetectorParameters.create()
    print("Successfully created detector parameters using cv2.aruco.DetectorParameters.create()")
except Exception as e:
    print(f"Failed with cv2.aruco.DetectorParameters.create(): {str(e)}")

# Third method (constructor approach)
try:
    detector_params = cv2.aruco.DetectorParameters()
    print("Successfully created detector parameters using cv2.aruco.DetectorParameters()")
except Exception as e:
    print(f"Failed with cv2.aruco.DetectorParameters(): {str(e)}")

# Test detector creation
print("\nTesting detector creation:")
try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    print("Successfully created ArucoDetector")
except Exception as e:
    print(f"Failed to create ArucoDetector: {str(e)}")

# Test marker detection methods
print("\nTesting marker detection methods:")

# Create a blank image for testing
img = np.zeros((400, 400), dtype=np.uint8)

# Try newer API (ArucoDetector class)
try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, rejected = detector.detectMarkers(img)
    print("Successfully used detector.detectMarkers() method")
except Exception as e:
    print(f"Failed with detector.detectMarkers(): {str(e)}")

# Try older API (function-based)
try:
    corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=detector_params)
    print("Successfully used cv2.aruco.detectMarkers() function")
except Exception as e:
    print(f"Failed with cv2.aruco.detectMarkers(): {str(e)}")

# Test pose estimation
print("\nTesting pose estimation methods:")

# Create dummy data for testing
dummy_corners = [np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]], dtype=np.float32)]
dummy_ids = np.array([[0]])
camera_matrix = np.array([[800, 0, 200], [0, 800, 200], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((1, 5), dtype=np.float32)

# Test estimatePoseSingleMarkers
try:
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        dummy_corners, 0.05, camera_matrix, dist_coeffs
    )
    print("Successfully used cv2.aruco.estimatePoseSingleMarkers()")
except Exception as e:
    print(f"Failed with cv2.aruco.estimatePoseSingleMarkers(): {str(e)}")

# Test solvePnP as alternative
try:
    objPoints = np.array([
        [-0.025, 0.025, 0],
        [0.025, 0.025, 0],
        [0.025, -0.025, 0],
        [-0.025, -0.025, 0]
    ], dtype=np.float32)
    
    imgPoints = dummy_corners[0][0].astype(np.float32)
    
    success, rvec, tvec = cv2.solvePnP(
        objPoints, imgPoints, camera_matrix, dist_coeffs
    )
    print("Successfully used cv2.solvePnP() as alternative")
except Exception as e:
    print(f"Failed with cv2.solvePnP(): {str(e)}")

# Test axis drawing
print("\nTesting axis drawing methods:")

# Try drawing axis with aruco.drawAxis
try:
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    dummy_rvec = np.zeros((3, 1), dtype=np.float32)
    dummy_tvec = np.array([[0], [0], [1]], dtype=np.float32)
    
    cv2.aruco.drawAxis(
        dummy_img, camera_matrix, dist_coeffs, dummy_rvec, dummy_tvec, 0.1
    )
    print("Successfully used cv2.aruco.drawAxis()")
except Exception as e:
    print(f"Failed with cv2.aruco.drawAxis(): {str(e)}")

# Try drawing axis with cv2.drawFrameAxes
try:
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.drawFrameAxes(
        dummy_img, camera_matrix, dist_coeffs, dummy_rvec, dummy_tvec, 0.1
    )
    print("Successfully used cv2.drawFrameAxes()")
except Exception as e:
    print(f"Failed with cv2.drawFrameAxes(): {str(e)}")

print("\nArUco module testing completed")
