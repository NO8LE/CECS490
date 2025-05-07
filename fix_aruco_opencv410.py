#!/usr/bin/env python3

"""
Enhanced ArUco detector for OpenCV 4.10.0

This script provides a specialized implementation for detecting ArUco markers
with OpenCV 4.10.0, which has API changes compared to 4.5.5. It includes:
1. Stricter filtering to prevent false positives
2. Custom parameter configurations
3. Post-detection validation to ensure markers are valid
4. Fixed pose estimation for OpenCV 4.10.0
"""

import cv2
import numpy as np

def detect_aruco_markers(frame, aruco_dict, aruco_params, camera_matrix=None, dist_coeffs=None):
    """
    Enhanced ArUco marker detection for OpenCV 4.10.0
    
    This function handles the API changes in OpenCV 4.10.0 and adds additional
    validation to prevent false positives.
    
    Args:
        frame: Input RGB image
        aruco_dict: ArUco dictionary
        aruco_params: Detection parameters
        camera_matrix: Camera calibration matrix (optional)
        dist_coeffs: Distortion coefficients (optional)
        
    Returns:
        markers_frame: Image with markers drawn
        corners: Detected marker corners
        ids: Detected marker IDs
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Set detection parameters for OpenCV 4.10.0
    # These are critical to prevent false positives
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.minMarkerPerimeterRate = 0.05  # Increased - critical for filtering
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.03  # More accurate corner detection
    aruco_params.minCornerDistanceRate = 0.05  # Minimum distance between corners
    aruco_params.minDistanceToBorder = 3  # Minimum distance from borders
    aruco_params.minOtsuStdDev = 5.0  # Filters out low-contrast regions
    aruco_params.errorCorrectionRate = 0.8  # Higher than default (0.6)
    
    # Create detector (OpenCV 4.10.0 uses ArucoDetector)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Handle different corner formats
    if ids is not None and len(ids) > 0:
        # Ensure corners is a list of arrays with shape (1, 4, 2)
        if not isinstance(corners, list):
            corners_list = []
            for i in range(len(ids)):
                corners_list.append(corners[i].reshape(1, 4, 2))
            corners = corners_list
        
        # Filter out false positives
        valid_indices = []
        valid_corners = []
        valid_ids = []
        
        # Apply validation checks to each detected marker
        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_corners = corners[i]
            
            # 1. Verify marker has a valid perimeter (minimum size)
            perimeter = cv2.arcLength(marker_corners[0], True)
            min_perimeter = gray.shape[0] * 0.03  # At least 3% of image height
            
            # 2. Verify marker is roughly square
            width = np.linalg.norm(marker_corners[0][0] - marker_corners[0][1])
            height = np.linalg.norm(marker_corners[0][1] - marker_corners[0][2])
            if width > 0 and height > 0:
                aspect_ratio = width/height
                aspect_valid = 0.7 < aspect_ratio < 1.3  # Square should have aspect ratio near 1
            else:
                aspect_valid = False
            
            # 3. Verify corner angles (should be close to 90 degrees for square)
            angles_valid = True
            for j in range(4):
                p1 = marker_corners[0][j]
                p2 = marker_corners[0][(j+1) % 4]
                p3 = marker_corners[0][(j+2) % 4]
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle
                dot = np.sum(v1 * v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                # Avoid division by zero
                if norm1 > 0 and norm2 > 0:
                    cos_angle = dot / (norm1 * norm2)
                    # Limit to valid range due to numerical errors
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle = np.abs(np.arccos(cos_angle) * 180 / np.pi)
                    
                    # In a perfect square, opposite corners should have angle close to 90 degrees
                    if abs(angle - 90) > 25:  # Stricter angle threshold (was 35)
                        angles_valid = False
                        break
                else:
                    angles_valid = False
                    break
            
            # Add to valid markers if it passes all tests
            if perimeter >= min_perimeter and aspect_valid and angles_valid:
                valid_indices.append(i)
                valid_corners.append(corners[i])
                valid_ids.append(ids[i])
        
        # Update with validated markers only
        if len(valid_indices) > 0:
            corners = valid_corners
            ids = np.array(valid_ids)
            print(f"Detected {len(ids)} valid markers out of {len(ids)} candidates")
        else:
            corners = []
            ids = None
            print("All detected markers were filtered out as false positives")
    else:
        corners = []
        ids = None
    
    # Copy original frame for drawing
    markers_frame = frame.copy()
    
    # Draw markers if any were detected and validated
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(markers_frame, corners, ids)
        
        # Estimate pose if camera is calibrated
        if camera_matrix is not None and dist_coeffs is not None:
            # OpenCV 4.10.0 requires us to handle pose estimation differently
            # In 4.10, we can still use estimatePoseSingleMarkers but with different format
            try:
                # First try standard approach for 4.10
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    0.3048,  # 12 inches in meters (marker size)
                    camera_matrix,
                    dist_coeffs
                )
                
                # Draw axes for each marker
                for i in range(len(ids)):
                    # Draw axes for the marker
                    try:
                        # First try cv2.aruco.drawAxis
                        cv2.aruco.drawAxis(
                            markers_frame,
                            camera_matrix,
                            dist_coeffs,
                            rvecs[i],
                            tvecs[i],
                            0.1  # Axis length
                        )
                    except Exception:
                        # If that fails, try cv2.drawFrameAxes
                        cv2.drawFrameAxes(
                            markers_frame,
                            camera_matrix,
                            dist_coeffs,
                            rvecs[i],
                            tvecs[i],
                            0.1  # Axis length
                        )
                    
            except Exception as e:
                print(f"Error with estimatePoseSingleMarkers: {e}")
                print("Falling back to manual pose estimation")
                
                # Fallback to manual solvePnP approach
                rvecs = []
                tvecs = []
                
                # Process each marker individually
                marker_size = 0.3048  # 12 inches in meters
                for i in range(len(corners)):
                    # Create object points for a square marker
                    objPoints = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)
                    
                    # Get image points from corners
                    imgPoints = corners[i][0].astype(np.float32)
                    
                    # Use solvePnP to get pose
                    success, rvec, tvec = cv2.solvePnP(
                        objPoints,
                        imgPoints,
                        camera_matrix,
                        dist_coeffs
                    )
                    
                    if success:
                        rvecs.append(rvec)
                        tvecs.append(tvec)
                        
                        # Draw axis for the marker
                        try:
                            cv2.drawFrameAxes(
                                markers_frame,
                                camera_matrix,
                                dist_coeffs,
                                rvec,
                                tvec,
                                0.1  # Axis length
                            )
                        except Exception as e2:
                            print(f"Could not draw axes: {e2}")
    
    return markers_frame, corners, ids

# Example usage:
if __name__ == "__main__":
    # Load an image
    img = cv2.imread("test_image.jpg")
    
    # Create dictionary
    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    
    # Create parameters
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Detect markers
    markers_img, corners, ids = detect_aruco_markers(img, aruco_dict, aruco_params)
    
    # Display result
    cv2.imshow("Markers", markers_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
