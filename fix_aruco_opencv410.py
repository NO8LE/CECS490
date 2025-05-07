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
import os
import sys

# Global variable to track which dictionary method was used for marker generation
generation_dict_method = None

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
    # Adjusted for better detection of markers on CharUco boards
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.minMarkerPerimeterRate = 0.03  # Reduced to detect smaller markers
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.05  # Increased for better detection
    aruco_params.minCornerDistanceRate = 0.05
    aruco_params.minDistanceToBorder = 3
    aruco_params.minOtsuStdDev = 5.0
    aruco_params.errorCorrectionRate = 0.6  # Default value
    
    # Additional parameters for better detection
    if hasattr(aruco_params, 'cornerRefinementMethod'):
        aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
    if hasattr(aruco_params, 'cornerRefinementWinSize'):
        aruco_params.cornerRefinementWinSize = 5
    if hasattr(aruco_params, 'cornerRefinementMaxIterations'):
        aruco_params.cornerRefinementMaxIterations = 30
    
    # Create detector (OpenCV 4.10.0 uses ArucoDetector)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Print initial detection results
    if ids is not None:
        print(f"Initial detection found {len(ids)} markers with IDs: {ids.flatten()}")
    else:
        print("Initial detection found no markers")
    
    # Handle different corner formats
    if ids is not None and len(ids) > 0:
        # Ensure corners is a list of arrays with shape (1, 4, 2)
        if not isinstance(corners, list):
            corners_list = []
            for i in range(len(ids)):
                corners_list.append(corners[i].reshape(1, 4, 2))
            corners = corners_list
        
        # Store original count for logging
        original_count = len(ids)
        
        # Filter out false positives with more relaxed criteria
        valid_indices = []
        valid_corners = []
        valid_ids = []
        
        # Apply validation checks to each detected marker
        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_corners = corners[i]
            
            # Print raw detection info for debugging
            print(f"Raw detection - Marker {i}: ID={marker_id}, Corners shape={marker_corners.shape}")
            
            # 1. Verify marker has a valid perimeter (minimum size)
            perimeter = cv2.arcLength(marker_corners[0], True)
            min_perimeter = gray.shape[0] * 0.01  # Reduced to 1% of image height
            perimeter_valid = perimeter >= min_perimeter
            
            # 2. Verify marker is roughly square (with more tolerance)
            width = np.linalg.norm(marker_corners[0][0] - marker_corners[0][1])
            height = np.linalg.norm(marker_corners[0][1] - marker_corners[0][2])
            if width > 0 and height > 0:
                aspect_ratio = width/height
                aspect_valid = 0.5 < aspect_ratio < 2.0  # More tolerant aspect ratio
            else:
                aspect_valid = False
                
            # Print validation details for debugging
            print(f"  Validation - Perimeter: {perimeter:.2f} (min: {min_perimeter:.2f}, valid: {perimeter_valid})")
            print(f"  Validation - Aspect ratio: {aspect_ratio:.2f} (valid: {aspect_valid})")
            
            # 3. Verify corner angles with more tolerance
            angles_valid = True
            angles = []
            
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
                    angles.append(angle)
                    
                    # More tolerant angle threshold
                    if abs(angle - 90) > 40:  # Increased from 25 to 40
                        angles_valid = False
                        break
                else:
                    angles_valid = False
                    angles.append(0)  # Invalid angle
                    break
            
            # Print angle validation details
            print(f"  Validation - Angles: {[f'{a:.1f}' for a in angles]} (valid: {angles_valid})")
            
            # Add to valid markers if it passes all tests
            validation_passed = perimeter_valid and aspect_valid and angles_valid
            print(f"  Overall validation result: {validation_passed}")
            
            if validation_passed:
                valid_indices.append(i)
                valid_corners.append(corners[i])
                valid_ids.append(ids[i])
                print(f"  Marker {i} with ID {marker_id} passed validation")
            else:
                print(f"  Marker {i} with ID {marker_id} failed validation")
        
        # Update with validated markers only
        if len(valid_indices) > 0:
            corners = valid_corners
            ids = np.array(valid_ids)
            # Only print if there are markers to report
            if len(ids) > 0:
                print(f"Detected {len(ids)} valid markers with IDs: {ids.flatten()}")
        else:
            corners = []
            ids = None
            # Only print this message if there were actually candidates detected
            if original_count > 0:
                print(f"No valid markers found among {original_count} candidates")
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

def create_test_image(width=640, height=480):
    """Create a synthetic test image with ArUco markers"""
    # Use the global variable to track which dictionary method was used
    global generation_dict_method
    
    # Create a base image
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Add some background noise to make detection more challenging
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Create markers with different IDs and positions
    marker_size = 100
    
    # Create markers with different IDs and positions
    for i, position in enumerate([
        (width//4, height//4),          # Top-left
        (width*3//4, height//4),        # Top-right
        (width//2, height//2),          # Center
    ]):
        # Create a marker image
        marker_id = i
        
        # Create a blank marker image
        marker_img = None
        success = False
        
        # Method 1: Using getPredefinedDictionary and generateImageMarker (OpenCV 4.10+)
        try:
            # Get a predefined dictionary - store this for later use in detection
            generation_dict_method = "getPredefinedDictionary"
            
            # Get a predefined dictionary
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            # Generate marker image directly with the function
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
            success = True
            print(f"Created marker ID {marker_id} using getPredefinedDictionary and generateImageMarker")
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: Alternative dictionary creation with Dictionary_get
        if not success:
            try:
                # Try using Dictionary_get
                generation_dict_method = "Dictionary_get"
                
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
                marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
                success = True
                print(f"Created marker ID {marker_id} using Dictionary_get and generateImageMarker")
            except Exception as e:
                print(f"Method 2 failed: {e}")
        
        # Method 3: Try with our original Dictionary constructor
        if not success:
            try:
                generation_dict_method = "Dictionary"
                
                aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
                marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
                success = True
                print(f"Created marker ID {marker_id} using Dictionary constructor and generateImageMarker")
            except Exception as e:
                print(f"Method 3 failed: {e}")
        
        # Method 4: Try creating a marker manually based on the ArUco pattern
        if not success:
            try:
                generation_dict_method = "manual"
                
                # Create a blank white image
                marker_img = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
                
                # Create a simple ArUco-like pattern
                # Black border
                border = int(marker_size * 0.1)
                marker_img[border:marker_size-border, border:marker_size-border] = 0
                
                # White inner area
                inner_border = int(marker_size * 0.2)
                marker_img[inner_border:marker_size-inner_border, inner_border:marker_size-inner_border] = 255
                
                # Add a pattern based on marker_id (simple binary pattern)
                cell_size = int((marker_size - 2*inner_border) / 4)
                for bit_pos in range(8):  # Use 8 bits for the pattern
                    bit_value = (marker_id >> bit_pos) & 1
                    row = 2 + (bit_pos // 4)
                    col = 1 + (bit_pos % 4)
                    if bit_value:
                        r1 = inner_border + (row-1)*cell_size
                        c1 = inner_border + (col-1)*cell_size
                        r2 = r1 + cell_size
                        c2 = c1 + cell_size
                        marker_img[r1:r2, c1:c2] = 0
                
                success = True
                print(f"Created marker ID {marker_id} using manual ArUco-like pattern")
            except Exception as e:
                print(f"Method 4 failed: {e}")
        
        # Fallback method: Create a simple visual marker with text
        if not success or marker_img is None:
            generation_dict_method = "text_fallback"
            
            print(f"All methods failed for marker ID {marker_id}, using text fallback")
            marker_img = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
            # Draw a black square border
            cv2.rectangle(marker_img, (10, 10), (marker_size-10, marker_size-10), 0, 2)
            # Draw marker ID
            cv2.putText(
                marker_img,
                f"ID: {marker_id}",
                (marker_size//4, marker_size//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                0,  # Black text
                2
            )
        
        # Ensure marker_img is valid
        if marker_img is None:
            print(f"Warning: Failed to create marker ID {marker_id}")
            continue
            
        # Make sure marker_img is grayscale
        if len(marker_img.shape) == 3:
            marker_gray = cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY)
        else:
            marker_gray = marker_img
            
        # Convert to RGB for placing in the color image
        marker_rgb = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
        
        # Calculate position to place marker
        x, y = position
        x = x - marker_size // 2
        y = y - marker_size // 2
        
        # Make sure the marker fits in the image
        if (x >= 0 and y >= 0 and
            x + marker_size < width and
            y + marker_size < height):
            
            # Place marker in the image
            img[y:y+marker_size, x:x+marker_size] = marker_rgb
            print(f"Placed marker ID {marker_id} at position ({x}, {y})")
        else:
            print(f"Warning: Marker ID {marker_id} position ({x}, {y}) is out of bounds")
    
    return img

# Example usage:
if __name__ == "__main__":
    print(f"OpenCV version: {cv2.__version__}")
    
    if not cv2.__version__.startswith("4.10"):
        print("Warning: This script is designed for OpenCV 4.10.x")
        print(f"Current version: {cv2.__version__}")
        print("It may still work, but results might vary.")
    
    # Check if test image exists, if not create a synthetic one
    test_image_path = "test_image.jpg"
    img = None  # Initialize img to None
    
    if os.path.exists(test_image_path):
        print(f"Loading test image from {test_image_path}")
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"Error loading {test_image_path}, creating synthetic test image instead")
            img = create_test_image()
            # Verify the synthetic image was created
            if img is not None:
                print("Successfully created synthetic test image")
    else:
        print(f"Test image {test_image_path} not found, creating synthetic test image")
        img = create_test_image()
        # Verify the synthetic image was created
        if img is not None:
            print("Successfully created synthetic test image")
            # Save the test image for future use
            try:
                cv2.imwrite(test_image_path, img)
                print(f"Saved synthetic test image to {test_image_path}")
            except Exception as e:
                print(f"Warning: Could not save synthetic test image: {e}")
    
    # Double-check that we have a valid image before proceeding
    if img is None:
        print("Error: Could not create or load test image")
        sys.exit(1)
        
    # Verify image dimensions
    print(f"Image dimensions: {img.shape}")
    
    # Create dictionary - use the same method that was successful for marker generation
    try:
        # Check if we have a global variable indicating which method worked for generation
        if 'generation_dict_method' in globals() and generation_dict_method == "getPredefinedDictionary":
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            print("Using getPredefinedDictionary for detection (same as generation)")
        else:
            # Fallback to Dictionary constructor
            aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
            print("Using Dictionary constructor for detection")
    except Exception as e:
        print(f"Error creating dictionary: {e}")
        # Fallback to Dictionary constructor
        aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
        print("Using Dictionary constructor for detection (fallback)")
    
    # Create parameters
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Detect markers
    print("Running ArUco detection with enhanced detector...")
    markers_img, corners, ids = detect_aruco_markers(img, aruco_dict, aruco_params)
    
    if ids is not None:
        print(f"Detected {len(ids)} markers with IDs: {ids.flatten()}")
    else:
        print("No markers detected")
    
    # Display result if not in headless mode
    if os.environ.get('DISPLAY'):
        cv2.imshow("Enhanced ArUco Detector", markers_img)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save the result
    result_path = "aruco_detection_result.jpg"
    cv2.imwrite(result_path, markers_img)
    print(f"Detection result saved to {result_path}")
