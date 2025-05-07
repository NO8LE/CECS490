#!/usr/bin/env python3

"""
Test script for the ArUco OpenCV 4.8.0 compatibility fix.

This script creates a synthetic test image with ArUco markers and tests
the enhanced detection algorithm to verify it correctly identifies 
markers and rejects false positives.

Usage:
  python3 test_aruco_fix.py

This will create a synthetic test image, run both the original and enhanced
detectors, and show the results side by side.
"""

import cv2
import numpy as np
import os
import sys

# Try to import the enhanced detector
try:
    from fix_aruco_opencv48 import detect_aruco_markers
    print("Enhanced ArUco detector imported successfully")
except ImportError:
    print("Error: Could not import enhanced ArUco detector.")
    print("Make sure fix_aruco_opencv48.py is in the same directory.")
    sys.exit(1)

def create_test_image(width=1280, height=720):
    """Create a synthetic test image with ArUco markers and distractors"""
    # Create a base image
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Add some background noise to make detection more challenging
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Add some random shapes as distractors
    for _ in range(20):
        # Random position and size
        x = np.random.randint(50, width-50)
        y = np.random.randint(50, height-50)
        size = np.random.randint(20, 100)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Random shape type
        shape_type = np.random.randint(0, 3)
        
        if shape_type == 0:  # Rectangle
            pt1 = (x, y)
            pt2 = (x + size, y + size)
            cv2.rectangle(img, pt1, pt2, color, 2)
        elif shape_type == 1:  # Circle
            cv2.circle(img, (x, y), size//2, color, 2)
        else:  # Triangle
            pts = np.array([[x, y-size//2], [x-size//2, y+size//2], [x+size//2, y+size//2]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
    
    # Create a dictionary for ArUco markers
    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    
    # Create and add actual ArUco markers
    marker_size = 120
    
    # Create markers with different IDs
    for i, position in enumerate([
        (width//4, height//4),          # Top-left
        (width*3//4, height//4),        # Top-right
        (width//4, height*3//4),        # Bottom-left
        (width*3//4, height*3//4),      # Bottom-right
        (width//2, height//2)           # Center
    ]):
        # Create a marker image
        marker_id = i
        marker_img = aruco_dict.drawMarker(marker_id, marker_size)
        
        # Convert to RGB
        marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
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
    
    return img

def test_original_detector(image):
    """Test the original OpenCV detector (may produce false positives)"""
    # Create a dictionary for ArUco markers
    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    
    # Create detector parameters
    params = cv2.aruco.DetectorParameters()
    
    # Create detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)
    
    # Draw results
    result_img = image.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(result_img, corners, ids)
        print(f"Original detector found {len(ids)} markers with IDs: {ids.flatten()}")
    else:
        print("Original detector found no markers")
    
    return result_img

def test_enhanced_detector(image):
    """Test our enhanced detector (should filter out false positives)"""
    # Create a dictionary for ArUco markers
    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    
    # Create detector parameters
    params = cv2.aruco.DetectorParameters()
    
    # Detect markers using our enhanced method
    result_img, corners, ids = detect_aruco_markers(image, aruco_dict, params)
    
    if ids is not None:
        print(f"Enhanced detector found {len(ids)} markers with IDs: {ids.flatten()}")
    else:
        print("Enhanced detector found no markers")
    
    return result_img

def main():
    print(f"OpenCV version: {cv2.__version__}")
    
    if not cv2.__version__.startswith("4.8"):
        print("Warning: This test is designed for OpenCV 4.8.x")
        print(f"Current version: {cv2.__version__}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Create a test image
    print("Creating test image with ArUco markers and distractors...")
    test_img = create_test_image()
    
    # Save the test image
    cv2.imwrite("aruco_test_image.jpg", test_img)
    print("Test image saved as 'aruco_test_image.jpg'")
    
    # Test original detector (may show false positives)
    print("\nTesting original OpenCV detector...")
    original_result = test_original_detector(test_img)
    
    # Test enhanced detector (should filter out false positives)
    print("\nTesting enhanced detector...")
    enhanced_result = test_enhanced_detector(test_img)
    
    # Display results side by side if not in headless mode
    if os.environ.get('DISPLAY'):
        # Create a side-by-side comparison
        h, w = test_img.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = original_result
        comparison[:, w:] = enhanced_result
        
        # Add labels
        cv2.putText(comparison, "Original Detector", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Enhanced Detector", (w+20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the comparison
        cv2.imshow("Detector Comparison", comparison)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save results
    cv2.imwrite("aruco_original_detector.jpg", original_result)
    cv2.imwrite("aruco_enhanced_detector.jpg", enhanced_result)
    print("\nResults saved as:")
    print("- aruco_original_detector.jpg")
    print("- aruco_enhanced_detector.jpg")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
