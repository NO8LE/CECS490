#!/usr/bin/env python3

"""
Demonstration of how to use the OpenCV 4.10.0 ArUco fixes in your code
Shows migration from broken standard API to fixed implementation
"""

import cv2
import numpy as np
from opencv410_aruco_fix import (
    create_dictionary_fixed,
    generate_marker_fixed,
    create_charuco_board_fixed,
    generate_charuco_board_image_fixed,
    OpenCV410ArUcoFix
)

def demo_marker_generation():
    """Demonstrate fixed marker generation"""
    print("=== Marker Generation Demo ===")
    
    # OLD WAY (broken in OpenCV 4.10.0):
    # dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    # marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)  # FAILS!
    
    # NEW WAY (fixed):
    dictionary = create_dictionary_fixed(cv2.aruco.DICT_6X6_250)
    marker = generate_marker_fixed(dictionary, 0, 200)
    
    # Save the marker
    cv2.imwrite("demo_marker_0.png", marker)
    print("✓ Generated marker ID 0 (saved as demo_marker_0.png)")
    
    # Generate multiple markers
    for marker_id in range(1, 5):
        marker = generate_marker_fixed(dictionary, marker_id, 200)
        cv2.imwrite(f"demo_marker_{marker_id}.png", marker)
        print(f"✓ Generated marker ID {marker_id}")

def demo_charuco_board_generation():
    """Demonstrate fixed CharucoBoard generation"""
    print("\n=== CharucoBoard Generation Demo ===")
    
    # OLD WAY (broken in OpenCV 4.10.0):
    # dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
    # board = cv2.aruco.CharucoBoard((6, 8), 0.04, 0.03, dictionary)
    # board_img = board.generateImage((600, 800))  # FAILS!
    
    # NEW WAY (fixed):
    board = create_charuco_board_fixed(
        squares_x=6,
        squares_y=8,
        square_length=0.04,
        marker_length=0.03,
        dict_type=cv2.aruco.DICT_6X6_250
    )
    
    board_img = generate_charuco_board_image_fixed(board, image_size=(600, 800))
    cv2.imwrite("demo_charuco_board.png", board_img)
    print("✓ Generated CharucoBoard (saved as demo_charuco_board.png)")

def demo_marker_detection():
    """Demonstrate marker detection with fixed detector"""
    print("\n=== Marker Detection Demo ===")
    
    # Create detector with fixed dictionary
    detector, parameters = OpenCV410ArUcoFix.create_detector(cv2.aruco.DICT_6X6_250)
    
    # Create a test image with a marker
    dictionary = create_dictionary_fixed(cv2.aruco.DICT_6X6_250)
    marker = generate_marker_fixed(dictionary, 42, 200)
    
    # Place marker in a larger image
    test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    test_image[100:300, 100:300] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    
    # Detect markers
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        print(f"✓ Detected {len(ids)} marker(s): IDs = {ids.flatten()}")
        
        # Draw detected markers
        detected_image = cv2.aruco.drawDetectedMarkers(test_image.copy(), corners, ids)
        cv2.imwrite("demo_detection_result.png", detected_image)
        print("✓ Saved detection result (demo_detection_result.png)")
    else:
        print("✗ No markers detected")

def demo_complete_workflow():
    """Complete workflow example: generation to detection"""
    print("\n=== Complete Workflow Demo ===")
    
    # 1. Generate markers
    print("1. Generating markers...")
    dictionary = create_dictionary_fixed(cv2.aruco.DICT_6X6_250)
    markers = {}
    for i in range(4):
        markers[i] = generate_marker_fixed(dictionary, i, 150)
    
    # 2. Create a scene with multiple markers
    print("2. Creating test scene...")
    scene = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Place markers at different positions
    positions = [(50, 50), (600, 50), (50, 400), (600, 400)]
    for i, (x, y) in enumerate(positions):
        marker_bgr = cv2.cvtColor(markers[i], cv2.COLOR_GRAY2BGR)
        scene[y:y+150, x:x+150] = marker_bgr
    
    cv2.imwrite("demo_scene.png", scene)
    print("✓ Created scene with 4 markers")
    
    # 3. Detect markers in the scene
    print("3. Detecting markers...")
    detector, parameters = OpenCV410ArUcoFix.create_detector(cv2.aruco.DICT_6X6_250)
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray_scene)
    
    if ids is not None:
        print(f"✓ Detected {len(ids)} markers: {sorted(ids.flatten())}")
        
        # Draw detection results
        result = cv2.aruco.drawDetectedMarkers(scene.copy(), corners, ids)
        
        # Add text labels
        for i, marker_id in enumerate(ids.flatten()):
            corner = corners[i][0]
            center = corner.mean(axis=0).astype(int)
            cv2.putText(result, f"ID: {marker_id}", 
                       (center[0]-20, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite("demo_workflow_result.png", result)
        print("✓ Saved workflow result (demo_workflow_result.png)")
    else:
        print("✗ No markers detected")

def demo_migration_guide():
    """Show side-by-side comparison of old vs new code"""
    print("\n=== Migration Guide ===")
    print("Replace your broken OpenCV 4.10.0 ArUco code as follows:\n")
    
    print("1. Dictionary Creation:")
    print("   OLD: dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)")
    print("   NEW: dictionary = create_dictionary_fixed(cv2.aruco.DICT_6X6_250)")
    
    print("\n2. Marker Generation:")
    print("   OLD: marker = cv2.aruco.generateImageMarker(dictionary, 0, 200)")
    print("   NEW: marker = generate_marker_fixed(dictionary, 0, 200)")
    
    print("\n3. CharucoBoard Creation:")
    print("   OLD: board = cv2.aruco.CharucoBoard((6, 8), 0.04, 0.03, dictionary)")
    print("   NEW: board = create_charuco_board_fixed(6, 8, 0.04, 0.03, cv2.aruco.DICT_6X6_250)")
    
    print("\n4. CharucoBoard Image:")
    print("   OLD: board_img = board.generateImage((600, 800))")
    print("   NEW: board_img = generate_charuco_board_image_fixed(board, (600, 800))")
    
    print("\n5. Detector Creation:")
    print("   OLD: params = cv2.aruco.DetectorParameters()")
    print("        detector = cv2.aruco.ArucoDetector(dictionary, params)")
    print("   NEW: detector, params = OpenCV410ArUcoFix.create_detector(cv2.aruco.DICT_6X6_250)")

if __name__ == "__main__":
    print("OpenCV 4.10.0 ArUco Fix Demonstration")
    print(f"OpenCV version: {cv2.__version__}")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_marker_generation()
        demo_charuco_board_generation()
        demo_marker_detection()
        demo_complete_workflow()
        demo_migration_guide()
        
        print("\n" + "=" * 50)
        print("✓ All demos completed successfully!")
        print("Check the generated image files to see the results.")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
