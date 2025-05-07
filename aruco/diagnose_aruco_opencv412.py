#!/usr/bin/env python3

"""
Diagnostic script for ArUco module in OpenCV 4.12.0-dev
This script tests the most basic ArUco operations with detailed error reporting
"""

import cv2
import numpy as np
import sys

def print_separator():
    print("\n" + "="*50 + "\n")

def print_opencv_info():
    """Print detailed OpenCV version and build information"""
    print("OPENCV INFORMATION:")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Print ArUco-related attributes
    print("\nArUco module attributes:")
    aruco_attrs = [attr for attr in dir(cv2.aruco) if not attr.startswith('_')]
    for attr in sorted(aruco_attrs):
        try:
            value = getattr(cv2.aruco, attr)
            if isinstance(value, int):
                print(f"  {attr} = {value}")
            else:
                print(f"  {attr} = {type(value).__name__}")
        except Exception as e:
            print(f"  {attr} = Error: {e}")

def test_dictionary_creation():
    """Test different ways to create an ArUco dictionary"""
    print_separator()
    print("TESTING DICTIONARY CREATION:")
    
    # Test all dictionary types
    dict_types = [
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50, 4),
        ("DICT_4X4_100", cv2.aruco.DICT_4X4_100, 4),
        ("DICT_4X4_250", cv2.aruco.DICT_4X4_250, 4),
        ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000, 4),
        ("DICT_5X5_50", cv2.aruco.DICT_5X5_50, 5),
        ("DICT_5X5_100", cv2.aruco.DICT_5X5_100, 5),
        ("DICT_5X5_250", cv2.aruco.DICT_5X5_250, 5),
        ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000, 5),
        ("DICT_6X6_50", cv2.aruco.DICT_6X6_50, 6),
        ("DICT_6X6_100", cv2.aruco.DICT_6X6_100, 6),
        ("DICT_6X6_250", cv2.aruco.DICT_6X6_250, 6),
        ("DICT_6X6_1000", cv2.aruco.DICT_6X6_1000, 6),
        ("DICT_7X7_50", cv2.aruco.DICT_7X7_50, 7),
        ("DICT_7X7_100", cv2.aruco.DICT_7X7_100, 7),
        ("DICT_7X7_250", cv2.aruco.DICT_7X7_250, 7),
        ("DICT_7X7_1000", cv2.aruco.DICT_7X7_1000, 7),
        ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL, 6)
    ]
    
    successful_dicts = []
    
    for dict_name, dict_type, marker_size in dict_types:
        print(f"\nTesting {dict_name} (marker size: {marker_size}):")
        
        # Try with different marker sizes
        for test_size in [marker_size, marker_size-1, marker_size+1]:
            try:
                dictionary = cv2.aruco.Dictionary(dict_type, test_size)
                print(f"  ✓ Created with marker size {test_size}")
                
                # Try to get dictionary properties
                try:
                    marker_size_prop = dictionary.getMarkerSize()
                    print(f"  ✓ Marker size property: {marker_size_prop}")
                except Exception as e:
                    print(f"  ✗ Cannot get marker size: {e}")
                
                try:
                    max_correction_bits = dictionary.getMaxCorrectionBits()
                    print(f"  ✓ Max correction bits: {max_correction_bits}")
                except Exception as e:
                    print(f"  ✗ Cannot get max correction bits: {e}")
                
                # If we got here, this dictionary works
                if test_size == marker_size:  # Only add the correct size to successful dicts
                    successful_dicts.append((dict_name, dict_type, marker_size))
            except Exception as e:
                print(f"  ✗ Failed with marker size {test_size}: {e}")
    
    print("\nSUCCESSFUL DICTIONARIES:")
    if successful_dicts:
        for dict_name, dict_type, marker_size in successful_dicts:
            print(f"  {dict_name} (marker size: {marker_size})")
        return successful_dicts
    else:
        print("  None")
        return []

def test_marker_generation(successful_dicts):
    """Test marker generation with successful dictionaries"""
    print_separator()
    print("TESTING MARKER GENERATION:")
    
    if not successful_dicts:
        print("No successful dictionaries to test marker generation.")
        return []
    
    successful_markers = []
    
    for dict_name, dict_type, marker_size in successful_dicts:
        print(f"\nTesting marker generation with {dict_name}:")
        
        try:
            # Create dictionary
            dictionary = cv2.aruco.Dictionary(dict_type, marker_size)
            
            # Try to generate a marker
            marker_size_px = 200
            marker_image = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
            
            try:
                # Try dictionary's drawMarker method if available
                marker_image = dictionary.drawMarker(0, marker_size_px, marker_image, 1)
                print(f"  ✓ Generated marker using dictionary.drawMarker")
                
                # Save the marker
                filename = f"test_marker_{dict_name}.png"
                cv2.imwrite(filename, marker_image)
                print(f"  ✓ Saved marker as {filename}")
                
                successful_markers.append((dict_name, dict_type, marker_size))
            except Exception as e:
                print(f"  ✗ dictionary.drawMarker failed: {e}")
                
                # Try alternative methods
                try:
                    # Try using ArucoDetector if available
                    detector = cv2.aruco.ArucoDetector(dictionary)
                    marker_image = detector.generateImageMarker(0, marker_size_px, marker_image, 1)
                    print(f"  ✓ Generated marker using detector.generateImageMarker")
                    
                    # Save the marker
                    filename = f"test_marker_{dict_name}_alt.png"
                    cv2.imwrite(filename, marker_image)
                    print(f"  ✓ Saved marker as {filename}")
                    
                    successful_markers.append((dict_name, dict_type, marker_size))
                except Exception as e2:
                    print(f"  ✗ detector.generateImageMarker failed: {e2}")
                    
                    # Try cv2.aruco.drawMarker if available
                    try:
                        marker_image = cv2.aruco.drawMarker(dictionary, 0, marker_size_px, marker_image, 1)
                        print(f"  ✓ Generated marker using cv2.aruco.drawMarker")
                        
                        # Save the marker
                        filename = f"test_marker_{dict_name}_cv2.png"
                        cv2.imwrite(filename, marker_image)
                        print(f"  ✓ Saved marker as {filename}")
                        
                        successful_markers.append((dict_name, dict_type, marker_size))
                    except Exception as e3:
                        print(f"  ✗ cv2.aruco.drawMarker failed: {e3}")
                        print(f"  ✗ All marker generation methods failed")
        except Exception as e:
            print(f"  ✗ Dictionary creation failed: {e}")
    
    print("\nSUCCESSFUL MARKER GENERATION:")
    if successful_markers:
        for dict_name, dict_type, marker_size in successful_markers:
            print(f"  {dict_name} (marker size: {marker_size})")
        return successful_markers
    else:
        print("  None")
        return []

def test_detector_creation(successful_dicts):
    """Test detector creation with successful dictionaries"""
    print_separator()
    print("TESTING DETECTOR CREATION:")
    
    if not successful_dicts:
        print("No successful dictionaries to test detector creation.")
        return []
    
    successful_detectors = []
    
    for dict_name, dict_type, marker_size in successful_dicts:
        print(f"\nTesting detector creation with {dict_name}:")
        
        try:
            # Create dictionary
            dictionary = cv2.aruco.Dictionary(dict_type, marker_size)
            
            # Create detector parameters
            try:
                parameters = cv2.aruco.DetectorParameters()
                print(f"  ✓ Created detector parameters")
                
                # Try to create detector
                try:
                    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
                    print(f"  ✓ Created ArucoDetector")
                    successful_detectors.append((dict_name, dict_type, marker_size))
                except Exception as e:
                    print(f"  ✗ ArucoDetector creation failed: {e}")
            except Exception as e:
                print(f"  ✗ DetectorParameters creation failed: {e}")
        except Exception as e:
            print(f"  ✗ Dictionary creation failed: {e}")
    
    print("\nSUCCESSFUL DETECTOR CREATION:")
    if successful_detectors:
        for dict_name, dict_type, marker_size in successful_detectors:
            print(f"  {dict_name} (marker size: {marker_size})")
        return successful_detectors
    else:
        print("  None")
        return []

def test_charuco_board_creation(successful_dicts):
    """Test CharucoBoard creation with successful dictionaries"""
    print_separator()
    print("TESTING CHARUCOBOARD CREATION:")
    
    if not successful_dicts:
        print("No successful dictionaries to test CharucoBoard creation.")
        return []
    
    successful_boards = []
    
    for dict_name, dict_type, marker_size in successful_dicts:
        print(f"\nTesting CharucoBoard creation with {dict_name}:")
        
        try:
            # Create dictionary
            dictionary = cv2.aruco.Dictionary(dict_type, marker_size)
            
            # Create CharucoBoard
            try:
                board = cv2.aruco.CharucoBoard(
                    (6, 6),  # (squaresX, squaresY) as a tuple
                    0.04,    # squareLength (in arbitrary units)
                    0.03,    # markerLength (in arbitrary units)
                    dictionary
                )
                print(f"  ✓ Created CharucoBoard")
                
                # Try to generate board image
                try:
                    board_size = (600, 600)
                    board_img = np.zeros((board_size[1], board_size[0]), dtype=np.uint8)
                    board_img = board.generateImage(board_size, board_img, marginSize=10)
                    print(f"  ✓ Generated board image")
                    
                    # Save the board
                    filename = f"test_charuco_{dict_name}.png"
                    cv2.imwrite(filename, board_img)
                    print(f"  ✓ Saved board as {filename}")
                    
                    successful_boards.append((dict_name, dict_type, marker_size))
                except Exception as e:
                    print(f"  ✗ Board image generation failed: {e}")
            except Exception as e:
                print(f"  ✗ CharucoBoard creation failed: {e}")
        except Exception as e:
            print(f"  ✗ Dictionary creation failed: {e}")
    
    print("\nSUCCESSFUL CHARUCOBOARD CREATION:")
    if successful_boards:
        for dict_name, dict_type, marker_size in successful_boards:
            print(f"  {dict_name} (marker size: {marker_size})")
        return successful_boards
    else:
        print("  None")
        return []

def main():
    print("\nDIAGNOSING ARUCO MODULE IN OPENCV 4.12.0-DEV\n")
    
    # Print OpenCV information
    print_opencv_info()
    
    # Test dictionary creation
    successful_dicts = test_dictionary_creation()
    
    # Test marker generation
    successful_markers = test_marker_generation(successful_dicts)
    
    # Test detector creation
    successful_detectors = test_detector_creation(successful_dicts)
    
    # Test CharucoBoard creation
    successful_boards = test_charuco_board_creation(successful_dicts)
    
    # Print summary
    print_separator()
    print("DIAGNOSTIC SUMMARY:")
    print(f"Successful dictionaries: {len(successful_dicts)}")
    print(f"Successful marker generation: {len(successful_markers)}")
    print(f"Successful detector creation: {len(successful_detectors)}")
    print(f"Successful CharucoBoard creation: {len(successful_boards)}")
    
    if not successful_dicts:
        print("\nCRITICAL ISSUE: No ArUco dictionaries could be created.")
        print("This suggests a fundamental incompatibility with the ArUco module in this OpenCV version.")
    elif not successful_markers:
        print("\nCRITICAL ISSUE: No markers could be generated.")
        print("This suggests issues with the marker generation functions in this OpenCV version.")
    elif not successful_detectors:
        print("\nCRITICAL ISSUE: No detectors could be created.")
        print("This suggests issues with the detector creation in this OpenCV version.")
    elif not successful_boards:
        print("\nCRITICAL ISSUE: No CharucoBoards could be created.")
        print("This suggests issues with the CharucoBoard implementation in this OpenCV version.")
    
    if successful_dicts and successful_markers and successful_detectors:
        print("\nRECOMMENDATION:")
        dict_name, dict_type, marker_size = successful_markers[0]
        print(f"Use dictionary {dict_name} with marker size {marker_size} for best compatibility.")
        print(f"Example code:")
        print(f"    dictionary = cv2.aruco.Dictionary({dict_type}, {marker_size})")
        print(f"    parameters = cv2.aruco.DetectorParameters()")
        print(f"    detector = cv2.aruco.ArucoDetector(dictionary, parameters)")

if __name__ == "__main__":
    main()