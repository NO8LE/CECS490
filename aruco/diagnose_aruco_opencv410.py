#!/usr/bin/env python3

"""
Diagnostic script for ArUco module in OpenCV 4.10.0
This script tests the most basic ArUco operations with detailed error reporting
specifically tailored for OpenCV 4.10.0
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
    
    # Test methods specific to OpenCV 4.10
    print("\nTesting OpenCV 4.10 dictionary creation methods:")
    
    # Method 1: Dictionary constructor with marker size
    try:
        test_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
        print("  ✓ Method 1: Dictionary constructor with marker size works")
    except Exception as e:
        print(f"  ✗ Method 1: Dictionary constructor with marker size fails: {e}")
    
    # Method 2: getPredefinedDictionary
    try:
        test_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        print("  ✓ Method 2: getPredefinedDictionary works")
    except Exception as e:
        print(f"  ✗ Method 2: getPredefinedDictionary fails: {e}")
        
    # Method 3: Dictionary_get (older method)
    try:
        test_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        print("  ✓ Method 3: Dictionary_get works")
    except Exception as e:
        print(f"  ✗ Method 3: Dictionary_get fails: {e}")
    
    for dict_name, dict_type, marker_size in dict_types:
        print(f"\nTesting {dict_name} (marker size: {marker_size}):")
        
        # Try with actual marker size (most important test)
        try:
            # For OpenCV 4.10, try first with Dictionary constructor and marker size
            try:
                dictionary = cv2.aruco.Dictionary(dict_type, marker_size)
                print(f"  ✓ Created with Dictionary constructor and marker size {marker_size}")
                
                # Try to get dictionary properties if available
                try:
                    # These might not be available in 4.10
                    marker_size_prop = dictionary.getMarkerSize()
                    print(f"  ✓ Marker size property: {marker_size_prop}")
                except Exception as e:
                    print(f"  ℹ Cannot get marker size (normal for 4.10): {e}")
                
                # If we got here, this dictionary works
                successful_dicts.append((dict_name, dict_type, marker_size, "Dictionary"))
            except Exception as e:
                print(f"  ✗ Dictionary constructor failed: {e}")
                
                # Try with getPredefinedDictionary
                try:
                    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
                    print(f"  ✓ Created with getPredefinedDictionary")
                    successful_dicts.append((dict_name, dict_type, marker_size, "getPredefinedDictionary"))
                except Exception as e2:
                    print(f"  ✗ getPredefinedDictionary failed: {e2}")
                    
                    # Last resort: Try with Dictionary_get
                    try:
                        dictionary = cv2.aruco.Dictionary_get(dict_type)
                        print(f"  ✓ Created with Dictionary_get")
                        successful_dicts.append((dict_name, dict_type, marker_size, "Dictionary_get"))
                    except Exception as e3:
                        print(f"  ✗ Dictionary_get failed: {e3}")
        except Exception as e:
            print(f"  ✗ All dictionary creation methods failed: {e}")
    
    print("\nSUCCESSFUL DICTIONARIES:")
    if successful_dicts:
        for dict_name, dict_type, marker_size, method in successful_dicts:
            print(f"  {dict_name} (marker size: {marker_size}, method: {method})")
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
    
    for dict_name, dict_type, marker_size, dict_method in successful_dicts:
        print(f"\nTesting marker generation with {dict_name} (using {dict_method}):")
        
        try:
            # Create dictionary using the method that worked
            if dict_method == "Dictionary":
                dictionary = cv2.aruco.Dictionary(dict_type, marker_size)
            elif dict_method == "getPredefinedDictionary":
                dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
            else:  # Dictionary_get
                dictionary = cv2.aruco.Dictionary_get(dict_type)
            
            # Try to generate a marker
            marker_size_px = 200
            marker_image = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
            
            # For OpenCV 4.10, try these methods in order:
            
            # Method 1: Try generateImageMarker first (should work in 4.10)
            try:
                marker_image = cv2.aruco.generateImageMarker(dictionary, 0, marker_size_px)
                print(f"  ✓ Generated marker using cv2.aruco.generateImageMarker")
                
                # Save the marker
                filename = f"test_marker_{dict_name}_4.10.png"
                cv2.imwrite(filename, marker_image)
                print(f"  ✓ Saved marker as {filename}")
                
                successful_markers.append((dict_name, dict_type, marker_size, dict_method, "generateImageMarker"))
            except Exception as e:
                print(f"  ✗ cv2.aruco.generateImageMarker failed: {e}")
                
                # Method 2: Try drawMarker method
                try:
                    marker_image = cv2.aruco.drawMarker(dictionary, 0, marker_size_px, marker_image, 1)
                    print(f"  ✓ Generated marker using cv2.aruco.drawMarker")
                    
                    # Save the marker
                    filename = f"test_marker_{dict_name}_4.10_drawMarker.png"
                    cv2.imwrite(filename, marker_image)
                    print(f"  ✓ Saved marker as {filename}")
                    
                    successful_markers.append((dict_name, dict_type, marker_size, dict_method, "drawMarker"))
                except Exception as e2:
                    print(f"  ✗ cv2.aruco.drawMarker failed: {e2}")
                    
                    # Method 3: Last resort - try dictionary's own drawMarker if available
                    try:
                        marker_image = dictionary.drawMarker(0, marker_size_px, marker_image, 1)
                        print(f"  ✓ Generated marker using dictionary.drawMarker")
                        
                        # Save the marker
                        filename = f"test_marker_{dict_name}_4.10_dictMethod.png"
                        cv2.imwrite(filename, marker_image)
                        print(f"  ✓ Saved marker as {filename}")
                        
                        successful_markers.append((dict_name, dict_type, marker_size, dict_method, "dictionary.drawMarker"))
                    except Exception as e3:
                        print(f"  ✗ dictionary.drawMarker failed: {e3}")
                        print(f"  ✗ All marker generation methods failed")
        except Exception as e:
            print(f"  ✗ Dictionary creation failed: {e}")
    
    print("\nSUCCESSFUL MARKER GENERATION:")
    if successful_markers:
        for dict_name, dict_type, marker_size, dict_method, gen_method in successful_markers:
            print(f"  {dict_name} (marker size: {marker_size}, using {dict_method} with {gen_method})")
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
    
    for dict_name, dict_type, marker_size, dict_method in successful_dicts:
        print(f"\nTesting detector creation with {dict_name} (using {dict_method}):")
        
        try:
            # Create dictionary using the method that worked
            if dict_method == "Dictionary":
                dictionary = cv2.aruco.Dictionary(dict_type, marker_size)
            elif dict_method == "getPredefinedDictionary":
                dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
            else:  # Dictionary_get
                dictionary = cv2.aruco.Dictionary_get(dict_type)
            
            # Create detector parameters - in OpenCV 4.10 try both approaches
            
            # Method 1: DetectorParameters constructor
            try:
                parameters = cv2.aruco.DetectorParameters()
                print(f"  ✓ Created detector parameters using DetectorParameters()")
                
                # Try to create detector with ArucoDetector
                try:
                    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
                    print(f"  ✓ Created ArucoDetector")
                    successful_detectors.append((dict_name, dict_type, marker_size, dict_method, "ArucoDetector"))
                except Exception as e:
                    print(f"  ✗ ArucoDetector creation failed: {e}")
            except Exception as e:
                print(f"  ✗ DetectorParameters() creation failed: {e}")
                
                # Method 2: Try DetectorParameters.create()
                try:
                    parameters = cv2.aruco.DetectorParameters.create()
                    print(f"  ✓ Created detector parameters using DetectorParameters.create()")
                    
                    # We don't try ArucoDetector again, as it likely won't work if it failed before
                    successful_detectors.append((dict_name, dict_type, marker_size, dict_method, "DetectorParameters.create()"))
                except Exception as e:
                    print(f"  ✗ DetectorParameters.create() failed: {e}")
                    
                    # Method 3: Try DetectorParameters_create()
                    try:
                        parameters = cv2.aruco.DetectorParameters_create()
                        print(f"  ✓ Created detector parameters using DetectorParameters_create()")
                        successful_detectors.append((dict_name, dict_type, marker_size, dict_method, "DetectorParameters_create()"))
                    except Exception as e:
                        print(f"  ✗ DetectorParameters_create() failed: {e}")
            
        except Exception as e:
            print(f"  ✗ Dictionary creation failed: {e}")
    
    print("\nSUCCESSFUL DETECTOR CREATION:")
    if successful_detectors:
        for dict_name, dict_type, marker_size, dict_method, params_method in successful_detectors:
            print(f"  {dict_name} (marker size: {marker_size}, using {dict_method} with {params_method})")
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
    
    for dict_name, dict_type, marker_size, dict_method in successful_dicts:
        print(f"\nTesting CharucoBoard creation with {dict_name} (using {dict_method}):")
        
        try:
            # Create dictionary using the method that worked
            if dict_method == "Dictionary":
                dictionary = cv2.aruco.Dictionary(dict_type, marker_size)
            elif dict_method == "getPredefinedDictionary":
                dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
            else:  # Dictionary_get
                dictionary = cv2.aruco.Dictionary_get(dict_type)
            
            # For OpenCV 4.10, try multiple CharucoBoard creation methods
            
            # Method 1: Modern CharucoBoard constructor
            try:
                board = cv2.aruco.CharucoBoard(
                    (6, 6),  # (squaresX, squaresY) as a tuple
                    0.04,    # squareLength (in arbitrary units)
                    0.03,    # markerLength (in arbitrary units)
                    dictionary
                )
                print(f"  ✓ Created CharucoBoard using constructor")
                
                # Try to generate board image
                try:
                    # Method 1: Try draw method first
                    try:
                        board_size = (600, 600)
                        board_img = board.draw(board_size)
                        print(f"  ✓ Generated board image using draw method")
                        
                        # Save the board
                        filename = f"test_charuco_{dict_name}_4.10.png"
                        cv2.imwrite(filename, board_img)
                        print(f"  ✓ Saved board as {filename}")
                        
                        successful_boards.append((dict_name, dict_type, marker_size, dict_method, "constructor", "draw"))
                    except Exception as e:
                        print(f"  ✗ Board draw method failed: {e}")
                        
                        # Method 2: Try generateImage method
                        try:
                            board_size = (600, 600)
                            board_img = np.zeros((board_size[1], board_size[0]), dtype=np.uint8)
                            board_img = board.generateImage(board_size, board_img, marginSize=10)
                            print(f"  ✓ Generated board image using generateImage method")
                            
                            # Save the board
                            filename = f"test_charuco_{dict_name}_4.10_generateImage.png"
                            cv2.imwrite(filename, board_img)
                            print(f"  ✓ Saved board as {filename}")
                            
                            successful_boards.append((dict_name, dict_type, marker_size, dict_method, "constructor", "generateImage"))
                        except Exception as e2:
                            print(f"  ✗ Board generateImage method failed: {e2}")
                except Exception as e:
                    print(f"  ✗ Board image generation failed: {e}")
            except Exception as e:
                print(f"  ✗ CharucoBoard constructor failed: {e}")
                
                # Method 2: Try CharucoBoard.create
                try:
                    board = cv2.aruco.CharucoBoard.create(
                        squaresX=6,
                        squaresY=6,
                        squareLength=0.04,
                        markerLength=0.03,
                        dictionary=dictionary
                    )
                    print(f"  ✓ Created CharucoBoard using CharucoBoard.create")
                    
                    # Try to generate board image using draw method
                    try:
                        board_size = (600, 600)
                        board_img = board.draw(board_size)
                        print(f"  ✓ Generated board image using draw method")
                        
                        # Save the board
                        filename = f"test_charuco_{dict_name}_4.10_create.png"
                        cv2.imwrite(filename, board_img)
                        print(f"  ✓ Saved board as {filename}")
                        
                        successful_boards.append((dict_name, dict_type, marker_size, dict_method, "CharucoBoard.create", "draw"))
                    except Exception as e:
                        print(f"  ✗ Board draw method failed: {e}")
                except Exception as e:
                    print(f"  ✗ CharucoBoard.create failed: {e}")
                    
                    # Method 3: Try CharucoBoard_create
                    try:
                        board = cv2.aruco.CharucoBoard_create(
                            squaresX=6,
                            squaresY=6,
                            squareLength=0.04,
                            markerLength=0.03,
                            dictionary=dictionary
                        )
                        print(f"  ✓ Created CharucoBoard using CharucoBoard_create")
                        
                        # Try to generate board image using draw method
                        try:
                            board_size = (600, 600)
                            board_img = board.draw(board_size)
                            print(f"  ✓ Generated board image using draw method")
                            
                            # Save the board
                            filename = f"test_charuco_{dict_name}_4.10_create_old.png"
                            cv2.imwrite(filename, board_img)
                            print(f"  ✓ Saved board as {filename}")
                            
                            successful_boards.append((dict_name, dict_type, marker_size, dict_method, "CharucoBoard_create", "draw"))
                        except Exception as e:
                            print(f"  ✗ Board draw method failed: {e}")
                    except Exception as e:
                        print(f"  ✗ CharucoBoard_create failed: {e}")
        except Exception as e:
            print(f"  ✗ Dictionary creation failed: {e}")
    
    print("\nSUCCESSFUL CHARUCOBOARD CREATION:")
    if successful_boards:
        for dict_name, dict_type, marker_size, dict_method, board_method, draw_method in successful_boards:
            print(f"  {dict_name} (marker size: {marker_size}, using {dict_method} with {board_method} and {draw_method})")
        return successful_boards
    else:
        print("  None")
        return []

def main():
    print("\nDIAGNOSING ARUCO MODULE IN OPENCV 4.10.0\n")
    
    # Check OpenCV version
    if not cv2.__version__.startswith("4.10"):
        print(f"WARNING: This script is designed for OpenCV 4.10.0 but you're running {cv2.__version__}")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
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
        print("This suggests a fundamental incompatibility with the ArUco module in OpenCV 4.10.0.")
    elif not successful_markers:
        print("\nCRITICAL ISSUE: No markers could be generated.")
        print("This suggests issues with the marker generation functions in OpenCV 4.10.0.")
    elif not successful_detectors:
        print("\nCRITICAL ISSUE: No detectors could be created.")
        print("This suggests issues with the detector creation in OpenCV 4.10.0.")
    elif not successful_boards:
        print("\nCRITICAL ISSUE: No CharucoBoards could be created.")
        print("This suggests issues with the CharucoBoard implementation in OpenCV 4.10.0.")
    
    if successful_dicts and successful_markers and successful_detectors:
        print("\nRECOMMENDATION FOR OPENCV 4.10.0:")
        if successful_markers:
            dict_name, dict_type, marker_size, dict_method, gen_method = successful_markers[0]
            print(f"Use dictionary {dict_name} with marker size {marker_size} for best compatibility.")
            print(f"Dictionary creation method: {dict_method}")
            print(f"Marker generation method: {gen_method}")
            print(f"Example code:")
            
            if dict_method == "Dictionary":
                print(f"    dictionary = cv2.aruco.Dictionary({dict_type}, {marker_size})")
            elif dict_method == "getPredefinedDictionary":
                print(f"    dictionary = cv2.aruco.getPredefinedDictionary({dict_type})")
            else:  # Dictionary_get
                print(f"    dictionary = cv2.aruco.Dictionary_get({dict_type})")
            
            if successful_detectors:
                _, _, _, _, params_method = successful_detectors[0]
                if params_method == "ArucoDetector":
                    print(f"    parameters = cv2.aruco.DetectorParameters()")
                    print(f"    detector = cv2.aruco.ArucoDetector(dictionary, parameters)")
                elif params_method == "DetectorParameters.create()":
                    print(f"    parameters = cv2.aruco.DetectorParameters.create()")
                else:  # DetectorParameters_create
                    print(f"    parameters = cv2.aruco.DetectorParameters_create()")

if __name__ == "__main__":
    main()