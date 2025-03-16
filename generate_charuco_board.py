#!/usr/bin/env python3

"""
CharucoBoard Generator

This script generates a CharucoBoard pattern and saves it as a PNG file.
The CharucoBoard is a hybrid between a chessboard and ArUco markers,
which provides more robust and accurate camera calibration.

By default, it generates a board with 6x6 ArUco markers that fits on a
standard 8.5x11 inch (letter size) sheet of printer paper.

Usage:
  python3 generate_charuco_board.py [squares_x] [squares_y] [square_length] [marker_length] [output_size]

  squares_x: Number of squares in X direction (default: 6)
  squares_y: Number of squares in Y direction (default: 8)
  square_length: Length of square side in pixels (default: 200)
  marker_length: Length of marker side in pixels (default: 150)
  output_size: Size of the output image in pixels (default: 2000)

Example:
  python3 generate_charuco_board.py 6 8 200 150 2000
  This will generate a CharucoBoard with 6x8 squares, each 200 pixels,
  with markers of 150 pixels, in a 2000x2000 pixel image.
"""

import os
import sys
import numpy as np

# Import OpenCV
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("Error: OpenCV (cv2) not found.")
    print("Please install OpenCV:")
    print("  pip install opencv-python opencv-contrib-python")
    sys.exit(1)

# Import ArUco module - try different approaches for different OpenCV installations
try:
    # First try direct import from cv2
    if hasattr(cv2, 'aruco'):
        print("Using cv2.aruco")
        aruco = cv2.aruco
    else:
        # Try to import as a separate module (older OpenCV versions)
        try:
            from cv2 import aruco
            print("Using from cv2 import aruco")
            # Make it available as cv2.aruco for consistency
            cv2.aruco = aruco
        except ImportError:
            # For Jetson with custom OpenCV builds
            try:
                sys.path.append('/usr/lib/python3/dist-packages/cv2/python-3.10')
                from cv2 import aruco
                print("Using Jetson-specific aruco import")
                cv2.aruco = aruco
            except (ImportError, FileNotFoundError):
                print("Error: OpenCV ArUco module not found.")
                print("Please ensure opencv-contrib-python is installed:")
                print("  pip install opencv-contrib-python")
                print("\nFor Jetson platforms, you might need to install it differently:")
                print("  sudo apt-get install python3-opencv")
                sys.exit(1)
except Exception as e:
    print(f"Error importing ArUco module: {str(e)}")
    print("Please ensure opencv-contrib-python is installed correctly")
    sys.exit(1)

# Verify ArUco module is working
try:
    # Try to access a dictionary to verify ArUco is working
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    print("ArUco module successfully loaded and verified (using Dictionary_get)")
    # Store the method to use later
    dictionary_method = "old"
except Exception as e:
    # If Dictionary_get fails, try the newer API
    try:
        aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_6X6_250)
        print("ArUco module successfully loaded and verified (using Dictionary.get)")
        # Store the method to use later
        dictionary_method = "new"
    except Exception as e2:
        # If both methods fail, try to create a dictionary directly
        try:
            # Try to create a dictionary directly (OpenCV 4.x approach)
            aruco_dict = cv2.aruco.Dictionary.create(cv2.aruco.DICT_6X6_250)
            print("ArUco module successfully loaded and verified (using Dictionary.create)")
            # Store the method to use later
            dictionary_method = "create"
        except Exception as e3:
            # Last resort: try to create a dictionary with parameters
            try:
                # Try to create a dictionary with parameters (another approach)
                aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250)
                print("ArUco module successfully loaded and verified (using Dictionary constructor)")
                # Store the method to use later
                dictionary_method = "constructor"
            except Exception as e4:
                print(f"Error verifying ArUco module: {str(e4)}")
                print("ArUco module found but not working correctly")
                print("\nDetailed error information:")
                print(f"Dictionary_get error: {str(e)}")
                print(f"Dictionary.get error: {str(e2)}")
                print(f"Dictionary.create error: {str(e3)}")
                print(f"Dictionary constructor error: {str(e4)}")
                print("\nPlease check your OpenCV installation and version.")
                sys.exit(1)

# Create output directory if it doesn't exist
OUTPUT_DIR = "calibration_patterns"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_aruco_dictionary(dictionary_id=cv2.aruco.DICT_6X6_250):
    """
    Get the ArUco dictionary using the method that worked during initialization
    
    Args:
        dictionary_id: ArUco dictionary ID to use
        
    Returns:
        The ArUco dictionary
    """
    if dictionary_method == "old":
        return cv2.aruco.Dictionary_get(dictionary_id)
    elif dictionary_method == "new":
        return cv2.aruco.Dictionary.get(dictionary_id)
    elif dictionary_method == "create":
        return cv2.aruco.Dictionary.create(dictionary_id)
    elif dictionary_method == "constructor":
        return cv2.aruco.Dictionary(dictionary_id)
    else:
        # Fallback to trying all methods
        try:
            return cv2.aruco.Dictionary_get(dictionary_id)
        except:
            try:
                return cv2.aruco.Dictionary.get(dictionary_id)
            except:
                try:
                    return cv2.aruco.Dictionary.create(dictionary_id)
                except:
                    return cv2.aruco.Dictionary(dictionary_id)

def create_charuco_board(squares_x=6, squares_y=8, square_length=200, marker_length=150, dictionary_id=cv2.aruco.DICT_6X6_250):
    """
    Create a CharucoBoard object
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Length of square side in pixels
        marker_length: Length of marker side in pixels
        dictionary_id: ArUco dictionary ID to use
        
    Returns:
        The CharucoBoard object
    """
    # Get the ArUco dictionary
    aruco_dict = get_aruco_dictionary(dictionary_id)
    
    # Create the CharucoBoard
    try:
        # Try the newer API first
        board = cv2.aruco.CharucoBoard.create(
            squaresX=squares_x,
            squaresY=squares_y,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=aruco_dict
        )
        print("Created CharucoBoard using CharucoBoard.create")
        return board
    except Exception as e:
        try:
            # Try the older API
            board = cv2.aruco.CharucoBoard_create(
                squaresX=squares_x,
                squaresY=squares_y,
                squareLength=square_length,
                markerLength=marker_length,
                dictionary=aruco_dict
            )
            print("Created CharucoBoard using CharucoBoard_create")
            return board
        except Exception as e2:
            print(f"Error creating CharucoBoard: {str(e)}")
            print(f"Second error: {str(e2)}")
            print("Please check your OpenCV installation and version.")
            sys.exit(1)

def generate_charuco_board_image(board, output_size=2000):
    """
    Generate a CharucoBoard image
    
    Args:
        board: CharucoBoard object
        output_size: Size of the output image in pixels
        
    Returns:
        The CharucoBoard image
    """
    # Generate the board image
    try:
        # Try the newer API first
        board_image = board.draw((output_size, output_size))
        print("Generated board image using board.draw")
    except Exception as e:
        try:
            # Try the older API
            board_image = board.draw((output_size, output_size))
            print("Generated board image using board.draw (older API)")
        except Exception as e2:
            print(f"Error generating board image: {str(e)}")
            print(f"Second error: {str(e2)}")
            print("Please check your OpenCV installation and version.")
            sys.exit(1)
    
    return board_image

def main():
    # Parse command line arguments
    squares_x = 6
    squares_y = 8
    square_length = 200
    marker_length = 150
    output_size = 2000
    
    if len(sys.argv) > 1:
        squares_x = int(sys.argv[1])
    if len(sys.argv) > 2:
        squares_y = int(sys.argv[2])
    if len(sys.argv) > 3:
        square_length = int(sys.argv[3])
    if len(sys.argv) > 4:
        marker_length = int(sys.argv[4])
    if len(sys.argv) > 5:
        output_size = int(sys.argv[5])
    
    # Validate input
    if squares_x < 2 or squares_y < 2:
        print("Error: Number of squares must be at least 2x2.")
        sys.exit(1)
    
    if marker_length >= square_length:
        print("Error: Marker length must be smaller than square length.")
        sys.exit(1)
    
    if output_size < 500 or output_size > 8000:
        print("Error: Output size must be between 500 and 8000 pixels.")
        sys.exit(1)
    
    # Create the CharucoBoard
    print(f"Creating CharucoBoard with {squares_x}x{squares_y} squares...")
    board = create_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dictionary_id=cv2.aruco.DICT_6X6_250
    )
    
    # Generate the board image
    print(f"Generating board image with size {output_size}x{output_size} pixels...")
    board_image = generate_charuco_board_image(board, output_size)
    
    # Save the board image
    filename = os.path.join(OUTPUT_DIR, f"charuco_board_{squares_x}x{squares_y}.png")
    cv2.imwrite(filename, board_image)
    print(f"CharucoBoard saved to {filename}")
    
    # Save board configuration
    config_filename = os.path.join(OUTPUT_DIR, f"charuco_board_{squares_x}x{squares_y}_config.txt")
    with open(config_filename, 'w') as f:
        f.write(f"Dictionary: DICT_6X6_250\n")
        f.write(f"Squares X: {squares_x}\n")
        f.write(f"Squares Y: {squares_y}\n")
        f.write(f"Square Length: {square_length}\n")
        f.write(f"Marker Length: {marker_length}\n")
    print(f"Board configuration saved to {config_filename}")
    
    print("\nPrinting Instructions:")
    print("1. Print this CharucoBoard on a standard 8.5x11 inch (letter size) sheet of paper")
    print("2. Measure the actual size of the squares on your printed board")
    print("3. When using for calibration, specify the actual physical size of the squares:")
    print(f"   python3 calibrate_camera.py --charuco {squares_x} {squares_y} [measured_square_size_in_meters]")
    print("\nNote: The default 6x8 configuration is designed to fit optimally on letter paper")
    print("with the correct aspect ratio (portrait orientation).")

if __name__ == "__main__":
    main()
