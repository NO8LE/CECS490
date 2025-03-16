#!/usr/bin/env python3

"""
CharucoBoard Generator for Drone Applications

This script generates a CharucoBoard pattern optimized for drone-based detection
at ranges from 0.5m to 12m with 12-inch (0.3048m) ArUco markers. By default, it
generates a board that fits exactly on a 12x12 inch (0.3048m x 0.3048m) square.

Usage:
  python3 generate_charuco_board_for_drone.py [squares_x] [squares_y] [output_size] [--high-contrast]

  squares_x: Number of squares in X direction (default: 6)
  squares_y: Number of squares in Y direction (default: 6)
  output_size: Size of the output image in pixels (default: 3000)
  --high-contrast: Generate a high-contrast version for better long-range detection

Example:
  python3 generate_charuco_board_for_drone.py 6 6 4000 --high-contrast
  This will generate a high-contrast CharucoBoard with 6x6 squares in a 4000x4000 pixel image.
"""

import os
import sys
import numpy as np
import argparse

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
    ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
    print("ArUco module successfully loaded and verified (using Dictionary_get)")
    # Store the method to use later
    dictionary_method = "old"
except Exception as e:
    # If Dictionary_get fails, try the newer API
    try:
        ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
        aruco_dict = cv2.aruco.Dictionary.get(ARUCO_DICT_TYPE)
        print("ArUco module successfully loaded and verified (using Dictionary.get)")
        # Store the method to use later
        dictionary_method = "new"
    except Exception as e2:
        # If both methods fail, try to create a dictionary directly
        try:
            # Try to create a dictionary directly (OpenCV 4.x approach)
            ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
            aruco_dict = cv2.aruco.Dictionary.create(ARUCO_DICT_TYPE)
            print("ArUco module successfully loaded and verified (using Dictionary.create)")
            # Store the method to use later
            dictionary_method = "create"
        except Exception as e3:
            # Last resort: try to create a dictionary with parameters
            try:
                # Try to create a dictionary with parameters (another approach)
                ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
                aruco_dict = cv2.aruco.Dictionary(ARUCO_DICT_TYPE)
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

def create_charuco_board(squares_x=6, squares_y=6, square_length=0.04, marker_length=0.03, dictionary_id=cv2.aruco.DICT_6X6_250):
    """
    Create a CharucoBoard object
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Length of square side in meters
        marker_length: Length of marker side in meters
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

def generate_charuco_board_image(board, output_size=3000, high_contrast=False):
    """
    Generate a CharucoBoard image
    
    Args:
        board: CharucoBoard object
        output_size: Size of the output image in pixels
        high_contrast: Whether to generate a high-contrast version
        
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
    
    # Apply high contrast if requested
    if high_contrast:
        # Convert to binary image with strong contrast
        _, board_image = cv2.threshold(board_image, 127, 255, cv2.THRESH_BINARY)
        
        # Add a white border
        border_size = output_size // 20
        bordered_image = np.ones((output_size + 2 * border_size, output_size + 2 * border_size), dtype=np.uint8) * 255
        bordered_image[border_size:border_size+output_size, border_size:border_size+output_size] = board_image
        board_image = bordered_image
    
    return board_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CharucoBoard Generator for Drone Applications')
    parser.add_argument('squares_x', nargs='?', type=int, default=6, help='Number of squares in X direction (default: 6)')
    parser.add_argument('squares_y', nargs='?', type=int, default=6, help='Number of squares in Y direction (default: 6)')
    parser.add_argument('output_size', nargs='?', type=int, default=3000, help='Size of the output image in pixels (default: 3000)')
    parser.add_argument('--high-contrast', action='store_true', help='Generate a high-contrast version for better long-range detection')
    args = parser.parse_args()
    
    # Validate input
    if args.squares_x < 2 or args.squares_y < 2:
        print("Error: Number of squares must be at least 2x2.")
        sys.exit(1)
    
    if args.output_size < 1000 or args.output_size > 8000:
        print("Error: Output size must be between 1000 and 8000 pixels.")
        sys.exit(1)
    
    # Calculate optimal square and marker sizes
    # For a 12x12 inch (0.3048m x 0.3048m) board
    square_length = 0.3048 / args.squares_x  # Square size in meters
    marker_length = square_length * 0.75     # Marker size is 75% of square size
    
    # Create the CharucoBoard
    print(f"Creating CharucoBoard with {args.squares_x}x{args.squares_y} squares...")
    print(f"Square size: {square_length*100:.1f}cm, Marker size: {marker_length*100:.1f}cm")
    board = create_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dictionary_id=cv2.aruco.DICT_6X6_250
    )
    
    # Generate the board image
    print(f"Generating board image with size {args.output_size}x{args.output_size} pixels...")
    board_image = generate_charuco_board_image(board, args.output_size, args.high_contrast)
    
    # Save the board image
    contrast_suffix = "_high_contrast" if args.high_contrast else ""
    filename = os.path.join(OUTPUT_DIR, f"charuco_board_drone_{args.squares_x}x{args.squares_y}{contrast_suffix}.png")
    cv2.imwrite(filename, board_image)
    print(f"CharucoBoard saved to {filename}")
    
    # Save board configuration
    config_filename = os.path.join(OUTPUT_DIR, f"charuco_board_drone_{args.squares_x}x{args.squares_y}_config.txt")
    with open(config_filename, 'w') as f:
        f.write(f"Dictionary: DICT_6X6_250\n")
        f.write(f"Squares X: {args.squares_x}\n")
        f.write(f"Squares Y: {args.squares_y}\n")
        f.write(f"Square Length: {square_length:.6f} meters\n")
        f.write(f"Marker Length: {marker_length:.6f} meters\n")
        f.write(f"Optimized for 12-inch (0.3048m) markers at ranges from 0.5m to 12m\n")
    print(f"Board configuration saved to {config_filename}")
    
    print("\nPrinting Instructions:")
    print("1. Print this CharucoBoard at exactly 12x12 inches (30.48x30.48 cm)")
    print("2. Measure the actual size of the squares on your printed board")
    print("3. Use the measured square size when calibrating:")
    print(f"   python3 calibrate_camera.py --charuco {args.squares_x} {args.squares_y} [measured_square_size_in_meters]")
    print("\nFor drone-based detection, mount the board on a flat surface")
    print("and ensure it's visible from various angles during calibration.")

if __name__ == "__main__":
    main()
