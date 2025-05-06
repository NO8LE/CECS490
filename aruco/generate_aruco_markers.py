#!/usr/bin/env python3

"""
ArUco 6x6 Marker Generator

This script generates ArUco markers from the 6x6_250 dictionary and saves them as PNG files.
These markers can be used with the OAK-D ArUco 6x6 Marker Detector.

Usage:
  python3 generate_aruco_markers.py [start_id] [end_id] [size]

  start_id: First marker ID to generate (default: 0)
  end_id: Last marker ID to generate (default: 9)
  size: Size of the marker image in pixels (default: 300)

Example:
  python3 generate_aruco_markers.py 0 5 500
  This will generate markers with IDs 0 through 5, each 500x500 pixels.
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
OUTPUT_DIR = "aruco_markers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_aruco_marker(marker_id, dictionary_id=cv2.aruco.DICT_6X6_250, size=300):
    """
    Generate an ArUco marker image
    
    Args:
        marker_id: ID of the marker to generate
        dictionary_id: ArUco dictionary to use
        size: Size of the marker image in pixels
        
    Returns:
        The marker image
    """
    # Get the ArUco dictionary using the method that worked during initialization
    if dictionary_method == "old":
        aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
    elif dictionary_method == "new":
        aruco_dict = cv2.aruco.Dictionary.get(dictionary_id)
    elif dictionary_method == "create":
        aruco_dict = cv2.aruco.Dictionary.create(dictionary_id)
    elif dictionary_method == "constructor":
        aruco_dict = cv2.aruco.Dictionary(dictionary_id)
    else:
        # Fallback to trying all methods
        try:
            aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
        except:
            try:
                aruco_dict = cv2.aruco.Dictionary.get(dictionary_id)
            except:
                try:
                    aruco_dict = cv2.aruco.Dictionary.create(dictionary_id)
                except:
                    aruco_dict = cv2.aruco.Dictionary(dictionary_id)
    
    # Generate the marker
    marker_image = np.zeros((size, size), dtype=np.uint8)
    marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, size, marker_image, 1)
    
    # Add a white border
    border_size = size // 10
    bordered_image = np.ones((size + 2 * border_size, size + 2 * border_size), dtype=np.uint8) * 255
    bordered_image[border_size:border_size+size, border_size:border_size+size] = marker_image
    
    return bordered_image

def main():
    # Parse command line arguments
    start_id = 0
    end_id = 9
    size = 300
    
    if len(sys.argv) > 1:
        start_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        end_id = int(sys.argv[2])
    if len(sys.argv) > 3:
        size = int(sys.argv[3])
    
    # Validate input
    if start_id < 0 or end_id < start_id or end_id > 249:
        print("Error: Invalid marker ID range. Must be between 0 and 249.")
        sys.exit(1)
    
    if size < 50 or size > 2000:
        print("Error: Invalid size. Must be between 50 and 2000 pixels.")
        sys.exit(1)
    
    # Generate markers
    print(f"Generating ArUco 6x6 markers with IDs {start_id} to {end_id}, size {size}x{size} pixels...")
    
    for marker_id in range(start_id, end_id + 1):
        # Generate the marker
        marker_image = generate_aruco_marker(marker_id, cv2.aruco.DICT_6X6_250, size)
        
        # Save the marker
        filename = os.path.join(OUTPUT_DIR, f"aruco_6x6_250_{marker_id}.png")
        cv2.imwrite(filename, marker_image)
        print(f"Generated marker ID {marker_id}: {filename}")
    
    print(f"\nAll markers saved to the '{OUTPUT_DIR}' directory.")
    print("Print these markers and measure their physical size accurately for best results.")
    print("Update the MARKER_SIZE constant in oak_d_aruco_6x6_detector.py to match your printed marker size.")

if __name__ == "__main__":
    main()
