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
                # Try with marker size parameter for OpenCV 4.12.0-dev and newer
                try:
                    # In OpenCV 4.12.0-dev, Dictionary constructor needs marker size (6 for 6x6 dict)
                    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
                    print("ArUco module successfully loaded and verified (using Dictionary with markerSize)")
                    # Store the method to use later
                    dictionary_method = "constructor_with_size"
                except Exception as e5:
                    print(f"Error verifying ArUco module: {str(e5)}")
                    print("ArUco module found but not working correctly")
                    print("\nDetailed error information:")
                    print(f"Dictionary_get error: {str(e)}")
                    print(f"Dictionary.get error: {str(e2)}")
                    print(f"Dictionary.create error: {str(e3)}")
                    print(f"Dictionary constructor error: {str(e4)}")
                    print(f"Dictionary with markerSize error: {str(e5)}")
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
    # For OpenCV 4.10, use getPredefinedDictionary which is known to work
    print(f"Generating marker ID {marker_id} with size {size}...")
    
    try:
        # This is the method that works in fix_aruco_opencv410.py
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        print("Using getPredefinedDictionary for dictionary creation")
        
        # Generate marker image
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
        print(f"Successfully generated marker ID {marker_id} using generateImageMarker")
        
        # Verify the marker image is not blank
        if marker_image is not None:
            white_pixels = np.sum(marker_image == 255)
            black_pixels = np.sum(marker_image == 0)
            print(f"Marker image stats - White pixels: {white_pixels}, Black pixels: {black_pixels}")
            if black_pixels == 0:
                print("WARNING: Marker appears to be blank (no black pixels)")
    except Exception as e:
        print(f"Error in primary generation method: {e}")
        
        # Fallback method if the primary method fails
        try:
            # Try with Dictionary constructor with marker size parameter
            aruco_dict = cv2.aruco.Dictionary(dictionary_id, 6)
            print("Fallback: Using Dictionary constructor with marker size")
            marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
        except Exception as e2:
            print(f"Error in fallback method: {e2}")
            # Create a blank marker with the ID as text as last resort
            marker_image = np.ones((size, size), dtype=np.uint8) * 255  # White background
            cv2.putText(
                marker_image,
                f"ID: {marker_id}",
                (size//4, size//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                0,  # Black text
                2
            )
            print(f"Using text fallback for marker ID {marker_id}")
    
    # Add a white border
    border_size = size // 10
    bordered_image = np.ones((size + 2 * border_size, size + 2 * border_size), dtype=np.uint8) * 255
    bordered_image[border_size:border_size+size, border_size:border_size+size] = marker_image
    
    # Verify the bordered image is not blank
    white_pixels = np.sum(bordered_image == 255)
    black_pixels = np.sum(bordered_image == 0)
    print(f"Bordered image stats - White pixels: {white_pixels}, Black pixels: {black_pixels}")
    if black_pixels == 0:
        print("WARNING: Final bordered image appears to be blank (no black pixels)")
    
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
