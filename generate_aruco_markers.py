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

import cv2
import os
import sys
import numpy as np

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
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)
    
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
