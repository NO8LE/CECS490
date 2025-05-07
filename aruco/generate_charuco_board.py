#!/usr/bin/env python3

"""
CharucoBoard Generator for Camera Calibration

This script generates a ChArUco board pattern and saves it as a PDF for printing.
The board combines ArUco markers embedded in a chessboard pattern, which provides
more accurate camera calibration and pose estimation.

Usage:
  python3 generate_charuco_board.py [--squares SQUARES] [--size SIZE] [--output FILENAME] [--drone]

Options:
  --squares, -s  Squares in X and Y direction (default: 6x6)
  --size, -z     Size of the printed board in inches (default: 12)
  --output, -o   Output filename (default: charuco_board.pdf)
  --drone, -d    Enable drone-optimized settings (larger margins, detection range info)

Examples:
  python3 generate_charuco_board.py
  python3 generate_charuco_board.py --squares 7 --size 15 --output custom_board.pdf
  python3 generate_charuco_board.py --drone --size 24 --output drone_board.pdf

Notes:
  For drone applications (--drone flag), larger boards with fewer squares are recommended:
  - 12x12 inches (30.5x30.5 cm) board is good for ranges of 0.5-8m
  - 24x24 inches (61x61 cm) board is recommended for ranges of 3-12m
"""

import numpy as np
import cv2
import argparse
import os
import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Create calibration patterns directory if it doesn't exist
PATTERNS_DIR = "calibration_patterns"
os.makedirs(PATTERNS_DIR, exist_ok=True)

def generate_charuco_board(squares_x=6, squares_y=6, board_size_inches=12.0, 
                           drone_mode=False, dpi=300):
    """
    Generate a CharucoBoard pattern.
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        board_size_inches: Total board size in inches
        drone_mode: Whether to use drone-optimized settings
        dpi: DPI for the output image
        
    Returns:
        The board image suitable for printing
    """
    # Calculate square size in inches
    square_length_inches = board_size_inches / squares_x
    # Marker size is 75% of square size
    marker_length_inches = square_length_inches * 0.75
    
    # Convert sizes to pixels at the given DPI
    square_length = int(square_length_inches * dpi)
    marker_length = int(marker_length_inches * dpi)
    
    # Create the ArUco dictionary - using 6x6 250
    # Check OpenCV version first to use the appropriate API
    if cv2.__version__.startswith("4.10") or cv2.__version__.startswith("4.11") or cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
        # For OpenCV 4.12.0-dev and newer
        try:
            # Create dictionary with marker size parameter
            aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
            print(f"Using OpenCV {cv2.__version__} ArUco Dictionary with markerSize")
        except Exception as e:
            print(f"Error creating ArUco dictionary for OpenCV 4.12+: {str(e)}")
            sys.exit(1)
    else:
        # For older OpenCV versions
        try:
            # Try new API first
            aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_6X6_250)
        except:
            # Fall back to old API
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    
    # Create the CharucoBoard
    # Check OpenCV version first to use the appropriate API
    if cv2.__version__.startswith("4.10") or cv2.__version__.startswith("4.11") or cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
        # For OpenCV 4.12.0-dev and newer
        try:
            # Create CharucoBoard with the constructor
            charuco_board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y),  # (squaresX, squaresY) as a tuple
                square_length,  # squareLength
                marker_length,  # markerLength
                aruco_dict
            )
            print(f"Created CharucoBoard using OpenCV {cv2.__version__} constructor")
        except Exception as e:
            print(f"Error creating CharucoBoard for OpenCV 4.12+: {str(e)}")
            sys.exit(1)
    else:
        # For older OpenCV versions
        try:
            # Try new API first
            charuco_board = cv2.aruco.CharucoBoard.create(
                squaresX=squares_x,
                squaresY=squares_y,
                squareLength=square_length,
                markerLength=marker_length,
                dictionary=aruco_dict
            )
        except:
            # Fall back to old API
            charuco_board = cv2.aruco.CharucoBoard_create(
                squaresX=squares_x,
                squaresY=squares_y,
                squareLength=square_length,
                markerLength=marker_length,
                dictionary=aruco_dict
            )
    
    # Generate the board image - handle OpenCV 4.12+ differently
    if cv2.__version__.startswith("4.10") or cv2.__version__.startswith("4.11") or cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
        try:
            # For OpenCV 4.12+, use the draw method with size parameter
            board_size = (int(squares_x * square_length), int(squares_y * square_length))
            board_img = np.zeros((board_size[1], board_size[0]), dtype=np.uint8)
            board_img = charuco_board.generateImage(board_size, board_img, marginSize=0)
            print("Generated board image using generateImage method")
        except Exception as e:
            print(f"Error generating board image with OpenCV 4.12+: {e}")
            # Fallback - create a blank image with text
            board_size = (int(squares_x * square_length), int(squares_y * square_length))
            board_img = np.ones((board_size[1], board_size[0]), dtype=np.uint8) * 255
            
            # Add text explaining the error
            cv2.putText(
                board_img,
                "CharucoBoard generation failed with OpenCV 4.12+",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                0,
                1
            )
            cv2.putText(
                board_img,
                f"Error: {str(e)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                0,
                1
            )
            print("Using fallback blank board with error message")
    else:
        # For older OpenCV versions, use the traditional API
        board_img = charuco_board.draw(
            (squares_x * square_length, squares_y * square_length)
        )
    
    # Add margins for printing (larger margin for drone boards)
    margin = int((1.0 if drone_mode else 0.5) * dpi)
    img_with_margin = np.ones((
        board_img.shape[0] + 2 * margin,
        board_img.shape[1] + 2 * margin
    ), dtype=np.uint8) * 255
    
    # Place the board in the center
    img_with_margin[margin:margin+board_img.shape[0], 
                    margin:margin+board_img.shape[1]] = board_img
    
    # Add information text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = 0  # Black
    
    if drone_mode:
        # Add drone-specific information
        # Title and specifications
        cv2.putText(img_with_margin, 
                    f"DRONE CALIBRATION PATTERN", 
                    (margin, margin - 50), 
                    font, 1.2, text_color, 3)
        
        cv2.putText(img_with_margin, 
                    f"CharucoBoard {squares_x}x{squares_y} squares, ArUco 6x6 dictionary", 
                    (margin, margin - 25), 
                    font, 0.8, text_color, 2)
                    
        # Add physical size information
        sq_cm = square_length_inches * 2.54  # convert to cm
        mk_cm = marker_length_inches * 2.54  # convert to cm
        
        cv2.putText(img_with_margin, 
                    f"Board size: {board_size_inches}x{board_size_inches} inches ({board_size_inches*2.54:.1f}x{board_size_inches*2.54:.1f} cm)", 
                    (margin, margin - 0), 
                    font, 0.7, text_color, 2)
                    
        cv2.putText(img_with_margin, 
                    f"Square size: {square_length_inches:.2f}in ({sq_cm:.1f}cm), Marker size: {marker_length_inches:.2f}in ({mk_cm:.1f}cm)", 
                    (margin, margin + 25), 
                    font, 0.7, text_color, 2)
        
        # Add calibration instructions
        instructions = [
            "DRONE CALIBRATION INSTRUCTIONS:",
            "1. Print this pattern at 100% scale (verify measurements)",
            "2. Mount on a rigid, flat surface",
            "3. Capture frames from multiple distances (0.5m to 12m)",
            "4. Run: python3 calibrate_camera.py --charuco " + 
               f"{squares_x} {squares_y} {square_length_inches * 0.0254:.4f}" + " --drone"
        ]
        
        for i, line in enumerate(instructions):
            cv2.putText(img_with_margin, 
                        line, 
                        (margin, img_with_margin.shape[0] - margin - 150 + i * 30), 
                        font, 0.7, text_color, 2)
                        
        # Add distance optimization guidance
        cv2.putText(img_with_margin, 
                    "DETECTION RANGE GUIDANCE:", 
                    (margin, img_with_margin.shape[0] - margin - 30), 
                    font, 0.7, text_color, 2)
                    
        # Calculate rough detection range based on board size
        max_range = board_size_inches / 1.5  # rough estimate 
        cv2.putText(img_with_margin, 
                    f"This {board_size_inches}\" board is optimized for detection at {0.5}-{max_range:.1f}m range", 
                    (margin, img_with_margin.shape[0] - margin), 
                    font, 0.7, text_color, 2)
    else:
        # Add standard calibration information
        cv2.putText(img_with_margin, 
                    f"CharucoBoard {squares_x}x{squares_y} squares, ArUco 6x6 dictionary", 
                    (margin, margin - 30), 
                    font, 1, text_color, 2)
        cv2.putText(img_with_margin, 
                    f"Square size: {square_length_inches:.3f}in, Marker size: {marker_length_inches:.3f}in", 
                    (margin, margin - 5), 
                    font, 0.7, text_color, 2)
        
        # Add calibration instructions
        instructions = [
            "Instructions for calibration:",
            "1. Print this pattern at accurate size",
            "2. Mount on a rigid, flat surface",
            "3. Run: python3 calibrate_camera.py --charuco " + 
               f"{squares_x} {squares_y} {square_length_inches * 0.0254:.4f}"
        ]
        
        for i, line in enumerate(instructions):
            cv2.putText(img_with_margin, 
                        line, 
                        (margin, img_with_margin.shape[0] - margin + 5 + i * 25), 
                        font, 0.7, text_color, 2)
    
    return img_with_margin

def save_as_pdf(img, filename, dpi=300):
    """Save the image as a PDF file"""
    with PdfPages(filename) as pdf:
        plt.figure(figsize=(img.shape[1]/dpi, img.shape[0]/dpi), dpi=dpi)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        pdf.savefig()
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate CharucoBoard pattern for camera calibration')
    parser.add_argument('--squares', '-s', type=int, default=6,
                        help='Number of squares in both X and Y directions (default: 6)')
    parser.add_argument('--size', '-z', type=float, default=12.0,
                        help='Total board size in inches (default: 12)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename (default: charuco_board.pdf or drone_charuco_board.pdf if --drone is used)')
    parser.add_argument('--drone', '-d', action='store_true',
                        help='Enable drone-optimized settings')
    args = parser.parse_args()
    
    # Set default output filename based on mode
    if args.output is None:
        args.output = "drone_charuco_board.pdf" if args.drone else "charuco_board.pdf"
    
    # Calculate board parameters
    squares = args.squares
    board_size = args.size
    square_length_inches = board_size / squares  # inches per square
    marker_length_inches = square_length_inches * 0.75  # 75% of square size
    
    # Print generation information
    mode_str = "Drone-Optimized " if args.drone else ""
    print(f"Generating {mode_str}{squares}x{squares} CharucoBoard pattern")
    print(f"Total size: {board_size}in x {board_size}in ({board_size*2.54:.1f}cm x {board_size*2.54:.1f}cm)")
    print(f"Square size: {square_length_inches:.3f}in ({square_length_inches*2.54:.1f}cm)")
    print(f"Marker size: {marker_length_inches:.3f}in ({marker_length_inches*2.54:.1f}cm)")
    
    # For drone mode, show detection range estimate
    if args.drone:
        min_range = 0.5  # meters
        max_range = board_size / 1.5  # rough estimate based on board size
        print(f"Optimized for detection range: {min_range}-{max_range:.1f} meters")
        
        if board_size < 12:
            print("\nWARNING: Small board size may limit detection range")
            print("For long-range drone applications, board sizes of 12-24 inches are recommended")
    
    # Generate the board
    board_img = generate_charuco_board(
        squares_x=squares,
        squares_y=squares,
        board_size_inches=board_size,
        drone_mode=args.drone
    )
    
    # Get current date for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output filenames with paths in the calibration_patterns directory
    if args.output is None:
        base_filename = f"charuco_{squares}x{squares}_{board_size}in_{timestamp}"
        if args.drone:
            base_filename = f"drone_{base_filename}"
    else:
        # Extract base filename without extension
        base_filename = os.path.splitext(os.path.basename(args.output))[0]
    
    # Create file paths
    pdf_file = os.path.join(PATTERNS_DIR, base_filename + ".pdf")
    png_file = os.path.join(PATTERNS_DIR, base_filename + ".png")
    config_file = os.path.join(PATTERNS_DIR, base_filename + "_config.txt")
    
    # Save as PDF
    save_as_pdf(board_img, pdf_file)
    print(f"CharucoBoard saved as {pdf_file}")
    
    # Also save as PNG for preview
    cv2.imwrite(png_file, board_img)
    print(f"Preview saved as {png_file}")
    
    # Calculate calibration parameters
    square_size_meters = (board_size / squares) * 0.0254  # convert to meters
    marker_size_meters = square_size_meters * 0.75
    
    # Save configuration file
    with open(config_file, 'w') as f:
        f.write("# CharucoBoard Configuration\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Board Type: {'Drone-Optimized ' if args.drone else ''}CharucoBoard\n")
        f.write(f"Squares: {squares}x{squares}\n")
        f.write(f"Dictionary: DICT_6X6_250\n\n")
        
        f.write("# Physical Dimensions\n")
        f.write(f"Board Size (inches): {board_size}x{board_size}\n")
        f.write(f"Board Size (cm): {board_size*2.54:.1f}x{board_size*2.54:.1f}\n")
        f.write(f"Square Size (inches): {square_length_inches:.4f}\n")
        f.write(f"Square Size (cm): {square_length_inches*2.54:.4f}\n")
        f.write(f"Square Size (meters): {square_size_meters:.6f}\n")
        f.write(f"Marker Size (inches): {marker_length_inches:.4f}\n")
        f.write(f"Marker Size (cm): {marker_length_inches*2.54:.4f}\n")
        f.write(f"Marker Size (meters): {marker_size_meters:.6f}\n\n")
        
        if args.drone:
            f.write("# Drone Detection Range\n")
            f.write(f"Estimated Detection Range: 0.5m - {board_size/1.5:.1f}m\n\n")
        
        f.write("# Calibration Command\n")
        drone_flag = " --drone" if args.drone else ""
        f.write(f"python3 calibrate_camera.py --charuco {squares} {squares} {square_size_meters:.6f}{drone_flag}\n")
    
    print(f"Configuration saved as {config_file}")
    
    # Print calibration command for user reference
    drone_flag = " --drone" if args.drone else ""
    print(f"\nTo calibrate with this board, run:")
    print(f"python3 calibrate_camera.py --charuco {squares} {squares} {square_size_meters:.6f}{drone_flag}")

if __name__ == "__main__":
    main()
