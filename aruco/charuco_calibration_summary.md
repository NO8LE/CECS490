# ChArUco Board Calibration for OAK-D Camera

This document explains the updates made to the camera calibration system, replacing the traditional chessboard calibration with a ChArUco board approach for improved accuracy and robustness in drone applications.

## What is a ChArUco Board?

A ChArUco board combines the advantages of chessboard patterns and ArUco markers:

- **ChArUco = Chessboard + ArUco markers**
- ArUco markers are embedded inside the chessboard squares
- The markers can be detected even with partial occlusion or at extreme angles
- The system can accurately interpolate chessboard corners even when only some markers are visible

## Advantages Over Traditional Chessboard

1. **Improved robustness**: 
   - Works in more challenging conditions like variable lighting and long distances
   - Partial detection is possible (doesn't need to see the entire board)
   - Better suited for drone applications where the camera is moving and distance varies

2. **Higher accuracy**:
   - More reliable corner detection than traditional chessboards
   - Sub-pixel accuracy for calibration points
   - Better pose estimation for moving cameras

3. **Automatic orientation detection**:
   - ArUco markers provide unique IDs that eliminate orientation ambiguity
   - Makes the calibration process more user-friendly

## Key Changes to the Calibration System

1. **New unified board generator**:
   - `generate_charuco_board.py`: Creates a printable ChArUco board pattern
   - Supports both standard and drone-optimized boards (with `--drone` flag)
   - Outputs both PDF (for printing) and PNG (for preview)
   - Configurable size and square count for different application needs

2. **Updated calibration script**:
   - Modified `calibrate_camera.py` to use ChArUco calibration by default
   - Added compatibility for different OpenCV versions
   - Enhanced with drone-specific optimizations for variable distances
   - Stores board parameters in calibration file for consistent usage

3. **Runtime detection improvements**:
   - `oak_d_aruco_6x6_detector.py` now recognizes ChArUco boards
   - More accurate pose estimation at variable distances
   - Improved performance optimization with simplified detection options

## Usage Instructions

### 1. Generate a ChArUco Board

```bash
# Standard calibration board
python3 generate_charuco_board.py --squares 6 --size 12 --output charuco_board.pdf

# Drone-optimized board
python3 generate_charuco_board.py --squares 6 --size 12 --drone
```

Parameters:
- `--squares`: Number of squares in each direction (default: 6)
- `--size`: Total board size in inches (default: 12)
- `--output`: Output filename (default: charuco_board.pdf or drone_charuco_board.pdf if --drone is used)
- `--drone`: Enable drone-optimized settings with larger margins and detection range guidance

### 2. Print and Prepare the Board

- Print the PDF at 100% scale (no scaling)
- Mount on a rigid, flat surface for best results
- Measure the actual size of squares to verify scale

### 3. Calibrate the Camera

```bash
python3 calibrate_camera.py --charuco 6 6 0.0508 --drone
```

Parameters:
- `--charuco`: Followed by: number of squares in X direction, number of squares in Y direction, and square size in meters
- `--drone`: Optional flag to optimize for drone applications with variable distances

### 4. Use with the Detector

The calibration data is automatically saved and will be used by the ArUco detector when it runs.

## Technical Details

The calibration data is stored in `camera_calibration/calibration.npz` and includes:

- Camera matrix (intrinsic parameters)
- Distortion coefficients
- ChArUco board parameters (squares_x, squares_y, square_length)
- Drone optimization data (if applicable)

For optimal drone-based detection at various distances (0.5m to 12m), it's recommended to capture calibration frames at multiple distances throughout this range.
