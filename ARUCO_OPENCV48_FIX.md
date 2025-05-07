# ArUco Detector Fix for OpenCV 4.8.0

## Issue Overview

When using the OAK-D ArUco 6x6 Marker Detector with OpenCV 4.8.0 (upgraded from 4.5.5.62 for CUDA support), you may encounter the following issues:

1. **False positive detections**: The detector identifies non-ArUco objects as markers, usually with ID=0
2. **Random object tracing**: Dozens of small shapes are traced over random objects in the frame
3. **Failed ArUco identification**: Legitimate ArUco markers are not properly detected
4. **Broken pose estimation**: 3D pose estimation fails to work correctly

These issues occur because:

1. The ArUco API changed significantly in OpenCV 4.8.0, with different parameter handling and detection methods
2. The pose estimation API (`estimatePoseSingleMarkers`) is deprecated in 4.8.0 and needs to be replaced with `solvePnP`
3. The default detection parameters in 4.8.0 are more prone to false positives without proper validation

## Solution

This package provides an enhanced ArUco detector specifically designed to work with OpenCV 4.8.0. It includes:

1. A standalone implementation (`fix_aruco_opencv48.py`) with:
   - Stricter parameter settings to prevent false positives
   - Post-detection validation of marker geometry
   - Fixed pose estimation using `solvePnP` instead of `estimatePoseSingleMarkers`
   - Additional filters to ensure detected markers are actual ArUco codes

2. A wrapper script (`oak_d_aruco_wrapper.py`) that:
   - Dynamically patches the original detector at runtime
   - Preserves all OAK-D camera functionality
   - Makes the original detector work correctly with OpenCV 4.8.0

## Using the Fix

### Method 1: Use the Wrapper Script (Recommended)

The simplest approach is to use the wrapper script which automatically integrates the enhanced detector with the original OAK-D detector:

```bash
# Run with the same arguments as you would with the original detector
python3 oak_d_aruco_wrapper.py --cuda
```

This script:
1. Creates a backup of the original detector if one doesn't exist
2. Dynamically patches the `detect_aruco_markers` function at runtime
3. Runs the original detector with all its functionality (OAK-D camera, depth sensing, etc.)
4. Uses the enhanced detection method when OpenCV 4.8.0 is detected

You can use all the same command-line arguments as the original detector:

```bash
# Examples
python3 oak_d_aruco_wrapper.py --target 5 --cuda
python3 oak_d_aruco_wrapper.py --resolution high --cuda --performance
python3 oak_d_aruco_wrapper.py --headless --stream
```

### Method 2: Integrate the Enhanced Detector Manually

If you prefer to modify your codebase directly, you can:

1. Import the enhanced detector in your code:
   ```python
   from fix_aruco_opencv48 import detect_aruco_markers
   ```

2. Replace your ArUco detection code with calls to the enhanced detector:
   ```python
   markers_frame, corners, ids = detect_aruco_markers(
       frame,
       aruco_dict,
       aruco_params,
       camera_matrix,
       dist_coeffs
   )
   ```

## Calibration File Compatibility

The calibration file from your previous OpenCV 4.5.5 build should work fine with this fix. The issue is not with the calibration data format but with the API changes in the ArUco module.

## How the Fix Works

The enhanced detector makes several improvements:

1. **Strict Parameter Configuration**:
   - Increased minimum marker perimeter threshold (prevents random small detections)
   - More accurate corner detection parameters
   - Higher error correction rate
   - Added minimum corner distance constraints
   - Added edge case handling for border proximity

2. **Geometric Validation**:
   - Verifies marker aspect ratio is close to square (0.7-1.3)
   - Checks that corner angles are approximately 90° (within 25° tolerance)
   - Enforces minimum size relative to image dimensions
   - Validates overall marker shape and proportions

3. **Fixed Pose Estimation**:
   - Replaces the deprecated `estimatePoseSingleMarkers` with manual handling via `solvePnP`
   - Correctly sets up 3D-2D point correspondences for accurate pose calculation
   - Properly handles rotation and translation vectors

4. **Robust Error Handling**:
   - Better handling of edge cases and potential detection errors
   - Fallbacks for API differences between OpenCV versions
   - Detailed logging about detection results and filtering

## Requirements

- OpenCV 4.8.0 with ArUco module
- NumPy
- OAK-D camera and DepthAI library
- Python 3.6+

This fix should help your ArUco detection work correctly with OpenCV 4.8.0 CUDA support while eliminating the false positive detections.
