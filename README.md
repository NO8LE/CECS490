# OAK-D ArUco 6x6 Marker Detector for Drone Applications

This Python project uses the Luxonis OAK-D camera to detect 6x6 ArUco markers, calculate their 3D position, and visualize the results. It has been optimized for drone-based detection at ranges from 0.5m to 12m with 12-inch (0.3048m) markers.

This repository is a submodule of the CECS490-Final-Project Organization:
https://github.com/CECS490-Final-Project

## Key Features

- **Long-range detection**: Optimized for detecting markers at distances up to 12m
- **Adaptive processing**: Automatically adjusts parameters based on estimated distance
- **Jetson Orin Nano optimization**: Leverages hardware acceleration for real-time performance
- **CharucoBoard calibration**: More robust and accurate camera calibration
- **Multi-scale detection**: Improves marker detection at varying distances
- **Performance monitoring**: Adapts processing based on system capabilities
- **Target prioritization**: Ability to track and prioritize a specific marker among many
- **Visual targeting guidance**: Provides directional guidance to center on a target marker
- **Real-time video streaming**: Stream annotated frames to GCS over Wi-Fi using H.264/RTP

## Requirements

- Luxonis OAK-D camera
- Python 3.6 or higher
- OpenCV with ArUco support
- DepthAI library
- NumPy
- SciPy

## Installation

1. Run the installation script to check and install required packages:

```bash
# Check and install dependencies
python3 install_dependencies.py
```

This script will:
- Check if all required packages are installed
- Install specific versions of packages known to work together
- Verify OpenCV ArUco module compatibility
- Handle NumPy compatibility issues
- Provide Jetson-specific instructions when needed

You can also force reinstallation of all packages:

```bash
python3 install_dependencies.py --force
```

2. Alternatively, install the required Python packages manually:

```bash
# Install specific versions known to work together
pip install numpy==1.26.4 opencv-contrib-python==4.5.5.62 depthai==2.24.0.0 scipy==1.15.2
```

3. Connect your OAK-D camera to your computer.

4. Clone this repository or download the scripts.

## Setup Instructions

### 1. Generate a CharucoBoard for Calibration

```bash
python3 generate_charuco_board_for_drone.py 5 7 4000 --high-contrast
```

This will create a high-contrast CharucoBoard with 5x7 squares in a 4000x4000 pixel image, saved to the `calibration_patterns` directory.

### 2. Print and Measure the CharucoBoard

1. Print the generated CharucoBoard at the largest size possible
2. Measure the actual size of the squares on your printed board in meters
3. Mount the board on a flat surface for calibration

### 3. Calibrate the Camera

```bash
python3 calibrate_camera.py --charuco 5 7 0.12 --drone
```

Replace `0.12` with the actual measured square size in meters. Move the board around to capture different angles and positions during calibration. For drone applications, be sure to capture frames at various distances (0.5m to 12m).

Alternatively, you can use a traditional chessboard pattern:

```bash
# Calibrate using the default 9x6 chessboard pattern
python3 calibrate_camera.py --chessboard 9 6
```

The calibration file is stored in the `camera_calibration` directory as `calibration.npz` and will be automatically used by the ArUco marker detector script.

### 4. Generate ArUco Markers

The script is configured to detect 6x6 ArUco markers from the DICT_6X6_250 dictionary. You can generate these markers using the included generator script:

```bash
# Generate markers with IDs 0-9 (default)
python3 generate_aruco_markers.py

# Generate markers with IDs 0-5, each 500x500 pixels
python3 generate_aruco_markers.py 0 5 500
```

The generated markers will be saved in the `aruco_markers` directory. Print these markers and measure their physical size accurately.

For drone applications, the default marker size is set to 12 inches (0.3048 meters). If your markers are a different size, update the `MARKER_SIZE` constant in the script.

### 5. Run the Detector

```bash
python3 oak_d_aruco_6x6_detector.py --resolution adaptive --cuda --performance
```

Options:
- `--target, -t MARKER_ID`: Specify a target marker ID to highlight and track
- `--resolution, -r RESOLUTION`: Specify resolution (low, medium, high, adaptive)
- `--cuda, -c`: Enable CUDA acceleration if available
- `--performance, -p`: Enable high performance mode on Jetson
- `--stream, -st`: Enable video streaming over RTP/UDP
- `--stream-ip IP`: IP address to stream to (default: 192.168.2.1)
- `--stream-port PORT`: Port to stream to (default: 5600)
- `--stream-bitrate BITRATE`: Streaming bitrate in bits/sec (default: 4000000)
- `--headless`: Run in headless mode (no GUI windows, for SSH sessions)

For targeting a specific marker:
```bash
python3 oak_d_aruco_6x6_detector.py --target 5
```
This will prioritize marker ID 5, providing visual guidance to center on it.

For streaming video to the GCS:
```bash
python3 oak_d_aruco_6x6_detector.py --stream --stream-ip 192.168.2.1
```
This will stream the annotated video feed from the Jetson to the GCS.

For headless operation over SSH (without X11 forwarding):
```bash
python3 oak_d_aruco_6x6_detector.py --headless
```
This will run without attempting to create GUI windows, preventing crashes on SSH sessions.

For headless operation with streaming (common for drone deployments):
```bash
python3 oak_d_aruco_6x6_detector.py --headless --stream --stream-ip 192.168.2.1
```
This allows you to run the script over SSH and still view the processed video on another machine.

## SSH and Headless Operation

When connecting to the Jetson over SSH, you need to use the `--headless` flag to prevent the script from attempting to create GUI windows, which would cause crashes in environments without a display server:

```bash
# Run the detector headlessly over SSH
python3 oak_d_aruco_6x6_detector.py --headless

# For convenience, use the start_streaming.sh helper script
./video_streaming/start_streaming.sh --headless
```

The `--headless` flag disables all GUI operations (cv2.imshow/waitKey) but still processes frames and performs marker detection. When combined with the `--stream` flag, you can view the processed video on another machine:

```bash
# Run headlessly but still view the results via streaming
python3 oak_d_aruco_6x6_detector.py --headless --stream --stream-ip 192.168.2.1
```

**Note:** The `--headless` and `--stream` flags serve different purposes and need to be specified separately if you want both behaviors.

## Performance Optimization

### Resolution Profiles

- **Low** (640x400): For close markers (0.5-3m)
- **Medium** (1280x720): For mid-range markers (3-8m)
- **High** (1920x1080): For distant markers (8-12m)

The detector automatically selects the appropriate resolution based on the estimated distance when using the `adaptive` resolution mode.

### Detection Profiles

The detector automatically adjusts detection parameters based on the estimated distance:

- **Close** (0.5-3m): Optimized for nearby markers
- **Medium** (3-8m): Balanced parameters
- **Far** (8-12m): Optimized for distant markers

### Jetson Orin Nano Optimization

When using the `--performance` flag, the detector will:

1. Set the Jetson to maximum performance mode during detection
2. Utilize hardware acceleration where available
3. Automatically adjust processing based on system load
4. Reset to power-saving mode when exiting

## Compatibility Notes

### OpenCV ArUco API

The scripts have been updated to work with newer versions of OpenCV (4.5.0+) that use a different ArUco API. If you encounter errors like:

```
AttributeError: module 'cv2.aruco' has no attribute 'Dictionary_get'
```

Run the `install_dependencies.py` script to check your OpenCV installation and get recommendations for fixing compatibility issues.

### NumPy 2.x Incompatibility

OpenCV is not compatible with NumPy 2.x. If you encounter errors like:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.3 as it may crash.
```

or

```
AttributeError: _ARRAY_API not found
```

You need to downgrade NumPy to a version below 2.0. The `install_dependencies.py` script will automatically install a compatible version of NumPy (1.26.4).

You can also manually install a compatible version:

```bash
pip install numpy==1.26.4
```

### Missing ArUco Module

If you encounter errors about the ArUco module not being found despite having opencv-contrib-python installed:

```
Error: OpenCV ArUco module not found.
Please install opencv-contrib-python:
  pip install opencv-contrib-python
```

The scripts now include robust ArUco module import handling that tries multiple approaches:
1. Direct access via `cv2.aruco`
2. Importing from `cv2` with `from cv2 import aruco`
3. Jetson-specific import paths

### Jetson Platform Considerations

When running on NVIDIA Jetson platforms:

1. OpenCV is typically pre-installed system-wide with CUDA support
2. Use system packages when possible:
   ```bash
   sudo apt-get install python3-opencv python3-numpy
   ```

3. If using virtual environments, create them with access to system packages:
   ```bash
   python3 -m venv --system-site-packages my_env
   ```

4. The scripts include Jetson-specific import paths and compatibility checks

## Troubleshooting

### Marker Not Detected at Long Range

1. Ensure the marker is well-lit and not obscured
2. Try using the `--resolution high` option
3. Increase the physical size of the marker if possible
4. Ensure the camera is properly calibrated using the CharucoBoard

### Poor Performance

1. Use the `--cuda` flag if your system supports CUDA
2. Try a lower resolution with `--resolution low` or `--resolution medium`
3. Ensure the Jetson is not overheating (check thermal throttling)

### Calibration Issues

1. Make sure the CharucoBoard is printed at a large size
2. Ensure good lighting conditions during calibration
3. Capture frames at various distances for drone applications
4. Measure the square size accurately

## Technical Details

### Marker Size

The system is optimized for 12-inch (0.3048m) markers. If using different sized markers, update the `MARKER_SIZE` constant in `oak_d_aruco_6x6_detector.py`.

### Camera Calibration

The CharucoBoard calibration provides more robust and accurate results than traditional chessboard calibration, especially for varying distances. The calibration data is saved to `camera_calibration/calibration.npz` and is automatically loaded by the detector.

### Multi-scale Detection

The detector uses a multi-scale approach to improve marker detection:
1. First attempts detection on the full image
2. If no markers are found, tries with enhanced parameters
3. If still no markers, tries with a scaled version of the image

This approach significantly improves detection reliability at varying distances.

### Target Tracking and Prioritization

When multiple markers are in the field of view, the system can prioritize a specific marker:

1. **Target Selection**: Use the `--target` flag to specify a marker ID to track
2. **Visual Highlighting**: The target marker is highlighted with:
   - Red bounding box (vs. yellow for other markers)
   - "TARGET" label
   - Crosshair overlay
   - Larger corner points
3. **Spatial Prioritization**: The spatial calculator focuses on the target marker
4. **Targeting Guidance**: Visual indicators show how to center the drone on the target:
   - Direction arrow pointing to the target
   - Text instructions (UP, DOWN, LEFT, RIGHT)
   - "TARGET CENTERED" confirmation when aligned

This functionality is particularly useful for drone navigation and autonomous targeting applications.

## Video Streaming

The system supports real-time video streaming from the Jetson to a Ground Control Station (GCS) over Wi-Fi:

- **H.264 encoding**: Uses software-based x264 encoding (compatible with all systems)
- **RTP over UDP**: Low-latency transport protocol suitable for real-time applications 
- **Up to 30 FPS**: Configurable frame rate with adaptive quality
- **Multiple viewing options**: Support for VLC, ffplay, or custom OpenCV-based viewer

To enable streaming:

```bash
python3 oak_d_aruco_6x6_detector.py --stream --stream-ip 192.168.2.1
```

On the GCS side, you can view the stream using the included receiver:

```bash
python3 gcs_video_receiver.py
```

### Network Configuration

The system is configured for the following network setup:
- Jetson (UAV): 192.168.2.2
- GCS: 192.168.2.1

A network configuration check script is provided to verify connectivity:

```bash
./check_network_config.sh --ip 192.168.2.1
```

For detailed instructions on setting up and using the video streaming functionality, see [VIDEO_STREAMING.md](VIDEO_STREAMING.md) or the quick start guide [STREAMING_QUICK_START.md](STREAMING_QUICK_START.md).

## License

This project is open source and available under the MIT License.
