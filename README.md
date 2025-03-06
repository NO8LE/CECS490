This repository is a submodule of the 
CECS490-Final-Project  Organization:
https://github.com/CECS490-Final-Project


# OAK-D ArUco 6x6 Marker Detector

This Python script uses the Luxonis OAK-D camera to detect 6x6 ArUco markers, calculate their 3D position, and visualize the results.

## Features

- Detects 6x6 ArUco markers using the OpenCV ArUco library
- Calculates the 3D position and orientation of detected markers
- Uses the OAK-D stereo depth capabilities for accurate spatial location
- Displays RGB and depth visualization with marker information
- Automatically tracks markers and updates the spatial calculation ROI
- Handles camera calibration for accurate pose estimation

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

## Usage

1. Run the script:

```bash
python3 oak_d_aruco_6x6_detector.py
```

Or use the executable directly:

```bash
./oak_d_aruco_6x6_detector.py
```

2. The script will open two windows:
   - RGB: Shows the color camera feed with detected markers, their IDs, orientation, and 3D position
   - Depth: Shows the depth map from the stereo cameras

3. Press 'q' to exit the program.

## Camera Calibration

For better accuracy, you should calibrate your camera using the included calibration script:

```bash
# Calibrate using the default 9x6 chessboard pattern
./calibrate_camera.py

# Calibrate using a 7x5 chessboard pattern
./calibrate_camera.py 7 5
```

You'll need to print a chessboard pattern (you can find many online) and hold it in front of the camera at different angles and positions. The script will capture frames when the chessboard is detected and use them to calculate the camera calibration parameters.

The calibration file is stored in the `camera_calibration` directory as `calibration.npz` and will be automatically used by the ArUco marker detector script.

If you don't calibrate your camera, the detector will use default calibration values, which may not be accurate for your specific camera.

## ArUco Markers

The script is configured to detect 6x6 ArUco markers from the DICT_6X6_250 dictionary. You can generate these markers using the included generator script:

```bash
# Generate markers with IDs 0-9 (default)
./generate_aruco_markers.py

# Generate markers with IDs 0-5, each 500x500 pixels
./generate_aruco_markers.py 0 5 500
```

The generated markers will be saved in the `aruco_markers` directory. Print these markers and measure their physical size accurately.

You can also generate markers using online tools like:
- [ArUco Marker Generator](https://chev.me/arucogen/)

Make sure to select the 6x6 dictionary with 250 markers.

The default marker size is set to 5 cm (0.05 meters). If your markers are a different size, update the `MARKER_SIZE` constant in the script.

## Customization

You can modify the following parameters in the script:

- `ARUCO_DICT_TYPE`: Change the ArUco dictionary type
- `MARKER_SIZE`: Set the physical size of your markers in meters
- Camera resolution and FPS in the `initialize_pipeline` method
- Depth thresholds and ROI size in various methods

## Troubleshooting

- If the script can't find your OAK-D camera, make sure it's properly connected and recognized by your system.
- If markers are detected but the pose estimation is inaccurate, try calibrating your camera.
- If the depth information is noisy, try adjusting the depth thresholds in the script.

## License

This project is open source and available under the MIT License.
