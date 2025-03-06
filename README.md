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

1. Install the required Python packages:

```bash
# Install dependencies individually
pip install opencv-python numpy scipy depthai

# Or use the provided requirements.txt file
pip install -r requirements.txt
```

2. Connect your OAK-D camera to your computer.

3. Clone this repository or download the scripts.

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
