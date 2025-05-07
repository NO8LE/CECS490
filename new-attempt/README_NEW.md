# OAK-D ArUco Detection System for OpenCV 4.10

A streamlined ArUco marker detection system for the OAK-D camera with OpenCV 4.10 compatibility. This system provides robust marker detection, tracking, and pose estimation with a focus on performance and accuracy.

## Features

- **OpenCV 4.10 Compatibility**: Properly handles API changes in OpenCV 4.10
- **Performance Optimization**: Adaptive parameters and CUDA acceleration
- **Target Selection**: Ability to specify and track specific markers
- **Range Calculation**: Accurate distance measurement and 3D positioning
- **Calibration Management**: Loading, creation, and management of camera calibration
- **Robust Validation**: Prevents false positives in marker detection
- **Real-time Visualization**: Visual feedback with performance statistics

## System Requirements

- Python 3.8 or higher
- OpenCV 4.10.0 with ArUco module and CUDA support (optional)
- DepthAI library for OAK-D camera
- NumPy for numerical operations
- OAK-D camera (or compatible Luxonis device)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/oak-d-aruco-system.git
   cd oak-d-aruco-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your OAK-D camera is connected and recognized by your system.

## Usage

### Command-Line Interface

The system can be run directly from the command line with various configuration options:

```bash
python oak_d_aruco_system.py [OPTIONS]
```

#### Options:

- `--target`, `-t`: Target marker ID to highlight
- `--resolution`, `-r`: Resolution mode (`low`, `medium`, `high`, or `adaptive`)
- `--cuda`, `-c`: Enable CUDA acceleration if available
- `--performance`, `-p`: Enable performance optimizations
- `--marker-size`, `-m`: Marker size in meters (default: 0.3048m / 12 inches)
- `--calib-dir`: Directory for calibration files
- `--no-validation`: Disable marker validation (not recommended)
- `--headless`: Run in headless mode (no GUI)

#### Examples:

Basic usage:
```bash
python oak_d_aruco_system.py
```

Track a specific marker with performance optimizations:
```bash
python oak_d_aruco_system.py --target 5 --performance
```

High-resolution mode with CUDA acceleration:
```bash
python oak_d_aruco_system.py --resolution high --cuda
```

### Programmatic Usage

The system can also be used programmatically in your own Python code:

```python
from oak_d_aruco_system import OakDArUcoSystem

# Create system with custom configuration
system = OakDArUcoSystem(
    target_id=5,
    resolution_mode="adaptive",
    use_cuda=True,
    performance_mode=True
)

# Start the system
system.start()

# Process frames in a loop
while True:
    # Process a single frame
    system.process_frame()
    
    # Get information about tracked markers
    markers = system.get_tracked_markers()
    
    # Get performance statistics
    stats = system.get_performance_stats()
    
    # Do something with the marker information
    for marker_id, info in markers.items():
        print(f"Marker {marker_id}: Distance = {info.get('distance', 'unknown')}m")
    
    # Check for exit condition
    if should_exit():
        break

# Stop the system
system.stop()
```

## System Components

The system consists of several modular components that work together:

### 1. ArUco Detector (`aruco_detector.py`)

Handles ArUco marker detection with OpenCV 4.10 compatibility. Features include:
- Proper initialization of ArUco dictionary and detector parameters
- Robust marker validation to prevent false positives
- CUDA acceleration with automatic fallback to CPU processing
- Adaptive parameter selection based on estimated distance
- Comprehensive error handling for API changes

### 2. OAK-D Camera Interface (`oak_d_camera.py`)

Provides an interface for the OAK-D camera using DepthAI. Features include:
- Setup of RGB and depth streams
- Spatial location calculator for 3D positioning
- Adaptive resolution selection
- Frame acquisition and processing

### 3. Marker Tracker (`marker_tracker.py`)

Tracks markers across frames and implements target selection. Features include:
- Temporal filtering for stable tracking
- Confidence scoring based on detection consistency
- Target marker selection and prioritization
- Visualization with targeting guidance

### 4. Calibration Manager (`calibration_manager.py`)

Manages camera calibration data for accurate pose estimation. Features include:
- Loading calibration from files
- Creating default calibration when needed
- Calibration from CharucoBoard or checkerboard
- Import/export to different formats

### 5. Main System (`oak_d_aruco_system.py`)

Integrates all components into a complete system. Features include:
- Unified interface for all components
- Main processing loop
- Performance monitoring and optimization
- Command-line interface

## Camera Calibration

For accurate pose estimation, it's recommended to calibrate your camera:

1. Print a ChArUco board:
   ```bash
   python aruco/generate_charuco_board.py
   ```

2. Capture multiple images of the board from different angles.

3. Run the calibration script:
   ```bash
   python aruco/calibrate_camera.py
   ```

4. The calibration will be saved to `aruco/camera_calibration/calibration.npz`.

## Troubleshooting

### Camera Not Detected

- Ensure the OAK-D camera is properly connected
- Check that the DepthAI library is installed correctly
- Try reconnecting the camera or restarting your computer

### Poor Detection Performance

- Ensure adequate lighting conditions
- Use the appropriate resolution for your distance
- Enable performance mode for better real-time performance
- Consider using CUDA acceleration if available

### Inaccurate Pose Estimation

- Calibrate your camera for better accuracy
- Ensure the marker size is set correctly
- Use larger markers for better detection at distance
- Enable validation to prevent false positives

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV team for the ArUco implementation
- Luxonis for the OAK-D camera and DepthAI library