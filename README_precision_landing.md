# Autonomous Precision Landing System

This system enables a quadcopter with a Jetson Orin Nano, OAK-D camera, and CubePilot Orange+ running ArduCopter 4.6 to autonomously locate and land on a specific ArUco marker.

## Overview

The autonomous precision landing system implements a complete mission flow:

1. **Takeoff**: The drone takes off to a search altitude
2. **Search**: A lawnmower search pattern is executed to locate the target ArUco marker
3. **Validation**: Multiple detections confirm the marker's identity and position
4. **Precision Loiter**: The drone centers above the marker and stabilizes
5. **Final Approach**: Controlled descent while maintaining position over the marker
6. **Precision Landing**: Final landing phase with optical flow assistance

## Components

- **mavlink_controller.py**: MAVLink interface for vehicle control and state monitoring
- **autonomous_precision_landing_mission.py**: Main mission controller
- **oak_d_aruco_wrapper410.py**: Wrapper for OAK-D camera ArUco detection
- **opencv410_aruco_fix.py**: Fixes for OpenCV 4.10 ArUco implementation

## Requirements

- Hardware:
  - Jetson Orin Nano onboard computer
  - OAK-D camera
  - CubePilot Orange+ flight controller
  - Optical flow sensor (optional but recommended)
  
- Software:
  - Python 3.8+
  - PyMAVLink
  - OpenCV 4.x (with ArUco support)
  - DepthAI library (for OAK-D camera)
  - NumPy
  - PyYAML

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/precision-landing.git
cd precision-landing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Connect your OAK-D camera and ensure it's detected:
```bash
python -c "import depthai; print(depthai.__version__)"
```

4. Connect to your CubePilot flight controller (typically via USB or UART)

## Usage

### Basic Usage

Run the mission with default settings:

```bash
python autonomous_precision_landing_mission.py
```

### Command Line Options

```bash
python autonomous_precision_landing_mission.py --target-id 5 --search-alt 15.0 --device /dev/ttyACM0 --verbose
```

- `--target-id`: ID of the target ArUco marker (default: 5)
- `--search-alt`: Search altitude in meters (default: 15.0)
- `--device`: MAVLink connection string (default: /dev/ttyACM0)
- `--simulation`: Run in simulation mode without hardware
- `--config`: Path to custom YAML configuration file
- `--verbose`: Enable detailed logging

### Custom Configuration

Create a YAML configuration file for custom settings:

```yaml
# config.yaml
search_altitude: 10.0
landing_start_altitude: 5.0
final_approach_altitude: 1.5
search_area_size: 20.0
target_marker_id: 3
marker_size: 0.3048  # 12 inches in meters
```

Then run with:
```bash
python autonomous_precision_landing_mission.py --config config.yaml
```

## ArUco Marker Preparation

1. Generate ArUco markers using the included script:
```bash
python aruco/generate_aruco_markers.py --dictionary 6x6_250 --id 5 --size 300
```

2. Print the marker at a size of at least 30cm x 30cm (12in x 12in) on non-reflective material
3. Place the marker on a flat, clear area for landing

## Camera Calibration

For best precision, calibrate your OAK-D camera:

1. Print the calibration pattern:
```bash
python aruco/generate_charuco_board.py
```

2. Run the calibration script:
```bash
python aruco/calibrate_camera.py
```

The calibration will be saved to `aruco/camera_calibration/calibration.npz`.

## Safety Considerations

- Always have a safety pilot ready with a manual controller
- Start with higher altitudes and gradually reduce as confidence in the system grows
- Ensure landing area is clear of obstacles
- Monitor battery voltage - the system will abort if voltage falls below threshold
- Test in simulation or controlled environment before real-world deployment

## Troubleshooting

- **No markers detected**: Check camera is working, lighting conditions, and marker size
- **MAVLink connection issues**: Verify connection string, baudrate, and cable connection
- **Inconsistent position hold**: Check for wind, ensure EKF is healthy, verify camera calibration
- **Landing accuracy problems**: Consider recalibrating camera, check marker size, review optical flow quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ArduPilot team for ArduCopter firmware
- OpenCV team for ArUco marker implementation
- Luxonis for OAK-D camera and DepthAI library
