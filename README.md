# Autonomous Precision Landing System for Quadcopters

This project implements a vision-based autonomous precision landing system for quadcopters using ArUco markers. The system leverages a Jetson Orin Nano, OAK-D camera, and ArduCopter 4.6 running on a CubePilot Orange Plus flight controller.

## System Architecture

The system uses the following hardware components:
- Jetson Orin Nano (onboard computer)
- OAK-D Camera (downward-facing stereo depth camera)
- CubePilot Orange Plus (flight controller running ArduCopter 4.6)
- Quadcopter airframe (X configuration)

The software stack includes:
- ArUco marker detection with OpenCV 4.10
- DepthAI for OAK-D camera interfacing
- MAVLink for drone communication and control
- Autonomous mission planning and execution
- Safety monitoring and emergency protocols

## Features

- **Autonomous Search**: Systematically search a 30×30 yard area for target markers
- **Marker Detection**: Real-time detection of ArUco markers using computer vision
- **Precision Loiter**: Center and hold position over detected target marker
- **Precision Landing**: Land accurately on the target marker
- **Safety Features**: Battery monitoring, altitude bounds, connection health checks
- **Simulation Environment**: Test and validate the system without hardware

## Components

### 1. ArUco Marker Detection System

The `oak_d_aruco_6x6_detector.py` provides robust marker detection with:
- Long-range detection (up to 12m)
- Adaptive resolution and parameter selection
- Multi-scale detection
- Target prioritization and tracking

### 2. Autonomous Mission Controller

The `autonomous_precision_landing.py` implements:
- Mission state machine with well-defined states
- MAVLink communication for flight control
- Coordinate transformation for precision positioning
- Safety monitoring and emergency protocols

### 3. Simulation Environment

The `simulate_precision_landing.py` provides:
- Physics-based drone simulation
- ArUco marker detection simulation
- MAVLink control simulation
- 2D and 3D visualization
- Mission monitoring

## Installation

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/autonomous-precision-landing
cd autonomous-precision-landing

# Install Python dependencies
pip install numpy==1.26.4 opencv-contrib-python==4.5.5.62 depthai==2.24.0.0 pymavlink==2.4.43 scipy==1.15.2 matplotlib
```

### 2. Set Up ArduCopter

Ensure your flight controller is running ArduCopter 4.6 or later with the following parameters:
- `PLND_ENABLED = 1` (Enable precision landing)
- `PLND_TYPE = 1` (MAVLink landing target type)
- `PLND_EST_TYPE = 0` (Precision landing estimator type)
- `PLND_BUS = -1` (Landing sensor bus)
- `PLND_LAG = 0.02` (Precision landing lag time)
- `PLND_XY_DRIFT_MAX = 0.2` (Maximum horizontal drift during landing)
- `PLND_STRICT = 1` (Strict landing requirements)

## Usage

### Real Hardware Operation

```bash
# Run the autonomous landing system
python autonomous_precision_landing.py --target 5 --connection 'udp:192.168.2.1:14550'

# With video streaming to GCS
python autonomous_precision_landing.py --target 5 --stream --stream-ip 192.168.2.1

# For headless operation (e.g., over SSH)
python autonomous_precision_landing.py --target 5 --headless
```

### Simulation

```bash
# Run simulation with default settings
python simulate_precision_landing.py 

# With custom parameters
python simulate_precision_landing.py --target 5 --wind-speed 2.0 --search-alt 15
```

## Mission Workflow

1. **Initialization**: Connect to MAVLink, initialize camera and detection systems
2. **Takeoff**: Ascend to search altitude
3. **Search**: Execute search pattern over the target area
4. **Target Acquisition**: Detect and validate the target marker
5. **Precision Loiter**: Center and maintain position over the target
6. **Precision Landing**: Descend while maintaining marker in center of FOV
7. **Mission Complete**: Safely landed on target

## Safety Features

- **Battery Monitoring**: Abort mission if battery level gets too low
- **Altitude Limits**: Maintain safe minimum and maximum altitudes
- **Connection Health**: Monitor MAVLink heartbeat and connection status
- **GPS Fallback**: Use GPS position if vision-based positioning fails
- **Emergency Protocols**: RTL or emergency land if critical issues occur

## Simulation

The simulation environment allows testing the autonomous landing system without real hardware:

- Simulates drone flight dynamics and physics
- Simulates ArUco marker detection
- Simulates MAVLink communication
- Provides 2D and 3D visualization
- Monitors mission state and progress

## ArUco Markers

The system uses 6x6 ArUco markers from the DICT_6X6_250 dictionary. The default marker size is 12 inches (0.3048m), which provides good detection range while remaining portable.

Generate markers using:
```bash
python aruco/generate_aruco_markers.py 0 10 500
```
This generates markers with IDs 0-9, each 500x500 pixels. Print these markers and place the target marker in the search area.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- The OpenCV team for ArUco implementation
- Luxonis for DepthAI and OAK-D camera
- ArduPilot for ArduCopter firmware
