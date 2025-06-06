# Flow-Enhanced Autonomous Precision Landing System for Quadcopters

This project implements a vision-based autonomous precision landing system for quadcopters using ArUco markers and optical flow. The system leverages a Jetson Orin Nano, OAK-D camera, HereFlow optical flow sensor, and ArduCopter 4.6 running on a CubePilot Orange Plus flight controller.

## System Architecture

The system uses the following hardware components:
- Jetson Orin Nano (onboard computer)
- OAK-D Camera (downward-facing stereo depth camera)
- HereFlow optical flow sensor (precision low-altitude positioning)
- CubePilot Orange Plus (flight controller running ArduCopter 4.6)
- Quadcopter airframe (X configuration)

The software stack includes:
- ArUco marker detection with OpenCV 4.10
- DepthAI for OAK-D camera interfacing
- Optical flow integration with ArduCopter
- Multi-sensor fusion for enhanced positioning
- MAVLink for drone communication and control
- Autonomous mission planning and execution
- Safety monitoring and emergency protocols
- QGroundControl mission integration

## Features

- **Autonomous Search**: Systematically search a 30×30 yard area for target markers
- **Marker Detection**: Real-time detection of ArUco markers using computer vision
- **Precision Loiter**: Center and hold position over detected target marker
- **Flow-Enhanced Final Approach**: Use optical flow for highly stable descent
- **Multi-Sensor Fusion**: Combine vision and flow data for reliable positioning
- **Precision Landing**: Land accurately on the target marker with cm-level precision
- **Adaptive Control**: Automatically adjust descent rate based on sensor quality
- **Safety Features**: Battery monitoring, altitude bounds, connection health checks, EKF monitoring
- **Enhanced Simulation**: Test and validate the system with simulated optical flow
- **QGC Mission Integration**: Integrate precision landing into QGroundControl missions
- **RTK GPS Acquisition**: Record precise coordinates at landing locations
- **UGV Coordination**: Optional handoff to UGV at landing locations

## Components

### 1. ArUco Marker Detection System

The `oak_d_aruco_6x6_detector.py` provides robust marker detection with:
- Long-range detection (up to 12m)
- Adaptive resolution and parameter selection
- Multi-scale detection
- Target prioritization and tracking

### 2. Basic Autonomous Mission Controller

The `autonomous_precision_landing.py` implements:
- Mission state machine with well-defined states
- MAVLink communication for flight control
- Coordinate transformation for precision positioning
- Safety monitoring and emergency protocols

### 3. Flow-Enhanced Autonomous Mission Controller

The `autonomous_precision_landing_with_flow.py` extends the basic controller with:
- HereFlow optical flow sensor integration
- Enhanced state machine with flow-optimized final approach phase
- Adaptive descent rate based on flow quality and altitude
- Multi-sensor fusion for reliable positioning
- Advanced EKF status monitoring
- Graceful degradation when sensor quality drops

### 4. QGC Mission Integration

The `qgc_mission_integration` module provides:
- Integration with QGroundControl mission plans
- Ability to detect ArUco markers during mission execution
- Precision landing capability during ongoing missions
- RTK GPS coordinate acquisition at landing sites
- Mission resumption after precision landing
- Comprehensive safety protections and error handling
- Optional UGV coordination at landing sites

### 5. Basic Simulation Environment

The `simulate_precision_landing.py` provides:
- Physics-based drone simulation
- ArUco marker detection simulation
- MAVLink control simulation
- 2D and 3D visualization
- Mission monitoring

### 6. Flow-Enhanced Simulation Environment

The `simulate_precision_landing_with_flow.py` extends the basic simulation with:
- Simulated optical flow sensor with configurable parameters
- EKF simulation for position estimation
- Flow quality visualization and monitoring
- Adaptive descent control simulation
- Enhanced telemetry and status monitoring

### 7. Gazebo Simulation Environment

The `gazebo_simulation` directory contains:
- Gazebo world files for precision landing scenarios
- ArUco marker models for gazebo
- Integration scripts for ROS and Gazebo
- SITL (Software In The Loop) support
- Detailed simulation instructions

## Installation

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/autonomous-precision-landing
cd autonomous-precision-landing

# Install Python dependencies
pip install numpy==1.26.4 opencv-contrib-python==4.5.5.62 depthai==2.24.0.0 pymavlink==2.4.43 scipy==1.15.2 matplotlib pyyaml requests
```

### 2. Set Up ArduCopter

Ensure your flight controller is running ArduCopter 4.6 or later with the following parameters:

#### Precision Landing Parameters
- `PLND_ENABLED = 1` (Enable precision landing)
- `PLND_TYPE = 1` (MAVLink landing target type)
- `PLND_EST_TYPE = 0` (Precision landing estimator type)
- `PLND_BUS = -1` (Landing sensor bus)
- `PLND_LAG = 0.02` (Precision landing lag time)
- `PLND_XY_DRIFT_MAX = 0.2` (Maximum horizontal drift during landing)
- `PLND_STRICT = 1` (Strict landing requirements)

#### Optical Flow Parameters (For Flow-Enhanced Version)
- `FLOW_TYPE = 6` (HereFlow sensor)
- `FLOW_ORIENT_YAW = 0` (Flow sensor orientation)
- `FLOW_POS_X = 0.0` (X position on vehicle)
- `FLOW_POS_Y = 0.0` (Y position on vehicle)
- `FLOW_POS_Z = 0.0` (Z position on vehicle)
- `EK3_SRC1_VELXY = 5` (Use optical flow for horizontal velocity)
- `LAND_SPEED = 30` (Slower landing speed in cm/s)

### 3. Set Up Network

For QGC Mission Integration with RTK GPS and UGV coordination:

- Set up a WiFi network between the drone and GCS
- Configure static IPs (drone: 192.168.2.2, GCS: 192.168.2.1)
- Ensure the RTK GPS server is running on the GCS
- Configure USB connection to the flight controller

For detailed setup instructions, see `qgc_mission_integration/utils/DRONE_SETUP.md`

## Usage

### Real Hardware Operation

#### Basic Version (ArUco-only)
```bash
# Run the basic autonomous landing system
python autonomous_precision_landing.py --target 5 --connection 'udp:192.168.2.1:14550'

# With video streaming to GCS
python autonomous_precision_landing.py --target 5 --stream --stream-ip 192.168.2.1

# For headless operation (e.g., over SSH)
python autonomous_precision_landing.py --target 5 --headless
```

#### Flow-Enhanced Version
```bash
# Run the flow-enhanced landing system
python autonomous_precision_landing_with_flow.py --target 5 --connection 'udp:192.168.2.1:14550'

# With custom final approach altitude
python autonomous_precision_landing_with_flow.py --target 5 --final-approach-alt 1.2

# With video streaming and headless mode
python autonomous_precision_landing_with_flow.py --target 5 --stream --headless
```

#### QGC Mission Integration
```bash
# Run QGC Mission Integrator with a mission file
cd qgc_mission_integration
python main.py --mission ../misc/aruco_finder_test_flight_long_beach.plan --headless

# Specify connection and config
python main.py --mission your_mission.plan --connection '/dev/ttyACM0' --config config.yaml

# Test connections before mission
cd utils
./test_connections.py

# Set up drone environment
./drone_setup.sh
```

### Simulation

#### Basic Simulation
```bash
# Run basic simulation with default settings
python simulate_precision_landing.py 

# With custom parameters
python simulate_precision_landing.py --target 5 --wind-speed 2.0 --search-alt 15
```

#### Flow-Enhanced Simulation
```bash
# Run flow-enhanced simulation with default settings
python simulate_precision_landing_with_flow.py

# With custom flow parameters
python simulate_precision_landing_with_flow.py --flow-quality 150 --flow-max-alt 3.0 --flow-noise 0.3

# With higher wind for testing robustness
python simulate_precision_landing_with_flow.py --wind-speed 2.0 --verbose
```

#### Gazebo Simulation
```bash
# Start the Gazebo simulation environment
cd gazebo_simulation
./scripts/setup_gazebo_native.py

# Run precision landing mission in Gazebo
./scripts/gazebo_precision_landing_mission_native.py
```

## Mission Workflow

### Basic Mission (ArUco-only)
1. **Initialization**: Connect to MAVLink, initialize camera and detection systems
2. **Takeoff**: Ascend to search altitude
3. **Search**: Execute search pattern over the target area
4. **Target Acquisition**: Detect and validate the target marker
5. **Precision Loiter**: Center and maintain position over the target
6. **Precision Landing**: Descend while maintaining marker in center of FOV
7. **Mission Complete**: Safely landed on target

### Flow-Enhanced Mission
1. **Initialization**: Connect to MAVLink, initialize camera, flow sensor, and detection systems
2. **Takeoff**: Ascend to search altitude
3. **Search**: Execute search pattern over the target area
4. **Target Acquisition**: Detect and validate the target marker
5. **Precision Loiter**: Center and maintain position over the target
6. **Final Approach**: Enter flow-controlled descent mode at defined altitude (default: 1.0m)
   - Adaptive descent rate based on flow quality and marker visibility
   - EKF monitoring for position estimation quality
7. **Precision Landing**: Final landing phase with flow-optimized control
8. **Mission Complete**: Safely landed on target with high precision

### QGC Integrated Mission
1. **Initialization**: Connect to MAVLink, initialize camera, detection systems, and mission monitor
2. **Mission Execution**: Execute the QGC mission loaded from a .plan file
3. **Monitoring**: Monitor for ArUco markers during mission execution
4. **Target Detection**: When a marker is detected, validate it
5. **Mission Interruption**: Safely interrupt the mission
6. **Precision Landing**: Perform precision landing on the marker
7. **RTK Acquisition**: Record RTK GPS coordinates at the landing site
8. **UGV Coordination** (Optional): Coordinate with UGV at landing site
9. **Mission Resumption**: Resume the original mission
10. **Mission Complete**: Return to launch or continue with mission

## Safety Features

### Basic Safety Features
- **Battery Monitoring**: Abort mission if battery level gets too low
- **Altitude Limits**: Maintain safe minimum and maximum altitudes
- **Connection Health**: Monitor MAVLink heartbeat and connection status
- **GPS Fallback**: Use GPS position if vision-based positioning fails
- **Emergency Protocols**: RTL or emergency land if critical issues occur

### Enhanced Safety Features (Flow-Enhanced Version)
- **EKF Status Monitoring**: Monitor Extended Kalman Filter health for reliable positioning
- **Flow Quality Assessment**: Monitor optical flow quality and adapt behavior accordingly
- **Adaptive Descent Rate**: Slow down or pause descent when sensor quality degrades
- **Sensor Fusion Prioritization**: Automatically prioritize the most reliable sensor data
- **Graceful Degradation**: Fall back to simpler control modes when sensor data is unreliable
- **Position Variance Monitoring**: Track position estimate uncertainty and adapt accordingly

### QGC Integration Safety Features
- **Completion State Protection**: Prevents re-attempting precision landing after completion
- **Altitude-Based Protection**: Blocks landing attempts above a certain altitude (e.g., during RTL)
- **Mission Status Guard**: Prevents unwanted state transitions after mission completion
- **Flight Mode Protection**: Blocks precision landing during RTL mode
- **Attempt Limiting**: Configurable limit on precision landing attempts per mission
- **Comprehensive Logging**: Detailed event logging for post-mission analysis

## Simulation

### Basic Simulation
The basic simulation environment allows testing the autonomous landing system without real hardware:

- Simulates drone flight dynamics and physics
- Simulates ArUco marker detection
- Simulates MAVLink communication
- Provides 2D and 3D visualization
- Monitors mission state and progress

### Flow-Enhanced Simulation
The enhanced simulation adds optical flow simulation for comprehensive testing:

- Simulates optical flow sensor with configurable parameters
- Models flow quality degradation with altitude (realistic behavior)
- Simulates EKF position estimation
- Visualizes flow quality and EKF variances in real-time
- Tests adaptive control strategies based on sensor quality
- Provides comprehensive diagnostics for system performance

### Gazebo Simulation
The Gazebo simulation provides a realistic 3D environment:

- Full 3D physics simulation
- Realistic camera and sensor models
- ArduCopter SITL integration
- ROS integration for monitoring and control
- Reproducible test scenarios

## ArUco Markers

The system uses 6x6 ArUco markers from the DICT_6X6_250 dictionary. The default marker size is 12 inches (0.3048m), which provides good detection range while remaining portable.

Generate markers using:
```bash
python aruco/generate_aruco_markers.py 0 10 500
```
This generates markers with IDs 0-9, each 500x500 pixels. Print these markers and place the target marker in the search area.

## QGC Mission Integration Network Architecture

The QGC Mission Integration system uses the following network architecture:

```
┌───────────────────────┐           ┌──────────────────────┐
│                       │           │                      │
│      DRONE (UAV)      │           │  Ground Control      │
│    192.168.2.2        │◄─────────►│  Station (GCS)       │
│                       │    WiFi   │  192.168.2.1         │
│  ┌─────────────────┐  │           │                      │
│  │ Jetson Orin Nano│  │           │ ┌────────────────┐  │
│  │ - Python code   │  │           │ │ QGroundControl │  │
│  │ - ArUco detector│  │           │ │ - Mission ctrl │  │
│  └───────┬─────────┘  │           │ └────────────────┘  │
│          │            │           │         ▲           │
│          │ USB        │           │         │           │
│          ▼            │           │         │           │
│  ┌─────────────────┐  │           │ ┌───────┴────────┐  │
│  │ CubePilot Orange│  │           │ │ RTK GPS Server │  │
│  │ - ArduCopter 4.6│  │           │ │ - Coordinates  │  │
│  └─────────────────┘  │           │ └────────────────┘  │
│                       │           │                      │
└───────────────────────┘           └──────────────────────┘
```

For detailed setup instructions, see `qgc_mission_integration/utils/DRONE_SETUP.md`

## License

This project is open source and available under the MIT License.

## Acknowledgments

- The OpenCV team for ArUco implementation
- Luxonis for DepthAI and OAK-D camera
- ArduPilot for ArduCopter firmware
- HolyBro for the HereFlow optical flow sensor
