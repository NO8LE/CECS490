# QGroundControl Mission Integration for Precision Landing

This module provides integration between QGroundControl missions, ArUco marker detection, and precision landing capabilities. It enables a drone to execute a QGC mission, detect a target ArUco marker during the mission, perform a precision landing on the marker, record RTK GPS coordinates, and then resume the mission.

## Features

- **QGroundControl Mission Integration**: Works with standard .plan mission files from QGroundControl
- **ArUco Marker Detection**: Detects ArUco markers using OAK-D camera during mission execution
- **Precision Landing**: Executes precise landing on detected ArUco markers
- **RTK GPS Coordinate Acquisition**: Records accurate RTK GPS coordinates at landing site
- **Mission Resumption**: Continues mission after precision landing and RTK GPS acquisition
- **Safety Monitoring**: Comprehensive safety checks and emergency protocols
- **Video Streaming**: Optional streaming of detection video to ground station

## System Architecture

The system consists of several components:

- **QGCMissionIntegrator**: The main controller that coordinates all components
- **MissionMonitor**: Monitors and interacts with QGroundControl missions
- **ArUcoDetectionManager**: Manages ArUco marker detection using OAK-D camera
- **RTKGPSInterface**: Interfaces with RTK GPS server to acquire precise coordinates
- **SafetyManager**: Monitors safety parameters and handles emergency protocols

## Requirements

- Jetson Orin Nano or equivalent onboard computer
- OAK-D camera
- CubePilot Orange+ running ArduCopter 4.6+
- RTK GPS system (for coordinate acquisition)
- Python 3.8+
- QGroundControl ground station

## Installation

1. Ensure all hardware is properly connected
2. Install Python dependencies:

```bash
pip install numpy opencv-contrib-python depthai pymavlink pyyaml requests
```

3. Clone the repository:

```bash
git clone https://github.com/yourusername/precision-landing.git
cd precision-landing
```

## Usage

### Basic Usage

```bash
python qgc_mission_integration/main.py --mission my_mission.plan --target-id 5
```

### Command Line Options

```
usage: main.py [-h] [--config CONFIG] [--mission MISSION] [--connection CONNECTION]
               [--headless] [--log-level {DEBUG,INFO,WARNING,ERROR}] [--log-file LOG_FILE]
               [--target-id TARGET_ID] [--rtk-server RTK_SERVER]

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to configuration file
  --mission MISSION, -m MISSION
                        Path to mission file
  --connection CONNECTION
                        MAVLink connection string
  --headless            Run in headless mode
  --log-level {DEBUG,INFO,WARNING,ERROR}, -l {DEBUG,INFO,WARNING,ERROR}
                        Logging level
  --log-file LOG_FILE   Path to log file
  --target-id TARGET_ID, -t TARGET_ID
                        Target ArUco marker ID
  --rtk-server RTK_SERVER
                        RTK GPS server URL
```

### Configuration

Create a YAML configuration file to customize the system:

```yaml
# General settings
target_marker_id: 5
log_level: INFO
log_file: qgc_mission_integrator.log

# MAVLink settings
mavlink_connection: udp:192.168.2.1:14550
mavlink_baudrate: 921600
mavlink_timeout: 10

# Mission settings
mission_file: mission.plan
mission_check_interval: 1.0

# Detection settings
detection_confidence_threshold: 0.7
detection_required_confirmations: 5
detection_resolution: adaptive
detection_use_cuda: true

# Precision landing settings
landing_start_altitude: 10.0
landing_final_approach_altitude: 1.0
landing_descent_rate: 0.3
landing_center_tolerance: 0.3
landing_timeout: 120

# Safety settings
safety_min_battery_voltage: 22.0
safety_min_battery_remaining: 15
safety_max_mission_time: 600
safety_max_altitude: 30.0
safety_min_altitude: 2.0

# RTK GPS settings
rtk_server_url: http://localhost:8000
rtk_check_interval: 1.0

# Video streaming settings
enable_streaming: false
stream_ip: 192.168.2.1
stream_port: 5600
```

## Mission Workflow

1. **Initialization**: Connect to MAVLink, initialize camera, RTK GPS server
2. **Mission Monitoring**: Monitor QGC mission execution
3. **Target Detection**: Detect ArUco marker during mission
4. **Mission Interruption**: Safely interrupt mission when target is found
5. **Precision Loiter**: Center drone over the target
6. **Precision Landing**: Land precisely on the target
7. **RTK GPS Acquisition**: Record precise coordinates at landing site
8. **Mission Resumption**: Take off and resume the original mission
9. **Completion**: Return to launch after mission completion

## State Machine

The precision landing integrator operates using a state machine:

1. `IDLE`: Initial state, waiting for mission
2. `MONITORING`: Monitoring mission for target detection
3. `TARGET_DETECTION`: Target detection in progress
4. `TARGET_VALIDATION`: Validating detected target
5. `MISSION_INTERRUPTION`: Interrupting mission
6. `PRECISION_LOITER`: Loitering over target
7. `PRECISION_LANDING`: Executing precision landing
8. `RTK_ACQUISITION`: Acquiring RTK coordinates
9. `MISSION_RESUMPTION`: Resuming mission
10. `COMPLETE`: Precision landing completed
11. `ERROR`: Error state

## Example Mission

1. Create a mission in QGroundControl
2. Save the mission as a .plan file
3. Run the integrator:

```bash
python qgc_mission_integration/main.py --mission my_mission.plan --target-id 5 --rtk-server http://localhost:8000
```

4. The drone will execute the mission, detect the ArUco marker, land, acquire coordinates, and resume the mission
5. Recorded RTK GPS coordinates will be saved in `rtk_coordinates.json`

## Troubleshooting

- **Camera issues**: Verify OAK-D camera is properly connected and recognized
- **MAVLink connection**: Check connection string and network/USB connectivity
- **ArUco detection**: Ensure marker is visible and lighting conditions are suitable
- **RTK GPS**: Verify RTK GPS server is running and accessible

## Safety Considerations

- Always have a safety pilot ready with a manual controller
- Test in a controlled environment before real-world deployment
- Monitor battery levels to ensure sufficient power for mission completion
- Start with smaller missions and gradually increase complexity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ArduPilot team for ArduCopter firmware
- Luxonis for OAK-D camera and DepthAI library
- QGroundControl team for mission planning capabilities
