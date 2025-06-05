# Gazebo Simulation for Autonomous Precision Landing

This directory contains a Gazebo simulation environment for testing the autonomous precision landing system without physical hardware. The simulation allows you to test and validate the precision landing algorithms in a controlled virtual environment before deploying to a real drone.

## Directory Structure

- **config/** - Configuration files for the simulation
- **launch/** - ROS launch files for starting the simulation
- **models/** - 3D models used in the simulation, including ArUco markers
- **scripts/** - Python scripts for running and configuring the simulation
- **worlds/** - Gazebo world definitions for the precision landing environment

## Key Components

### Scripts

- **setup_gazebo_simulation.py**: Sets up the simulation environment
  - Configures and places ArUco markers in the world
  - Sets up the drone model with camera and sensors
  - Generates a complete launch file for the simulation

- **generate_aruco_marker_models.py**: Creates 3D models of ArUco markers for the simulation
  - Generates SDF model files for use in Gazebo
  - Creates texture images for the markers

- **gazebo_aruco_detector.py**: ArUco marker detection for the simulated camera
  - Processes images from the simulated camera
  - Detects and tracks ArUco markers
  - Provides position and orientation data

- **gazebo_precision_landing_mission.py**: Mission controller for the simulation
  - Implements the same state machine as the real-world system
  - Interfaces with the Gazebo simulation instead of real hardware
  - Uses simulated sensors and actuators

### Worlds

- **precision_landing_world.world**: Main simulation world
  - Defines physics settings
  - Sets up lighting and ground plane
  - Includes markers and other objects

## Prerequisites

- ROS Noetic or newer
- Gazebo 11 or newer
- Python 3.8+
- Required Python packages: `rospy`, `gazebo_msgs`, `cv_bridge`, `sensor_msgs`, `geometry_msgs`
- ArduPilot SITL (Software In The Loop)

## Installation

1. Make sure you have ROS and Gazebo installed:
```bash
sudo apt install ros-noetic-desktop-full python3-rosdep python3-catkin-tools
```

2. Install ArduPilot SITL:
```bash
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
```

3. Build ArduPilot for SITL:
```bash
cd ardupilot
./waf configure --board sitl
./waf copter
```

4. Install MAVROS:
```bash
sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
```

## Setting Up the Simulation

1. Configure the simulation environment:
```bash
python3 scripts/setup_gazebo_simulation.py --target_id 5 --num_markers 10
```

Options:
- `--target_id, -t`: ID of the landing target marker (default: 5)
- `--num_markers, -n`: Total number of markers to place (default: 10)
- `--random_placement`: Randomly place markers (default: true)
- `--output_dir, -o`: Output directory for generated files (default: ../)
- `--verbose, -v`: Enable verbose output

2. Generate ArUco marker models:
```bash
python3 scripts/generate_aruco_marker_models.py --start_id 0 --count 10
```

## Running the Simulation

1. Start ROS core:
```bash
roscore
```

2. In a new terminal, start Gazebo with the precision landing world:
```bash
roslaunch gazebo_simulation/launch/precision_landing_simulation.launch
```

3. In a new terminal, start ArduPilot SITL:
```bash
cd ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map
```

4. In a new terminal, start the precision landing mission:
```bash
python3 scripts/gazebo_precision_landing_mission.py --target_id 5
```

## Monitoring and Visualization

1. View camera feed:
```bash
rosrun image_view image_view image:=/drone/camera/image_raw
```

2. View ArUco marker detections:
```bash
rosrun image_view image_view image:=/drone/aruco/image_detections
```

3. Use RViz for 3D visualization:
```bash
rosrun rviz rviz -d gazebo_simulation/config/precision_landing.rviz
```

## Integration with the Real System

The simulation is designed to match the real system as closely as possible:

- The `gazebo_precision_landing_mission.py` script follows the same state machine and logic as `autonomous_precision_landing_mission.py`
- The ArUco detection parameters are configured to match the real OAK-D camera
- The drone model's physics and control characteristics are tuned to match the real CubePilot behavior

This allows you to test your precision landing algorithms in simulation before deploying them on real hardware.

## Common Issues and Troubleshooting

- **Simulation is slow**: Reduce the number of markers or simplify the world
- **Markers not detecting**: Check camera parameters and ArUco dictionary settings
- **Physics instabilities**: Adjust physics parameters in the world file
- **SITL connection issues**: Verify MAVROS is running and UDP ports are correct
- **ROS node communication problems**: Check topic names and message types

## Customizing the Simulation

- Edit `worlds/precision_landing_world.world` to modify the environment
- Add new models to the `models/` directory
- Modify camera parameters in the drone model
- Adjust ArUco marker sizes and placement in `setup_gazebo_simulation.py`
- Change wind and disturbance settings for more realistic testing
