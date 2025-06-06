# Gazebo Simulation for Autonomous Precision Landing (Native Ubuntu)

This document explains how to run the Gazebo simulation for autonomous precision landing on Ubuntu without requiring ROS. These adapted scripts will work on a Jetson Orin Nano or any Ubuntu system with Gazebo installed.

## Prerequisites

- Ubuntu 20.04 or newer, or Debian 12 (bookworm)
- Gazebo 11.0 or newer (standalone installation, not from ROS)
- Python 3.8+
- ArduPilot SITL (optional, for full flight control)

## Installation

1. Install Gazebo standalone (if not already installed):

### For Ubuntu 20.04 (Focal):
```bash
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update
sudo apt install gazebo libgazebo-dev gazebo-common
```

### For newer Ubuntu versions:
Use Gazebo Harmonic (the newer Gazebo system):
```bash
# Install Gazebo Harmonic
sudo apt-get update
sudo apt-get install gz-harmonic

# Or install the development packages for a specific version
# Note: The version numbers for sim and msgs can be different, as they're separate components
# Check available packages with: apt search libgz-sim apt search libgz-msgs
sudo apt install libgz-sim9-dev libgz-msgs11-dev gazebo-common
```

### For Debian 12 (bookworm):
```bash
# Add Gazebo repository
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/debian-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# If the above key method doesn't work on newer Debian, use this instead:
# wget https://packages.osrfoundation.org/gazebo.key -O /tmp/gazebo.key
# gpg --dearmor < /tmp/gazebo.key | sudo tee /usr/share/keyrings/gazebo-archive-keyring.gpg > /dev/null
# echo "deb [signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/debian-stable `lsb_release -cs` main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list

sudo apt update

# Install Gazebo packages
sudo apt install gazebo libgazebo-dev gazebo-common

# If the classic Gazebo isn't available, you can install Gazebo Garden or newer:
# sudo apt install gz-garden
# or check available versions:
# apt search gz-garden
```

2. Install required Python packages:
```bash
pip install numpy opencv-python pymavlink pyyaml scipy
```

3. Install ArduPilot SITL (optional):
```bash
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
./waf configure --board sitl
./waf copter
```

## Setting Up the Simulation

1. Configure the simulation environment using the native setup script:
```bash
cd gazebo_simulation/scripts
python3 setup_gazebo_native.py --target_id 5 --num_markers 10
```

Options:
- `--target_id, -t`: ID of the landing target marker (default: 5)
- `--num_markers, -n`: Total number of markers to place (default: 10)
- `--random_placement`: Randomly place markers (default: true)
- `--output_dir, -o`: Output directory for generated files (default: ../)
- `--verbose, -v`: Enable verbose output

2. Generate ArUco marker models:
```bash
python3 generate_aruco_marker_models.py --start_id 0 --count 10
```

## Running the Simulation

The setup script will create a `start_native_simulation.sh` script that you can use to start the simulation:

```bash
cd gazebo_simulation/scripts
bash start_native_simulation.sh
```

This script:
1. Sets up the Gazebo environment variables
2. Starts Gazebo with the precision landing world
3. Spawns the drone model
4. Starts ArduPilot SITL (if available)
5. Starts the ArUco detector and mission controller

### Running Components Individually

If you prefer to run the components individually:

1. Start Gazebo with the world file:
```bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(realpath ../models)
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$(realpath ..)
gazebo ../worlds/precision_landing_with_markers.world
```

2. Spawn the drone model (in a new terminal):
```bash
gz model --spawn-file=$(realpath ../models/quad_camera_native/model.sdf) --model-name=quad_camera -x 0 -y 0 -z 0.2
```

3. Start ArduPilot SITL (in a new terminal, if available):
```bash
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map
```

4. Start the ArUco detector (in a new terminal):
```bash
python3 gazebo_aruco_detector_native.py --target 5 --verbose
```

5. Start the mission controller (in a new terminal):
```bash
python3 gazebo_precision_landing_mission_native.py --target-id 5 --verbose
```

## Key Differences from ROS Version

This native implementation differs from the ROS version in several ways:

1. **No ROS Dependencies**: All communication happens through direct socket connections to Gazebo and MAVLink.
2. **Simplified Plugin Architecture**: Uses standard Gazebo plugins instead of ROS-specific ones.
3. **Direct Socket Communication**: Implements a custom socket interface to Gazebo's transport system.
4. **Visualization**: Uses OpenCV windows for visualization rather than RViz or ROS topics.

## Customization

The native scripts support the same configuration options as the ROS versions:

- Modify mission parameters in `gazebo_precision_landing_mission_native.py`
- Adjust camera and detection settings in `gazebo_aruco_detector_native.py`
- Change the world configuration in the world file

## Troubleshooting

- **Gazebo Not Starting**: Ensure Gazebo is properly installed with `gazebo --version`
- **Camera Image Issues**: Check that the camera topic name matches what Gazebo is publishing
- **ArUco Detection Problems**: Verify that the OpenCV version is compatible (4.2+ recommended)
- **MAVLink Connection Failures**: Check the connection string parameters match your SITL configuration
- **Model Not Spawning**: Ensure `GAZEBO_MODEL_PATH` includes the path to your models directory

## Architecture

The non-ROS implementation uses three main components:

1. **gazebo_aruco_detector_native.py**: Connects directly to Gazebo's camera sensor, processes images to detect ArUco markers, and communicates with the drone via MAVLink.

2. **gazebo_precision_landing_mission_native.py**: Implements the mission state machine, manages the drone's behavior, and makes decisions based on marker detections.

3. **gazebo_interface.py**: A helper library that provides a simplified interface to Gazebo's transport system through sockets.

These components work together to provide a complete simulation environment without requiring ROS.
