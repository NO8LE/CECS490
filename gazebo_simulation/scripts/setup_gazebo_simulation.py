#!/usr/bin/env python3

"""
Gazebo Simulation Setup for Autonomous Precision Landing

This script sets up a Gazebo simulation for testing the autonomous precision landing system.
It:
1. Configures and places ArUco markers in the world
2. Sets up the drone model with camera and sensors
3. Generates a complete launch file for the simulation

Usage:
  python3 setup_gazebo_simulation.py [--target_id ID] [--num_markers COUNT]

Options:
  --target_id, -t      ID of the landing target marker (default: 5)
  --num_markers, -n    Total number of markers to place (default: 10)
  --random_placement   Randomly place markers (default: true)
  --output_dir, -o     Output directory for generated files (default: ../)
  --verbose, -v        Enable verbose output
"""

import os
import sys
import argparse
import random
import math
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Setup Gazebo simulation for autonomous precision landing')
    
    parser.add_argument('--target_id', '-t', type=int, default=5,
                        help='ID of the landing target marker (default: 5)')
    
    parser.add_argument('--num_markers', '-n', type=int, default=10,
                        help='Total number of markers to place (default: 10)')
    
    parser.add_argument('--random_placement', action='store_true', default=True,
                        help='Randomly place markers (default: true)')
    
    parser.add_argument('--output_dir', '-o', type=str, default='../',
                        help='Output directory for generated files')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def generate_random_positions(num_positions, target_id, area_size=27.4):
    """Generate random positions for markers within the search area"""
    positions = []
    
    # First, generate target marker position near center (within 1/3 of area)
    center_bound = area_size / 6  # 1/6 of area size on each side of center
    target_x = random.uniform(-center_bound, center_bound)
    target_y = random.uniform(-center_bound, center_bound)
    target_z = 0.001  # Just above ground
    target_yaw = random.uniform(0, 2*math.pi)
    
    positions.append({
        'id': target_id,
        'pose': (target_x, target_y, target_z, 0, 0, target_yaw),
        'is_target': True
    })
    
    # Now generate remaining random markers
    for i in range(num_positions - 1):
        marker_id = i
        if marker_id >= target_id:
            marker_id += 1  # Skip target ID
            
        # Keep trying until we get a position not too close to others
        while True:
            x = random.uniform(-area_size/2 * 0.9, area_size/2 * 0.9)
            y = random.uniform(-area_size/2 * 0.9, area_size/2 * 0.9)
            z = 0.001  # Just above ground
            yaw = random.uniform(0, 2*math.pi)
            
            # Check distance from other markers
            valid_position = True
            min_distance = 1.5  # Minimum 1.5m between markers
            
            for pos in positions:
                existing_x, existing_y = pos['pose'][0], pos['pose'][1]
                distance = math.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if distance < min_distance:
                    valid_position = False
                    break
                    
            if valid_position:
                positions.append({
                    'id': marker_id,
                    'pose': (x, y, z, 0, 0, yaw),
                    'is_target': False
                })
                break
                
    return positions

def create_world_file(marker_positions, output_dir, verbose=False):
    """Create a world file with ArUco markers"""
    # Get base world file
    base_world_path = os.path.join(output_dir, 'worlds/precision_landing_world.world')
    if not os.path.exists(base_world_path):
        print(f"Error: Base world file not found at {base_world_path}")
        return None
        
    # Parse the world file
    try:
        tree = ET.parse(base_world_path)
        root = tree.getroot()
        world = root.find('.//world')
    except Exception as e:
        print(f"Error parsing world file: {e}")
        return None
        
    # Add marker includes to world
    for marker in marker_positions:
        marker_id = marker['id']
        pose = marker['pose']
        pose_str = f"{pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]}"
        
        # Create the include element
        include = ET.SubElement(world, 'include')
        uri = ET.SubElement(include, 'uri')
        uri.text = f"model://aruco_marker_{marker_id}"
        
        # Add pose
        marker_pose = ET.SubElement(include, 'pose')
        marker_pose.text = pose_str
        
        # Add name to make it unique
        name = ET.SubElement(include, 'name')
        target_str = "_target" if marker['is_target'] else ""
        name.text = f"aruco_marker_{marker_id}{target_str}"
        
        if verbose:
            print(f"Added marker {marker_id} at pose {pose_str}")
            
    # Save the modified world file
    output_world_path = os.path.join(output_dir, 'worlds/precision_landing_with_markers.world')
    
    # Use minidom to pretty print the XML
    xml_str = ET.tostring(root, 'utf-8')
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")
    
    with open(output_world_path, 'w') as f:
        f.write(pretty_xml)
        
    print(f"Created world file with {len(marker_positions)} markers at {output_world_path}")
    return output_world_path

def create_drone_model(output_dir, verbose=False):
    """Create a simulated drone model with camera and sensors"""
    model_dir = os.path.join(output_dir, 'models/quad_camera')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model.config
    config_path = os.path.join(model_dir, 'model.config')
    with open(config_path, 'w') as f:
        f.write('''<?xml version="1.0"?>
<model>
  <name>Quadcopter with Camera</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Autonomous Systems Lab</name>
    <email>info@example.com</email>
  </author>
  <description>
    Quadcopter model with downward-facing camera and optical flow sensor
    for autonomous precision landing simulation.
  </description>
</model>
''')
    
    # Create SDF model file with camera and sensors
    model_path = os.path.join(model_dir, 'model.sdf')
    with open(model_path, 'w') as f:
        f.write('''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="quad_camera">
    <pose>0 0 0.2 0 0 0</pose>
    
    <link name="base_link">
      <inertial>
        <mass>1.5</mass>
        <inertia>
          <ixx>0.029125</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.029125</iyy>
          <iyz>0</iyz>
          <izz>0.055225</izz>
        </inertia>
        <pose>0 0 0 0 0 0</pose>
      </inertial>
      
      <collision name="collision">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>0.47 0.47 0.11</size>
          </box>
        </geometry>
      </collision>
      
      <visual name="visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>model://iris_base/meshes/iris.dae</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <script>
            <name>Gazebo/DarkGrey</name>
          </script>
        </material>
      </visual>
      
      <!-- Downward-facing camera -->
      <sensor name="camera_sensor" type="camera">
        <pose>0 0 -0.05 0 1.5708 0</pose>
        <always_on>1</always_on>
        <visualize>true</visualize>
        <update_rate>30</update_rate>
        <camera name="camera">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>1280</width>
            <height>720</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.005</stddev>
          </noise>
          <distortion>
            <k1>0.0</k1>
            <k2>0.0</k2>
            <k3>0.0</k3>
            <p1>0.0</p1>
            <p2>0.0</p2>
            <center>0.5 0.5</center>
          </distortion>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <alwaysOn>true</alwaysOn>
          <updateRate>30.0</updateRate>
          <cameraName>drone/camera</cameraName>
          <imageTopicName>image_raw</imageTopicName>
          <cameraInfoTopicName>camera_info</cameraInfoTopicName>
          <frameName>camera_link</frameName>
          <hackBaseline>0.0</hackBaseline>
          <distortionK1>0.0</distortionK1>
          <distortionK2>0.0</distortionK2>
          <distortionK3>0.0</distortionK3>
          <distortionT1>0.0</distortionT1>
          <distortionT2>0.0</distortionT2>
        </plugin>
      </sensor>
      
      <!-- Optical flow sensor -->
      <sensor name="optical_flow_sensor" type="camera">
        <pose>0 0 -0.05 0 1.5708 0</pose>
        <always_on>1</always_on>
        <visualize>false</visualize>
        <update_rate>20</update_rate>
        <camera name="flow_camera">
          <horizontal_fov>0.5236</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>L8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>30</far>
          </clip>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.007</stddev>
          </noise>
        </camera>
        <plugin name="optical_flow_plugin" filename="libgazebo_ros_optical_flow.so">
          <robotNamespace></robotNamespace>
          <qualityThreshold>50</qualityThreshold>
          <frameId>base_link</frameId>
          <topicName>/mavros/px4flow/raw/optical_flow_rad</topicName>
        </plugin>
      </sensor>
    </link>
    
    <!-- IMU Sensor -->
    <link name="imu_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.01</mass>
        <inertia>
          <ixx>0.000001</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.000001</iyy>
          <iyz>0</iyz>
          <izz>0.000001</izz>
        </inertia>
      </inertial>
      <sensor name="imu_sensor" type="imu">
        <always_on>true</always_on>
        <update_rate>200</update_rate>
        <imu>
          <angular_velocity>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.005</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.005</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.005</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </z>
          </angular_velocity>
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.01</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </z>
          </linear_acceleration>
        </imu>
        <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
          <robotNamespace></robotNamespace>
          <topicName>/mavros/imu/data</topicName>
          <bodyName>imu_link</bodyName>
          <updateRateHZ>200.0</updateRateHZ>
          <gaussianNoise>0.01</gaussianNoise>
          <xyzOffset>0 0 0</xyzOffset>
          <rpyOffset>0 0 0</rpyOffset>
          <frameName>imu_link</frameName>
        </plugin>
      </sensor>
    </link>
    
    <joint name="imu_joint" type="fixed">
      <parent>base_link</parent>
      <child>imu_link</child>
    </joint>
    
    <!-- MAVROS Interface Plugin -->
    <plugin name="mavlink_interface" filename="libgazebo_mavlink_interface.so">
      <robotNamespace></robotNamespace>
      <imuSubTopic>/imu</imuSubTopic>
      <mavlink_addr>INADDR_ANY</mavlink_addr>
      <mavlink_udp_port>14560</mavlink_udp_port>
      <serialEnabled>false</serialEnabled>
      <serialDevice>/dev/ttyACM0</serialDevice>
      <baudRate>921600</baudRate>
      <qgc_addr>INADDR_ANY</qgc_addr>
      <qgc_udp_port>14550</qgc_udp_port>
      <sdk_addr>INADDR_ANY</sdk_addr>
      <sdk_udp_port>14540</sdk_udp_port>
      <hil_mode>false</hil_mode>
      <hil_state_level>false</hil_state_level>
      <vehicle_is_tailsitter>false</vehicle_is_tailsitter>
      <send_vision_estimation>true</send_vision_estimation>
      <send_odometry>false</send_odometry>
      <motorSpeedCommandPubTopic>/gazebo/command/motor_speed</motorSpeedCommandPubTopic>
      <control_channels>
        <channel name="rotor1">
          <input_index>0</input_index>
          <input_offset>0</input_offset>
          <input_scaling>1000</input_scaling>
          <zero_position_disarmed>0</zero_position_disarmed>
          <zero_position_armed>100</zero_position_armed>
          <joint_control_type>velocity</joint_control_type>
        </channel>
        <channel name="rotor2">
          <input_index>1</input_index>
          <input_offset>0</input_offset>
          <input_scaling>1000</input_scaling>
          <zero_position_disarmed>0</zero_position_disarmed>
          <zero_position_armed>100</zero_position_armed>
          <joint_control_type>velocity</joint_control_type>
        </channel>
        <channel name="rotor3">
          <input_index>2</input_index>
          <input_offset>0</input_offset>
          <input_scaling>1000</input_scaling>
          <zero_position_disarmed>0</zero_position_disarmed>
          <zero_position_armed>100</zero_position_armed>
          <joint_control_type>velocity</joint_control_type>
        </channel>
        <channel name="rotor4">
          <input_index>3</input_index>
          <input_offset>0</input_offset>
          <input_scaling>1000</input_scaling>
          <zero_position_disarmed>0</zero_position_disarmed>
          <zero_position_armed>100</zero_position_armed>
          <joint_control_type>velocity</joint_control_type>
        </channel>
      </control_channels>
    </plugin>
    
    <!-- Multicopter Dynamics Plugin -->
    <plugin name="multicopter_dynamics" filename="libgazebo_multicopter_dynamics.so">
      <robotNamespace></robotNamespace>
      <rotorConfiguration>
        <rotor>
          <position>0.13 0.13 0</position>
          <direction>1</direction>
          <maxThrust>4.5</maxThrust>
        </rotor>
        <rotor>
          <position>-0.13 -0.13 0</position>
          <direction>1</direction>
          <maxThrust>4.5</maxThrust>
        </rotor>
        <rotor>
          <position>-0.13 0.13 0</position>
          <direction>-1</direction>
          <maxThrust>4.5</maxThrust>
        </rotor>
        <rotor>
          <position>0.13 -0.13 0</position>
          <direction>-1</direction>
          <maxThrust>4.5</maxThrust>
        </rotor>
      </rotorConfiguration>
    </plugin>
  </model>
</sdf>
''')
    
    print(f"Created drone model at {model_path}")
    return model_path

def create_launch_file(world_file, output_dir, verbose=False):
    """Create a launch file for the simulation"""
    launch_dir = os.path.join(output_dir, 'launch')
    os.makedirs(launch_dir, exist_ok=True)
    
    launch_path = os.path.join(launch_dir, 'precision_landing_simulation.launch')
    with open(launch_path, 'w') as f:
        f.write(f'''<?xml version="1.0"?>
<launch>
  <!-- Gazebo settings -->
  <arg name="headless" default="false"/>
  <arg name="gui" default="true"/>
  <arg name="paused" default="false"/>
  <arg name="verbose" default="{str(verbose).lower()}"/>
  
  <!-- Mission configuration -->
  <arg name="target_id" default="5"/>
  <arg name="search_altitude" default="15.0"/>
  <arg name="final_approach_altitude" default="1.0"/>
  
  <!-- Start Gazebo with the precision landing world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="{world_file}"/>
    <arg name="headless" value="$(arg headless)"/>
    <arg name="gui" value="$(arg gui)"/>
    <arg name="paused" value="$(arg paused)"/>
    <arg name="verbose" value="$(arg verbose)"/>
  </include>
  
  <!-- Spawn the quadcopter model -->
  <node name="spawn_quad" pkg="gazebo_ros" type="spawn_model"
        args="-sdf -file $(find precision_landing_gazebo)/models/quad_camera/model.sdf -model quad_camera"
        output="screen"/>
  
  <!-- Start ArduPilot SITL -->
  <node name="ardupilot_sitl" pkg="precision_landing_gazebo" type="start_ardupilot_sitl.sh"
        args="copter" output="screen"/>
  
  <!-- MAVROS -->
  <include file="$(find mavros)/launch/apm.launch">
    <arg name="fcu_url" value="udp://localhost:14560@localhost:14550"/>
    <arg name="gcs_url" value=""/>
    <arg name="tgt_system" value="1"/>
    <arg name="tgt_component" value="1"/>
  </include>
  
  <!-- ArUco detection node -->
  <node name="aruco_detector" pkg="precision_landing_gazebo" type="gazebo_aruco_detector.py"
        output="screen">
    <param name="target_id" value="$(arg target_id)"/>
    <param name="camera_topic" value="/drone/camera/image_raw"/>
    <param name="camera_info_topic" value="/drone/camera/camera_info"/>
  </node>
  
  <!-- Mission control node -->
  <node name="mission_control" pkg="precision_landing_gazebo" type="gazebo_precision_landing_mission.py"
        output="screen">
    <param name="target_id" value="$(arg target_id)"/>
    <param name="search_altitude" value="$(arg search_altitude)"/>
    <param name="final_approach_altitude" value="$(arg final_approach_altitude)"/>
    <param name="use_optical_flow" value="true"/>
  </node>
</launch>
''')
    
    print(f"Created launch file at {launch_path}")
    return launch_path

def create_startup_script(output_dir):
    """Create a startup script for the simulation"""
    scripts_dir = os.path.join(output_dir, 'scripts')
    
    script_path = os.path.join(scripts_dir, 'start_ardupilot_sitl.sh')
    with open(script_path, 'w') as f:
        f.write('''#!/bin/bash

# Start ArduPilot SITL with specific parameters for precision landing

VEHICLE=${1:-copter}
INSTANCE=${2:-1}
TARGET_PORT=$((14550 + INSTANCE))

# Set up environment variables
export PATH=$PATH:$HOME/ardupilot/Tools/autotest
export PATH=/usr/lib/ccache:$PATH

# Make sure log directory exists
mkdir -p $HOME/ardupilot_logs

# Navigate to ArduPilot directory
cd $HOME/ardupilot

# Start SITL with appropriate parameters
echo "Starting ArduPilot SITL for $VEHICLE instance $INSTANCE on port $TARGET_PORT"
sim_vehicle.py -v $VEHICLE \\
    --instance $INSTANCE \\
    --mavproxy-args="--out=udp:localhost:$TARGET_PORT --out=udp:localhost:14540" \\
    --add-param-file=$HOME/precision_landing_params.param \\
    --out=udp:localhost:14560 \\
    --map \\
    --console
''')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created ArduPilot SITL startup script at {script_path}")
    return script_path

def create_mission_config(output_dir):
    """Create mission configuration file"""
    config_dir = os.path.join(output_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, 'mission_config.yaml')
    with open(config_path, 'w') as f:
        f.write('''# Mission configuration for autonomous precision landing simulation

# Area and altitude settings
search_altitude: 15.0           # meters - safe search height
landing_start_altitude: 8.0     # begin precision landing descent
final_approach_altitude: 1.0    # begin flow-based final approach (m)
min_altitude: 1.0               # minimum safe altitude
max_altitude: 30.0              # maximum mission altitude
search_area_size: 27.4          # 30 yards converted to meters
search_pattern_spacing: 5.0     # meters between search legs

# Target and detection settings  
target_marker_id: 5             # specific landing marker ID
confirmation_frames: 5          # frames to confirm detection
validation_time: 3.0            # seconds to validate target
center_tolerance: 0.3           # acceptable center deviation
min_marker_size: 50             # minimum marker size in pixels
max_detection_distance: 12000   # maximum detection range (mm)

# Optical flow settings
use_optical_flow: true          # Enable optical flow sensor integration
final_descent_rate: 0.3         # Final descent rate (m/s) during flow landing
flow_quality_threshold: 50      # Minimum flow quality (0-255)
position_variance_threshold: 0.5# Maximum acceptable position variance (m²)

# Mission timing and safety
max_mission_time: 600           # 10 minute mission timeout
search_timeout: 300             # 5 minute search timeout  
landing_timeout: 120            # 2 minute landing timeout
min_battery_voltage: 22.0       # abort threshold
connection_timeout: 5.0         # MAVLink connection timeout

# EKF settings
ekf_pos_horiz_variance_threshold: 1.0  # Maximum acceptable horizontal position variance (m²)
ekf_pos_vert_variance_threshold: 1.0   # Maximum acceptable vertical position variance (m²)
''')
    
    print(f"Created mission configuration file at {config_path}")
    return config_path

def create_arducopter_params(output_dir):
    """Create ArduCopter parameter file for the simulation"""
    config_dir = os.path.join(output_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    param_path = os.path.join(config_dir, 'arducopter_params.param')
    with open(param_path, 'w') as f:
        f.write('''# ArduCopter parameters for precision landing simulation

# Precision landing parameters
PLND_ENABLED,1
PLND_TYPE,1
PLND_EST_TYPE,0
PLND_LAG,0.02
PLND_XY_DRIFT_MAX,0.2
PLND_STRICT,1

# Optical flow parameters
FLOW_TYPE,6
FLOW_ORIENT_YAW,0
FLOW_POS_X,0.0
FLOW_POS_Y,0.0
FLOW_POS_Z,0.0
EK3_SRC1_VELXY,5
LAND_SPEED,30

# General settings
AHRS_ORIENTATION,0
AHRS_EKF_TYPE,3
EK3_ENABLE,1
GPS_TYPE,0
GPS_AUTO_CONFIG,0
COMPASS_USE,0
COMPASS_USE2,0
COMPASS_USE3,0
ARMING_CHECK,0
ARMING_REQUIRE,0
RC_FEEL_RP,30
RC1_MAX,2000
RC1_MIN,1000
RC1_TRIM,1500
RC2_MAX,2000
RC2_MIN,1000
RC2_TRIM,1500
RC3_MAX,2000
RC3_MIN,1000
RC3_TRIM,1500
RC4_MAX,2000
RC4_MIN,1000
RC4_TRIM,1500
''')
    
    print(f"Created ArduCopter parameter file at {param_path}")
    return param_path

def main():
    """Main function"""
    args = parse_arguments()
    
    # Get base directory
    base_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.abspath(os.path.join(base_dir, args.output_dir))
    
    # Generate random marker positions
    marker_positions = generate_random_positions(
        args.num_markers, args.target_id
    )
    
    # Create world file with markers
    world_file = create_world_file(marker_positions, output_dir, args.verbose)
    if not world_file:
        print("Failed to create world file")
        return 1
        
    # Create drone model
    drone_model = create_drone_model(output_dir, args.verbose)
    if not drone_model:
        print("Failed to create drone model")
        return 1
        
    # Create launch file
    launch_file = create_launch_file(world_file, output_dir, args.verbose)
    if not launch_file:
        print("Failed to create launch file")
        return 1
        
    # Create startup script
    startup_script = create_startup_script(output_dir)
    if not startup_script:
        print("Failed to create startup script")
        return 1
        
    # Create mission configuration
    mission_config = create_mission_config(output_dir)
    if not mission_config:
        print("Failed to create mission configuration")
        return 1
        
    # Create ArduCopter parameters
    param_file = create_arducopter_params(output_dir)
    if not param_file:
        print("Failed to create ArduCopter parameters")
        return 1
        
    print("\nGazebo simulation setup complete!")
    print(f"World file: {world_file}")
    print(f"Drone model: {drone_model}")
    print(f"Launch file: {launch_file}")
    print(f"Startup script: {startup_script}")
    print(f"Mission config: {mission_config}")
    print(f"ArduCopter params: {param_file}")
    print("\nTo run the simulation, use the following commands:")
    print("1. Export Gazebo model path:")
    print(f"   export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:{os.path.join(output_dir, 'models')}")
    print("2. Copy parameter file:")
    print(f"   cp {param_file} $HOME/precision_landing_params.param")
    print("3. Launch the simulation:")
    print(f"   roslaunch precision_landing_gazebo precision_landing_simulation.launch")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
