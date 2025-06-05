#!/usr/bin/env python3

"""
Native Gazebo Simulation Setup for Autonomous Precision Landing (No ROS)

This script sets up a Gazebo simulation for testing the autonomous precision landing system
without requiring ROS. It:
1. Configures and places ArUco markers in the world
2. Sets up the drone model with camera and sensors using standard Gazebo plugins
3. Generates appropriate launch files for native Gazebo

Usage:
  python3 setup_gazebo_native.py [--target_id ID] [--num_markers COUNT]

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
import subprocess

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Setup native Gazebo simulation for autonomous precision landing')
    
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
        
        # Remove ROS wind plugin if it exists
        for plugin in world.findall('.//plugin'):
            if 'libgazebo_ros_wind.so' in plugin.attrib.get('filename', ''):
                world.remove(plugin)
                
        # Add standard Gazebo wind plugin
        wind_plugin = ET.SubElement(world, 'plugin')
        wind_plugin.set('name', 'wind')
        wind_plugin.set('filename', 'libgazebo_wind.so')
        
        # Configure wind parameters
        wind_velocity = ET.SubElement(wind_plugin, 'velocity')
        wind_velocity.text = '2.0'
        wind_direction = ET.SubElement(wind_plugin, 'direction')
        wind_direction.text = '1 1 0'
        
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
    """Create a simulated drone model with camera and sensors without ROS dependencies"""
    model_dir = os.path.join(output_dir, 'models/quad_camera_native')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model.config
    config_path = os.path.join(model_dir, 'model.config')
    with open(config_path, 'w') as f:
        f.write('''<?xml version="1.0"?>
<model>
  <name>Quadcopter with Camera (Native)</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Autonomous Systems Lab</name>
    <email>info@example.com</email>
  </author>
  <description>
    Quadcopter model with downward-facing camera and sensors
    for autonomous precision landing simulation without ROS.
  </description>
</model>
''')
    
    # Create SDF model file with camera and sensors (using standard Gazebo plugins)
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
          <box>
            <size>0.47 0.47 0.11</size>
          </box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1.0</ambient>
          <diffuse>0.2 0.2 0.2 1.0</diffuse>
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
        <!-- Use standard Gazebo camera plugin instead of ROS plugin -->
        <plugin name="camera_controller" filename="libgazebo_camera.so">
          <alwaysOn>true</alwaysOn>
          <updateRate>30.0</updateRate>
          <cameraName>drone/camera</cameraName>
          <frameName>camera_link</frameName>
        </plugin>
      </sensor>
      
      <!-- IMU Sensor -->
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
        <!-- Use standard Gazebo IMU plugin instead of ROS plugin -->
        <plugin name="imu_plugin" filename="libgazebo_imu.so">
          <updateRateHZ>200.0</updateRateHZ>
          <gaussianNoise>0.01</gaussianNoise>
          <xyzOffset>0 0 0</xyzOffset>
          <rpyOffset>0 0 0</rpyOffset>
        </plugin>
      </sensor>
    </link>
    
    <!-- MAVLINK Interface Plugin -->
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
    
    print(f"Created native drone model at {model_path}")
    return model_path

def create_startup_script(output_dir, world_file):
    """Create a native Gazebo startup script"""
    scripts_dir = os.path.join(output_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    
    script_path = os.path.join(scripts_dir, 'start_native_simulation.sh')
    with open(script_path, 'w') as f:
        f.write(f'''#!/bin/bash

# Start Gazebo simulation for precision landing without ROS

# Ensure environment is set up properly
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(realpath {os.path.join(output_dir, 'models')})
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$(realpath {output_dir})

# Start Gazebo with the world file
gazebo {world_file} --verbose &
GAZEBO_PID=$!

# Wait for Gazebo to start up
sleep 5

# Spawn the drone model
DRONE_MODEL_PATH=$(realpath {os.path.join(output_dir, 'models/quad_camera_native/model.sdf')})
echo "Spawning drone model from $DRONE_MODEL_PATH"
gz model --spawn-file=$DRONE_MODEL_PATH --model-name=quad_camera -x 0 -y 0 -z 0.2

# Start ArduPilot SITL if available
if command -v sim_vehicle.py &> /dev/null; then
    echo "Starting ArduPilot SITL"
    cd $HOME/ardupilot/ArduCopter 2>/dev/null || cd $HOME
    sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map &
    ARDUPILOT_PID=$!
    
    # Wait for ArduPilot to start
    sleep 5
    
    # Start the ArUco detector
    cd $(realpath {scripts_dir})
    python3 gazebo_aruco_detector.py --target {args.target_id} --verbose &
    DETECTOR_PID=$!
    
    # Start the mission controller
    cd $(realpath {scripts_dir})
    python3 gazebo_precision_landing_mission.py --target-id {args.target_id} --verbose &
    MISSION_PID=$!
    
    # Register cleanup handler
    trap "kill $GAZEBO_PID $ARDUPILOT_PID $DETECTOR_PID $MISSION_PID 2>/dev/null" EXIT
else
    echo "ArduPilot SITL not found. Install ardupilot and run sim_vehicle.py manually."
    # Register cleanup handler
    trap "kill $GAZEBO_PID 2>/dev/null" EXIT
fi

# Wait for user to press Ctrl+C
echo "Simulation running. Press Ctrl+C to exit."
wait $GAZEBO_PID
''')
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created native Gazebo startup script at {script_path}")
    return script_path

def create_gazebo_interface_script(output_dir):
    """Create a helper script to interface with Gazebo"""
    scripts_dir = os.path.join(output_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    
    script_path = os.path.join(scripts_dir, 'gazebo_interface.py')
    with open(script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3

"""
Gazebo Interface Helper for Native Simulation

This script provides helper functions to interact with Gazebo directly
without requiring ROS. It includes:
1. Camera subscription and image retrieval
2. Model manipulation
3. Physics control

Usage:
  Imported as a module in other scripts
"""

import socket
import struct
import time
import threading
import numpy as np
import cv2

class GazeboInterface:
    """Interface to Gazebo for camera, models, and physics"""
    
    def __init__(self, host="localhost", port=11345):
        """Initialize Gazebo interface"""
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.subscribed_topics = {}
        self.lock = threading.Lock()
        
    def connect(self):
        """Connect to Gazebo server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(("0.0.0.0", 0))
            self.socket.settimeout(1.0)
            self.running = True
            return True
        except Exception as e:
            print(f"Failed to connect to Gazebo: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from Gazebo server"""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
            
    def subscribe_camera(self, camera_topic, callback=None):
        """Subscribe to camera topic"""
        if not self.socket:
            print("Not connected to Gazebo")
            return False
            
        # Store subscription info
        with self.lock:
            self.subscribed_topics[camera_topic] = {
                "type": "camera",
                "callback": callback,
                "data": None,
                "last_update": 0
            }
            
        # Send subscription message
        sub_msg = f"sub:{camera_topic}"
        self.socket.sendto(sub_msg.encode(), (self.host, self.port))
        
        # Start receiver thread if not already running
        if not hasattr(self, 'receiver_thread') or not self.receiver_thread.is_alive():
            self.receiver_thread = threading.Thread(target=self._receive_loop)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()
            
        return True
        
    def _receive_loop(self):
        """Background thread to receive messages from Gazebo"""
        while self.running and self.socket:
            try:
                data, addr = self.socket.recvfrom(1024 * 1024)  # Large buffer for images
                
                # Parse the message type and topic
                if len(data) < 8:
                    continue
                    
                # Extract header information (simplified - real implementation would need to match Gazebo format)
                header_size = 16
                if len(data) <= header_size:
                    continue
                    
                # Try to identify the topic from the message
                # This is a simplification - actual Gazebo messages have more complex headers
                topic = None
                for t in self.subscribed_topics.keys():
                    # In a real implementation, the topic would be in the header
                    # Here we're just checking each subscription
                    topic = t
                    break
                    
                if not topic:
                    continue
                    
                # Process based on topic type
                with self.lock:
                    if topic in self.subscribed_topics:
                        topic_info = self.subscribed_topics[topic]
                        
                        if topic_info["type"] == "camera":
                            # Extract image data (simplified)
                            try:
                                # Parse image dimensions (would need to match Gazebo's format)
                                width, height = struct.unpack('II', data[0:8])
                                
                                # Convert to OpenCV format (assuming RGB)
                                img_data = data[header_size:]
                                img_array = np.frombuffer(img_data, dtype=np.uint8)
                                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                
                                # Store the image
                                topic_info["data"] = img
                                topic_info["last_update"] = time.time()
                                
                                # Call callback if registered
                                if topic_info["callback"]:
                                    topic_info["callback"](img)
                            except Exception as e:
                                print(f"Error processing camera data: {e}")
            except socket.timeout:
                # This is normal, just retry
                continue
            except Exception as e:
                print(f"Error in receive loop: {e}")
                time.sleep(0.1)
                
    def get_camera_image(self, camera_topic):
        """Get the latest image from a subscribed camera topic"""
        with self.lock:
            if camera_topic in self.subscribed_topics:
                return self.subscribed_topics[camera_topic]["data"]
        return None
        
    def get_model_pose(self, model_name):
        """Get the pose of a model in the simulation"""
        if not self.socket:
            return None
            
        # Construct request message (this is a simplified approach)
        req_msg = f"get_model_pose:{model_name}"
        self.socket.sendto(req_msg.encode(), (self.host, self.port))
        
        # In a real implementation, we would wait for and parse the response
        # Here we just return a placeholder
        return [0, 0, 0, 0, 0, 0]  # x, y, z, roll, pitch, yaw
        
    def set_model_pose(self, model_name, pose):
        """Set the pose of a model in the simulation"""
        if not self.socket:
            return False
            
        # Construct request message (this is a simplified approach)
        pose_str = " ".join(str(p) for p in pose)
        req_msg = f"set_model_pose:{model_name}:{pose_str}"
        self.socket.sendto(req_msg.encode(), (self.host, self.port))
        
        return True
        
    def spawn_model(self, model_file, model_name, pose):
        """Spawn a model in the simulation"""
        # This would normally use the Gazebo API
        # For now, we use a simplified approach
        pose_str = " ".join(str(p) for p in pose)
        cmd = f"gz model --spawn-file={model_file} --model-name={model_name} -x {pose[0]} -y {pose[1]} -z {pose[2]}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            return True
        except Exception as e:
            print(f"Error spawning model: {e}")
            return False
            
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
        
    # Create startup script
    startup_script = create_startup_script(output_dir, world_file)
    if not startup_script:
        print("Failed to create startup script")
        return 1
        
    # Create Gazebo interface helper
    interface_script = create_gazebo_interface_script(output_dir)
    if not interface_script:
        print("Failed to create Gazebo interface script")
        return 1
        
    print("\nNative Gazebo simulation setup complete!")
    print(f"World file: {world_file}")
    print(f"Drone model: {drone_model}")
    print(f"Startup script: {startup_script}")
    print(f"Interface helper: {interface_script}")
    print("\nTo run the simulation, use the following commands:")
    print("1. Export Gazebo model path:")
    print(f"   export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:{os.path.join(output_dir, 'models')}")
    print("2. Run the simulation:")
    print(f"   bash {startup_script}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
''')
    
    print(f"Created Gazebo interface helper at {script_path}")
    return script_path
