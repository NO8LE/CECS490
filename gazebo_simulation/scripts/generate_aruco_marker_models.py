#!/usr/bin/env python3

"""
ArUco Marker Model Generator for Gazebo Simulation

This script generates ArUco marker SDF models for use in Gazebo simulation.
It creates:
1. ArUco marker images using OpenCV
2. Gazebo model files for each marker
3. Material files for proper visualization

Usage:
  python3 generate_aruco_marker_models.py [--dictionary DICT] [--marker_size SIZE]

Options:
  --dictionary, -d  Dictionary to use (default: DICT_6X6_250)
  --marker_size, -s Size of marker in meters (default: 0.3048)
  --output_dir, -o  Output directory (default: ../models/aruco_markers)
  --count, -c       Number of markers to generate (default: 10)
"""

import os
import sys
import argparse
import cv2
import numpy as np

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate ArUco marker models for Gazebo')
    
    parser.add_argument('--dictionary', '-d', type=str, default='DICT_6X6_250',
                        choices=['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
                                 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
                                 'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
                                 'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000'],
                        help='ArUco dictionary to use')
    
    parser.add_argument('--marker_size', '-s', type=float, default=0.3048,
                        help='Marker size in meters (default: 0.3048m / 12 inches)')
    
    parser.add_argument('--output_dir', '-o', type=str, default='../models/aruco_markers',
                        help='Output directory for model files')
    
    parser.add_argument('--count', '-c', type=int, default=10,
                        help='Number of markers to generate')
    
    return parser.parse_args()

def get_aruco_dict(dict_name):
    """Get ArUco dictionary by name"""
    # Dictionary mapping
    dict_map = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
        'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
        'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
        'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
        'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000
    }
    
    # Get dictionary ID
    dict_id = dict_map.get(dict_name)
    if dict_id is None:
        print(f"Error: Dictionary {dict_name} not found. Using DICT_6X6_250.")
        dict_id = cv2.aruco.DICT_6X6_250
        
    # Create and return dictionary
    try:
        return cv2.aruco.getPredefinedDictionary(dict_id)
    except Exception as e:
        print(f"Error creating dictionary: {e}")
        # Fallback for OpenCV 4.x
        return cv2.aruco.Dictionary_get(dict_id)

def generate_marker_image(aruco_dict, marker_id, image_size=1000):
    """Generate ArUco marker image"""
    # Create the marker image
    try:
        # Try OpenCV 4.x method
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, image_size)
    except AttributeError:
        # Fallback for older OpenCV versions
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, image_size)
        
    # Apply a white border (10% of the image size)
    border_size = int(image_size * 0.1)
    bordered_img = cv2.copyMakeBorder(marker_img, border_size, border_size, border_size, border_size, 
                                      cv2.BORDER_CONSTANT, value=255)
    
    return bordered_img

def create_model_files(marker_id, marker_size, output_dir, image_path):
    """Create Gazebo model files for the marker"""
    model_name = f"aruco_marker_{marker_id}"
    model_dir = os.path.join(output_dir, model_name)
    
    # Create model directory structure
    os.makedirs(os.path.join(model_dir, "materials", "textures"), exist_ok=True)
    
    # Copy the marker image to the textures directory
    img_basename = os.path.basename(image_path)
    texture_path = os.path.join(model_dir, "materials", "textures", img_basename)
    os.system(f"cp {image_path} {texture_path}")
    
    # Create model.config file
    config_path = os.path.join(model_dir, "model.config")
    with open(config_path, 'w') as f:
        f.write(f'''<?xml version="1.0"?>
<model>
  <name>ArUco Marker {marker_id}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Autonomous Systems Lab</name>
    <email>info@example.com</email>
  </author>
  <description>
    ArUco marker with ID {marker_id} from the 6x6_250 dictionary.
    Size: {marker_size} meters.
  </description>
</model>
''')
    
    # Create material script file
    material_script_path = os.path.join(model_dir, "materials", "scripts")
    os.makedirs(material_script_path, exist_ok=True)
    
    with open(os.path.join(material_script_path, "marker.material"), 'w') as f:
        f.write(f'''material ArUco/Marker{marker_id}
{{
  technique
  {{
    pass
    {{
      texture_unit
      {{
        texture {img_basename}
        filtering none
        scale 1.0 1.0
      }}
    }}
  }}
}}
''')
    
    # Create SDF model file
    half_size = marker_size / 2
    thickness = 0.001  # Very thin box
    
    with open(os.path.join(model_dir, "model.sdf"), 'w') as f:
        f.write(f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="aruco_marker_{marker_id}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>{marker_size} {marker_size} {thickness}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{marker_size} {marker_size} {thickness}</size>
          </box>
        </geometry>
        <material>
          <script>
            <uri>model://aruco_marker_{marker_id}/materials/scripts</uri>
            <uri>model://aruco_marker_{marker_id}/materials/textures</uri>
            <name>ArUco/Marker{marker_id}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
''')
    
    print(f"Created model files for marker {marker_id} in {model_dir}")
    return model_dir

def main():
    """Main function"""
    args = parse_arguments()
    
    # Get ArUco dictionary
    aruco_dict = get_aruco_dict(args.dictionary)
    
    # Ensure output directory exists
    base_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.abspath(os.path.join(base_dir, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.count} ArUco marker models with size {args.marker_size}m")
    print(f"Output directory: {output_dir}")
    
    # Generate markers
    for marker_id in range(args.count):
        # Generate marker image
        marker_img = generate_marker_image(aruco_dict, marker_id)
        
        # Save image
        image_path = os.path.join(output_dir, f"aruco_{marker_id}.png")
        cv2.imwrite(image_path, marker_img)
        
        # Create model files
        create_model_files(marker_id, args.marker_size, output_dir, image_path)
    
    # Create a special target marker (ID 5 by default)
    target_id = 5
    target_img = generate_marker_image(aruco_dict, target_id)
    target_path = os.path.join(output_dir, f"aruco_target_{target_id}.png")
    cv2.imwrite(target_path, target_img)
    target_dir = create_model_files(target_id, args.marker_size, output_dir, target_path)
    
    print("\nModel generation complete!")
    print("To use these models in Gazebo:")
    print(f"1. Add {output_dir} to your GAZEBO_MODEL_PATH")
    print("2. Use the following format to include them in your world:")
    print(f"   <include><uri>model://aruco_marker_X</uri><pose>X Y Z R P Y</pose></include>")
    print("\nTarget marker (for landing) is ID:", target_id)

if __name__ == "__main__":
    main()
