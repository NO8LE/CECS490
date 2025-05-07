#!/usr/bin/env python3

"""
Calibration Manager Module

This module provides functionality for managing camera calibration data,
including loading from files, creating default calibration, and saving
updated calibration data.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np
import os
import json
from typing import Tuple, List, Optional, Dict, Any, Union


class CalibrationManager:
    """
    Manager class for camera calibration data
    
    This class handles loading, saving, and managing camera calibration
    data for accurate pose estimation.
    """
    
    def __init__(self, 
                 calib_dir: str = "aruco/camera_calibration",
                 default_calib_file: str = "calibration.npz"):
        """
        Initialize the calibration manager
        
        Args:
            calib_dir: Directory for calibration files
            default_calib_file: Default calibration filename
        """
        self.calib_dir = calib_dir
        self.default_calib_file = default_calib_file
        
        # Create calibration directory if it doesn't exist
        os.makedirs(self.calib_dir, exist_ok=True)
        
        # Camera calibration matrices
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Calibration metadata
        self.calibration_info = {}
        
        # Load or create calibration
        self.load_calibration()
    
    def get_calibration_path(self) -> str:
        """
        Get the path to the calibration file
        
        Returns:
            str: Path to the calibration file
        """
        return os.path.join(self.calib_dir, self.default_calib_file)
    
    def load_calibration(self) -> bool:
        """
        Load camera calibration from file or create default
        
        Returns:
            bool: True if calibration was loaded from file, False if default was created
        """
        calib_file = self.get_calibration_path()
        
        # Check if calibration file exists
        if os.path.exists(calib_file):
            try:
                # Load calibration from file
                print(f"Loading camera calibration from {calib_file}")
                data = np.load(calib_file, allow_pickle=True)
                
                # Load camera matrix and distortion coefficients
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                
                # Load additional calibration info if available
                if 'calibration_info' in data:
                    self.calibration_info = data['calibration_info'].item()
                else:
                    # Create basic info
                    self.calibration_info = {
                        'source': 'file',
                        'file': calib_file,
                        'type': 'loaded'
                    }
                
                # Check if this is a CharucoBoard calibration
                if 'charuco_calibration' in data:
                    print("Detected CharucoBoard calibration data")
                    self.calibration_info['charuco'] = True
                    
                    if 'squares_x' in data and 'squares_y' in data:
                        print(f"CharucoBoard: {data['squares_x']}x{data['squares_y']} squares")
                        self.calibration_info['squares_x'] = data['squares_x']
                        self.calibration_info['squares_y'] = data['squares_y']
                
                print("Calibration loaded successfully")
                return True
            
            except Exception as e:
                print(f"Error loading calibration: {e}")
                print("Creating default calibration")
                self._create_default_calibration()
                return False
        else:
            # Use default calibration if no file exists
            print(f"No calibration file found at {calib_file}")
            print("Creating default calibration")
            self._create_default_calibration()
            return False
    
    def _create_default_calibration(self):
        """
        Create default camera calibration when no calibration file exists
        """
        # Default calibration for OAK-D RGB camera (approximate values)
        self.camera_matrix = np.array([
            [860.0, 0, 640.0],
            [0, 860.0, 360.0],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((1, 5))
        
        # Create calibration info
        self.calibration_info = {
            'source': 'default',
            'type': 'approximate',
            'camera': 'OAK-D',
            'stream': 'RGB',
            'warning': 'This is an approximate calibration. For better accuracy, please calibrate your camera.'
        }
        
        # Save default calibration
        self.save_calibration()
        
        print("Default calibration created")
        print("For better accuracy, consider calibrating your camera using aruco/calibrate_camera.py")
    
    def save_calibration(self) -> bool:
        """
        Save current calibration to file
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Error: No calibration data to save")
            return False
        
        calib_file = self.get_calibration_path()
        
        try:
            # Update timestamp in calibration info
            self.calibration_info['last_saved'] = np.datetime64('now').astype(str)
            
            # Save calibration data
            np.savez(
                calib_file,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                calibration_info=self.calibration_info
            )
            
            print(f"Calibration saved to {calib_file}")
            return True
        
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def update_calibration(self, 
                          camera_matrix: np.ndarray, 
                          dist_coeffs: np.ndarray,
                          info: Optional[Dict] = None) -> bool:
        """
        Update calibration with new data
        
        Args:
            camera_matrix: New camera calibration matrix
            dist_coeffs: New distortion coefficients
            info: Additional calibration information (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if camera_matrix is None or dist_coeffs is None:
            print("Error: Invalid calibration data")
            return False
        
        # Update calibration matrices
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Update calibration info
        if info is not None:
            self.calibration_info.update(info)
        
        # Update source and timestamp
        self.calibration_info['source'] = 'updated'
        self.calibration_info['updated_at'] = np.datetime64('now').astype(str)
        
        # Save updated calibration
        return self.save_calibration()
    
    def get_camera_matrix(self) -> np.ndarray:
        """
        Get the camera calibration matrix
        
        Returns:
            np.ndarray: Camera calibration matrix
        """
        return self.camera_matrix
    
    def get_dist_coeffs(self) -> np.ndarray:
        """
        Get the distortion coefficients
        
        Returns:
            np.ndarray: Distortion coefficients
        """
        return self.dist_coeffs
    
    def get_calibration_info(self) -> Dict:
        """
        Get calibration information
        
        Returns:
            Dict: Calibration information
        """
        return self.calibration_info
    
    def is_calibrated(self) -> bool:
        """
        Check if camera is calibrated
        
        Returns:
            bool: True if calibrated, False otherwise
        """
        return self.camera_matrix is not None and self.dist_coeffs is not None
    
    def is_default_calibration(self) -> bool:
        """
        Check if current calibration is the default
        
        Returns:
            bool: True if default calibration, False otherwise
        """
        return self.calibration_info.get('source') == 'default'
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort an image using the current calibration
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Undistorted image
        """
        if not self.is_calibrated():
            print("Warning: Camera not calibrated, returning original image")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def calibrate_from_charuco(self, 
                              corners_list: List[np.ndarray], 
                              ids_list: List[np.ndarray],
                              image_size: Tuple[int, int],
                              board_size: Tuple[int, int] = (6, 8),
                              square_length: float = 0.0508,  # 2 inches in meters
                              marker_length: float = 0.0381   # 1.5 inches in meters
                             ) -> bool:
        """
        Calibrate camera from CharucoBoard detections
        
        Args:
            corners_list: List of detected charuco corners from multiple images
            ids_list: List of detected charuco IDs from multiple images
            image_size: Size of the images (width, height)
            board_size: Size of the CharucoBoard (squares_x, squares_y)
            square_length: Length of the squares in meters
            marker_length: Length of the markers in meters
            
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        if len(corners_list) < 3 or len(ids_list) < 3:
            print("Error: Not enough detections for calibration")
            print("Need at least 3 images with CharucoBoard detections")
            return False
        
        try:
            # Create CharucoBoard
            dictionary = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
            board = cv2.aruco.CharucoBoard(
                board_size,
                square_length,
                marker_length,
                dictionary
            )
            
            # Prepare object points and image points
            obj_points = []
            img_points = []
            
            # Process each detection
            for corners, ids in zip(corners_list, ids_list):
                # Get object and image points
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, image_size, board
                )
                
                if charuco_corners is not None and len(charuco_corners) > 4:
                    obj_points.append(board.getChessboardCorners())
                    img_points.append(charuco_corners)
            
            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                image_size,
                None,
                None
            )
            
            if ret:
                # Update calibration
                info = {
                    'source': 'charuco',
                    'type': 'calibrated',
                    'board_size': board_size,
                    'square_length': square_length,
                    'marker_length': marker_length,
                    'image_size': image_size,
                    'num_images': len(corners_list),
                    'calibration_error': ret
                }
                
                return self.update_calibration(camera_matrix, dist_coeffs, info)
            else:
                print("Calibration failed")
                return False
        
        except Exception as e:
            print(f"Error during calibration: {e}")
            return False
    
    def calibrate_from_checkerboard(self, 
                                   corners_list: List[np.ndarray],
                                   image_size: Tuple[int, int],
                                   board_size: Tuple[int, int] = (9, 6),
                                   square_size: float = 0.0254  # 1 inch in meters
                                  ) -> bool:
        """
        Calibrate camera from checkerboard detections
        
        Args:
            corners_list: List of detected checkerboard corners from multiple images
            image_size: Size of the images (width, height)
            board_size: Size of the checkerboard (width, height) in inner corners
            square_size: Size of the squares in meters
            
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        if len(corners_list) < 3:
            print("Error: Not enough detections for calibration")
            print("Need at least 3 images with checkerboard detections")
            return False
        
        try:
            # Prepare object points
            objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
            
            # Arrays to store object points and image points
            obj_points = []
            img_points = []
            
            # Add each detection
            for corners in corners_list:
                obj_points.append(objp)
                img_points.append(corners)
            
            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                image_size,
                None,
                None
            )
            
            if ret:
                # Update calibration
                info = {
                    'source': 'checkerboard',
                    'type': 'calibrated',
                    'board_size': board_size,
                    'square_size': square_size,
                    'image_size': image_size,
                    'num_images': len(corners_list),
                    'calibration_error': ret
                }
                
                return self.update_calibration(camera_matrix, dist_coeffs, info)
            else:
                print("Calibration failed")
                return False
        
        except Exception as e:
            print(f"Error during calibration: {e}")
            return False
    
    def export_calibration(self, format_type: str = 'json') -> Optional[str]:
        """
        Export calibration to different formats
        
        Args:
            format_type: Format type ('json', 'yaml', or 'xml')
            
        Returns:
            str: Path to exported file or None if export failed
        """
        if not self.is_calibrated():
            print("Error: No calibration data to export")
            return None
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(self.calib_dir, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        # Prepare data for export
        export_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'calibration_info': self.calibration_info
        }
        
        try:
            if format_type.lower() == 'json':
                # Export to JSON
                export_file = os.path.join(export_dir, 'calibration.json')
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=4)
                
                print(f"Calibration exported to {export_file}")
                return export_file
            
            elif format_type.lower() == 'yaml':
                # Export to YAML
                try:
                    import yaml
                    export_file = os.path.join(export_dir, 'calibration.yaml')
                    with open(export_file, 'w') as f:
                        yaml.dump(export_data, f)
                    
                    print(f"Calibration exported to {export_file}")
                    return export_file
                except ImportError:
                    print("Error: PyYAML not installed. Cannot export to YAML.")
                    print("Install with: pip install pyyaml")
                    return None
            
            elif format_type.lower() == 'xml':
                # Export to XML
                export_file = os.path.join(export_dir, 'calibration.xml')
                fs = cv2.FileStorage(export_file, cv2.FILE_STORAGE_WRITE)
                
                # Write camera matrix
                fs.write("camera_matrix", self.camera_matrix)
                
                # Write distortion coefficients
                fs.write("dist_coeffs", self.dist_coeffs)
                
                # Write calibration info
                for key, value in self.calibration_info.items():
                    if isinstance(value, (int, float, str, bool)):
                        fs.write(key, value)
                
                fs.release()
                
                print(f"Calibration exported to {export_file}")
                return export_file
            
            else:
                print(f"Error: Unsupported export format '{format_type}'")
                print("Supported formats: json, yaml, xml")
                return None
        
        except Exception as e:
            print(f"Error exporting calibration: {e}")
            return None
    
    def import_calibration(self, file_path: str) -> bool:
        """
        Import calibration from file
        
        Args:
            file_path: Path to calibration file
            
        Returns:
            bool: True if import was successful, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return False
        
        try:
            # Determine file type from extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.npz':
                # Import from NPZ
                data = np.load(file_path, allow_pickle=True)
                
                if 'camera_matrix' in data and 'dist_coeffs' in data:
                    camera_matrix = data['camera_matrix']
                    dist_coeffs = data['dist_coeffs']
                    
                    # Get calibration info if available
                    info = None
                    if 'calibration_info' in data:
                        info = data['calibration_info'].item()
                    else:
                        info = {
                            'source': 'imported',
                            'file': file_path,
                            'format': 'npz'
                        }
                    
                    return self.update_calibration(camera_matrix, dist_coeffs, info)
                else:
                    print("Error: Invalid NPZ calibration file")
                    return False
            
            elif ext == '.json':
                # Import from JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'camera_matrix' in data and 'dist_coeffs' in data:
                    camera_matrix = np.array(data['camera_matrix'])
                    dist_coeffs = np.array(data['dist_coeffs'])
                    
                    # Get calibration info if available
                    info = data.get('calibration_info', {
                        'source': 'imported',
                        'file': file_path,
                        'format': 'json'
                    })
                    
                    return self.update_calibration(camera_matrix, dist_coeffs, info)
                else:
                    print("Error: Invalid JSON calibration file")
                    return False
            
            elif ext == '.yaml' or ext == '.yml':
                # Import from YAML
                try:
                    import yaml
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    if 'camera_matrix' in data and 'dist_coeffs' in data:
                        camera_matrix = np.array(data['camera_matrix'])
                        dist_coeffs = np.array(data['dist_coeffs'])
                        
                        # Get calibration info if available
                        info = data.get('calibration_info', {
                            'source': 'imported',
                            'file': file_path,
                            'format': 'yaml'
                        })
                        
                        return self.update_calibration(camera_matrix, dist_coeffs, info)
                    else:
                        print("Error: Invalid YAML calibration file")
                        return False
                except ImportError:
                    print("Error: PyYAML not installed. Cannot import from YAML.")
                    print("Install with: pip install pyyaml")
                    return False
            
            elif ext == '.xml':
                # Import from XML
                fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
                
                # Read camera matrix
                camera_matrix = fs.getNode("camera_matrix").mat()
                
                # Read distortion coefficients
                dist_coeffs = fs.getNode("dist_coeffs").mat()
                
                # Create calibration info
                info = {
                    'source': 'imported',
                    'file': file_path,
                    'format': 'xml'
                }
                
                fs.release()
                
                return self.update_calibration(camera_matrix, dist_coeffs, info)
            
            else:
                print(f"Error: Unsupported file format '{ext}'")
                print("Supported formats: .npz, .json, .yaml, .yml, .xml")
                return False
        
        except Exception as e:
            print(f"Error importing calibration: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create calibration manager
    calib_manager = CalibrationManager()
    
    # Print calibration info
    print("\nCalibration Information:")
    print(f"Camera Matrix:\n{calib_manager.get_camera_matrix()}")
    print(f"Distortion Coefficients: {calib_manager.get_dist_coeffs().flatten()}")
    
    info = calib_manager.get_calibration_info()
    print("\nCalibration Metadata:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Check if using default calibration
    if calib_manager.is_default_calibration():
        print("\nWarning: Using default calibration")
        print("For better accuracy, please calibrate your camera")
    
    # Export calibration to JSON
    export_path = calib_manager.export_calibration('json')
    if export_path:
        print(f"\nCalibration exported to: {export_path}")