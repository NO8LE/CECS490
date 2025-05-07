#!/usr/bin/env python3

"""
ArUco Detector Module for OpenCV 4.10

This module provides a specialized implementation for detecting ArUco markers
with OpenCV 4.10.0, handling API changes and adding validation to prevent
false positives. It is designed to be used with the OAK-D camera but can
work with any RGB and depth input.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any, Union


class ArUcoDetector:
    """
    ArUco marker detector compatible with OpenCV 4.10.0
    
    This class handles the detection of ArUco markers using OpenCV 4.10.0's
    updated API. It includes validation to prevent false positives and
    supports CUDA acceleration when available.
    """
    
    def __init__(self, 
                 dictionary_type: int = cv2.aruco.DICT_6X6_250,
                 marker_size: float = 0.3048,  # 12 inches in meters
                 use_cuda: bool = False,
                 enable_validation: bool = True):
        """
        Initialize the ArUco detector
        
        Args:
            dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
            marker_size: Physical size of the marker in meters (default: 0.3048m / 12 inches)
            use_cuda: Whether to use CUDA acceleration if available (default: False)
            enable_validation: Whether to enable validation to prevent false positives (default: True)
        """
        self.dictionary_type = dictionary_type
        self.marker_size = marker_size
        self.enable_validation = enable_validation
        
        # Check CUDA availability
        self.use_cuda = use_cuda and self._check_cuda_availability()
        
        # Initialize ArUco dictionary and parameters
        self._initialize_aruco_detector()
        
        # Detection profiles for different distances
        self.detection_profiles = {
            "close": {  # 0.5-3m
                "adaptiveThreshConstant": 7,
                "minMarkerPerimeterRate": 0.1,
                "maxMarkerPerimeterRate": 4.0,
                "polygonalApproxAccuracyRate": 0.05,
                "cornerRefinementMethod": 1,  # CORNER_REFINE_SUBPIX
                "cornerRefinementWinSize": 5,
                "cornerRefinementMaxIterations": 30,
                "cornerRefinementMinAccuracy": 0.1,
                "minOtsuStdDev": 5.0,
                "errorCorrectionRate": 0.6
            },
            "medium": {  # 3-8m
                "adaptiveThreshConstant": 9,
                "minMarkerPerimeterRate": 0.05,
                "maxMarkerPerimeterRate": 4.0,
                "polygonalApproxAccuracyRate": 0.08,
                "cornerRefinementMethod": 1,  # CORNER_REFINE_SUBPIX
                "cornerRefinementWinSize": 5,
                "cornerRefinementMaxIterations": 30,
                "cornerRefinementMinAccuracy": 0.1,
                "minOtsuStdDev": 5.0,
                "errorCorrectionRate": 0.6
            },
            "far": {  # 8-12m
                "adaptiveThreshConstant": 11,
                "minMarkerPerimeterRate": 0.03,
                "maxMarkerPerimeterRate": 4.0,
                "polygonalApproxAccuracyRate": 0.1,
                "cornerRefinementMethod": 1,  # CORNER_REFINE_SUBPIX
                "cornerRefinementWinSize": 5,
                "cornerRefinementMaxIterations": 30,
                "cornerRefinementMinAccuracy": 0.1,
                "minOtsuStdDev": 5.0,
                "errorCorrectionRate": 0.6
            }
        }
        
        # Set default profile
        self.current_profile = "medium"
        self.apply_detection_profile(self.current_profile)
        
        # Performance monitoring
        self.detection_times = []
        
    def _check_cuda_availability(self) -> bool:
        """
        Check if CUDA is available for OpenCV
        
        Returns:
            bool: True if CUDA is available, False otherwise
        """
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
                return True
            else:
                print("No CUDA devices found, using CPU")
                return False
        except Exception as e:
            print(f"Error checking CUDA availability: {e}")
            print("CUDA not available in OpenCV, using CPU")
            return False
            
    def _initialize_aruco_detector(self):
        """
        Initialize the ArUco dictionary and detector parameters
        
        This method handles the initialization of the ArUco dictionary and
        detector parameters for OpenCV 4.10.0.
        """
        try:
            # Create ArUco dictionary
            self.aruco_dict = cv2.aruco.Dictionary(self.dictionary_type, 6)
            
            # Create detector parameters
            self.aruco_params = cv2.aruco.DetectorParameters()
            
            # Configure parameters with stricter settings to prevent false positives
            # Adaptive thresholding parameters
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 23
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_params.adaptiveThreshConstant = 7
            
            # Contour filtering parameters
            self.aruco_params.minMarkerPerimeterRate = 0.05
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.03
            self.aruco_params.minCornerDistanceRate = 0.05
            self.aruco_params.minDistanceToBorder = 3
            
            # Corner refinement parameters
            self.aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
            self.aruco_params.cornerRefinementWinSize = 5
            self.aruco_params.cornerRefinementMaxIterations = 30
            self.aruco_params.cornerRefinementMinAccuracy = 0.1
            
            # Error correction parameters
            self.aruco_params.errorCorrectionRate = 0.6
            self.aruco_params.minOtsuStdDev = 5.0
            self.aruco_params.perspectiveRemovePixelPerCell = 4
            self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
            
            # Create the detector
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            print("Successfully created ArucoDetector for OpenCV 4.10+")
            
        except Exception as e:
            print(f"Error initializing ArUco detector: {e}")
            print("Falling back to basic initialization")
            
            # Basic initialization as fallback
            self.aruco_dict = cv2.aruco.Dictionary(self.dictionary_type, 6)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = None
    
    def apply_detection_profile(self, profile_name: str):
        """
        Apply a detection parameter profile
        
        Args:
            profile_name: Name of the profile to apply ("close", "medium", or "far")
        """
        if profile_name not in self.detection_profiles:
            print(f"Warning: Unknown profile '{profile_name}'. Using 'medium' profile.")
            profile_name = "medium"
            
        profile = self.detection_profiles[profile_name]
        for param, value in profile.items():
            try:
                setattr(self.aruco_params, param, value)
            except Exception as e:
                print(f"Warning: Could not set parameter {param}: {e}")
                
        # Update the detector with new parameters
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except Exception as e:
            print(f"Warning: Could not update detector with new parameters: {e}")
            
        self.current_profile = profile_name
        print(f"Applied detection profile: {profile_name}")
    
    def select_profile_for_distance(self, distance_mm: float) -> str:
        """
        Select the appropriate detection profile based on distance
        
        Args:
            distance_mm: Estimated distance to marker in millimeters
            
        Returns:
            str: Name of the selected profile
        """
        if distance_mm > 8000:  # > 8m
            return "far"
        elif distance_mm > 3000:  # 3-8m
            return "medium"
        else:  # < 3m
            return "close"
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve marker detection
        
        Args:
            frame: Input RGB image
            
        Returns:
            np.ndarray: Preprocessed grayscale image
        """
        if self.use_cuda:
            return self._preprocess_image_gpu(frame)
        else:
            return self._preprocess_image_cpu(frame)
    
    def _preprocess_image_cpu(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess image on CPU
        
        Args:
            frame: Input RGB image
            
        Returns:
            np.ndarray: Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        return gray
    
    def _preprocess_image_gpu(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess image on GPU using CUDA
        
        Args:
            frame: Input RGB image
            
        Returns:
            np.ndarray: Preprocessed grayscale image
        """
        try:
            # Upload to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization on GPU
            gpu_gray = cv2.cuda.equalizeHist(gpu_gray)
            
            # Download result
            gray = gpu_gray.download()
            return gray
        except Exception as e:
            print(f"Error in GPU preprocessing: {e}")
            print("Falling back to CPU preprocessing")
            return self._preprocess_image_cpu(frame)
    
    def detect_markers(self, 
                      frame: np.ndarray, 
                      camera_matrix: Optional[np.ndarray] = None, 
                      dist_coeffs: Optional[np.ndarray] = None,
                      estimated_distance: float = 5000.0,
                      simple_detection: bool = False) -> Tuple[np.ndarray, List, Optional[np.ndarray]]:
        """
        Detect ArUco markers in the frame
        
        Args:
            frame: Input RGB image
            camera_matrix: Camera calibration matrix (optional)
            dist_coeffs: Distortion coefficients (optional)
            estimated_distance: Estimated distance to markers in mm (default: 5000.0)
            simple_detection: Whether to use simplified detection for performance (default: False)
            
        Returns:
            Tuple containing:
                - markers_frame: Image with markers drawn
                - corners: Detected marker corners
                - ids: Detected marker IDs
        """
        start_time = time.time()
        
        # Update detection profile based on estimated distance
        profile = self.select_profile_for_distance(estimated_distance)
        if profile != self.current_profile:
            self.apply_detection_profile(profile)
        
        # Preprocess image
        if simple_detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.preprocess_image(frame)
        
        # Detect markers
        try:
            # Make sure we have a detector
            if self.aruco_detector is None:
                # Create detector on-the-fly if needed
                self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            
            # Use the detector
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            # Validate detections if enabled
            if self.enable_validation and ids is not None and len(ids) > 0:
                corners, ids = self._validate_detections(corners, ids, gray)
        except Exception as e:
            print(f"Error in marker detection: {e}")
            corners = []
            ids = None
        
        # Copy original frame for drawing
        markers_frame = frame.copy()
        
        # Draw markers if any were detected and validated
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(markers_frame, corners, ids)
            
            # Estimate pose if camera is calibrated
            if camera_matrix is not None and dist_coeffs is not None:
                try:
                    # Estimate pose for each marker
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners,
                        self.marker_size,
                        camera_matrix,
                        dist_coeffs
                    )
                    
                    # Draw axes for each marker
                    for i in range(len(ids)):
                        try:
                            # Draw axes for the marker
                            cv2.aruco.drawAxis(
                                markers_frame,
                                camera_matrix,
                                dist_coeffs,
                                rvecs[i],
                                tvecs[i],
                                0.1  # Axis length
                            )
                        except Exception:
                            # If that fails, try cv2.drawFrameAxes
                            try:
                                cv2.drawFrameAxes(
                                    markers_frame,
                                    camera_matrix,
                                    dist_coeffs,
                                    rvecs[i],
                                    tvecs[i],
                                    0.1  # Axis length
                                )
                            except Exception as e2:
                                print(f"Could not draw axes: {e2}")
                except Exception as e:
                    print(f"Error with pose estimation: {e}")
                    print("Falling back to manual pose estimation")
                    
                    # Fallback to manual solvePnP approach
                    rvecs = []
                    tvecs = []
                    
                    # Process each marker individually
                    for i in range(len(corners)):
                        # Create object points for a square marker
                        objPoints = np.array([
                            [-self.marker_size/2, self.marker_size/2, 0],
                            [self.marker_size/2, self.marker_size/2, 0],
                            [self.marker_size/2, -self.marker_size/2, 0],
                            [-self.marker_size/2, -self.marker_size/2, 0]
                        ], dtype=np.float32)
                        
                        # Get image points from corners
                        imgPoints = corners[i][0].astype(np.float32)
                        
                        # Use solvePnP to get pose
                        success, rvec, tvec = cv2.solvePnP(
                            objPoints,
                            imgPoints,
                            camera_matrix,
                            dist_coeffs
                        )
                        
                        if success:
                            rvecs.append(rvec)
                            tvecs.append(tvec)
                            
                            # Draw axis for the marker
                            try:
                                cv2.drawFrameAxes(
                                    markers_frame,
                                    camera_matrix,
                                    dist_coeffs,
                                    rvec,
                                    tvec,
                                    0.1  # Axis length
                                )
                            except Exception as e2:
                                print(f"Could not draw axes: {e2}")
        
        # Record detection time
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 30:
            self.detection_times.pop(0)
        
        return markers_frame, corners, ids
    
    def _validate_detections(self, 
                            corners: List, 
                            ids: np.ndarray, 
                            gray: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Validate marker detections to filter out false positives
        
        Args:
            corners: Detected marker corners
            ids: Detected marker IDs
            gray: Grayscale image
            
        Returns:
            Tuple containing:
                - valid_corners: Validated marker corners
                - valid_ids: Validated marker IDs
        """
        # Ensure corners is a list of arrays with shape (1, 4, 2)
        if not isinstance(corners, list):
            corners_list = []
            for i in range(len(ids)):
                corners_list.append(corners[i].reshape(1, 4, 2))
            corners = corners_list
        
        # Apply validation checks to each detected marker
        valid_indices = []
        valid_corners = []
        valid_ids = []
        
        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_corners = corners[i]
            
            # 1. Verify marker has a valid perimeter (minimum size)
            perimeter = cv2.arcLength(marker_corners[0], True)
            min_perimeter = gray.shape[0] * 0.01  # Reduced to 1% of image height
            perimeter_valid = perimeter >= min_perimeter
            
            # 2. Verify marker is roughly square (with more tolerance)
            width = np.linalg.norm(marker_corners[0][0] - marker_corners[0][1])
            height = np.linalg.norm(marker_corners[0][1] - marker_corners[0][2])
            if width > 0 and height > 0:
                aspect_ratio = width/height
                aspect_valid = 0.5 < aspect_ratio < 2.0  # More tolerant aspect ratio
            else:
                aspect_valid = False
            
            # 3. Verify corner angles with more tolerance
            angles_valid = True
            
            for j in range(4):
                p1 = marker_corners[0][j]
                p2 = marker_corners[0][(j+1) % 4]
                p3 = marker_corners[0][(j+2) % 4]
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle
                dot = np.sum(v1 * v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                # Avoid division by zero
                if norm1 > 0 and norm2 > 0:
                    cos_angle = dot / (norm1 * norm2)
                    # Limit to valid range due to numerical errors
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle = np.abs(np.arccos(cos_angle) * 180 / np.pi)
                    
                    # More tolerant angle threshold
                    if abs(angle - 90) > 40:  # Increased from 25 to 40
                        angles_valid = False
                        break
                else:
                    angles_valid = False
                    break
            
            # Add to valid markers if it passes all tests
            validation_passed = perimeter_valid and aspect_valid and angles_valid
            
            if validation_passed:
                valid_indices.append(i)
                valid_corners.append(corners[i])
                valid_ids.append(ids[i])
        
        # Update with validated markers only
        if len(valid_indices) > 0:
            return valid_corners, np.array(valid_ids)
        else:
            return [], np.array([])
    
    def estimate_pose(self, 
                     corners: List, 
                     ids: np.ndarray, 
                     camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> Tuple[List, List]:
        """
        Estimate pose of detected markers
        
        Args:
            corners: Detected marker corners
            ids: Detected marker IDs
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            Tuple containing:
                - rvecs: Rotation vectors
                - tvecs: Translation vectors
        """
        if ids is None or len(ids) == 0:
            return [], []
        
        try:
            # Estimate pose for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                camera_matrix,
                dist_coeffs
            )
            return rvecs, tvecs
        except Exception as e:
            print(f"Error with estimatePoseSingleMarkers: {e}")
            print("Falling back to manual pose estimation")
            
            # Fallback to manual solvePnP approach
            rvecs = []
            tvecs = []
            
            # Process each marker individually
            for i in range(len(corners)):
                # Create object points for a square marker
                objPoints = np.array([
                    [-self.marker_size/2, self.marker_size/2, 0],
                    [self.marker_size/2, self.marker_size/2, 0],
                    [self.marker_size/2, -self.marker_size/2, 0],
                    [-self.marker_size/2, -self.marker_size/2, 0]
                ], dtype=np.float32)
                
                # Get image points from corners
                imgPoints = corners[i][0].astype(np.float32)
                
                # Use solvePnP to get pose
                success, rvec, tvec = cv2.solvePnP(
                    objPoints,
                    imgPoints,
                    camera_matrix,
                    dist_coeffs
                )
                
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
            
            return rvecs, tvecs
    
    def get_marker_distances(self, 
                           rvecs: List, 
                           tvecs: List) -> Dict[int, float]:
        """
        Calculate distances to markers from pose estimation results
        
        Args:
            rvecs: Rotation vectors
            tvecs: Translation vectors
            
        Returns:
            Dict mapping marker indices to distances in meters
        """
        distances = {}
        for i, tvec in enumerate(tvecs):
            # Calculate Euclidean distance
            distance = np.linalg.norm(tvec)
            distances[i] = distance
        return distances
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics
        
        Returns:
            Dict containing performance statistics
        """
        if not self.detection_times:
            return {"avg_detection_time": 0.0, "max_detection_time": 0.0, "min_detection_time": 0.0}
        
        avg_time = sum(self.detection_times) / len(self.detection_times)
        max_time = max(self.detection_times)
        min_time = min(self.detection_times)
        
        return {
            "avg_detection_time": avg_time,
            "max_detection_time": max_time,
            "min_detection_time": min_time
        }


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = ArUcoDetector(use_cuda=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers
        markers_frame, corners, ids = detector.detect_markers(frame)
        
        # Display performance stats
        stats = detector.get_performance_stats()
        cv2.putText(
            markers_frame,
            f"Avg detection time: {stats['avg_detection_time']*1000:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Display result
        cv2.imshow("ArUco Detector", markers_frame)
        
        # Check for key press
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()