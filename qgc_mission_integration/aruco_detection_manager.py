#!/usr/bin/env python3

"""
ArUco Marker Detection Manager for QGC Mission Integration

This module provides a wrapper around the OAK-D ArUco detector to integrate
it with the QGC mission system for target detection and precision landing.
"""

import os
import sys
import time
import logging
import threading
import queue
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

# Set up logging
logger = logging.getLogger("ArUcoDetectionManager")

# Import OAK-D ArUco detector
try:
    # Add the project root to the path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root_dir)
    
    # First check for OpenCV 4.10 ArUco fixes
    try:
        from aruco.opencv410_aruco_fix import OpenCV410ArUcoFix
        USE_ARUCO_FIX = True
        logger.info("Using OpenCV 4.10 ArUco fixes")
    except ImportError:
        USE_ARUCO_FIX = False
        logger.info("OpenCV 4.10 ArUco fixes not found, using standard OpenCV ArUco")

    # Import OAK-D detector
    from oak_d_aruco_6x6_detector import OakDArUcoDetector
    logger.info("OAK-D ArUco detector imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure oak_d_aruco_6x6_detector.py is in the project root directory")
    raise

class ArUcoDetectionManager:
    """
    Manages ArUco marker detection for mission integration
    
    This class wraps the OAK-D ArUco detector and provides additional
    functionality for target validation, tracking, and integration with
    the mission controller.
    """
    
    def __init__(self, config: Dict[str, Any], headless: bool = False, 
                enable_streaming: bool = False):
        """
        Initialize the ArUco detection manager
        
        Args:
            config: Configuration dictionary with detection parameters
            headless: Whether to run without GUI
            enable_streaming: Whether to enable video streaming
        """
        self.config = config
        self.target_marker_id = config.get('target_marker_id', 5)
        self.min_confidence = config.get('min_detection_confidence', 0.7)
        self.required_detections = config.get('required_detection_count', 5)
        self.detector = None
        self.detection_thread = None
        self.stop_event = threading.Event()
        self.detection_queue = queue.Queue(maxsize=30)
        self.latest_detections = {}
        self.detection_history = {}  # Track recent detections by ID
        self.target_confirmation_count = 0
        self.callbacks = []
        self.target_detected = False
        self.headless = headless
        self.enable_streaming = enable_streaming
        
        # Initialize detector
        self._initialize_detector()
        
    def _initialize_detector(self) -> None:
        """Initialize the OAK-D ArUco detector"""
        try:
            logger.info(f"Initializing OAK-D ArUco detector, target ID: {self.target_marker_id}")
            
            # Create detector instance
            self.detector = OakDArUcoDetector(
                target_id=self.target_marker_id,
                resolution=self.config.get('resolution', 'adaptive'),
                use_cuda=self.config.get('use_cuda', True),
                high_performance=self.config.get('high_performance', True),
                headless=self.headless,
                enable_streaming=self.enable_streaming,
                stream_ip=self.config.get('stream_ip', '192.168.2.1'),
                stream_port=self.config.get('stream_port', 5600),
                quiet=not self.config.get('verbose', False)
            )
            
            logger.info("OAK-D ArUco detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OAK-D ArUco detector: {e}")
            raise
        
    def start(self) -> None:
        """Start the detection process"""
        if self.detector is None:
            logger.error("Cannot start detection: detector not initialized")
            return
            
        # Start the detector
        logger.info("Starting OAK-D ArUco detector")
        self.detector.start()
        
        # Start detection thread
        self.stop_event.clear()
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        logger.info("ArUco detection manager started")
        
    def stop(self) -> None:
        """Stop the detection process"""
        logger.info("Stopping ArUco detection manager")
        
        # Stop detection thread
        if self.detection_thread and self.detection_thread.is_alive():
            self.stop_event.set()
            self.detection_thread.join(timeout=2.0)
            
        # Stop detector
        if self.detector:
            self.detector.stop()
            
        logger.info("ArUco detection manager stopped")
        
    def _detection_loop(self) -> None:
        """Main detection loop in separate thread"""
        logger.info("Detection loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get frame
                rgb_frame = self.detector.get_rgb_frame()
                if rgb_frame is None:
                    time.sleep(0.01)
                    continue
                    
                # Detect markers
                markers_frame, corners, ids = self.detector.detect_aruco_markers(rgb_frame)
                
                # Process detections
                self._process_detections(markers_frame, corners, ids)
                
                # Sleep to control detection rate
                detection_rate = self.config.get('detection_rate', 20.0)
                time.sleep(1.0 / detection_rate)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(0.1)
                
        logger.info("Detection loop stopped")
        
    def _process_detections(self, frame, corners, ids) -> None:
        """Process marker detections"""
        # Package detection data
        detection_data = {
            'timestamp': time.time(),
            'markers': {},
            'frame': frame
        }
        
        # Clear target detected flag
        self.target_detected = False
        
        if ids is not None and len(ids) > 0:
            # Process each detected marker
            for i, marker_id in enumerate(ids):
                marker_id_val = marker_id[0]
                
                # Get 3D position using spatial data
                spatial_data = self.detector.get_spatial_data()
                if i < len(spatial_data):
                    coords = spatial_data[i].spatialCoordinates
                    position_3d = (coords.x, coords.y, coords.z)
                else:
                    position_3d = None
                    
                # Calculate confidence based on marker size and quality
                confidence = self._calculate_confidence(corners[i]) if corners else 0.0
                
                # Store detection data
                marker_data = {
                    'id': marker_id_val,
                    'corners': corners[i][0] if corners else None,
                    'position_3d': position_3d,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                
                detection_data['markers'][marker_id_val] = marker_data
                
                # Update detection history for this marker
                if marker_id_val not in self.detection_history:
                    self.detection_history[marker_id_val] = []
                    
                self.detection_history[marker_id_val].append({
                    'position_3d': position_3d,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
                
                # Limit history size
                max_history = 20
                if len(self.detection_history[marker_id_val]) > max_history:
                    self.detection_history[marker_id_val] = self.detection_history[marker_id_val][-max_history:]
                    
                # Check if this is our target marker
                if marker_id_val == self.target_marker_id:
                    # Update target confirmation
                    if confidence >= self.min_confidence:
                        self.target_confirmation_count += 1
                    else:
                        # Low confidence detection - reduce count
                        self.target_confirmation_count = max(0, self.target_confirmation_count - 1)
                        
                    # Check if we have enough confirmations
                    if self.target_confirmation_count >= self.required_detections:
                        if not self.target_detected:
                            logger.info(f"Target marker {self.target_marker_id} confirmed")
                            self._notify_callbacks('target_confirmed', marker_data)
                            
                        self.target_detected = True
        else:
            # No markers detected - reduce confirmation count
            self.target_confirmation_count = max(0, self.target_confirmation_count - 1)
            
        # Update latest detections
        self.latest_detections = detection_data
        
        # Add to queue for external components
        try:
            if self.detection_queue.full():
                # Remove oldest item if queue is full
                self.detection_queue.get_nowait()
                
            self.detection_queue.put_nowait(detection_data)
        except queue.Full:
            pass
            
    def _calculate_confidence(self, corners) -> float:
        """Calculate confidence score for marker detection (0-1)"""
        if corners is None or len(corners) == 0:
            return 0.0
            
        # Calculate marker perimeter
        perimeter = 0
        for i in range(4):
            p1 = corners[0][i]
            p2 = corners[0][(i + 1) % 4]
            perimeter += np.linalg.norm(p1 - p2)
            
        # Normalize confidence based on perimeter
        # 0-200px: 0.1-0.3
        # 200-500px: 0.3-0.7
        # 500+px: 0.7-1.0
        if perimeter < 200:
            confidence = 0.1 + (perimeter / 200) * 0.2
        elif perimeter < 500:
            confidence = 0.3 + ((perimeter - 200) / 300) * 0.4
        else:
            confidence = min(0.7 + ((perimeter - 500) / 1000) * 0.3, 1.0)
            
        return confidence
        
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback function for detection events
        
        Args:
            event: Event type ('target_detected', 'target_confirmed', 'detection_update')
            callback: Function to call when event occurs
        """
        self.callbacks.append((event, callback))
        logger.debug(f"Registered callback for event: {event}")
        
    def _notify_callbacks(self, event: str, data: Any) -> None:
        """Notify registered callbacks for a specific event"""
        for evt, callback in self.callbacks:
            if evt == event:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for event {event}: {e}")
                    
    def get_latest_detections(self) -> Dict[int, Dict[str, Any]]:
        """Get the latest marker detections"""
        return self.latest_detections.get('markers', {})
        
    def get_target_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the target marker if detected
        
        Returns:
            Dictionary with target marker information or None if not detected
        """
        detections = self.get_latest_detections()
        if self.target_marker_id in detections:
            return detections[self.target_marker_id]
        return None
        
    def is_target_detected(self) -> bool:
        """Check if target marker is currently detected with high confidence"""
        return self.target_detected
        
    def get_target_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Get 3D position of target marker relative to camera
        
        Returns:
            Tuple of (x, y, z) in mm or None if not available
        """
        target_info = self.get_target_info()
        if target_info and 'position_3d' in target_info:
            return target_info['position_3d']
        return None
        
    def get_target_distance(self) -> Optional[float]:
        """
        Get distance to target marker in meters
        
        Returns:
            Distance in meters or None if not available
        """
        position = self.get_target_position()
        if position:
            # Z coordinate is distance in mm, convert to meters
            return position[2] / 1000.0
        return None
        
    def get_detection_frame(self) -> Optional[np.ndarray]:
        """Get the latest detection frame with markers highlighted"""
        if hasattr(self.detector, 'last_markers_frame'):
            return self.detector.last_markers_frame
        return None
        
    def validate_target_detection(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate target detection with additional checks
        
        Returns:
            Tuple of (is_valid, validation_info)
        """
        if not self.is_target_detected():
            return False, {'reason': 'Target not detected'}
            
        # Get target info
        target_info = self.get_target_info()
        if not target_info:
            return False, {'reason': 'Target info not available'}
            
        # Check confidence
        confidence = target_info.get('confidence', 0.0)
        if confidence < self.min_confidence:
            return False, {'reason': f'Confidence too low: {confidence:.2f}'}
            
        # Check 3D position
        position = target_info.get('position_3d')
        if not position:
            return False, {'reason': 'No 3D position available'}
            
        # Check distance
        distance_mm = position[2]
        max_distance = self.config.get('max_detection_distance', 12000)  # 12m default
        if distance_mm > max_distance:
            return False, {'reason': f'Target too far: {distance_mm/1000:.2f}m'}
            
        # Check detection stability from history
        if self.target_marker_id in self.detection_history:
            history = self.detection_history[self.target_marker_id]
            if len(history) < self.required_detections:
                return False, {'reason': f'Not enough detection history: {len(history)}'}
                
            # Check position variance
            positions = [h['position_3d'] for h in history if h['position_3d'] is not None]
            if len(positions) >= 3:  # Need at least 3 points for variance
                # Calculate variance of X, Y coordinates
                positions = np.array(positions)
                variance = np.var(positions, axis=0)
                max_variance = np.max(variance[:2])  # X, Y variance
                
                # Convert to meters and check against threshold
                max_variance_m = max_variance / 1000000  # mm² to m²
                threshold = self.config.get('position_variance_threshold', 0.5)
                if max_variance_m > threshold:
                    return False, {'reason': f'Position variance too high: {max_variance_m:.3f}m²'}
            
        # All checks passed
        return True, {
            'confidence': confidence,
            'distance': distance_mm / 1000.0,
            'position': position,
            'detection_count': self.target_confirmation_count
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configuration
    config = {
        'target_marker_id': 5,
        'min_detection_confidence': 0.7,
        'required_detection_count': 5,
        'resolution': 'adaptive',
        'use_cuda': True,
        'high_performance': True,
        'detection_rate': 20.0,
        'verbose': True
    }
    
    # Callback function
    def on_target_confirmed(data):
        print(f"Target confirmed at distance: {data['position_3d'][2]/1000:.2f}m")
    
    # Create detection manager
    detector = ArUcoDetectionManager(config, headless=False)
    
    # Register callback
    detector.register_callback('target_confirmed', on_target_confirmed)
    
    try:
        # Start detection
        detector.start()
        
        # Run for 60 seconds
        print("Running ArUco detection for 60 seconds...")
        time.sleep(60)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        detector.stop()
