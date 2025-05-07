#!/usr/bin/env python3

"""
Marker Tracker Module

This module provides functionality for tracking ArUco markers across frames,
selecting target markers, and implementing temporal filtering for more
stable detection and tracking.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any, Union
from collections import deque


class MarkerTracker:
    """
    Class for tracking ArUco markers across frames
    
    This class maintains the state of detected markers across frames,
    provides temporal filtering, and implements target selection logic.
    """
    
    def __init__(self, marker_id: int, corners: np.ndarray, timestamp: float):
        """
        Initialize a marker tracker
        
        Args:
            marker_id: ID of the marker to track
            corners: Initial corner positions of the marker
            timestamp: Timestamp of the detection
        """
        self.marker_id = marker_id
        self.corners = corners
        self.last_seen = timestamp
        
        # Store position history for temporal filtering
        self.positions = deque(maxlen=10)  # Store last 10 positions
        self.positions.append((corners, timestamp))
        
        # Calculate velocity for prediction
        self.velocity = np.zeros((4, 2))  # Velocity of each corner
        
        # Confidence score (0-1)
        self.confidence = 1.0
        
        # Tracking metrics
        self.detection_count = 1
        self.frame_count = 1
        self.detection_rate = 1.0
    
    def update(self, corners: np.ndarray, timestamp: float):
        """
        Update tracker with new detection
        
        Args:
            corners: New corner positions
            timestamp: Timestamp of the detection
        """
        dt = timestamp - self.last_seen
        if dt > 0:
            # Calculate velocity
            velocity = (corners - self.corners) / dt
            
            # Apply exponential smoothing to velocity
            alpha = 0.7  # Smoothing factor
            self.velocity = alpha * velocity + (1 - alpha) * self.velocity
        
        # Update corners with smoothing
        beta = 0.8  # Corner position smoothing factor
        self.corners = beta * corners + (1 - beta) * self.corners
        
        self.last_seen = timestamp
        self.positions.append((corners.copy(), timestamp))
        
        # Update confidence
        self.confidence = min(1.0, self.confidence + 0.1)
        
        # Update tracking metrics
        self.detection_count += 1
        self.frame_count += 1
        self.detection_rate = self.detection_count / self.frame_count
    
    def predict(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Predict marker position at given timestamp
        
        Args:
            timestamp: Timestamp to predict position for
            
        Returns:
            Predicted corner positions or None if prediction is unreliable
        """
        dt = timestamp - self.last_seen
        
        # If not seen for more than 0.5 seconds, prediction may be unreliable
        if dt > 0.5:
            return None
        
        # Predict using velocity
        predicted_corners = self.corners + self.velocity * dt
        return predicted_corners
    
    def missed_frame(self):
        """
        Update tracker when marker is not detected in a frame
        """
        self.frame_count += 1
        self.detection_rate = self.detection_count / self.frame_count
        
        # Decrease confidence
        self.confidence = max(0.0, self.confidence - 0.1)
    
    def is_valid(self, timestamp: float) -> bool:
        """
        Check if tracker is still valid
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            bool: True if tracker is valid, False otherwise
        """
        # Valid if seen recently and has reasonable detection rate
        time_valid = (timestamp - self.last_seen) < 1.0  # Valid for 1 second
        confidence_valid = self.confidence > 0.2
        return time_valid and confidence_valid
    
    def get_center(self) -> Tuple[float, float]:
        """
        Get the center point of the marker
        
        Returns:
            Tuple containing (x, y) coordinates of the center
        """
        center_x = np.mean([self.corners[0][0], self.corners[1][0], 
                           self.corners[2][0], self.corners[3][0]])
        center_y = np.mean([self.corners[0][1], self.corners[1][1], 
                           self.corners[2][1], self.corners[3][1]])
        return (center_x, center_y)
    
    def get_area(self) -> float:
        """
        Get the area of the marker in pixels
        
        Returns:
            float: Area of the marker
        """
        # Calculate area using shoelace formula
        x = [self.corners[i][0] for i in range(4)]
        y = [self.corners[i][1] for i in range(4)]
        
        area = 0.5 * abs(
            (x[0] * y[1] - x[1] * y[0]) +
            (x[1] * y[2] - x[2] * y[1]) +
            (x[2] * y[3] - x[3] * y[2]) +
            (x[3] * y[0] - x[0] * y[3])
        )
        
        return area


class MarkerTrackerManager:
    """
    Manager class for tracking multiple markers across frames
    
    This class manages multiple MarkerTracker instances, handles
    target selection, and provides temporal filtering for more
    stable detection and tracking.
    """
    
    def __init__(self, target_id: Optional[int] = None, max_markers: int = 10):
        """
        Initialize the marker tracker manager
        
        Args:
            target_id: ID of the target marker to prioritize (optional)
            max_markers: Maximum number of markers to track
        """
        self.target_id = target_id
        self.max_markers = max_markers
        self.trackers = {}  # Dict mapping marker IDs to MarkerTracker instances
        
        # Target marker information
        self.target_found = False
        self.target_center = None
        self.target_corners = None
        self.target_distance = None
        
        # Frame information
        self.frame_width = 640  # Default, will be updated
        self.frame_height = 480  # Default, will be updated
        self.last_frame_time = time.time()
    
    def set_target_id(self, target_id: Optional[int]):
        """
        Set the target marker ID
        
        Args:
            target_id: ID of the target marker to prioritize
        """
        self.target_id = target_id
        print(f"Target marker ID set to: {target_id}")
    
    def set_frame_size(self, width: int, height: int):
        """
        Set the frame size for calculations
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.frame_width = width
        self.frame_height = height
    
    def update(self, corners: List, ids: np.ndarray, distances: Optional[Dict[int, float]] = None):
        """
        Update trackers with new detections
        
        Args:
            corners: List of detected marker corners
            ids: Array of detected marker IDs
            distances: Dict mapping marker indices to distances (optional)
        """
        current_time = time.time()
        
        # Reset target information
        self.target_found = False
        self.target_center = None
        self.target_corners = None
        self.target_distance = None
        
        # Create set of detected marker IDs for quick lookup
        detected_ids = set()
        if ids is not None:
            detected_ids = {marker_id[0] for marker_id in ids}
        
        # Update existing trackers
        for marker_id in list(self.trackers.keys()):
            if marker_id in detected_ids:
                # Find the index of this marker in the detections
                idx = np.where(ids == marker_id)[0][0]
                
                # Update the tracker with new detection
                self.trackers[marker_id].update(corners[idx][0], current_time)
                
                # Update distance if available
                if distances is not None and idx in distances:
                    self.trackers[marker_id].distance = distances[idx]
                
                # Check if this is the target marker
                if marker_id == self.target_id:
                    self.target_found = True
                    self.target_corners = corners[idx][0]
                    self.target_center = self.trackers[marker_id].get_center()
                    if distances is not None and idx in distances:
                        self.target_distance = distances[idx]
            else:
                # Marker not detected in this frame
                self.trackers[marker_id].missed_frame()
                
                # Try to predict position
                predicted_corners = self.trackers[marker_id].predict(current_time)
                if predicted_corners is not None and marker_id == self.target_id:
                    # Use predicted position for target if not detected
                    self.target_found = True
                    self.target_corners = predicted_corners
                    self.target_center = self.trackers[marker_id].get_center()
                    self.target_distance = None  # Can't predict distance
        
        # Add new trackers for newly detected markers
        if ids is not None:
            for i, marker_id in enumerate(ids):
                marker_id_val = marker_id[0]
                if marker_id_val not in self.trackers:
                    # Create new tracker
                    self.trackers[marker_id_val] = MarkerTracker(
                        marker_id_val, corners[i][0], current_time
                    )
                    
                    # Update distance if available
                    if distances is not None and i in distances:
                        self.trackers[marker_id_val].distance = distances[i]
                    
                    # Check if this is the target marker
                    if marker_id_val == self.target_id:
                        self.target_found = True
                        self.target_corners = corners[i][0]
                        self.target_center = self.trackers[marker_id_val].get_center()
                        if distances is not None and i in distances:
                            self.target_distance = distances[i]
        
        # Clean up old trackers
        for marker_id in list(self.trackers.keys()):
            if not self.trackers[marker_id].is_valid(current_time):
                del self.trackers[marker_id]
        
        # Limit number of trackers if needed
        if len(self.trackers) > self.max_markers:
            # Sort trackers by confidence and keep only the top ones
            sorted_trackers = sorted(
                self.trackers.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )
            
            # Keep target marker if it exists
            if self.target_id in self.trackers:
                # Keep target and top (max_markers-1) trackers
                keep_ids = [self.target_id]
                for marker_id, _ in sorted_trackers:
                    if marker_id != self.target_id:
                        keep_ids.append(marker_id)
                        if len(keep_ids) >= self.max_markers:
                            break
                
                # Remove trackers not in keep_ids
                for marker_id in list(self.trackers.keys()):
                    if marker_id not in keep_ids:
                        del self.trackers[marker_id]
            else:
                # No target marker, just keep top max_markers trackers
                keep_trackers = dict(sorted_trackers[:self.max_markers])
                self.trackers = keep_trackers
        
        # Update last frame time
        self.last_frame_time = current_time
    
    def get_tracked_markers(self) -> Dict[int, Dict]:
        """
        Get information about all tracked markers
        
        Returns:
            Dict mapping marker IDs to information dictionaries
        """
        result = {}
        for marker_id, tracker in self.trackers.items():
            result[marker_id] = {
                "corners": tracker.corners,
                "center": tracker.get_center(),
                "confidence": tracker.confidence,
                "last_seen": tracker.last_seen,
                "is_target": marker_id == self.target_id,
                "detection_rate": tracker.detection_rate,
                "area": tracker.get_area()
            }
            
            # Add distance if available
            if hasattr(tracker, "distance") and tracker.distance is not None:
                result[marker_id]["distance"] = tracker.distance
        
        return result
    
    def draw_markers(self, frame: np.ndarray, draw_ids: bool = True, 
                    draw_centers: bool = True, highlight_target: bool = True) -> np.ndarray:
        """
        Draw tracked markers on the frame
        
        Args:
            frame: Input frame to draw on
            draw_ids: Whether to draw marker IDs
            draw_centers: Whether to draw marker centers
            highlight_target: Whether to highlight the target marker
            
        Returns:
            np.ndarray: Frame with markers drawn
        """
        result_frame = frame.copy()
        
        # Draw each tracked marker
        for marker_id, tracker in self.trackers.items():
            # Get corners as integer points for drawing
            corners = tracker.corners.astype(np.int32)
            
            # Determine color based on confidence and target status
            is_target = marker_id == self.target_id
            
            if is_target and highlight_target:
                # Target marker: red with intensity based on confidence
                color = (0, 0, int(255 * tracker.confidence))
                thickness = 2
            else:
                # Regular marker: green with intensity based on confidence
                color = (0, int(255 * tracker.confidence), 0)
                thickness = 1
            
            # Draw marker outline
            cv2.polylines(
                result_frame,
                [corners.reshape((-1, 1, 2))],
                True,
                color,
                thickness
            )
            
            # Draw marker ID if requested
            if draw_ids:
                center = tracker.get_center()
                cv2.putText(
                    result_frame,
                    f"ID: {marker_id}",
                    (int(center[0]), int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
            
            # Draw center point if requested
            if draw_centers:
                center = tracker.get_center()
                cv2.circle(
                    result_frame,
                    (int(center[0]), int(center[1])),
                    3,
                    color,
                    -1
                )
            
            # Add extra highlighting for target marker
            if is_target and highlight_target:
                center = tracker.get_center()
                
                # Draw crosshair
                cv2.line(
                    result_frame,
                    (int(center[0] - 20), int(center[1])),
                    (int(center[0] + 20), int(center[1])),
                    (0, 0, 255),
                    1
                )
                cv2.line(
                    result_frame,
                    (int(center[0]), int(center[1] - 20)),
                    (int(center[0]), int(center[1] + 20)),
                    (0, 0, 255),
                    1
                )
                
                # Draw target circle
                cv2.circle(
                    result_frame,
                    (int(center[0]), int(center[1])),
                    15,
                    (0, 0, 255),
                    1
                )
                
                # Add "TARGET" label
                cv2.putText(
                    result_frame,
                    "TARGET",
                    (int(center[0]) - 30, int(center[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
                
                # Add distance if available
                if hasattr(tracker, "distance") and tracker.distance is not None:
                    cv2.putText(
                        result_frame,
                        f"Dist: {tracker.distance:.2f}m",
                        (int(center[0]) - 30, int(center[1]) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
        
        return result_frame
    
    def draw_targeting_guidance(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw targeting guidance to help navigate to the target
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            np.ndarray: Frame with targeting guidance drawn
        """
        if not self.target_found or self.target_center is None:
            return frame
        
        result_frame = frame.copy()
        
        # Get frame center
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        
        # Calculate offset from center
        target_x, target_y = self.target_center
        offset_x = target_x - center_x
        offset_y = target_y - center_y
        
        # Draw guidance arrow
        arrow_length = min(100, max(20, int(np.sqrt(offset_x**2 + offset_y**2) / 5)))
        angle = np.arctan2(offset_y, offset_x)
        end_x = int(center_x + np.cos(angle) * arrow_length)
        end_y = int(center_y + np.sin(angle) * arrow_length)
        
        # Only draw if target is not centered
        if abs(offset_x) > 30 or abs(offset_y) > 30:
            # Draw direction arrow
            cv2.arrowedLine(
                result_frame,
                (center_x, center_y),
                (end_x, end_y),
                (0, 255, 0),
                2,
                tipLength=0.3
            )
            
            # Add guidance text
            direction_text = ""
            if abs(offset_y) > 30:
                direction_text += "UP " if offset_y < 0 else "DOWN "
            if abs(offset_x) > 30:
                direction_text += "LEFT" if offset_x < 0 else "RIGHT"
            
            cv2.putText(
                result_frame,
                direction_text,
                (center_x + 20, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            # Target is centered
            cv2.putText(
                result_frame,
                "TARGET CENTERED",
                (center_x - 100, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return result_frame


# Example usage
if __name__ == "__main__":
    # Create a simple example with synthetic data
    tracker_manager = MarkerTrackerManager(target_id=5)
    tracker_manager.set_frame_size(640, 480)
    
    # Create some synthetic marker data
    marker_ids = np.array([[0], [5], [10]])
    marker_corners = [
        np.array([[[100, 100], [150, 100], [150, 150], [100, 150]]]),
        np.array([[[300, 200], [350, 200], [350, 250], [300, 250]]]),
        np.array([[[500, 300], [550, 300], [550, 350], [500, 350]]])
    ]
    
    # Update tracker with synthetic data
    tracker_manager.update(marker_corners, marker_ids)
    
    # Create a blank frame for visualization
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw markers on the frame
    result_frame = tracker_manager.draw_markers(frame)
    
    # Draw targeting guidance
    result_frame = tracker_manager.draw_targeting_guidance(result_frame)
    
    # Display the result
    cv2.imshow("Marker Tracking", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()