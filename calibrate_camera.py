#!/usr/bin/env python3

"""
OAK-D Camera Calibration Script

This script calibrates the OAK-D camera using a chessboard pattern.
The calibration data is saved to a file that can be used by the ArUco marker detector.

Usage:
  python3 calibrate_camera.py [chessboard_width] [chessboard_height]

  chessboard_width: Number of inner corners in the chessboard width (default: 9)
  chessboard_height: Number of inner corners in the chessboard height (default: 6)

Instructions:
  1. Print a chessboard pattern and attach it to a flat surface
  2. Run this script and hold the chessboard in front of the camera
  3. Move the chessboard around to capture different angles and positions
  4. The script will capture frames when the chessboard is detected
  5. Press 'q' to exit and calculate the calibration

Example:
  python3 calibrate_camera.py 7 5
  This will calibrate using a chessboard with 7x5 inner corners.
"""

import cv2
import depthai as dai
import numpy as np
import os
import time

# Create calibration directory if it doesn't exist
CALIB_DIR = "camera_calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

# Calibration parameters
CALIB_FILE = os.path.join(CALIB_DIR, "calibration.npz")
MIN_FRAMES = 20  # Minimum number of frames to use for calibration
FRAME_INTERVAL = 1.0  # Minimum interval between captured frames (seconds)

class CameraCalibrator:
    def __init__(self, chessboard_size=(9, 6)):
        self.chessboard_size = chessboard_size
        self.pipeline = None
        self.device = None
        self.rgb_queue = None
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Captured frames counter
        self.frame_count = 0
        self.last_capture_time = 0
        
        # Initialize the pipeline
        self.initialize_pipeline()
        
    def initialize_pipeline(self):
        """
        Initialize the DepthAI pipeline for the OAK-D camera
        """
        # Create pipeline
        self.pipeline = dai.Pipeline()
        
        # Define sources and outputs
        rgb_cam = self.pipeline.create(dai.node.ColorCamera)
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        
        # Set stream names
        xout_rgb.setStreamName("rgb")
        
        # Properties
        rgb_cam.setPreviewSize(640, 400)
        rgb_cam.setInterleaved(False)
        rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb_cam.setFps(30)
        
        # Linking
        rgb_cam.preview.link(xout_rgb.input)
        
    def start(self):
        """
        Start the camera calibration process
        """
        # Connect to device and start pipeline
        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()
        
        # Get output queue
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        print(f"Camera started. Looking for a {self.chessboard_size[0]}x{self.chessboard_size[1]} chessboard pattern.")
        print("Move the chessboard around to capture different angles and positions.")
        print(f"Need at least {MIN_FRAMES} good frames for calibration.")
        print("Press 'q' to exit and calculate calibration.")
        
        # Main loop
        while True:
            # Get camera frame
            rgb_frame = self.get_rgb_frame()
            
            if rgb_frame is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(
                    gray, 
                    self.chessboard_size, 
                    None
                )
                
                # If found, add object points and image points
                display_frame = rgb_frame.copy()
                
                if ret and (time.time() - self.last_capture_time) > FRAME_INTERVAL:
                    # Refine corner positions
                    corners2 = cv2.cornerSubPix(
                        gray, 
                        corners, 
                        (11, 11), 
                        (-1, -1), 
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    
                    # Add to calibration data
                    self.objpoints.append(self.objp)
                    self.imgpoints.append(corners2)
                    
                    # Draw the corners
                    cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners2, ret)
                    
                    # Update counters
                    self.frame_count += 1
                    self.last_capture_time = time.time()
                    
                    # Display status
                    cv2.putText(
                        display_frame,
                        f"Captured frame {self.frame_count}/{MIN_FRAMES}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                else:
                    # Display status
                    if ret:
                        cv2.putText(
                            display_frame,
                            "Chessboard detected! Hold still...",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )
                    else:
                        cv2.putText(
                            display_frame,
                            "No chessboard detected",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
                
                # Display progress
                cv2.putText(
                    display_frame,
                    f"Frames: {self.frame_count}/{MIN_FRAMES}",
                    (20, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Display the frame
                cv2.imshow("Calibration", display_frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q') or (self.frame_count >= MIN_FRAMES):
                break
                
        # Clean up
        cv2.destroyAllWindows()
        self.device.close()
        
        # Calculate calibration if we have enough frames
        if self.frame_count >= MIN_FRAMES:
            self.calculate_calibration(gray.shape[::-1])
        else:
            print(f"Not enough frames captured ({self.frame_count}/{MIN_FRAMES}). Calibration aborted.")
        
    def get_rgb_frame(self):
        """
        Get the latest RGB frame from the camera
        """
        in_rgb = self.rgb_queue.tryGet()
        if in_rgb is not None:
            return in_rgb.getCvFrame()
        return None
        
    def calculate_calibration(self, img_size):
        """
        Calculate camera calibration from the collected points
        """
        print("\nCalculating camera calibration...")
        
        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, 
            self.imgpoints, 
            img_size, 
            None, 
            None
        )
        
        if ret:
            # Save calibration data
            np.savez(
                CALIB_FILE,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
            
            print(f"Calibration successful! Data saved to {CALIB_FILE}")
            print("\nCamera Matrix:")
            print(camera_matrix)
            print("\nDistortion Coefficients:")
            print(dist_coeffs)
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    self.objpoints[i], 
                    rvecs[i], 
                    tvecs[i], 
                    camera_matrix, 
                    dist_coeffs
                )
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            print(f"\nReprojection Error: {mean_error/len(self.objpoints)}")
            print("\nCalibration complete! You can now use the ArUco marker detector with improved accuracy.")
        else:
            print("Calibration failed. Please try again with a clearer chessboard pattern.")

def main():
    """
    Main function
    """
    # Parse command line arguments
    import sys
    
    chessboard_width = 9
    chessboard_height = 6
    
    if len(sys.argv) > 1:
        chessboard_width = int(sys.argv[1])
    if len(sys.argv) > 2:
        chessboard_height = int(sys.argv[2])
    
    print(f"Initializing OAK-D Camera Calibration with {chessboard_width}x{chessboard_height} chessboard...")
    calibrator = CameraCalibrator((chessboard_width, chessboard_height))
    calibrator.start()

if __name__ == "__main__":
    main()
