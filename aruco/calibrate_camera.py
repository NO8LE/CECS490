#!/usr/bin/env python3

"""
OAK-D Camera Calibration Script for Drone Applications

This script calibrates the OAK-D camera using a CharucoBoard pattern, optimized for
drone-based detection at ranges from 0.5m to 12m with 12-inch (0.3048m) markers.
The calibration data is saved to a file that can be used by the ArUco marker detector.

Usage:
  python3 calibrate_camera.py [--charuco squares_x squares_y square_length] [--drone]
  python3 calibrate_camera.py [--chessboard width height]

  --charuco: Use a CharucoBoard for calibration (recommended)
    squares_x: Number of squares in X direction (width) (default: 8)
    squares_y: Number of squares in Y direction (height) (default: 6)
    square_length: Physical size of each square in meters (default: 0.325)

  --drone: Optimize calibration for drone-based detection at various distances
    This increases the number of required calibration frames and enables
    distance-specific calibration profiles.

  --chessboard: Use a traditional chessboard for calibration (legacy)
    width: Number of inner corners in the chessboard width (default: 9)
    height: Number of inner corners in the chessboard height (default: 6)

Instructions:
  1. Generate and print a CharucoBoard pattern using generate_charuco_board_for_drone.py
  2. Measure the actual size of the squares on your printed board
  3. Run this script with the --charuco option and the correct square size
  4. Move the board around to capture different angles and positions
     For drone applications, be sure to capture frames at various distances (0.5m to 12m)
  5. The script will capture frames when the board is detected
  6. Press 'q' to exit and calculate the calibration

Examples:
  python3 calibrate_camera.py --charuco 8 6 0.325 --drone
  This will calibrate using a CharucoBoard with 8x6 squares (width x height), 
  each 32.5cm in size, optimized for drone-based detection at various distances.

  python3 calibrate_camera.py --charuco 6 8 0.025
  This will calibrate using a standard CharucoBoard with 6x8 squares (width x height), 
  each 2.5cm in size, designed to fit on a standard 8.5x11 inch paper.

  python3 calibrate_camera.py --chessboard 9 6
  This will calibrate using a traditional chessboard with 9x6 inner corners.
"""

import os
import sys
import numpy as np
import time
import argparse

# Import OpenCV
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("Error: OpenCV (cv2) not found.")
    print("Please install OpenCV:")
    print("  pip install opencv-python opencv-contrib-python")
    sys.exit(1)

# Import ArUco module - try different approaches for different OpenCV installations
try:
    # First try direct import from cv2
    if hasattr(cv2, 'aruco'):
        print("Using cv2.aruco")
        aruco = cv2.aruco
    else:
        # Try to import as a separate module (older OpenCV versions)
        try:
            from cv2 import aruco
            print("Using from cv2 import aruco")
            # Make it available as cv2.aruco for consistency
            cv2.aruco = aruco
        except ImportError:
            # For Jetson with custom OpenCV builds
            try:
                sys.path.append('/usr/lib/python3/dist-packages/cv2/python-3.10')
                from cv2 import aruco
                print("Using Jetson-specific aruco import")
                cv2.aruco = aruco
            except (ImportError, FileNotFoundError):
                print("Error: OpenCV ArUco module not found.")
                print("Please ensure opencv-contrib-python is installed:")
                print("  pip install opencv-contrib-python")
                print("\nFor Jetson platforms, you might need to install it differently:")
                print("  sudo apt-get install python3-opencv")
                sys.exit(1)
except Exception as e:
    print(f"Error importing ArUco module: {str(e)}")
    print("Please ensure opencv-contrib-python is installed correctly")
    sys.exit(1)

# Verify ArUco module is working
# Check OpenCV version first to use the appropriate API
if cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
    # For OpenCV 4.12.0-dev and newer
    try:
        # Create dictionary with marker size parameter
        aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250, 6)
        print(f"ArUco module successfully loaded for OpenCV {cv2.__version__} (using Dictionary with markerSize)")
        dictionary_method = "opencv4.12"
    except Exception as e:
        print(f"Error creating ArUco dictionary for OpenCV 4.12+: {str(e)}")
        print("Please check your OpenCV installation and version.")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Build information: {cv2.getBuildInformation()}")
        sys.exit(1)
else:
    # For older OpenCV versions
    try:
        # Try to access a dictionary to verify ArUco is working
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        print("ArUco module successfully loaded and verified (using Dictionary_get)")
        # Store the method to use later
        dictionary_method = "old"
    except Exception as e:
        # If Dictionary_get fails, try the newer API
        try:
            aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_6X6_250)
            print("ArUco module successfully loaded and verified (using Dictionary.get)")
            # Store the method to use later
            dictionary_method = "new"
        except Exception as e2:
            # If both methods fail, try to create a dictionary directly
            try:
                # Try to create a dictionary directly (OpenCV 4.x approach)
                aruco_dict = cv2.aruco.Dictionary.create(cv2.aruco.DICT_6X6_250)
                print("ArUco module successfully loaded and verified (using Dictionary.create)")
                # Store the method to use later
                dictionary_method = "create"
            except Exception as e3:
                # Last resort: try to create a dictionary with parameters
                try:
                    # Try to create a dictionary with parameters (another approach)
                    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250)
                    print("ArUco module successfully loaded and verified (using Dictionary constructor)")
                    # Store the method to use later
                    dictionary_method = "constructor"
                except Exception as e4:
                    print(f"Error verifying ArUco module: {str(e4)}")
                    print("ArUco module found but not working correctly")
                    print("\nDetailed error information:")
                    print(f"Dictionary_get error: {str(e)}")
                    print(f"Dictionary.get error: {str(e2)}")
                    print(f"Dictionary.create error: {str(e3)}")
                    print(f"Dictionary constructor error: {str(e4)}")
                    print("\nPlease check your OpenCV installation and version.")
                    sys.exit(1)

# Import DepthAI
try:
    import depthai as dai
    print(f"DepthAI version: {dai.__version__}")
except ImportError:
    print("Error: DepthAI module not found.")
    print("Please install DepthAI:")
    print("  pip install depthai")
    print("\nFor Jetson platforms, you might need to install it differently:")
    print("  sudo apt-get install python3-depthai")
    sys.exit(1)

# Create calibration directory if it doesn't exist
CALIB_DIR = "camera_calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

# Calibration parameters
CALIB_FILE = os.path.join(CALIB_DIR, "calibration.npz")
MIN_FRAMES = 30  # Minimum number of frames to use for calibration (increased for drone applications)
FRAME_INTERVAL = 1.0  # Minimum interval between captured frames (seconds)
DRONE_MODE = False  # Whether to optimize for drone-based detection

def get_aruco_dictionary(dictionary_id=cv2.aruco.DICT_6X6_250):
    """
    Get the ArUco dictionary using the method that worked during initialization
    
    Args:
        dictionary_id: ArUco dictionary ID to use
        
    Returns:
        The ArUco dictionary
    """
    # Check OpenCV version first to use the appropriate API
    if cv2.__version__.startswith("4.10") or cv2.__version__.startswith("4.11") or cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
        # For OpenCV 4.12.0-dev and newer, always use the Dictionary constructor with marker size
        try:
            # Create dictionary with marker size parameter (6 for 6x6 dictionary)
            return cv2.aruco.Dictionary(dictionary_id, 6)
        except Exception as e:
            print(f"Error creating ArUco dictionary for OpenCV 4.12+: {str(e)}")
            raise
    else:
        # For older OpenCV versions
        if dictionary_method == "old":
            return cv2.aruco.Dictionary_get(dictionary_id)
        elif dictionary_method == "new":
            return cv2.aruco.Dictionary.get(dictionary_id)
        elif dictionary_method == "create":
            return cv2.aruco.Dictionary.create(dictionary_id)
        elif dictionary_method == "constructor":
            return cv2.aruco.Dictionary(dictionary_id)
        elif dictionary_method == "opencv4.12":
            # This should not happen for older OpenCV versions, but just in case
            return cv2.aruco.Dictionary(dictionary_id, 6)
        else:
            # Fallback to trying all methods
            try:
                return cv2.aruco.Dictionary_get(dictionary_id)
            except:
                try:
                    return cv2.aruco.Dictionary.get(dictionary_id)
                except:
                    try:
                        return cv2.aruco.Dictionary.create(dictionary_id)
                    except:
                        try:
                            return cv2.aruco.Dictionary(dictionary_id)
                        except:
                            # Last resort for OpenCV 4.x
                            return cv2.aruco.Dictionary(dictionary_id, 6)

def create_charuco_board(squares_x=8, squares_y=6, square_length=0.325, marker_length=0.244, dictionary_id=cv2.aruco.DICT_6X6_250):
    """
    Create a CharucoBoard object
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Length of square side in meters
        marker_length: Length of marker side in meters
        dictionary_id: ArUco dictionary ID to use
        
    Returns:
        The CharucoBoard object
    """
    # Check OpenCV version first to use the appropriate API
    if cv2.__version__.startswith("4.10") or cv2.__version__.startswith("4.11") or cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
        # For OpenCV 4.12.0-dev and newer
        try:
            # Create dictionary with marker size parameter
            aruco_dict = cv2.aruco.Dictionary(dictionary_id, 6)
            
            # Create CharucoBoard with the constructor
            board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y),  # (squaresX, squaresY) as a tuple
                square_length,  # squareLength
                marker_length,  # markerLength
                aruco_dict
            )
            print("Created CharucoBoard using CharucoBoard constructor for OpenCV 4.12+")
            return board
        except Exception as e:
            print(f"Error creating CharucoBoard for OpenCV 4.12+: {str(e)}")
            print("Please check your OpenCV installation and version.")
            sys.exit(1)
    else:
        # For older OpenCV versions
        # Get the ArUco dictionary
        aruco_dict = get_aruco_dictionary(dictionary_id)
        
        # Create the CharucoBoard
        try:
            # Try the newer API first
            board = cv2.aruco.CharucoBoard.create(
                squaresX=squares_x,
                squaresY=squares_y,
                squareLength=square_length,
                markerLength=marker_length,
                dictionary=aruco_dict
            )
            print("Created CharucoBoard using CharucoBoard.create")
            return board
        except Exception as e:
            try:
                # Try the older API
                board = cv2.aruco.CharucoBoard_create(
                    squaresX=squares_x,
                    squaresY=squares_y,
                    squareLength=square_length,
                    markerLength=marker_length,
                    dictionary=aruco_dict
                )
                print("Created CharucoBoard using CharucoBoard_create")
                return board
            except Exception as e2:
                print(f"Error creating CharucoBoard: {str(e)}")
                print(f"Second error: {str(e2)}")
                print("Please check your OpenCV installation and version.")
                sys.exit(1)

class CameraCalibrator:
    def __init__(self, calibration_type="charuco", charuco_params=None, chessboard_size=None, drone_mode=False):
        self.calibration_type = calibration_type
        self.pipeline = None
        self.device = None
        self.rgb_queue = None
        
        # Captured frames counter
        self.frame_count = 0
        self.last_capture_time = 0
        
        # Drone mode settings
        self.drone_mode = drone_mode
        if self.drone_mode:
            print("Drone mode enabled: Optimizing for detection at ranges from 0.5m to 12m")
            print("Please capture frames at various distances for best results")
            # Store distance information for each frame
            self.distances = []
        
        if calibration_type == "charuco":
            if charuco_params is None:
                # Default CharucoBoard parameters
                self.squares_x = 8
                self.squares_y = 6
                self.square_length = 0.325  # 32.5 cm
                self.marker_length = 0.244  # 24.4 cm (75% of square size)
            else:
                self.squares_x = charuco_params[0]
                self.squares_y = charuco_params[1]
                self.square_length = charuco_params[2]
                self.marker_length = self.square_length * 0.75  # 75% of square size
            
            # Create CharucoBoard
            self.board = create_charuco_board(
                squares_x=self.squares_x,
                squares_y=self.squares_y,
                square_length=self.square_length,
                marker_length=self.marker_length
            )
            
            # Get the ArUco dictionary
            self.aruco_dict = get_aruco_dictionary()
            
            # Initialize detector parameters - handle OpenCV 4.10+ differently
            if cv2.__version__.startswith("4.10") or cv2.__version__.startswith("4.11") or cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
                try:
                    # For OpenCV 4.12.0-dev and newer
                    self.aruco_params = cv2.aruco.DetectorParameters()
                    print("Using cv2.aruco.DetectorParameters() for OpenCV 4.12+")
                    
                    # Configure parameters for better detection
                    self.aruco_params.adaptiveThreshWinSizeMin = 3
                    self.aruco_params.adaptiveThreshWinSizeMax = 23
                    self.aruco_params.adaptiveThreshWinSizeStep = 10
                    self.aruco_params.adaptiveThreshConstant = 7
                    self.aruco_params.minMarkerPerimeterRate = 0.03
                    self.aruco_params.maxMarkerPerimeterRate = 4.0
                    self.aruco_params.polygonalApproxAccuracyRate = 0.05
                    self.aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX
                except Exception as e:
                    print(f"Error creating detector parameters for OpenCV 4.12+: {str(e)}")
                    print("Using None for parameters")
                    self.aruco_params = None
            else:
                # For older OpenCV versions
                try:
                    self.aruco_params = cv2.aruco.DetectorParameters.create()
                except:
                    try:
                        self.aruco_params = cv2.aruco.DetectorParameters_create()
                    except:
                        try:
                            self.aruco_params = cv2.aruco.DetectorParameters()
                        except Exception as e:
                            print(f"Error creating detector parameters: {str(e)}")
                            print("Using default parameters")
                            self.aruco_params = None
            
            # Arrays to store calibration data
            self.all_charuco_corners = []
            self.all_charuco_ids = []
            self.all_corners = []
            self.all_ids = []
            
            # For drone mode, track estimated distance for each frame
            if self.drone_mode:
                self.frame_distances = []
            
            print(f"Using CharucoBoard with {self.squares_x}x{self.squares_y} squares")
            print(f"Square size: {self.square_length*1000:.1f}mm, Marker size: {self.marker_length*1000:.1f}mm")
            
        else:  # chessboard
            self.chessboard_size = chessboard_size if chessboard_size else (9, 6)
            
            # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
            self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
            
            # Arrays to store object points and image points
            self.objpoints = []  # 3D points in real world space
            self.imgpoints = []  # 2D points in image plane
            
            print(f"Using Chessboard with {self.chessboard_size[0]}x{self.chessboard_size[1]} inner corners")
        
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
        
        if self.calibration_type == "charuco":
            print(f"Camera started. Looking for a CharucoBoard with {self.squares_x}x{self.squares_y} squares.")
            print("Move the board around to capture different angles and positions.")
        else:
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
                
                # Process frame based on calibration type
                if self.calibration_type == "charuco":
                    display_frame = self.process_charuco_frame(rgb_frame, gray)
                else:
                    display_frame = self.process_chessboard_frame(rgb_frame, gray)
                
                # Display the frame
                cv2.imshow("Calibration", display_frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
            # For drone mode, require more frames for better calibration
            required_frames = MIN_FRAMES * (2 if self.drone_mode else 1)
            if self.frame_count >= required_frames:
                print(f"\nCaptured {self.frame_count} frames, which is sufficient for calibration.")
                print("Press 'q' to exit and calculate calibration, or continue capturing more frames.")
                
        # Clean up
        cv2.destroyAllWindows()
        self.device.close()
        
        # Calculate calibration if we have enough frames
        if self.frame_count >= MIN_FRAMES:
            if self.calibration_type == "charuco":
                self.calculate_charuco_calibration(gray.shape[::-1])
            else:
                self.calculate_chessboard_calibration(gray.shape[::-1])
        else:
            print(f"Not enough frames captured ({self.frame_count}/{MIN_FRAMES}). Calibration aborted.")
    
    def process_charuco_frame(self, frame, gray):
        """
        Process a frame to detect CharucoBoard
        """
        display_frame = frame.copy()
        
        # Estimate distance for drone mode
        estimated_distance = None
        
        # Detect ArUco markers - handle OpenCV 4.12+ differently
        if cv2.__version__.startswith("4.10") or cv2.__version__.startswith("4.11") or cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
            try:
                # For OpenCV 4.12+, use the ArucoDetector
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, rejected = detector.detectMarkers(gray)
                print("Using ArucoDetector for marker detection")
            except Exception as e:
                print(f"Error using ArucoDetector: {e}")
                # Fallback to direct detection
                try:
                    corners, ids, rejected = cv2.aruco.detectMarkers(
                        gray,
                        self.aruco_dict,
                        parameters=self.aruco_params
                    )
                    print("Fallback to detectMarkers")
                except Exception as e2:
                    print(f"Error in fallback detection: {e2}")
                    corners = []
                    ids = None
                    rejected = []
        else:
            # For older OpenCV versions
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )
        
        # If markers are detected
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            
            # Interpolate CharucoBoard corners - handle OpenCV 4.12+ differently
            try:
                if cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
                    try:
                        # For OpenCV 4.12+, use the CharucoDetector
                        charuco_params = cv2.aruco.CharucoParameters()
                        charuco_detector = cv2.aruco.CharucoDetector(self.board, charuco_params)
                        
                        # Detect the board
                        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
                        
                        # Check if we have enough corners
                        ret = charuco_corners is not None and len(charuco_corners) > 4
                        if ret:
                            print(f"Detected CharucoBoard with {len(charuco_corners)} corners using CharucoDetector")
                        else:
                            charuco_corners = None
                            charuco_ids = None
                            ret = False
                    except Exception as e:
                        print(f"Error with CharucoDetector: {e}")
                        # Fallback to traditional method
                        try:
                            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                                corners,
                                ids,
                                gray,
                                self.board
                            )
                            print("Fallback to interpolateCornersCharuco")
                        except Exception as e2:
                            print(f"Error in fallback interpolation: {e2}")
                            ret = False
                            charuco_corners = None
                            charuco_ids = None
                else:
                    # For older OpenCV versions
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners,
                        ids,
                        gray,
                        self.board
                    )
            except Exception as e:
                print(f"Error interpolating corners: {str(e)}")
                ret = False
                charuco_corners = None
                charuco_ids = None
            
            # If CharucoBoard corners are detected, enough time has passed, and we have at least 6 corners
            if ret and charuco_corners is not None and len(charuco_corners) >= 6 and (time.time() - self.last_capture_time) > FRAME_INTERVAL:
                # Draw CharucoBoard corners
                cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)
                
                # Estimate distance based on marker size in the image
                if self.drone_mode:
                    # Use the first marker to estimate distance
                    marker_corners = corners[0][0]
                    # Calculate the perimeter of the marker
                    perimeter = cv2.arcLength(marker_corners, True)
                    # Estimate distance based on marker size and perimeter
                    # This is a rough estimate, actual distance would be more accurate with pose estimation
                    estimated_distance = (4 * self.marker_length * frame.shape[1]) / perimeter
                    
                    # Display estimated distance
                    cv2.putText(
                        display_frame,
                        f"Est. Distance: {estimated_distance:.2f}m",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )
                
                # Add to calibration data
                self.all_charuco_corners.append(charuco_corners)
                self.all_charuco_ids.append(charuco_ids)
                self.all_corners.append(corners)
                self.all_ids.append(ids)
                
                # Store distance information for drone mode
                if self.drone_mode and estimated_distance is not None:
                    self.frame_distances.append(estimated_distance)
                
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
                    # Show how many corners were detected and how many are needed
                    if charuco_corners is not None:
                        corners_count = len(charuco_corners)
                        if corners_count < 6:
                            cv2.putText(
                                display_frame,
                                f"Need more corners! Detected: {corners_count}/6 required",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 100, 255),  # Orange
                                2
                            )
                        else:
                            cv2.putText(
                                display_frame,
                                f"CharucoBoard detected with {corners_count} corners! Hold still...",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),  # Yellow
                                2
                            )
                    else:
                        cv2.putText(
                            display_frame,
                            "CharucoBoard detected but corners not found",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )
                else:
                    cv2.putText(
                        display_frame,
                        "Not enough markers detected for CharucoBoard",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),  # Red
                        2
                    )
        else:
            # No markers detected
            cv2.putText(
                display_frame,
                "No markers detected",
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
        
        return display_frame
    
    def process_chessboard_frame(self, frame, gray):
        """
        Process a frame to detect chessboard
        """
        display_frame = frame.copy()
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size, 
            None
        )
        
        # If found, add object points and image points
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
        
        return display_frame
        
    def get_rgb_frame(self):
        """
        Get the latest RGB frame from the camera
        """
        in_rgb = self.rgb_queue.tryGet()
        if in_rgb is not None:
            return in_rgb.getCvFrame()
        return None
    
    def calculate_charuco_calibration(self, img_size):
        """
        Calculate camera calibration using CharucoBoard
        """
        print("\nCalculating camera calibration using CharucoBoard...")
        
        # For drone mode, analyze distance distribution
        if self.drone_mode and len(self.frame_distances) > 0:
            distances = np.array(self.frame_distances)
            print(f"\nDistance statistics:")
            print(f"  Min distance: {np.min(distances):.2f}m")
            print(f"  Max distance: {np.max(distances):.2f}m")
            print(f"  Mean distance: {np.mean(distances):.2f}m")
            print(f"  Std deviation: {np.std(distances):.2f}m")
            
            # Check if we have a good distribution of distances
            if np.max(distances) - np.min(distances) < 2.0:
                print("\nWARNING: Limited distance range in calibration frames.")
                print("For best results with drone applications, capture frames at various distances (0.5m to 12m).")
        
        # Check if we have enough data
        if len(self.all_charuco_corners) < 4:
            print("Not enough corners detected. Calibration failed.")
            return
        
        # Prepare arrays for calibration
        corners_all = []
        ids_all = []
        
        # Combine all detected corners and IDs, filtering out frames with too few corners
        valid_frames = 0
        for corners, ids in zip(self.all_charuco_corners, self.all_charuco_ids):
            if len(corners) >= 6:  # Ensure minimum of 6 corners for DLT algorithm
                corners_all.append(corners)
                ids_all.append(ids)
                valid_frames += 1
            else:
                print(f"Skipping a frame with only {len(corners)} corners (minimum 6 required)")
        
        print(f"Using {valid_frames} valid frames for calibration (each with 6+ corners)")
        
        # Check if we have enough valid frames
        if valid_frames < 4:  # Need at least 4 frames for a good calibration
            print("Not enough valid frames with 6+ corners. Please recapture with a clearer view of the CharucoBoard.")
            print("Tips for better calibration:")
            print("- Ensure good, even lighting on the board")
            print("- Avoid motion blur by holding the camera steady")
            print("- Make sure most of the board is visible in each frame")
            print("- Try different angles and distances")
            return
        
        # Perform calibration - handle OpenCV 4.12+ differently
        if cv2.__version__.startswith("4.12") or cv2.__version__.startswith("4.13") or cv2.__version__.startswith("4.14"):
            try:
                print("Using OpenCV 4.12+ calibration approach")
                
                # For OpenCV 4.12+, we need to use the CharucoDetector and calibrateCamera
                # First, prepare object points for each detected corner
                objPoints = []
                imgPoints = []
                
                # For each frame
                for frame_idx in range(len(corners_all)):
                    frame_corners = corners_all[frame_idx]
                    frame_ids = ids_all[frame_idx]
                    
                    # For each corner in this frame
                    frame_obj_points = []
                    frame_img_points = []
                    
                    for i in range(len(frame_corners)):
                        # Get the corner ID
                        corner_id = frame_ids[i][0]
                        
                        # Calculate 3D position based on square size
                        # For a 6x6 board, there are 7 corners in each direction
                        row = corner_id // 7
                        col = corner_id % 7
                        
                        # Use the board's square size to calculate 3D position
                        square_length = self.board.getSquareLength()
                        frame_obj_points.append([col * square_length, row * square_length, 0])
                        frame_img_points.append(frame_corners[i][0])
                    
                    # Add to the collection if we have enough points
                    if len(frame_obj_points) >= 6:
                        objPoints.append(np.array(frame_obj_points, dtype=np.float32))
                        imgPoints.append(np.array(frame_img_points, dtype=np.float32))
                
                # Use standard camera calibration with the prepared points
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objPoints,
                    imgPoints,
                    img_size,
                    None,
                    None
                )
                print("Calibration performed using cv2.calibrateCamera for OpenCV 4.12+")
            except Exception as e:
                print(f"Error during OpenCV 4.12+ calibration: {str(e)}")
                print("Falling back to standard calibration approach")
                try:
                    # Try standard calibration as fallback
                    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                        objPoints,
                        imgPoints,
                        img_size,
                        None,
                        None
                    )
                    print("Calibration performed using fallback cv2.calibrateCamera")
                except Exception as e2:
                    print(f"Fallback calibration also failed: {str(e2)}")
                    print("Calibration failed. Please try again with a clearer CharucoBoard pattern.")
                    return
        else:
            # For older OpenCV versions, use the traditional API
            try:
                # Try the newer API first
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                    charucoCorners=corners_all,
                    charucoIds=ids_all,
                    board=self.board,
                    imageSize=img_size,
                    cameraMatrix=None,
                    distCoeffs=None
                )
                print("Calibration performed using calibrateCameraCharuco")
            except Exception as e:
                try:
                    # Try the older API
                    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                        corners_all,
                        ids_all,
                        self.board,
                        img_size,
                        None,
                        None
                    )
                    print("Calibration performed using calibrateCameraCharuco (older API)")
                except Exception as e2:
                    print(f"Error during calibration: {str(e)}")
                    print(f"Second error: {str(e2)}")
                    print("Calibration failed. Please try again with a clearer CharucoBoard pattern.")
                    return
        
        if ret:
            # Save calibration data
            if self.drone_mode and len(self.frame_distances) > 0:
                # Save with distance information for drone applications and CharucoBoard parameters
                np.savez(
                    CALIB_FILE,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    drone_optimized=True,
                    min_distance=np.min(self.frame_distances),
                    max_distance=np.max(self.frame_distances),
                    mean_distance=np.mean(self.frame_distances),
                    charuco_calibration=True,
                    squares_x=self.squares_x,
                    squares_y=self.squares_y,
                    square_length=self.square_length,
                    marker_length=self.marker_length
                )
            else:
                # Standard save with CharucoBoard parameters
                np.savez(
                    CALIB_FILE,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    charuco_calibration=True,
                    squares_x=self.squares_x,
                    squares_y=self.squares_y,
                    square_length=self.square_length,
                    marker_length=self.marker_length
                )
            
            print(f"Calibration successful! Data saved to {CALIB_FILE}")
            print("\nCamera Matrix:")
            print(camera_matrix)
            print("\nDistortion Coefficients:")
            print(dist_coeffs)
            
            # Calculate reprojection error
            mean_error = 0
            total_points = 0
            
            for i in range(len(corners_all)):
                if len(corners_all[i]) > 0:
                    imgpoints2, _ = cv2.projectPoints(
                        self.board.chessboardCorners[ids_all[i]],
                        rvecs[i],
                        tvecs[i],
                        camera_matrix,
                        dist_coeffs
                    )
                    error = cv2.norm(corners_all[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                    total_points += 1
            
            if total_points > 0:
                print(f"\nReprojection Error: {mean_error/total_points}")
            
            print("\nCalibration complete! You can now use the ArUco marker detector with improved accuracy.")
        else:
            print("Calibration failed. Please try again with a clearer CharucoBoard pattern.")
    
    def calculate_chessboard_calibration(self, img_size):
        """
        Calculate camera calibration using chessboard
        """
        print("\nCalculating camera calibration using chessboard...")
        
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
    parser = argparse.ArgumentParser(description='OAK-D Camera Calibration for Drone Applications')
    
    # Create mutually exclusive group for calibration type
    calib_type = parser.add_mutually_exclusive_group()
    
    # CharucoBoard calibration arguments
    calib_type.add_argument('--charuco', nargs='+', type=float, 
                          help='Use CharucoBoard with squares_x squares_y square_length')
    
    # Chessboard calibration arguments
    calib_type.add_argument('--chessboard', nargs=2, type=int,
                          help='Use chessboard with width height inner corners')
                          
    # Drone mode flag
    parser.add_argument('--drone', action='store_true',
                      help='Optimize calibration for drone-based detection at various distances')
    
    args = parser.parse_args()
    
    # Determine calibration type and parameters
    if args.charuco is not None:
        # CharucoBoard calibration
        if len(args.charuco) < 2:
            print("Error: --charuco requires at least squares_x and squares_y")
            parser.print_help()
            sys.exit(1)
        
        squares_x = int(args.charuco[0])
        squares_y = int(args.charuco[1])
        
        # Square length is optional, default to 0.025m (2.5cm)
        square_length = args.charuco[2] if len(args.charuco) > 2 else 0.025
        
        print(f"Initializing OAK-D Camera Calibration with CharucoBoard ({squares_x}x{squares_y}, {square_length*1000:.1f}mm squares)...")
        calibrator = CameraCalibrator(
            calibration_type="charuco",
            charuco_params=[squares_x, squares_y, square_length],
            drone_mode=args.drone
        )
    elif args.chessboard is not None:
        # Chessboard calibration
        width, height = args.chessboard
        print(f"Initializing OAK-D Camera Calibration with chessboard ({width}x{height} inner corners)...")
        calibrator = CameraCalibrator(
            calibration_type="chessboard",
            chessboard_size=(width, height),
            drone_mode=args.drone
        )
    else:
        # Default to CharucoBoard calibration
        print("No calibration type specified, defaulting to CharucoBoard (8x6, 32.5cm squares)...")
        calibrator = CameraCalibrator(
            calibration_type="charuco",
            drone_mode=args.drone
        )
    
    # Start calibration
    calibrator.start()

if __name__ == "__main__":
    main()
