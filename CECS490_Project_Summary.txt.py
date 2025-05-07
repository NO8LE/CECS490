==============================================================================
                    CECS490 PROJECT SUMMARY
==============================================================================
                OAK-D ArUco Marker Detection System
                  for Drone-Based Applications
==============================================================================

------------------------------------------------------------------------------
                           PROJECT OVERVIEW
------------------------------------------------------------------------------

This project implements a sophisticated 3D marker detection system using the
Luxonis OAK-D camera to detect ArUco markers, calculate their precise 3D
positions in space, and visualize the results in real-time. The system has been
optimized for drone-based detection at ranges from 0.5m to 12m with 12-inch
(0.3048m) markers. It leverages the stereo depth capabilities of the OAK-D
camera for accurate spatial location and provides robust compatibility across
different platforms, including specialized support for NVIDIA Jetson Orin
hardware.

------------------------------------------------------------------------------
                          SYSTEM ARCHITECTURE
------------------------------------------------------------------------------

┌─────────────────────────────────────────────────────────────┐
│                     OAK-D Camera System                      │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     DepthAI Pipeline                         │
│  ┌───────────┐     ┌───────────┐     ┌───────────────────┐  │
│  │ RGB Camera├────►│ StereoDepth├────►│Spatial Calculator │  │
│  └───────────┘     └─────┬─────┘     └─────────┬─────────┘  │
│  ┌───────────┐           │                     │            │
│  │ Left Mono │           │                     │            │
│  │  Camera   ├───────────┘                     │            │
│  └───────────┘                                 │            │
│  ┌───────────┐                                 │            │
│  │Right Mono │                                 │            │
│  │  Camera   ├───────────────────────────────►│            │
│  └───────────┘                                 │            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                       │
│  ┌───────────┐     ┌───────────┐     ┌───────────────────┐  │
│  │ArUco Marker│    │ 3D Position│    │ Visualization &   │  │
│  │ Detection  ├───►│ Calculation├───►│   Display         │  │
│  └─────┬─────┘     └───────────┘     └───────────────────┘  │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────┐     ┌───────────┐     ┌───────────────────┐  │
│  │Multi-scale │    │ Adaptive   │    │ Target Tracking & │  │
│  │ Detection  ├───►│ Parameters ├───►│   Guidance        │  │
│  └───────────┘     └───────────┘     └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘

------------------------------------------------------------------------------
                           DATA FLOW DIAGRAM
------------------------------------------------------------------------------

┌──────────────┐    ┌───────────────┐    ┌────────────────┐
│ RGB Image    │    │ Grayscale     │    │ ArUco Marker   │
│ Acquisition  ├───►│ Conversion    ├───►│ Detection      │
└──────────────┘    └───────────────┘    └────────┬───────┘
                                                  │
┌──────────────┐    ┌───────────────┐             │
│ Stereo Depth │    │ Spatial       │             │
│ Calculation  ├───►│ Coordinates   │             │
└──────────────┘    └───────┬───────┘             │
                            │                     │
                            ▼                     ▼
                    ┌───────────────────────────────┐
                    │ Marker Pose Estimation        │
                    │ (Position & Orientation)      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Distance Estimation           │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Adaptive Parameter Selection  │
                    │ - Resolution profile          │
                    │ - Detection parameters        │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Target Marker Prioritization  │
                    │ (if target ID specified)      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Visualization & Display       │
                    │ - RGB with marker overlay     │
                    │ - Depth map                   │
                    │ - 3D coordinates              │
                    │ - Orientation angles          │
                    │ - Targeting guidance          │
                    └───────────────────────────────┘

------------------------------------------------------------------------------
                             STATE DIAGRAM
------------------------------------------------------------------------------

┌───────────────┐
│  INITIALIZING │
└───────┬───────┘
        │
        ▼
┌───────────────┐     Error     ┌───────────────┐
│  CONNECTING   ├──────────────►│  ERROR STATE  │
└───────┬───────┘               └───────┬───────┘
        │                               │
        │ Connected                     │ Retry
        ▼                               │
┌───────────────┐                       │
│  CALIBRATING  ├───────────────────────┘
└───────┬───────┘
        │
        │ Calibrated
        ▼
┌───────────────┐     No Markers    ┌───────────────┐
│  IDLE         ├────────────────┬─►│  SEARCHING    │
└───────┬───────┘                │  └───────┬───────┘
        │                        │          │
        │ Marker Detected        │          │ Marker Found
        ▼                        │          │
┌───────────────┐                │          │
│  TRACKING     ├────────────────┘          │
└───────┬───────┘ Marker Lost              │
        │                                   │
        │ Target Marker                     │
        ▼                                   │
┌───────────────┐                           │
│  TARGETING    ├───────────────────────────┘
└───────┬───────┘ Target Lost
        │
        │ Target Centered
        ▼
┌───────────────┐
│  CENTERED     ├───────────────────────────┐
└───────┬───────┘ Target Off-center         │
        │                                   │
        │ User Input 'q'                    │
        ▼                                   │
┌───────────────┐                           │
│  SHUTDOWN     │◄──────────────────────────┘
└───────────────┘

------------------------------------------------------------------------------
                            SYSTEM WORKFLOW
------------------------------------------------------------------------------

┌─────────────────┐
│ Start System    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Load Camera     │     │ Generate ArUco  │     │ Generate        │
│ Calibration     │◄────┤ Markers         │     │ CharucoBoard    │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │                                             ▲
         │                                             │
         ▼                                             │
┌─────────────────┐     ┌─────────────────┐           │
│ Initialize      │     │ Calibrate with  │           │
│ DepthAI Pipeline│     │ CharucoBoard    ├───────────┘
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Start Camera    │
│ Streams         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Process Frames  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Detect ArUco    │     │ Calculate 3D    │
│ Markers         ├────►│ Position        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
┌─────────────────┐     ┌────────▼────────┐
│ Estimate        │     │ Select Adaptive │
│ Distance        ├────►│ Parameters      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Update Spatial  │     │ Draw Marker     │
│ ROI             │◄────┤ Information     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
┌─────────────────┐     ┌────────▼────────┐
│ Prioritize      │     │ Generate Target │
│ Target Marker   ├────►│ Guidance        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Display RGB &   │     │ Wait for        │
│ Depth Frames    ├────►│ Key Press       │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Exit (on 'q')   │
                        └─────────────────┘

------------------------------------------------------------------------------
                            KEY COMPONENTS
------------------------------------------------------------------------------


----------------------------------

1. CHARUCO BOARD CALIBRATION SYSTEM
----------------------------------

The system includes a robust camera calibration module that uses a CharucoBoard
pattern to calculate intrinsic camera parameters, ensuring accurate pose
estimation at various distances.

Key Code:
```
# CharucoBoard calibration process
def calculate_charuco_calibration(self, img_size):
    # For drone mode, analyze distance distribution
    if self.drone_mode and len(self.frame_distances) > 0:
        distances = np.array(self.frame_distances)
        print(f"\nDistance statistics:")
        print(f"  Min distance: {np.min(distances):.2f}m")
        print(f"  Max distance: {np.max(distances):.2f}m")
        print(f"  Mean distance: {np.mean(distances):.2f}m")

    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=self.board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    # Save calibration data with distance information for drone applications
    np.savez(
        CALIB_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        drone_optimized=True,
        min_distance=np.min(self.frame_distances),
        max_distance=np.max(self.frame_distances)
    )
```

------------------------------------
2. MULTI-SCALE ARUCO MARKER DETECTION
------------------------------------

The system uses a multi-scale approach to detect ArUco markers at various
distances, improving detection reliability at long ranges.

Key Code:
```
# Multi-scale ArUco marker detection
def detect_aruco_markers(self, frame):
    # Preprocess image
    gray = self.preprocess_image(frame)

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        self.aruco_dict,
        parameters=self.aruco_params
    )

    # If no markers found and we're looking for distant markers, try with different parameters
    if (ids is None or len(ids) == 0) and self.current_profile == "far":
        # Try with enhanced parameters for distant detection
        backup_params = self.aruco_params
        try:
            enhanced_params = cv2.aruco.DetectorParameters.create()
            enhanced_params.adaptiveThreshConstant = 15
            enhanced_params.minMarkerPerimeterRate = 0.02
            enhanced_params.polygonalApproxAccuracyRate = 0.12

            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=enhanced_params
            )
        except:
            # Restore original parameters
            self.aruco_params = backup_params

    # If still no markers found, try with a scaled version of the image
    if ids is None or len(ids) == 0:
        # Try with 75% scale
        h, w = gray.shape
        scaled_gray = cv2.resize(gray, (int(w*0.75), int(h*0.75)))
        scaled_corners, scaled_ids, _ = cv2.aruco.detectMarkers(
            scaled_gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        # If markers found in scaled image, convert coordinates back to original scale
        if scaled_ids is not None and len(scaled_ids) > 0:
            scale_factor = 1/0.75
            for i in range(len(scaled_corners)):
                scaled_corners[i][0][:, 0] *= scale_factor
                scaled_corners[i][0][:, 1] *= scale_factor
            corners = scaled_corners
            ids = scaled_ids
```

3. ADAPTIVE PARAMETER SELECTION
-----------------------------

The system dynamically adjusts detection parameters based on the estimated
distance to the markers, optimizing for different ranges.

Key Code:
```python
# Update detection parameters based on estimated distance
if self.estimated_distance > 8000:  # > 8m
    if self.current_profile != "far":
        self.apply_detection_profile("far")
elif self.estimated_distance > 3000:  # 3-8m
    if self.current_profile != "medium":
        self.apply_detection_profile("medium")
else:  # < 3m
    if self.current_profile != "close":
        self.apply_detection_profile("close")

# Apply detection profile
def apply_detection_profile(self, profile_name):
    if self.aruco_params is None:
        return

    profile = DETECTION_PROFILES[profile_name]
    for param, value in profile.items():
        try:
            setattr(self.aruco_params, param, value)
        except Exception as e:
            print(f"Warning: Could not set parameter {param}: {e}")

    self.current_profile = profile_name
    print(f"Applied detection profile: {profile_name}")
```
```
-------------------------------
4. TARGET MARKER PRIORITIZATION
-------------------------------

The system can prioritize a specific marker among many, providing visual
guidance to center on the target.

Key Code:
```
# Draw targeting guidance
def _draw_targeting_guidance(self, frame):
    if not hasattr(self, 'target_found') or not self.target_found:
        return

    # Get frame center
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Calculate offset from center
    if hasattr(self, 'target_center'):
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
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)

            # Add guidance text
            direction_text = ""
            if abs(offset_y) > 30:
                direction_text += "UP " if offset_y < 0 else "DOWN "
            if abs(offset_x) > 30:
                direction_text += "LEFT" if offset_x < 0 else "RIGHT"

            cv2.putText(
                frame,
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
                frame,
                "TARGET CENTERED",
                (center_x - 100, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


```

----------------------------------
5. CUDA ACCELERATION FOR JETSON ORIN
----------------------------------

The system leverages CUDA acceleration on the Jetson Orin Nano for improved
performance.

Key Code:
```
# Check for CUDA support
USE_CUDA = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}")
        USE_CUDA = True
        print("CUDA acceleration enabled")
    else:
        print("No CUDA devices found, using CPU")
except:
    print("CUDA not available in OpenCV, using CPU")

# GPU-accelerated image preprocessing
def preprocess_image_gpu(self, frame):
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

```
--------------------------------------
6. PERFORMANCE MONITORING AND ADAPTATION
--------------------------------------

The system monitors performance and adapts processing to maintain real-time
operation.

Key Code:
```
# Monitor and adapt processing based on performance
def monitor_performance(self):
    if len(self.frame_times) < 10:
        return

    # Calculate average processing time
    avg_time = sum(self.frame_times) / len(self.frame_times)

    # Calculate detection success rate
    success_rate = sum(self.detection_success_rate) / len(self.detection_success_rate)

    print(f"Avg processing time: {avg_time*1000:.1f}ms, Success rate: {success_rate*100:.1f}%")

    # If processing is too slow, reduce resolution or simplify detection
    if avg_time > 0.1:  # More than 100ms per frame
        self.skip_frames = 1  # Process every other frame
        print("Performance warning: Processing time > 100ms, skipping frames")
    else:
        self.skip_frames = 0  # Process every frame


------------------------------------------------------------------------------
                  COMPATIBILITY AND DEPENDENCY MANAGEMENT
------------------------------------------------------------------------------

The project includes sophisticated dependency management to handle various
versions of OpenCV, NumPy, and other libraries, ensuring compatibility across
different platforms.

Key Code:
```python
# Verify ArUco module is working with multiple fallback methods
try:
    # Try old API first
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    dictionary_method = "old"
except Exception as e1:
    # If old API fails, try new API
    try:
        aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_6X6_250)
        dictionary_method = "new"
    except Exception as e2:
        # Try additional methods...
```

------------------------------------------------------------------------------
                         JETSON ORIN INTEGRATION
------------------------------------------------------------------------------

The project includes specialized support for NVIDIA Jetson Orin hardware, with
power management features for optimal performance.

Key Code:
```python
# Set Jetson power mode based on processing needs
def set_power_mode(self, high_performance=False):
    try:
        if high_performance:
            # Maximum performance when actively searching for markers
            os.system("sudo nvpmodel -m 0")  # Max performance mode
            os.system("sudo jetson_clocks")   # Max clock speeds
            print("Set Jetson to high performance mode")
        else:
            # Power saving when idle or markers are consistently tracked
            os.system("sudo nvpmodel -m 1")  # Power saving mode
            print("Set Jetson to power saving mode")
    except Exception as e:
        print(f"Failed to set power mode: {e}")
```

------------------------------------------------------------------------------
                         TECHNICAL ACHIEVEMENTS
------------------------------------------------------------------------------

1. LONG-RANGE DETECTION
   The system can reliably detect 6x6 ArUco markers at distances up to 12m
   using multi-scale detection and adaptive parameters.

2. DRONE-BASED OPTIMIZATION
   The system is specifically optimized for drone applications, with
   considerations for varying distances, motion, and power efficiency.

3. TARGET PRIORITIZATION
   When multiple markers are in view, the system can prioritize a specific
   target and provide visual guidance to center on it.

4. ADAPTIVE PROCESSING
   The system dynamically adjusts resolution, detection parameters, and
   processing intensity based on estimated distance and system performance.

5. CHARUCO CALIBRATION
   The system uses CharucoBoard calibration for improved accuracy at various
   distances, with distance-specific calibration profiles.

6. JETSON ORIN OPTIMIZATION
   The system leverages CUDA acceleration and power management features of the
   Jetson Orin Nano for optimal performance on drone platforms.

------------------------------------------------------------------------------
                            FUTURE DIRECTIONS
------------------------------------------------------------------------------

1. MARKER TRACKING WITH KALMAN FILTERING
   Enhance the marker tracking with Kalman filtering for improved stability
   during rapid drone movements.

2. INTEGRATION WITH DRONE FLIGHT CONTROL
   Add direct integration with drone flight control systems for autonomous
   navigation based on marker detection.

3. MULTI-CAMERA FUSION
   Implement fusion of multiple OAK-D cameras for 360-degree marker detection
   and improved accuracy.

4. DEEP LEARNING ENHANCEMENT
   Add deep learning-based marker detection for improved robustness in
   challenging lighting and environmental conditions.

5. EXTENDED RANGE DETECTION
   Further optimize for even longer detection ranges (15-20m) with larger
   markers and specialized detection algorithms.

==============================================================================
                               CONCLUSION
==============================================================================

This project demonstrates significant progress in computer vision and spatial
computing for drone applications. The OAK-D ArUco Marker Detection System
provides a robust foundation for drone-based marker detection at ranges from
0.5m to 12m, with specialized features for target prioritization, adaptive
processing, and performance optimization.

The combination of accurate 3D position estimation, long-range detection, and
target tracking makes this system suitable for a wide range of drone
applications, from autonomous navigation to inspection and monitoring tasks.

==============================================================================
