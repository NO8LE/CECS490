#!/usr/bin/env python3

"""
GCS Video Receiver for UAV Perception Stream

This script receives the H.264 RTP video stream from the UAV's Jetson
and displays it using OpenCV. It's designed to run on the Ground Control Station (GCS).

Usage:
  python3 gcs_video_receiver.py [--port PORT] [--display-info]

Options:
  --port PORT           UDP port to receive the stream (default: 5600)
  --display-info, -d    Display additional stream information

Example:
  python3 gcs_video_receiver.py --port 5600 --display-info

Press 'q' to exit the program.
"""

import cv2
import argparse
import time
import numpy as np

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GCS Video Receiver for UAV Perception Stream')
    parser.add_argument('--port', type=int, default=5600, help='UDP port to receive the stream (default: 5600)')
    parser.add_argument('--display-info', '-d', action='store_true', help='Display additional stream information')
    args = parser.parse_args()
    
    # Construct the GStreamer pipeline for receiving the RTP stream
    gst_pipeline = (
        f"udpsrc port={args.port} caps=\"application/x-rtp, media=(string)video, "
        f"clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96\" ! "
        f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
    )
    
    print(f"Starting video receiver on port {args.port}")
    print("Waiting for stream from UAV...")
    print("Press 'q' to exit")
    
    # Open the video capture with the GStreamer pipeline
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open video stream. Make sure GStreamer is installed correctly.")
        print("If using OpenCV from pip, it might not have GStreamer support.")
        print("Alternative: Use VLC or ffplay to view the stream.")
        print(f"ffplay command: ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental \"udp://@:{args.port}?buffer_size=120000\"")
        return
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Main loop
    while True:
        # Read a frame from the stream
        ret, frame = cap.read()
        
        if not ret:
            print("No frame received. Waiting...")
            time.sleep(0.1)
            continue
        
        # Update performance metrics
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Display additional information if requested
        if args.display_info:
            # Add FPS counter
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add frame dimensions
            height, width = frame.shape[:2]
            cv2.putText(
                frame,
                f"Resolution: {width}x{height}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add latency estimation (very approximate)
            # This is just a placeholder - real latency measurement would require timestamps
            cv2.putText(
                frame,
                "Stream latency: measuring...",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Display the frame
        cv2.imshow("UAV Video Stream", frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Video receiver stopped")

if __name__ == "__main__":
    main()