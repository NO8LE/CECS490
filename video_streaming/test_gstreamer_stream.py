#!/usr/bin/env python3

"""
GStreamer Streaming Test Script

This script tests the GStreamer H.264/RTP streaming pipeline without requiring
the OAK-D camera. It generates a test pattern or uses a webcam and streams it
using the same GStreamer pipeline as the main application.

Usage:
  python3 test_gstreamer_stream.py [--ip IP] [--port PORT] [--bitrate BITRATE] [--webcam]

Options:
  --ip IP               IP address to stream to (default: 192.168.1.100)
  --port PORT           Port to stream to (default: 5000)
  --bitrate BITRATE     Streaming bitrate in bits/sec (default: 4000000)
  --webcam, -w          Use webcam instead of test pattern
  --width WIDTH         Video width (default: 1280)
  --height HEIGHT       Video height (default: 720)
  --fps FPS             Frames per second (default: 30)

Example:
  python3 test_gstreamer_stream.py --ip 192.168.0.10 --port 5001
  python3 test_gstreamer_stream.py --webcam

Press 'q' to exit.
"""

import cv2
import numpy as np
import argparse
import time
import sys

def create_test_pattern(width, height, frame_count):
    """Create a test pattern with moving elements"""
    # Create a base pattern
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add a color gradient background
    for y in range(height):
        for x in range(width):
            b = int(255 * x / width)
            g = int(255 * y / height)
            r = int(255 * (x + y) / (width + height))
            pattern[y, x] = [b, g, r]
    
    # Make a copy to avoid modifying the base pattern
    frame = pattern.copy()
    
    # Add moving elements
    # 1. Moving circle
    circle_x = int(width/2 + width/4 * np.sin(frame_count * 0.05))
    circle_y = int(height/2 + height/4 * np.cos(frame_count * 0.05))
    cv2.circle(frame, (circle_x, circle_y), 50, (0, 255, 255), -1)
    
    # 2. Moving rectangle
    rect_x = int(width/2 + width/4 * np.cos(frame_count * 0.03))
    rect_y = int(height/2 + height/4 * np.sin(frame_count * 0.03))
    cv2.rectangle(frame, (rect_x-40, rect_y-40), (rect_x+40, rect_y+40), (255, 0, 255), -1)
    
    # 3. Pulsing center circle
    radius = int(50 + 20 * np.sin(frame_count * 0.1))
    cv2.circle(frame, (width//2, height//2), radius, (0, 255, 0), -1)
    
    # Add frame counter and timestamp
    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(
        frame,
        f"Time: {timestamp}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    # Add "GStreamer Test" label
    cv2.putText(
        frame,
        "GStreamer Test Stream",
        (width//2 - 200, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    return frame

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GStreamer Streaming Test Script')
    parser.add_argument('--ip', type=str, default='192.168.251.105', help='IP address to stream to (default: 192.168.251.105)')
    parser.add_argument('--port', type=int, default=5000, help='Port to stream to (default: 5000)')
    parser.add_argument('--bitrate', type=int, default=4000000, help='Streaming bitrate in bits/sec (default: 4000000)')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam instead of test pattern')
    parser.add_argument('--width', type=int, default=1280, help='Video width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Video height (default: 720)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    args = parser.parse_args()
    
    # Construct the GStreamer pipeline for streaming
    gst_pipeline = (
        f"appsrc ! video/x-raw,format=BGR ! videoconvert ! "
        f"video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        f"nvv4l2h264enc insert-sps-pps=1 bitrate={args.bitrate} preset-level=1 iframeinterval=30 ! "
        f"h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host={args.ip} port={args.port} sync=false"
    )
    
    # Alternative pipeline for systems without NVENC
    alt_pipeline = (
        f"appsrc ! video/x-raw,format=BGR ! videoconvert ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={args.bitrate//1000} ! "
        f"h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host={args.ip} port={args.port} sync=false"
    )
    
    print(f"Starting GStreamer test stream to {args.ip}:{args.port}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Bitrate: {args.bitrate} bps")
    print("Press 'q' to exit")
    
    # Initialize video source
    if args.webcam:
        print("Using webcam as video source")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
    else:
        print("Using test pattern as video source")
        cap = None
    
    # Initialize video writer with GStreamer pipeline
    try:
        writer = cv2.VideoWriter(
            gst_pipeline,
            cv2.CAP_GSTREAMER,
            0,  # Codec is ignored when using GStreamer
            float(args.fps),
            (args.width, args.height)
        )
        
        if not writer.isOpened():
            print("Failed to open primary GStreamer pipeline. Trying alternative pipeline...")
            writer = cv2.VideoWriter(
                alt_pipeline,
                cv2.CAP_GSTREAMER,
                0,
                float(args.fps),
                (args.width, args.height)
            )
            
            if not writer.isOpened():
                print("Error: Failed to open video writer pipeline.")
                print("Make sure GStreamer is properly installed.")
                return
    except Exception as e:
        print(f"Error initializing GStreamer pipeline: {e}")
        return
    
    # Print SDP information for client playback
    print("\nTo view the stream on the client (GCS), create a file named stream.sdp with these contents:")
    print("c=IN IP4 0.0.0.0")
    print(f"m=video {args.port} RTP/AVP 96")
    print("a=rtpmap:96 H264/90000")
    print("\nThen use VLC to open this file, or use ffplay with:")
    print(f"ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental \"udp://@:{args.port}?buffer_size=120000\"")
    
    # Main loop
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Get frame from webcam or generate test pattern
            if args.webcam:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from webcam")
                    break
                    
                # Resize frame if needed
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))
            else:
                # Generate test pattern
                frame = create_test_pattern(args.width, args.height, frame_count)
            
            # Add FPS information
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # Stream the frame
            writer.write(frame)
            
            # Display the frame locally
            cv2.imshow("Test Stream", frame)
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(f"Streaming at {fps:.1f} FPS")
            
            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Add a small delay to control frame rate if not using webcam
            if not args.webcam:
                time.sleep(1.0 / args.fps)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        if args.webcam and cap is not None:
            cap.release()
            
        if writer is not None:
            writer.release()
            
        cv2.destroyAllWindows()
        print("Test stream stopped")

if __name__ == "__main__":
    main()