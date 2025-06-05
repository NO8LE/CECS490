#!/usr/bin/env python3

"""
Advanced Stream Processor for UAV Video Feed

This script demonstrates how to receive the H.264 RTP video stream from the UAV
and perform additional processing on the frames. It can be used as a starting point
for integrating the video feed into custom GCS applications.

Features:
- Receives and decodes the H.264 RTP stream
- Displays the video with optional information overlay
- Performs basic image processing (edge detection, color filtering)
- Demonstrates how to extract information from the stream
- Provides a framework for custom processing algorithms

Usage:
  python3 advanced_stream_processor.py [--port PORT] [--mode MODE] [--record]

Options:
  --port PORT           UDP port to receive the stream (default: 5600)
  --mode MODE           Processing mode: normal, edges, hsv, or custom (default: normal)
  --record              Record the received video to a file

Example:
  python3 advanced_stream_processor.py --port 5600 --mode edges --record

Press 'q' to exit, 'm' to cycle through modes, 'r' to toggle recording.
"""

import cv2
import argparse
import time
import numpy as np
import os
from datetime import datetime

class StreamProcessor:
    def __init__(self, port=5600, mode="normal", record=False):
        self.port = port
        self.mode = mode
        self.record = record
        self.recording = False
        self.video_writer = None
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.modes = ["normal", "edges", "hsv", "custom"]
        
        # Create output directory for recordings
        self.output_dir = "stream_recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Construct the GStreamer pipeline for receiving the RTP stream
        self.gst_pipeline = (
            f"udpsrc port={self.port} caps=\"application/x-rtp, media=(string)video, "
            f"clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96\" ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
        )
        
        print(f"Starting advanced stream processor on port {self.port}")
        print(f"Initial processing mode: {self.mode}")
        print("Press 'q' to exit, 'm' to cycle through modes, 'r' to toggle recording")
        
    def start(self):
        # Open the video capture with the GStreamer pipeline
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("Error: Could not open video stream. Make sure GStreamer is installed correctly.")
            print("If using OpenCV from pip, it might not have GStreamer support.")
            return False
            
        return True
        
    def process_frame(self, frame):
        """Process the frame based on the current mode"""
        if self.mode == "normal":
            # Just return the original frame
            processed = frame
            
        elif self.mode == "edges":
            # Edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            processed = cv2.Canny(blurred, 50, 150)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
        elif self.mode == "hsv":
            # HSV color filtering (highlight green objects)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([80, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            processed = cv2.bitwise_and(frame, frame, mask=mask)
            
        elif self.mode == "custom":
            # Custom processing - modify this for your specific needs
            # This example adds a simple sharpening filter
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            processed = cv2.filter2D(frame, -1, kernel)
            
        else:
            processed = frame
            
        return processed
        
    def add_info_overlay(self, frame):
        """Add information overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Add FPS counter
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add processing mode
        cv2.putText(
            frame,
            f"Mode: {self.mode}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add recording status
        if self.recording:
            cv2.putText(
                frame,
                "REC",
                (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            # Add red circle for recording indicator
            cv2.circle(frame, (width - 120, 25), 10, (0, 0, 255), -1)
            
        return frame
        
    def toggle_recording(self, frame):
        """Toggle video recording"""
        if not self.recording and self.record:
            # Start recording
            self.recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"stream_{timestamp}.mp4")
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))
            
            print(f"Started recording to {output_file}")
            
        elif self.recording:
            # Stop recording
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                print("Stopped recording")
                
    def cycle_mode(self):
        """Cycle through processing modes"""
        current_index = self.modes.index(self.mode)
        next_index = (current_index + 1) % len(self.modes)
        self.mode = self.modes[next_index]
        print(f"Switched to mode: {self.mode}")
        
    def run(self):
        """Main processing loop"""
        if not self.start():
            return
            
        while True:
            # Read a frame from the stream
            ret, frame = self.cap.read()
            
            if not ret:
                print("No frame received. Waiting...")
                time.sleep(0.1)
                continue
                
            # Update performance metrics
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
                
            # Process the frame based on the current mode
            processed_frame = self.process_frame(frame)
            
            # Add information overlay
            display_frame = self.add_info_overlay(processed_frame.copy())
            
            # Record the processed frame if recording is enabled
            if self.recording and self.video_writer is not None:
                self.video_writer.write(processed_frame)
                
            # Display the frame
            cv2.imshow("Advanced Stream Processor", display_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.cycle_mode()
            elif key == ord('r'):
                self.toggle_recording(processed_frame)
                
        # Clean up
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            
        self.cap.release()
        cv2.destroyAllWindows()
        print("Stream processor stopped")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced Stream Processor for UAV Video Feed')
    parser.add_argument('--port', type=int, default=5600, help='UDP port to receive the stream (default: 5600)')
    parser.add_argument('--mode', choices=['normal', 'edges', 'hsv', 'custom'], default='normal',
                        help='Processing mode: normal, edges, hsv, or custom (default: normal)')
    parser.add_argument('--record', action='store_true', help='Enable video recording capability')
    args = parser.parse_args()
    
    # Print network configuration information
    print("Network Configuration:")
    print("  Jetson (sender): 192.168.2.2")
    print("  GCS (receiver): 192.168.2.1")
    print(f"  Receiving on port: {args.port}")
    print("")
    
    # Create and run the stream processor
    processor = StreamProcessor(port=args.port, mode=args.mode, record=args.record)
    processor.run()

if __name__ == "__main__":
    main()