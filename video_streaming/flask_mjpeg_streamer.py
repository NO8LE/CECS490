#!/usr/bin/env python3

"""
Flask MJPEG Streamer

This script demonstrates how to implement MJPEG streaming over HTTP using Flask
as an alternative to H.264/RTP streaming. This approach is more compatible with
browsers and doesn't require special players, but uses more bandwidth.

Usage:
  python3 flask_mjpeg_streamer.py [--port PORT] [--quality QUALITY] [--webcam]

Options:
  --port PORT           HTTP port to serve the stream (default: 5600)
  --quality QUALITY     JPEG quality (1-100, default: 80)
  --webcam, -w          Use webcam instead of test pattern
  --width WIDTH         Video width (default: 1280)
  --height HEIGHT       Video height (default: 720)
  --fps FPS             Target frames per second (default: 30)

Example:
  python3 flask_mjpeg_streamer.py --port 5600 --quality 70
  python3 flask_mjpeg_streamer.py --webcam

To view the stream, open a browser and navigate to:
  http://<jetson_ip>:5600/video_feed

Press Ctrl+C to exit.
"""

import cv2
import numpy as np
import argparse
import time
import threading
from flask import Flask, Response, render_template
import socket
import os

# Global variables
frame_lock = threading.Lock()
current_frame = None
frame_count = 0
stop_event = threading.Event()

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
    
    # Add "MJPEG Test" label
    cv2.putText(
        frame,
        "MJPEG Test Stream",
        (width//2 - 200, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    
    return frame

def frame_producer(args):
    """Thread function to produce frames"""
    global current_frame, frame_count
    
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
    
    # Performance monitoring
    start_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while not stop_event.is_set():
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
            
            # Update the current frame (thread-safe)
            with frame_lock:
                current_frame = frame.copy()
                frame_count += 1
            
            # Update FPS calculation
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
                print(f"Producing frames at {fps:.1f} FPS")
            
            # Add a small delay to control frame rate if not using webcam
            if not args.webcam:
                time.sleep(1.0 / args.fps)
    
    except Exception as e:
        print(f"Error in frame producer: {e}")
    finally:
        # Clean up
        if args.webcam and cap is not None:
            cap.release()
        print("Frame producer stopped")

def generate_frames(args):
    """Generator function for MJPEG streaming"""
    global current_frame
    
    while not stop_event.is_set():
        # Get the current frame (thread-safe)
        with frame_lock:
            if current_frame is None:
                # No frame available yet
                time.sleep(0.1)
                continue
            
            frame = current_frame.copy()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if not ret:
            continue
            
        # Convert to bytes
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def create_app(args):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Create a simple HTML template for the index page
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MJPEG Stream</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                text-align: center;
                background-color: #f0f0f0;
            }
            h1 {
                color: #333;
            }
            .video-container {
                margin: 20px auto;
                max-width: 100%;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            }
            img {
                max-width: 100%;
                height: auto;
            }
            .info {
                margin: 20px;
                padding: 10px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <h1>MJPEG Stream</h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        <div class="info">
            <p>Resolution: {{ width }}x{{ height }}</p>
            <p>JPEG Quality: {{ quality }}</p>
            <p>Source: {{ source }}</p>
        </div>
    </body>
    </html>
    """
    
    # Define routes
    @app.route('/')
    def index():
        source = "Webcam" if args.webcam else "Test Pattern"
        return render_template_string(
            html_template, 
            width=args.width, 
            height=args.height, 
            quality=args.quality,
            source=source
        )
    
    @app.route('/video_feed')
    def video_feed():
        return Response(
            generate_frames(args),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    # Helper function for render_template_string
    def render_template_string(template_string, **context):
        from flask import render_template_string as flask_render_template_string
        return flask_render_template_string(template_string, **context)
    
    return app

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flask MJPEG Streamer')
    parser.add_argument('--port', type=int, default=5600, help='HTTP port to serve the stream (default: 5600)')
    parser.add_argument('--quality', type=int, default=80, help='JPEG quality (1-100, default: 80)')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam instead of test pattern')
    parser.add_argument('--width', type=int, default=1280, help='Video width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Video height (default: 720)')
    parser.add_argument('--fps', type=int, default=30, help='Target frames per second (default: 30)')
    args = parser.parse_args()
    
    # Validate arguments
    if args.quality < 1 or args.quality > 100:
        print("Error: Quality must be between 1 and 100")
        return
    
    # Get the local IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    # Print information
    print(f"Starting Flask MJPEG Streamer on port {args.port}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"JPEG Quality: {args.quality}")
    print(f"Target FPS: {args.fps}")
    print(f"\nTo view the stream, open a browser and navigate to:")
    print(f"http://{local_ip}:{args.port}/")
    print(f"http://{local_ip}:{args.port}/video_feed (direct stream)")
    print("\nPress Ctrl+C to exit")
    
    # Start the frame producer thread
    producer_thread = threading.Thread(target=frame_producer, args=(args,))
    producer_thread.daemon = True
    producer_thread.start()
    
    # Create and run the Flask app
    app = create_app(args)
    
    try:
        # Run the Flask app
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Signal the producer thread to stop
        stop_event.set()
        producer_thread.join(timeout=2.0)
        print("Flask MJPEG Streamer stopped")

if __name__ == "__main__":
    main()