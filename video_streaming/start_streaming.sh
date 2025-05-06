#!/bin/bash

# Start Streaming Demo Script
# This script starts the ArUco detector with video streaming enabled

# Default values
STREAM_IP="192.168.2.1"
STREAM_PORT=5000
BITRATE=4000000
RESOLUTION="adaptive"
USE_CUDA=false
HIGH_PERFORMANCE=false
TARGET_ID=""
HEADLESS=false

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script starts the ArUco detector with video streaming enabled."
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -i, --ip IP                IP address to stream to (default: $STREAM_IP)"
    echo "  -p, --port PORT            Port to stream to (default: $STREAM_PORT)"
    echo "  -b, --bitrate BITRATE      Streaming bitrate in bits/sec (default: $BITRATE)"
    echo "  -r, --resolution RES       Resolution mode: low, medium, high, adaptive (default: $RESOLUTION)"
    echo "  -c, --cuda                 Enable CUDA acceleration if available"
    echo "  --performance              Enable high performance mode on Jetson"
    echo "  -t, --target ID            Specify a target marker ID to highlight"
    echo "  --headless                 Run in headless mode (no GUI windows, for SSH sessions)"
    echo ""
    echo "Example:"
    echo "  $0 --ip 192.168.2.1 --port 5001 --cuda --target 5"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--ip)
            STREAM_IP="$2"
            shift 2
            ;;
        -p|--port)
            STREAM_PORT="$2"
            shift 2
            ;;
        -b|--bitrate)
            BITRATE="$2"
            shift 2
            ;;
        -r|--resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        -c|--cuda)
            USE_CUDA=true
            shift
            ;;
        --performance)
            HIGH_PERFORMANCE=true
            shift
            ;;
        -t|--target)
            TARGET_ID="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the command
CMD="python3 oak_d_aruco_6x6_detector.py --stream --stream-ip $STREAM_IP --stream-port $STREAM_PORT --stream-bitrate $BITRATE --resolution $RESOLUTION"

# Add optional flags
if [ "$USE_CUDA" = true ]; then
    CMD="$CMD --cuda"
fi

if [ "$HIGH_PERFORMANCE" = true ]; then
    CMD="$CMD --performance"
fi

if [ ! -z "$TARGET_ID" ]; then
    CMD="$CMD --target $TARGET_ID"
fi

if [ "$HEADLESS" = true ]; then
    CMD="$CMD --headless"
fi

# Display the command
echo "Starting ArUco detector with streaming enabled..."
if [ "$HEADLESS" = true ]; then
    echo "Running in headless mode (no GUI windows)"
fi
echo "Command: $CMD"
echo ""
echo "To view the stream on another computer:"
echo "1. Make sure the stream.sdp file has the correct port ($STREAM_PORT)"
echo "2. Use VLC to open stream.sdp, or"
echo "3. Run: python3 gcs_video_receiver.py --port $STREAM_PORT, or"
echo "4. Run: ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental \"udp://@:$STREAM_PORT?buffer_size=120000\""
echo ""
echo "Press Ctrl+C to stop streaming"
echo ""

# Execute the command
eval $CMD