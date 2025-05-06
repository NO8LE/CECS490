#!/bin/bash

# Receive Stream Demo Script
# This script helps view the video stream from the UAV on the GCS

# Default values
STREAM_PORT=5000
DISPLAY_INFO=false
USE_FFPLAY=false
USE_VLC=false

# Network configuration
JETSON_IP="192.168.251.245"  # Jetson (sender) IP address
GCS_IP="192.168.251.105"     # GCS (receiver) IP address

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script helps view the video stream from the UAV on the GCS."
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -p, --port PORT            Port to receive the stream on (default: $STREAM_PORT)"
    echo "  -i, --info                 Display additional stream information"
    echo "  -f, --ffplay               Use ffplay instead of the Python receiver"
    echo "  -v, --vlc                  Use VLC instead of the Python receiver"
    echo ""
    echo "Example:"
    echo "  $0 --port 5001 --info"
    echo "  $0 --ffplay"
    echo "  $0 --vlc"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            STREAM_PORT="$2"
            shift 2
            ;;
        -i|--info)
            DISPLAY_INFO=true
            shift
            ;;
        -f|--ffplay)
            USE_FFPLAY=true
            shift
            ;;
        -v|--vlc)
            USE_VLC=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Update the SDP file with the correct port
update_sdp() {
    echo "Updating stream.sdp with port $STREAM_PORT..."
    cat > stream.sdp << EOF
c=IN IP4 0.0.0.0
m=video $STREAM_PORT RTP/AVP 96
a=rtpmap:96 H264/90000
EOF
    echo "SDP file updated."
    echo "Note: Expecting stream from Jetson at $JETSON_IP to GCS at $GCS_IP"
}

# Check if we have the required tools
check_requirements() {
    if [ "$USE_FFPLAY" = true ]; then
        if ! command -v ffplay &> /dev/null; then
            echo "Error: ffplay not found. Please install ffmpeg."
            exit 1
        fi
    elif [ "$USE_VLC" = true ]; then
        if ! command -v vlc &> /dev/null; then
            echo "Error: vlc not found. Please install VLC media player."
            exit 1
        fi
    else
        # Check for Python and OpenCV
        if ! command -v python3 &> /dev/null; then
            echo "Error: python3 not found. Please install Python 3."
            exit 1
        fi
        
        # Check if the receiver script exists
        if [ ! -f "gcs_video_receiver.py" ]; then
            echo "Error: gcs_video_receiver.py not found."
            exit 1
        fi
    fi
}

# Update the SDP file
update_sdp

# Check requirements
check_requirements

# Start the appropriate viewer
if [ "$USE_FFPLAY" = true ]; then
    echo "Starting ffplay to view the stream on port $STREAM_PORT..."
    echo "Press q to quit."
    ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental "udp://@:$STREAM_PORT?buffer_size=120000"
elif [ "$USE_VLC" = true ]; then
    echo "Starting VLC to view the stream using stream.sdp..."
    echo "Press Ctrl+C to quit."
    vlc stream.sdp
else
    # Build the Python command
    CMD="python3 gcs_video_receiver.py --port $STREAM_PORT"
    
    if [ "$DISPLAY_INFO" = true ]; then
        CMD="$CMD --display-info"
    fi
    
    echo "Starting Python receiver to view the stream on port $STREAM_PORT..."
    echo "Command: $CMD"
    echo "Press q to quit."
    echo ""
    
    # Execute the command
    eval $CMD
fi