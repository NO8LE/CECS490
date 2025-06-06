#!/bin/bash

# Drone Setup Script for QGC Mission Integration
# This script helps set up the drone environment for running the QGC Mission Integration software

# Display header
echo "=================================================="
echo "      QGC Mission Integration Drone Setup         "
echo "=================================================="
echo

# Make the script exit on error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if user is in a group
user_in_group() {
    groups $USER | grep -q "$1"
}

# Check if script is run as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  This script should not be run as root."
    echo "   Instead, it will use sudo when necessary."
    echo
    exit 1
fi

echo "🔍 Checking prerequisites..."

# Check for required tools
if ! command_exists python3; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check dialout group membership
if ! user_in_group "dialout"; then
    echo "⚠️  User is not in the dialout group. Adding..."
    sudo usermod -a -G dialout $USER
    echo "✅ Added user to dialout group. You may need to log out and back in for this to take effect."
else
    echo "✅ User is already in the dialout group."
fi

# Check for USB device
echo
echo "🔌 Checking USB connection to autopilot..."
if [ -e "/dev/ttyACM0" ]; then
    echo "✅ Found device at /dev/ttyACM0"
    
    # Check permissions
    if [ -r "/dev/ttyACM0" ] && [ -w "/dev/ttyACM0" ]; then
        echo "✅ Device has correct permissions."
    else
        echo "⚠️  Setting permissions for /dev/ttyACM0..."
        sudo chmod a+rw /dev/ttyACM0
        echo "✅ Permissions set."
    fi
else
    echo "❌ Device /dev/ttyACM0 not found."
    echo "   Please check USB connection to the autopilot."
    echo "   Looking for other possible devices..."
    
    # Look for other ACM devices
    OTHER_DEVICES=$(ls /dev/ttyACM* 2>/dev/null || echo "None")
    if [ "$OTHER_DEVICES" != "None" ]; then
        echo "   Found other devices: $OTHER_DEVICES"
        echo "   You might need to update the configuration to use one of these."
    else
        echo "   No ACM devices found. Is the autopilot connected via USB?"
    fi
fi

# Check network connectivity to GCS
echo
echo "🌐 Checking network connectivity to GCS (192.168.2.1)..."
if ping -c 1 -W 2 192.168.2.1 >/dev/null 2>&1; then
    echo "✅ GCS is reachable at 192.168.2.1"
else
    echo "❌ Cannot reach GCS at 192.168.2.1"
    echo "   Please check network connection and GCS IP address."
fi

# Check for RTK server
echo
echo "🛰️  Checking RTK GPS server..."
if command_exists curl; then
    if curl --max-time 2 -s http://192.168.2.1:8000/gps_location.json >/dev/null 2>&1; then
        echo "✅ RTK GPS server is accessible."
    else
        echo "❌ Cannot connect to RTK GPS server at http://192.168.2.1:8000"
        echo "   Please ensure the RTK server is running on the GCS."
    fi
else
    echo "⚠️  curl not found, skipping RTK server check."
    echo "   Install curl to enable this check: sudo apt-get install curl"
fi

# Create mission_logs directory if it doesn't exist
echo
echo "📁 Checking required directories..."
if [ ! -d "../mission_logs" ]; then
    echo "Creating mission_logs directory..."
    mkdir -p ../mission_logs
    echo "✅ Created mission_logs directory."
else
    echo "✅ mission_logs directory already exists."
fi

# Create a personalized config file if it doesn't exist
CONFIG_FILE="../config.yaml"
TEMPLATE_FILE="../config_template.yaml"

echo
echo "📝 Checking configuration..."
if [ ! -f "$CONFIG_FILE" ] && [ -f "$TEMPLATE_FILE" ]; then
    echo "Creating personalized config file from template..."
    cp "$TEMPLATE_FILE" "$CONFIG_FILE"
    echo "✅ Created personalized config file: $CONFIG_FILE"
else
    if [ -f "$CONFIG_FILE" ]; then
        echo "✅ Personalized config file already exists: $CONFIG_FILE"
    else
        echo "❌ Template config file not found: $TEMPLATE_FILE"
    fi
fi

# Make test_connections.py executable
if [ -f "./test_connections.py" ]; then
    chmod +x ./test_connections.py
    echo "✅ Made test_connections.py executable."
fi

echo
echo "=================================================="
echo "              Setup Complete! 🚁                  "
echo "=================================================="
echo
echo "Next steps:"
echo "1. If you were added to the dialout group, log out and back in."
echo "2. Run the connection test: ./test_connections.py"
echo "3. If all tests pass, you're ready to run the mission integration!"
echo "   cd .."
echo "   python main.py --mission your_mission.plan --headless"
echo
echo "For more information, see DRONE_SETUP.md"
echo
