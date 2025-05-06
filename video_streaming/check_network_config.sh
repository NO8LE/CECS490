#!/bin/bash

# Network Configuration Check Script
# This script helps verify that the network is properly configured for video streaming

# Default values
PORT=5000
MODE="rtp"  # rtp or http
CHECK_FIREWALL=true
CHECK_CONNECTIVITY=true
TARGET_IP=""

# Default IP addresses for this setup
JETSON_IP="192.168.251.245"
GCS_IP="192.168.251.105"

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script checks network configuration for video streaming."
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -p, --port PORT            Port to check (default: $PORT)"
    echo "  -m, --mode MODE            Streaming mode: rtp or http (default: $MODE)"
    echo "  -i, --ip IP                Target IP address to check connectivity with"
    echo "  --no-firewall              Skip firewall checks"
    echo "  --no-connectivity          Skip connectivity checks"
    echo ""
    echo "Example:"
    echo "  $0 --port 5001 --mode http --ip 192.168.1.100"
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
            PORT="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -i|--ip)
            TARGET_IP="$2"
            shift 2
            ;;
        --no-firewall)
            CHECK_FIREWALL=false
            shift
            ;;
        --no-connectivity)
            CHECK_CONNECTIVITY=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "rtp" && "$MODE" != "http" ]]; then
    echo "Error: Mode must be either 'rtp' or 'http'"
    exit 1
fi

# Set protocol based on mode
if [[ "$MODE" == "rtp" ]]; then
    PROTOCOL="udp"
else
    PROTOCOL="tcp"
fi

# Print header
echo "========================================"
echo "Network Configuration Check for Streaming"
echo "========================================"
echo "Checking port: $PORT/$PROTOCOL (${MODE^^} streaming)"
echo ""

# Get system information
echo "System Information:"
echo "-------------------"
OS=$(uname -s)
echo "Operating System: $OS"

# Get IP addresses
echo "IP Addresses:"
echo "-------------"
if [[ "$OS" == "Linux" ]]; then
    ip addr | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 | while read -r ip; do
        echo "  $ip"
    done
elif [[ "$OS" == "Darwin" ]]; then
    # macOS
    ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | while read -r ip; do
        echo "  $ip"
    done
else
    echo "  Unable to determine IP addresses on this OS"
fi
echo ""

# Check if port is already in use
echo "Port Usage Check:"
echo "----------------"
if command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$PORT "; then
        echo "WARNING: Port $PORT is already in use!"
        netstat -tuln | grep ":$PORT " | while read -r line; do
            echo "  $line"
        done
    else
        echo "Port $PORT is available (not currently in use)"
    fi
elif command -v ss &> /dev/null; then
    if ss -tuln | grep -q ":$PORT "; then
        echo "WARNING: Port $PORT is already in use!"
        ss -tuln | grep ":$PORT " | while read -r line; do
            echo "  $line"
        done
    else
        echo "Port $PORT is available (not currently in use)"
    fi
else
    echo "Unable to check port usage (netstat or ss command not found)"
fi
echo ""

# Check firewall configuration
if [[ "$CHECK_FIREWALL" == "true" ]]; then
    echo "Firewall Configuration:"
    echo "----------------------"
    
    if [[ "$OS" == "Linux" ]]; then
        # Check UFW (Ubuntu/Debian)
        if command -v ufw &> /dev/null; then
            echo "UFW Firewall:"
            if sudo -n ufw status 2>/dev/null | grep -q "Status: active"; then
                echo "  UFW is active"
                if sudo -n ufw status 2>/dev/null | grep -q "$PORT/$PROTOCOL"; then
                    echo "  Port $PORT/$PROTOCOL is allowed in UFW"
                else
                    echo "  WARNING: Port $PORT/$PROTOCOL is not explicitly allowed in UFW"
                    echo "  You may need to run: sudo ufw allow $PORT/$PROTOCOL"
                fi
            else
                echo "  UFW is inactive or not installed"
            fi
        fi
        
        # Check iptables
        if command -v iptables &> /dev/null; then
            echo "iptables Firewall:"
            if sudo -n iptables -L INPUT -n 2>/dev/null | grep -q "dpt:$PORT"; then
                echo "  Port $PORT appears to be allowed in iptables"
            else
                echo "  NOTE: Could not confirm if port $PORT is allowed in iptables"
                echo "  You may need to run: sudo iptables -A INPUT -p $PROTOCOL --dport $PORT -j ACCEPT"
            fi
        fi
        
        # Check firewalld
        if command -v firewall-cmd &> /dev/null; then
            echo "firewalld Firewall:"
            if sudo -n firewall-cmd --state 2>/dev/null | grep -q "running"; then
                echo "  firewalld is active"
                if sudo -n firewall-cmd --list-ports 2>/dev/null | grep -q "$PORT/$PROTOCOL"; then
                    echo "  Port $PORT/$PROTOCOL is allowed in firewalld"
                else
                    echo "  WARNING: Port $PORT/$PROTOCOL is not explicitly allowed in firewalld"
                    echo "  You may need to run: sudo firewall-cmd --permanent --add-port=$PORT/$PROTOCOL"
                    echo "  And then: sudo firewall-cmd --reload"
                fi
            else
                echo "  firewalld is inactive or not installed"
            fi
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        # macOS
        echo "macOS typically doesn't block incoming connections by default"
        echo "Check System Preferences > Security & Privacy > Firewall if you have issues"
    else
        echo "Unable to check firewall configuration on this OS"
    fi
    echo ""
fi

# Check connectivity
if [[ "$CHECK_CONNECTIVITY" == "true" && -n "$TARGET_IP" ]]; then
    echo "Connectivity Check:"
    echo "------------------"
    echo "Testing connectivity to $TARGET_IP:$PORT..."
    
    # Check if we can reach the target IP
    if ping -c 1 -W 2 "$TARGET_IP" &> /dev/null; then
        echo "  Target IP $TARGET_IP is reachable (ping successful)"
        
        # Check if the port is open
        if command -v nc &> /dev/null; then
            if nc -z -v -w 2 "$TARGET_IP" "$PORT" 2>&1 | grep -q "succeeded"; then
                echo "  Port $PORT on $TARGET_IP is open and accepting connections"
            else
                echo "  WARNING: Port $PORT on $TARGET_IP appears to be closed or blocked"
            fi
        else
            echo "  Unable to check if port is open (nc command not found)"
        fi
    else
        echo "  WARNING: Target IP $TARGET_IP is not reachable"
        echo "  Make sure both devices are on the same network"
    fi
    echo ""
fi

# Recommendations
echo "Recommendations:"
echo "---------------"
if [[ "$MODE" == "rtp" ]]; then
    echo "For RTP/UDP streaming:"
    echo "  - On the GCS (receiver, $GCS_IP), ensure port $PORT/udp is open for incoming connections"
    echo "  - On the Jetson (sender, $JETSON_IP), no specific inbound ports need to be opened"
    echo "  - Make sure both devices are on the same network (192.168.251.x subnet)"
    echo "  - If using a VPN, ensure it allows UDP traffic on port $PORT"
else
    echo "For HTTP/MJPEG streaming:"
    echo "  - On the Jetson (sender, $JETSON_IP), ensure port $PORT/tcp is open for incoming connections"
    echo "  - On the GCS (receiver, $GCS_IP), no specific inbound ports need to be opened"
    echo "  - Make sure both devices are on the same network (192.168.251.x subnet)"
fi

echo ""
echo "For more detailed information, see VIDEO_STREAMING.md"
echo ""