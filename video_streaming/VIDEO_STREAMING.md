# UAV Video Streaming Guide

This guide explains how to use the video streaming functionality added to the UAV perception system. The system streams annotated video frames from the Jetson to a Ground Control Station (GCS) over Wi-Fi using H.264 RTP/UDP.

## Overview

The streaming system uses:
- H.264 encoding with NVENC (NVIDIA hardware encoder) on the Jetson
- RTP over UDP for transport
- Low-latency configuration for real-time monitoring
- Up to 30 FPS streaming capability

## Jetson (UAV) Side Setup

### Starting the ArUco Detector with Streaming

To enable video streaming, use the `--stream` flag when starting the ArUco detector:

```bash
python3 oak_d_aruco_6x6_detector.py --stream
```

By default, this will stream to IP address `192.168.2.1` (GCS) on port `5600`. You can customize these settings:

```bash
python3 oak_d_aruco_6x6_detector.py --stream --stream-ip 192.168.2.1 --stream-port 5001
```

Note: The Jetson is expected to be on IP address `192.168.2.2` and the GCS on `192.168.2.1`.

### Additional Streaming Options

- `--stream-bitrate`: Set the streaming bitrate in bits/sec (default: 4000000)

Example with all options:

```bash
python3 oak_d_aruco_6x6_detector.py --stream --stream-ip 192.168.0.10 --stream-port 5001 --stream-bitrate 6000000
```

You can combine streaming with other detector options:

```bash
python3 oak_d_aruco_6x6_detector.py --target 5 --resolution high --cuda --performance --stream
```

## GCS (Ground Control Station) Side Setup

There are multiple ways to view the stream on the GCS:

### 1. Using the Python Receiver

The included Python receiver script provides a simple way to view the stream:

```bash
python3 gcs_video_receiver.py --port 5600
```

Options:
- `--port`: UDP port to receive the stream (default: 5600)
- `--display-info`: Display additional stream information like FPS and resolution

### 2. Using VLC Media Player

1. Make sure the included `stream.sdp` file has the correct port number (edit if needed)
2. Open VLC Media Player
3. Go to Media > Open File
4. Select the `stream.sdp` file
5. Click "Open"

### 3. Using ffplay

ffplay provides a low-latency option for viewing the stream:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental "udp://@:5600?buffer_size=120000"
```

Replace `5600` with your port number if you changed it.

## Network Configuration

### Required Ports

For the streaming functionality to work properly, certain network ports need to be accessible:

#### For H.264/RTP Streaming (UDP)

1. **On the Jetson (sender)**:
   - No specific inbound ports need to be opened
   - Outbound UDP traffic to the GCS IP on the specified port (default: 5600) must be allowed

2. **On the GCS (receiver)**:
   - Inbound UDP traffic on the specified port (default: 5600) must be allowed
   - Make sure your firewall allows incoming UDP traffic on this port

#### For MJPEG/HTTP Streaming (Flask)

1. **On the Jetson (sender)**:
   - Inbound TCP traffic on the HTTP port (default: 5600) must be allowed
   - Make sure your firewall allows incoming TCP connections on this port

2. **On the GCS (receiver)**:
   - No specific inbound ports need to be opened
   - Outbound HTTP (TCP) traffic to the Jetson IP on the specified port must be allowed

### Firewall Configuration Examples

#### Ubuntu/Debian (including Jetson)

To allow incoming UDP traffic on port 5600:
```bash
sudo ufw allow 5600/udp
```

To allow incoming TCP traffic on port 5600 (for MJPEG streaming):
```bash
sudo ufw allow 5600/tcp
```

#### Windows

1. Open Windows Defender Firewall with Advanced Security
2. Create a new Inbound Rule
3. Select "Port" as the rule type
4. Select "UDP" or "TCP" as the protocol
5. Enter "5600" as the specific local port
6. Allow the connection
7. Apply the rule to all network profiles
8. Name the rule (e.g., "UAV Stream")

### Network Configuration Check Script

A utility script is provided to help verify that your network is properly configured for streaming:

```bash
./check_network_config.sh --ip <target_ip> [options]
```

Options:
- `--port PORT`: Port to check (default: 5600)
- `--mode MODE`: Streaming mode: rtp or http (default: rtp)
- `--ip IP`: Target IP address to check connectivity with
- `--no-firewall`: Skip firewall checks
- `--no-connectivity`: Skip connectivity checks

Examples:

```bash
# Check RTP/UDP configuration for streaming to 192.168.2.1
./check_network_config.sh --ip 192.168.2.1

# Check HTTP/MJPEG configuration on port 8080
./check_network_config.sh --mode http --port 8080 --ip 192.168.2.1
```

The script will:
1. Display your system's IP addresses
2. Check if the port is already in use
3. Verify firewall configurations
4. Test connectivity to the target device
5. Provide recommendations for fixing any issues

## Troubleshooting

### No Stream Visible

1. Ensure the Jetson and GCS are on the same network
2. Check that any firewalls allow UDP traffic on the streaming port
3. Verify the correct IP address is being used (the GCS IP address)
4. Try increasing the bitrate if the video quality is poor
5. Use network diagnostic tools to verify connectivity:
   ```bash
   # Check if the port is open on the GCS
   nc -zvu <GCS_IP> 5600
   
   # Check firewall status on Ubuntu/Debian
   sudo ufw status
   
   # Check if the UDP port is in use
   netstat -tuln | grep 5600
   ```

### High Latency

1. Use the low-latency options in ffplay
2. Ensure the network has sufficient bandwidth
3. Try reducing the resolution with `--resolution low` on the Jetson side

### GStreamer Issues

If you encounter GStreamer pipeline errors:

1. Ensure GStreamer is properly installed on both systems
2. On the Jetson, verify NVENC is available and working
3. Check that OpenCV was built with GStreamer support

## Alternative: MJPEG Streaming

If H.264/RTP streaming is problematic, you can implement MJPEG streaming over HTTP using Flask as mentioned in the development prompt. This would require:

1. Adding Flask to requirements.txt
2. Implementing a Flask server in a separate thread
3. Using cv2.imencode('.jpg', frame) to compress frames
4. Serving frames via multipart MIME format

This approach is simpler but less efficient than H.264 streaming.

## Performance Considerations

- Streaming adds some overhead to the processing pipeline
- If performance issues arise, try:
  - Reducing resolution with `--resolution low`
  - Lowering the bitrate with `--stream-bitrate`
  - Disabling CUDA if it's causing conflicts with NVENC

## Network Requirements

- Stable Wi-Fi connection between Jetson and GCS
- Sufficient bandwidth (at least 4-6 Mbps for default settings)
- Low network congestion for minimal latency
