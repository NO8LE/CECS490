# UAV Video Streaming Quick Start Guide

This guide provides a quick overview of the different streaming options available in this project and how to use them.

## H.264/RTP Streaming (Primary Method)

### On the UAV (Jetson) - IP: 192.168.251.245

Start the ArUco detector with streaming enabled:

```bash
./start_streaming.sh --ip 192.168.251.105 --port 5000
```

Or manually:

```bash
python3 oak_d_aruco_6x6_detector.py --stream --stream-ip 192.168.251.105 --stream-port 5000
```

### On the GCS (Ground Control Station) - IP: 192.168.251.105

Receive and view the stream:

```bash
./receive_stream.sh --port 5000
```

Or choose a specific viewer:

```bash
# Using the Python receiver
python3 gcs_video_receiver.py --port 5000 --display-info

# Using ffplay
ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental "udp://@:5000?buffer_size=120000"

# Using VLC
vlc stream.sdp
```

## MJPEG/HTTP Streaming (Alternative Method)

If you encounter issues with H.264/RTP streaming, you can use MJPEG streaming over HTTP:

### On the UAV (Jetson)

```bash
python3 flask_mjpeg_streamer.py --port 5000 --quality 80
```

### On the GCS (Ground Control Station)

Open a web browser and navigate to:

```
http://<jetson_ip>:5000/
```

## Testing Tools

### Test GStreamer Pipeline

Test the GStreamer pipeline without using the OAK-D camera:

```bash
python3 test_gstreamer_stream.py --ip 192.168.1.100 --port 5000
```

### Advanced Stream Processing

Process the received stream with custom algorithms:

```bash
python3 advanced_stream_processor.py --port 5000 --mode edges
```

## Required Ports

### For H.264/RTP Streaming (UDP)

1. **On the Jetson (sender)**:
   - No specific inbound ports need to be opened
   - Outbound UDP traffic to the GCS IP on the specified port (default: 5000) must be allowed

2. **On the GCS (receiver)**:
   - Inbound UDP traffic on the specified port (default: 5000) must be allowed
   - Make sure your firewall allows incoming UDP traffic on this port

### For MJPEG/HTTP Streaming (Flask)

1. **On the Jetson (sender)**:
   - Inbound TCP traffic on the HTTP port (default: 5000) must be allowed
   - Make sure your firewall allows incoming TCP connections on this port

2. **On the GCS (receiver)**:
   - No specific inbound ports need to be opened
   - Outbound HTTP (TCP) traffic to the Jetson IP on the specified port must be allowed

### Network Configuration Check

Use the provided script to check if your network is properly configured:

```bash
# Check RTP/UDP configuration (default)
./check_network_config.sh --ip 192.168.251.105

# Check HTTP/MJPEG configuration
./check_network_config.sh --mode http --ip 192.168.251.105
```

This script will:
- Check if the required ports are available
- Verify firewall configurations
- Test connectivity between devices
- Provide recommendations for fixing issues

## Troubleshooting

1. **No stream visible**:
   - Ensure the Jetson and GCS are on the same network
   - Check that firewalls allow UDP traffic on the streaming port
   - Verify the correct IP address is being used
   - Use `sudo ufw status` or `sudo iptables -L` to check firewall rules

2. **GStreamer pipeline errors**:
   - Ensure GStreamer is properly installed on both systems
   - On the Jetson, verify NVENC is available and working

3. **High latency**:
   - Use the low-latency options in ffplay
   - Try reducing the resolution with `--resolution low`

## For More Information

See the detailed [VIDEO_STREAMING.md](VIDEO_STREAMING.md) guide for complete documentation.