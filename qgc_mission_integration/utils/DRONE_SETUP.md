# Drone Setup for QGC Mission Integration

This document outlines the hardware setup, network architecture, and configuration steps needed to run the QGC Mission Integration system successfully.

## Network Architecture

The system uses the following network architecture:

```
┌───────────────────────┐           ┌──────────────────────┐
│                       │           │                      │
│      DRONE (UAV)      │           │  Ground Control      │
│    192.168.2.2        │◄─────────►│  Station (GCS)       │
│                       │    WiFi   │  192.168.2.1         │
│  ┌─────────────────┐  │           │                      │
│  │ Jetson Orin Nano│  │           │ ┌────────────────┐  │
│  │ - Python code   │  │           │ │ QGroundControl │  │
│  │ - ArUco detector│  │           │ │ - Mission ctrl │  │
│  └───────┬─────────┘  │◄─────────►│ └────────────────┘  │
│          │            │   Serial  │         ▲           │
│          │ USB        │           │         │           │
│          ▼            │           │         │           │
│  ┌─────────────────┐  │           │ ┌───────┴────────┐  │
│  │ CubePilot Orange│  │           │ │ RTK GPS Server │  │
│  │ - ArduCopter 4.6│  │           │ │ - Coordinates  │  │
│  └─────────────────┘  │           │ └────────────────┘  │
│                       │           │                      │
└───────────────────────┘           └──────────────────────┘
```

### Key Components:

1. **Drone (UAV) - 192.168.2.2**
   - **Jetson Orin Nano**: Runs the mission integration code, ArUco detection
   - **CubePilot Orange+**: Flight controller running ArduCopter 4.6
   - The two are connected via **USB** (/dev/ttyACM0)

2. **Ground Control Station (GCS) - 192.168.2.1**
   - Runs **QGroundControl** for mission planning/monitoring
   - Hosts the **RTK GPS Server** for precise coordinates
   - WiFi connection to the drone

## Configuration Settings

The key configuration settings are in `config_template.yaml`:

### MAVLink Connection

```yaml
# When running on the drone, connect directly to the autopilot via USB
mavlink_connection: /dev/ttyACM0
mavlink_baudrate: 921600
```

### RTK GPS Connection

```yaml
# Connect to the RTK server running on the GCS (192.168.2.1)
rtk_server_url: http://192.168.2.1:8000
```

## Setup Steps

1. **Drone Setup**
   
   - Connect the Jetson Orin Nano to CubePilot Orange+ via USB
   - Ensure the user running the software is in the `dialout` group: 
     ```
     sudo usermod -a -G dialout $USER
     ```
   - Make sure the USB device is accessible:
     ```
     sudo chmod a+rw /dev/ttyACM0
     ```

2. **Network Setup**
   
   - Connect the drone to the GCS WiFi network
   - Ensure the drone has a static IP (192.168.2.2)
   - Ensure the GCS has a static IP (192.168.2.1)

3. **RTK GPS Server**
   
   - Ensure the RTK GPS server is running on the GCS
   - The server should be accessible at http://192.168.2.1:8000

## Testing Connections

A utility script is provided to test all connections:

```bash
# On the drone
cd qgc_mission_integration/utils
python test_connections.py
```

You can also test specific connections:

```bash
# Test only the MAVLink connection
python test_connections.py --mavlink-only

# Test only the RTK GPS server connection
python test_connections.py --rtk-only

# Test only the USB device access
python test_connections.py --usb-only
```

## Common Issues and Troubleshooting

### MAVLink Connection Failures

1. **USB Device Not Found**
   - Check if the device exists: `ls -l /dev/ttyACM0`
   - If not found, try unplugging and reconnecting the USB cable
   - Try other USB ports
   - Check if device appears with different name: `ls -l /dev/ttyACM*`

2. **Permission Denied**
   - Make sure user is in the dialout group: `groups $USER`
   - Set permissions: `sudo chmod a+rw /dev/ttyACM0`
   - Reboot the Jetson if needed

3. **No Heartbeat Received**
   - Check if ArduCopter is running on the flight controller
   - Check baudrate settings (921600 is standard for CubePilot)
   - Try a lower baudrate (115200) if having issues

### RTK GPS Server Issues

1. **Connection Refused**
   - Ensure the RTK server is running on the GCS
   - Check if you can ping the GCS: `ping 192.168.2.1`
   - Check the server port (default 8000)

2. **Invalid Data**
   - Check if the RTK GPS is properly connected to the GCS
   - Verify GPS has good satellite fix
   - Check server logs for errors

### Network Issues

1. **Can't Connect to GCS**
   - Verify WiFi connection between drone and GCS
   - Check IP addresses are correct
   - Ensure no firewall is blocking connections

## Running the System

Once connections are verified, you can run the main system:

```bash
# On the drone
cd qgc_mission_integration
python main.py --mission your_mission.plan --headless
```

For more options, see the main README.md file.
