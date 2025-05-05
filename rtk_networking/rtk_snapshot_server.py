from pymavlink import mavutil
from flask import Flask, send_file, jsonify
import time
import json
import os
import threading
import sys
import termios
import tty

app = Flask(__name__)
gps_data = {}
target_system_id = 2
target_mavlink = None
mode = 'manual'  # Start in manual mode
master = None
forward_mavlink = None

def connect_mavlink():
    print("[MAVLINK] Attempting connection to Cube...")
    ports_to_try = ['/dev/ttyACM0', '/dev/ttyACM1']

    for port in ports_to_try:
        try:
            print(f"[MAVLINK] Trying {port}...")
            m = mavutil.mavlink_connection(port, baud=115200)
            m.wait_heartbeat(timeout=5)
            print(f"[MAVLINK] Connected on {port}.")
            return m
        except Exception as e:
            print(f"[MAVLINK] Failed on {port}: {e}")

    print("[ERROR] No MAVLink connection found.")
    exit(1)

def open_mavlink_output():
    global target_mavlink, forward_mavlink
    print("[MAVLINK] Opening MAVLink output to second drone and localhost forwarder...")
    target_mavlink = mavutil.mavlink_connection('udpout:192.168.4.2:14550')  # Optional teammate
    forward_mavlink = mavutil.mavlink_connection('udpout:127.0.0.1:14445', source_system=1)
    print("[MAVLINK] Ready to send GPS_INPUT messages.")

def save_and_broadcast(data):
    global gps_data
    gps_data = data
    with open('gps_location.json', 'w') as f:
        json.dump(gps_data, f, indent=2)
    print("[GPS] Saved GPS data to gps_location.json")

    # Build GPS_INPUT packet
    gps_packet = {
        "time_usec": int(data["timestamp"] * 1e6),
        "gps_id": 0,
        "ignore_flags": 0,
        "time_week": 0,
        "time_week_ms": 0,
        "fix_type": data["fix_type"],
        "lat": int(data["latitude"] * 1e7),
        "lon": int(data["longitude"] * 1e7),
        "alt": int(data["altitude_m"] * 1000),
        "hdop": int(data["eph"] * 100),
        "vdop": 255,
        "vn": 0,
        "ve": 0,
        "vd": 0,
        "speed_accuracy": 1,
        "horiz_accuracy": int(data["eph"] * 1000),
        "vert_accuracy": 1,
        "satellites_visible": data["satellites"]
    }

    if target_mavlink:
        print(f"[MAVLINK → Drone {target_system_id}] Sending GPS_INPUT...")
        target_mavlink.mav.gps_input_send(**gps_packet)

    if forward_mavlink:
        print("[MAVLINK → localhost:14445, sysid 69] Forwarding GPS_INPUT...")
        forward_mavlink.target_system = 69
        forward_mavlink.mav.gps_input_send(**gps_packet)


def wait_for_landing_and_log():
    global gps_data, master

    landed_counter = 0
    airborne_counter = 0
    was_landed = False

    while True:
        last_fix = None
        print("[MODE] Waiting for landing with valid GPS...")

        while True:
            msg = master.recv_match(type=['GPS_RAW_INT', 'VFR_HUD', 'EXTENDED_SYS_STATE'], blocking=True)
            if not msg:
                continue

            if msg.get_type() == 'GPS_RAW_INT':
                fix_type = msg.fix_type
                if fix_type < 3:
                    continue

                eph = msg.eph / 100.0
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.alt / 1000.0
                sat_count = msg.satellites_visible

                if fix_type >= 5 and eph <= 0.02:
                    acc_status = "cm-level"
                elif fix_type == 4:
                    acc_status = "rtk-float"
                elif fix_type == 3:
                    acc_status = "3d-fix"
                else:
                    acc_status = "coarse"

                last_fix = {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_m": alt,
                    "fix_type": fix_type,
                    "eph": eph,
                    "satellites": sat_count,
                    "accuracy_status": acc_status,
                    "timestamp": time.time()
                }

            if msg.get_type() == 'EXTENDED_SYS_STATE':
                if hasattr(msg, 'landed_state'):
                    if msg.landed_state == 1 and last_fix:
                        if not was_landed:
                            print("[LANDING] Detected via EKF.")
                            save_and_broadcast(last_fix)
                        was_landed = True
                        if mode != 'auto':
                            return
                    elif msg.landed_state == 0:
                        was_landed = False

            if msg.get_type() == 'VFR_HUD' and last_fix:
                if msg.groundspeed < 0.5 and msg.alt < 2.0:
                    landed_counter += 1
                    airborne_counter = 0
                else:
                    airborne_counter += 1
                    landed_counter = 0

                if landed_counter > 5:
                    if not was_landed:
                        print("[LANDING] Detected via motion.")
                        save_and_broadcast(last_fix)
                    was_landed = True
                    if mode != 'auto':
                        return
                if airborne_counter > 5:
                    was_landed = False

            time.sleep(0.1)

def run_webserver():
    @app.route("/gps_location.json")
    def serve_gps():
        if os.path.exists("gps_location.json"):
            return send_file("gps_location.json", mimetype='application/json')
        else:
            return jsonify({"error": "No GPS fix yet"}), 404

    print("[WEB] Serving GPS data at http://0.0.0.0:8000/gps_location.json")
    app.run(host='0.0.0.0', port=8000)

def key_listener():
    global mode
    print("[INPUT] Press 'r' to reset GPS capture. Press 'a' to enter auto-hopping mode.")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            if os.read(fd, 1).decode() == 'r':
                print("[INPUT] Manual mode triggered.")
                mode = 'manual'
                wait_for_landing_and_log()
            elif os.read(fd, 1).decode() == 'a':
                print("[INPUT] Auto hopping mode activated.")
                mode = 'auto'
                threading.Thread(target=wait_for_landing_and_log, daemon=True).start()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

if __name__ == "__main__":
    master = connect_mavlink()
    open_mavlink_output()
    threading.Thread(target=key_listener, daemon=True).start()
    run_webserver()
