#!/usr/bin/env python3

"""
Mission Event Logger for QGC Mission Integration

This module provides a centralized logging system for capturing critical mission events
with timestamps for mission analysis and debugging.
"""

import os
import json
import time
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger("MissionEventLogger")

class MissionEventLogger:
    """
    Event logger for capturing mission-critical events with timestamps
    
    This class provides a centralized mechanism for logging events during mission
    execution, particularly for events not captured by standard flight logs.
    """
    
    def __init__(self, log_dir: str = "mission_logs", 
                log_file: str = None,
                enable_console: bool = True):
        """
        Initialize the event logger
        
        Args:
            log_dir: Directory to store log files
            log_file: Name of the log file (default: mission_events_YYYY-MM-DD_HH-MM-SS.json)
            enable_console: Whether to also log events to the console
        """
        self.events = []
        self.start_time = time.time()
        self.mission_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Generate log filename if not provided
        if log_file is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = f"mission_events_{timestamp}.json"
            
        self.log_path = os.path.join(log_dir, log_file)
        self.enable_console = enable_console
        
        logger.info(f"Mission event logger initialized. Log file: {self.log_path}")
        
        # Log mission start event
        self.log_event("MISSION_INITIALIZED", {
            "mission_id": self.mission_id,
            "log_file": self.log_path
        })
        
    def log_event(self, event_type: str, data: Dict[str, Any] = None, 
                timestamp: float = None, verbose: bool = True) -> Dict[str, Any]:
        """
        Log a mission event
        
        Args:
            event_type: Type of event (e.g., "UAV_START", "ARUCO_DISCOVERED")
            data: Additional data associated with the event
            timestamp: Event timestamp (defaults to current time)
            verbose: Whether to log to console even if enable_console is True
            
        Returns:
            The event entry that was logged
        """
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()
            
        # Ensure data is a dictionary
        if data is None:
            data = {}
            
        # Create event entry
        event = {
            "event_type": event_type,
            "timestamp": timestamp,
            "time_str": datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "mission_time": timestamp - self.start_time,
            "data": data
        }
        
        # Add to events list
        self.events.append(event)
        
        # Log to console if enabled
        if self.enable_console and verbose:
            msg = f"EVENT: {event_type} at {event['time_str']} (+{event['mission_time']:.3f}s)"
            if data:
                data_str = ", ".join([f"{k}={v}" for k, v in data.items() if k != "details"])
                msg += f" - {data_str}"
            logger.info(msg)
            
        # Write event to file
        self._write_event(event)
        
        return event
        
    def _write_event(self, event: Dict[str, Any]) -> None:
        """
        Write event to log file
        
        Args:
            event: Event data to write
        """
        try:
            # Read existing events
            events = []
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    try:
                        events = json.load(f)
                    except json.JSONDecodeError:
                        # File exists but is not valid JSON or is empty
                        events = []
            
            # Add new event
            events.append(event)
            
            # Write back to file
            with open(self.log_path, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error writing event to log file: {e}")
            
    def log_uav_start(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log UAV start event"""
        return self.log_event("UAV_START", data)
        
    def log_aruco_discovery(self, marker_id: int, position: Tuple[float, float, float] = None, 
                         confidence: float = None) -> Dict[str, Any]:
        """Log ArUco marker discovery event"""
        data = {"marker_id": marker_id}
        if position:
            data["position_3d"] = {"x": position[0], "y": position[1], "z": position[2]}
        if confidence:
            data["confidence"] = confidence
        return self.log_event("ARUCO_DISCOVERED", data)
        
    def log_aruco_location(self, lat: float, lon: float, alt: float, 
                        accuracy: str = None) -> Dict[str, Any]:
        """Log ArUco marker location (RTK GPS coordinates)"""
        data = {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
        }
        if accuracy:
            data["accuracy"] = accuracy
        return self.log_event("ARUCO_LOCATION", data)
        
    def log_ugv_start(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log UGV coordination start event"""
        return self.log_event("UGV_START", data)
        
    def log_ugv_command_sent(self, lat: float, lon: float, alt: float, 
                          ugv_ip: str = None) -> Dict[str, Any]:
        """Log UGV command sent event"""
        data = {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
        }
        if ugv_ip:
            data["ugv_ip"] = ugv_ip
        return self.log_event("UGV_COMMAND_SENT", data)
        
    def log_ugv_receipt_confirmed(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log UGV command receipt confirmation"""
        return self.log_event("UGV_RECEIPT_CONFIRMED", data)
        
    def log_ugv_delivery(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log UGV arrival at target location"""
        return self.log_event("UGV_DELIVERY", data)
        
    def log_ugv_end(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log UGV coordination end event"""
        return self.log_event("UGV_END", data)
        
    def log_uav_end(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log UAV end event"""
        return self.log_event("UAV_END", data)
        
    def log_state_transition(self, previous_state: str, current_state: str, 
                           details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log state machine transition"""
        data = {
            "previous_state": previous_state,
            "current_state": current_state
        }
        if details:
            data["details"] = details
        return self.log_event("STATE_TRANSITION", data)
        
    def log_uav_ugv_communication(self, message_type: str, direction: str, 
                               details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Log communication between UAV and UGV
        
        Args:
            message_type: Type of message (e.g., "HEARTBEAT", "SET_POSITION_TARGET")
            direction: Direction of message ("UAV_TO_UGV" or "UGV_TO_UAV")
            details: Additional message details
        """
        data = {
            "message_type": message_type,
            "direction": direction
        }
        if details:
            data["details"] = details
        return self.log_event("UAV_UGV_COMMUNICATION", data)
        
    def get_events(self, event_type: str = None) -> List[Dict[str, Any]]:
        """
        Get events of a specific type
        
        Args:
            event_type: Type of events to retrieve, or None for all events
            
        Returns:
            List of matching events
        """
        if event_type is None:
            return self.events
        
        return [e for e in self.events if e["event_type"] == event_type]
        
    def get_event_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline of all events sorted by timestamp
        
        Returns:
            List of events sorted by timestamp
        """
        return sorted(self.events, key=lambda e: e["timestamp"])
        
    def get_mission_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the mission
        
        Returns:
            Dictionary with mission summary data
        """
        if not self.events:
            return {"status": "No events recorded"}
            
        timeline = self.get_event_timeline()
        start_time = timeline[0]["timestamp"]
        end_time = timeline[-1]["timestamp"]
        duration = end_time - start_time
        
        # Extract key events
        uav_start = self.get_events("UAV_START")
        aruco_discovery = self.get_events("ARUCO_DISCOVERED")
        aruco_location = self.get_events("ARUCO_LOCATION")
        ugv_start = self.get_events("UGV_START")
        ugv_receipt = self.get_events("UGV_RECEIPT_CONFIRMED")
        ugv_delivery = self.get_events("UGV_DELIVERY")
        ugv_end = self.get_events("UGV_END")
        uav_end = self.get_events("UAV_END")
        
        summary = {
            "mission_id": self.mission_id,
            "start_time": datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "total_events": len(self.events),
            "key_events": {
                "uav_start": len(uav_start) > 0,
                "aruco_discovery": len(aruco_discovery) > 0,
                "aruco_location": len(aruco_location) > 0,
                "ugv_start": len(ugv_start) > 0,
                "ugv_receipt": len(ugv_receipt) > 0,
                "ugv_delivery": len(ugv_delivery) > 0,
                "ugv_end": len(ugv_end) > 0,
                "uav_end": len(uav_end) > 0
            }
        }
        
        # Calculate time deltas if events exist
        if uav_start and aruco_discovery:
            summary["time_to_aruco_discovery"] = aruco_discovery[0]["timestamp"] - uav_start[0]["timestamp"]
            
        if aruco_discovery and aruco_location:
            summary["time_to_aruco_location"] = aruco_location[0]["timestamp"] - aruco_discovery[0]["timestamp"]
            
        if aruco_location and ugv_start:
            summary["time_to_ugv_start"] = ugv_start[0]["timestamp"] - aruco_location[0]["timestamp"]
            
        if ugv_start and ugv_receipt:
            summary["time_to_ugv_receipt"] = ugv_receipt[0]["timestamp"] - ugv_start[0]["timestamp"]
            
        if ugv_receipt and ugv_delivery:
            summary["time_to_ugv_delivery"] = ugv_delivery[0]["timestamp"] - ugv_receipt[0]["timestamp"]
            
        if uav_start and uav_end:
            summary["total_uav_mission_time"] = uav_end[0]["timestamp"] - uav_start[0]["timestamp"]
            
        if ugv_start and ugv_end:
            summary["total_ugv_mission_time"] = ugv_end[0]["timestamp"] - ugv_start[0]["timestamp"]
            
        return summary
        
    def export_events(self, format: str = "json", filepath: str = None) -> Union[str, Dict[str, Any]]:
        """
        Export events to a file or return as a formatted string/object
        
        Args:
            format: Output format ("json", "csv", "text")
            filepath: Path to output file, or None to return data
            
        Returns:
            Exported data if filepath is None, otherwise None
        """
        if format == "json":
            data = self.events
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                return None
            return data
            
        elif format == "csv":
            lines = ["event_type,timestamp,time_str,mission_time,data"]
            for event in self.get_event_timeline():
                data_str = json.dumps(event["data"]).replace('"', '""')
                line = f"{event['event_type']},{event['timestamp']},{event['time_str']},{event['mission_time']},\"{data_str}\""
                lines.append(line)
                
            csv_data = "\n".join(lines)
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(csv_data)
                return None
            return csv_data
            
        elif format == "text":
            lines = ["MISSION EVENT LOG", "=" * 80]
            for event in self.get_event_timeline():
                line = f"{event['time_str']} (+{event['mission_time']:.3f}s) - {event['event_type']}"
                if event['data']:
                    data_str = ", ".join([f"{k}={v}" for k, v in event['data'].items() if k != "details"])
                    line += f": {data_str}"
                lines.append(line)
                
            text_data = "\n".join(lines)
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(text_data)
                return None
            return text_data
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def print_mission_summary(self) -> None:
        """Print a human-readable mission summary to the console"""
        summary = self.get_mission_summary()
        
        print("\n" + "=" * 80)
        print("MISSION SUMMARY".center(80))
        print("=" * 80)
        
        print(f"Mission ID: {summary['mission_id']}")
        print(f"Start Time: {summary['start_time']}")
        print(f"End Time: {summary['end_time']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Total Events: {summary['total_events']}")
        
        print("\nKey Events:")
        for event, occurred in summary['key_events'].items():
            status = "✓" if occurred else "✗"
            print(f"  {status} {event.upper().replace('_', ' ')}")
            
        print("\nKey Timings:")
        for key, value in summary.items():
            if key.startswith("time_to_") or key.startswith("total_"):
                event_name = key.replace("time_to_", "").replace("total_", "total ").replace("_", " ")
                print(f"  {event_name.capitalize()}: {value:.2f} seconds")
                
        print("=" * 80 + "\n")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create event logger
    event_logger = MissionEventLogger()
    
    # Log some example events
    event_logger.log_uav_start({"mode": "AUTO", "battery": "95%"})
    time.sleep(1.5)
    
    event_logger.log_aruco_discovery(marker_id=5, position=(120, 45, 2500), confidence=0.85)
    time.sleep(0.8)
    
    event_logger.log_aruco_location(lat=34.0522, lon=-118.2437, alt=10.5, accuracy="cm-level")
    time.sleep(2.0)
    
    event_logger.log_ugv_start({"ip": "192.168.2.6", "port": 14550})
    time.sleep(0.5)
    
    event_logger.log_ugv_command_sent(lat=34.0522, lon=-118.2437, alt=0.0, ugv_ip="192.168.2.6")
    time.sleep(1.2)
    
    event_logger.log_ugv_receipt_confirmed({"status": "acknowledged"})
    time.sleep(3.0)
    
    event_logger.log_ugv_delivery({"distance_to_target": 0.15})
    time.sleep(1.0)
    
    event_logger.log_ugv_end({"status": "mission_complete"})
    time.sleep(0.5)
    
    event_logger.log_uav_end({"mode": "RTL", "battery": "85%"})
    
    # Print mission summary
    event_logger.print_mission_summary()
    
    # Export events to different formats
    event_logger.export_events(format="json", filepath="example_mission_events.json")
    event_logger.export_events(format="csv", filepath="example_mission_events.csv")
    event_logger.export_events(format="text", filepath="example_mission_events.txt")
    
    print("Example event logs exported to example_mission_events.*")
