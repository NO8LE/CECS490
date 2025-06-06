#!/usr/bin/env python3

"""
QGroundControl Mission Plan Parser

This module provides functionality to parse QGC .plan files and extract waypoints,
geofence information, and other mission parameters.
"""

import json
import math
import logging
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logger = logging.getLogger("PlanParser")

class PlanParser:
    """Parser for QGroundControl mission plan files"""
    
    def __init__(self):
        self.plan_data = None
        self.waypoints = []
        self.home_position = None
        self.geofence = None
        self.mission_items = []
        
    def parse_file(self, file_path: str) -> bool:
        """
        Parse a QGC .plan file and extract mission information
        
        Args:
            file_path: Path to the .plan file
            
        Returns:
            bool: True if parsing was successful, False otherwise
        """
        try:
            logger.info(f"Parsing mission plan file: {file_path}")
            with open(file_path, 'r') as f:
                self.plan_data = json.load(f)
                
            # Extract mission information
            self._extract_mission_items()
            self._extract_home_position()
            self._extract_geofence()
            
            logger.info(f"Successfully parsed mission with {len(self.waypoints)} waypoints")
            return True
        except Exception as e:
            logger.error(f"Error parsing mission plan: {e}")
            return False
    
    def _extract_mission_items(self) -> None:
        """Extract mission items from the plan data"""
        if not self.plan_data or 'mission' not in self.plan_data:
            logger.warning("No mission data found in plan file")
            return
            
        mission = self.plan_data['mission']
        
        # Store the original mission items
        self.mission_items = mission.get('items', [])
        
        # Extract simple waypoints
        self.waypoints = []
        
        for item in self.mission_items:
            # Check if it's a simple waypoint
            if item.get('type') == 'SimpleItem':
                command = item.get('command')
                
                # Check for waypoint commands (16 = MAV_CMD_NAV_WAYPOINT)
                # 22 = MAV_CMD_NAV_TAKEOFF, 21 = MAV_CMD_NAV_LAND
                if command in [16, 21, 22]:
                    params = item.get('params', [])
                    # Extract lat, lon, alt from params (usually 4, 5, 6)
                    if len(params) >= 7:
                        lat = params[4]
                        lon = params[5]
                        alt = params[6]
                        
                        waypoint = {
                            'lat': lat,
                            'lon': lon,
                            'alt': alt,
                            'command': command,
                            'frame': item.get('frame', 3),  # Default to relative alt
                            'params': params,
                            'doJumpId': item.get('doJumpId', 0)
                        }
                        self.waypoints.append(waypoint)
            
            # Complex items like surveys
            elif item.get('type') == 'ComplexItem' and item.get('complexItemType') == 'survey':
                # Extract survey transect points
                if 'TransectStyleComplexItem' in item:
                    transect = item['TransectStyleComplexItem']
                    if 'VisualTransectPoints' in transect:
                        points = transect['VisualTransectPoints']
                        for i, point in enumerate(points):
                            if len(point) >= 2:
                                lat, lon = point[0], point[1]
                                alt = mission.get('globalPlanAltitudeMode', 0)
                                
                                waypoint = {
                                    'lat': lat,
                                    'lon': lon,
                                    'alt': alt,
                                    'command': 16,  # MAV_CMD_NAV_WAYPOINT
                                    'frame': 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
                                    'params': [0, 0, 0, 0, lat, lon, alt],
                                    'doJumpId': len(self.waypoints) + i + 1,
                                    'transect': True
                                }
                                self.waypoints.append(waypoint)
    
    def _extract_home_position(self) -> None:
        """Extract home position from the plan data"""
        if not self.plan_data or 'mission' not in self.plan_data:
            return
            
        mission = self.plan_data['mission']
        if 'plannedHomePosition' in mission:
            home_pos = mission['plannedHomePosition']
            if len(home_pos) >= 3:
                self.home_position = {
                    'lat': home_pos[0],
                    'lon': home_pos[1],
                    'alt': home_pos[2]
                }
    
    def _extract_geofence(self) -> None:
        """Extract geofence information from the plan data"""
        if not self.plan_data or 'geoFence' not in self.plan_data:
            return
            
        geofence = self.plan_data['geoFence']
        self.geofence = {
            'circles': geofence.get('circles', []),
            'polygons': geofence.get('polygons', []),
            'breach_return': geofence.get('breachReturn', None)
        }
    
    def get_waypoints(self) -> List[Dict[str, Any]]:
        """Get list of waypoints from the mission"""
        return self.waypoints
    
    def get_home_position(self) -> Optional[Dict[str, float]]:
        """Get the planned home position"""
        return self.home_position
    
    def get_geofence(self) -> Optional[Dict[str, Any]]:
        """Get the geofence configuration"""
        return self.geofence
    
    def get_mission_item_count(self) -> int:
        """Get total number of mission items"""
        return len(self.mission_items)
    
    def calculate_mission_distance(self) -> float:
        """Calculate total mission distance in meters"""
        total_distance = 0.0
        prev_point = None
        
        for waypoint in self.waypoints:
            lat = waypoint['lat']
            lon = waypoint['lon']
            
            if prev_point:
                prev_lat, prev_lon = prev_point
                distance = self._calculate_distance(prev_lat, prev_lon, lat, lon)
                total_distance += distance
                
            prev_point = (lat, lon)
            
        return total_distance
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Calculate distance between two lat/lon points in meters"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r
    
    def export_waypoints_to_json(self, output_file: str) -> bool:
        """Export waypoints to a JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.waypoints, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error exporting waypoints: {e}")
            return False

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plan_parser.py mission.plan")
        sys.exit(1)
        
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = PlanParser()
    if parser.parse_file(sys.argv[1]):
        print(f"Parsed {len(parser.get_waypoints())} waypoints")
        print(f"Home position: {parser.get_home_position()}")
        print(f"Total mission distance: {parser.calculate_mission_distance():.2f} meters")
    else:
        print("Failed to parse mission file")
