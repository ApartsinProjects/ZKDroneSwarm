"""
Simulator Helper Module

This module provides utility functions for extracting and managing data from simulation snapshots.
It serves as a data access layer between the raw snapshot data and the visualization components.
"""
from typing import Dict, Any, List, Tuple, Optional
from targets.state import TargetState
from drone.entity import Drone


def get_world_bounds(snap: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract world dimensions from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        Dictionary with width and height of the world.
    """
    world = snap.get("world", {})
    bounds_m = world.get("bounds_m", {"width": 1000.0, "height": 1000.0})
    width_m = bounds_m.get("width", 1000.0)
    height_m = bounds_m.get("height", 1000.0)
    
    return {
        "width": width_m,
        "height": height_m
    }


def get_targets(snap: Dict[str, Any]) -> List[TargetState]:
    """
    Extract and normalize target data from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        List of TargetState objects with normalized data.
    """
    world = snap.get("world", {})
    target_items = []
    
    if "target_spawn_region" in world and "target_instances" in world["target_spawn_region"]:
        target_instances = world["target_spawn_region"]["target_instances"]
        if "items" in target_instances:
            raw_targets = target_instances["items"]
            
            for target in raw_targets:
                # Get raw status value
                raw_status = target.get("status", 0.0)
                
                # Create a TargetState object
                target_state = TargetState(
                    id=target.get("id", "unknown"),
                    pos_m=(target.get("x", 0), target.get("y", 0)),
                    status=raw_status,
                    # Include TargetType attributes
                    name=target.get("type", "Default"),
                    resilience=target.get("resilience", "Unknown"),
                    resilience_score=target.get("resilience_score", 0.0)
                )
                
                target_items.append(target_state)
    
    return target_items


def get_drones(snap: Dict[str, Any]) -> List[Drone]:
    """
    Extract and normalize drone data from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        List of Drone objects with normalized data.
    """
    world = snap.get("world", {})
    drone_items = []
    
    if "drones" in world and "items" in world["drones"]:
        raw_drones = world["drones"]["items"]
        
        for drone in raw_drones:
            # Create a Drone object from the raw data
            drone_obj = Drone(
                id=drone.get("id", "unknown"),
                type=drone.get("type", "Default"),
                pos_m=(drone.get("x", 0), drone.get("y", 0)),
                battery_capacity_max=drone.get("battery_capacity_max", 0.0),
                speed_kmh=drone.get("speed_kmh", 0.0),
                max_payload_kg=drone.get("max_payload_kg", 0.0),
                status=drone.get("status", "active")
            )
            
            # Store detection range as an attribute in __dict__ for visualization purposes
            # This doesn't modify the class definition but allows us to access it for drawing
            if "detection_range_m" in drone:
                drone_obj.__dict__["detection_range_m"] = drone.get("detection_range_m", 0.0)
            
            # Store events as an attribute in __dict__ for visualization purposes
            if "events" in drone:
                drone_obj.__dict__["events"] = drone.get("events", {})
                
            drone_items.append(drone_obj)
    
    return drone_items


def get_resupply_station(snap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract resupply station data from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        Dictionary with resupply station data or None if not present.
    """
    world = snap.get("world", {})
    
    if "resupply_station" in world and "pos" in world["resupply_station"]:
        station = world["resupply_station"]
        pos = station["pos"]
        
        return {
            "x": pos.get("x", 0),
            "y": pos.get("y", 0),
            "missiles": station.get("missiles", {})
        }
    
    return None


def get_spawn_regions(snap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract spawn region data from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        Dictionary with spawn region data or None if not present.
    """
    world = snap.get("world", {})
    
    if "target_spawn_region" in world and "region" in world["target_spawn_region"]:
        region = world["target_spawn_region"]["region"]
        
        return {
            "x_fraction": region.get("x_fraction", [0, 1]),
            "y_fraction": region.get("y_fraction", [0, 1])
        }
    
    return None


def get_simulation_time(snap: Dict[str, Any]) -> float:
    """
    Get the current simulation time from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        Current simulation time in seconds.
    """
    meta = snap.get("meta", {})
    return meta.get("t_s", 0.0)


def get_wall_clock(snap: Dict[str, Any]) -> float:
    """
    Get the current wall clock time from a snapshot.

    Args:
        snap: The snapshot data dictionary.

    Returns:
        Current wall clock time in seconds.
    """
    meta = snap.get("meta", {})
    return meta.get("wall_clock_elapsed_ms", 0.0)


def get_missile_types(snap: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract missile type data from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        Dictionary of missile types with their properties.
    """
    if "catalogs" in snap and "missile_types" in snap["catalogs"]:
        return snap["catalogs"]["missile_types"]
    
    return {}


def get_target_types(snap: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract target type data from a snapshot.
    
    Args:
        snap: The snapshot data dictionary.
        
    Returns:
        Dictionary of target types with their properties.
    """
    if "catalogs" in snap and "target_types" in snap["catalogs"]:
        return snap["catalogs"]["target_types"]
    
    return {}
