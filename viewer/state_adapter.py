"""
State adapter for loading and extracting data from episode logs.
"""

import json
from typing import Dict, Any, List, Tuple, Optional


def load_episode(path: str) -> Dict[str, Any]:
    """
    Load episode log from JSON file.
    
    Args:
        path: Path to episode log JSON file
        
    Returns:
        Episode data dict
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(path, "r") as f:
        return json.load(f)


def extract_initial_state(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract initial world state for visualization.
    
    Handles both v1.0 (no config) and v1.1 (with config) episode logs.
    
    Args:
        episode_data: Episode data dict from load_episode()
        
    Returns:
        Dict with:
            - world_size: (width, height) tuple
            - drones: list of {id, position, weapon_type}
            - targets: list of {id, position, class_type}
            - version: episode log version
    """
    version = episode_data.get("version", "1.0")
    scenario = episode_data.get("scenario", {})
    config = episode_data.get("config", {})
    
    # Extract world size (v1.1 has it in config, v1.0 needs fallback)
    if "world_size" in config:
        world_size = tuple(config["world_size"])
    else:
        # Fallback: infer from positions or use default
        world_size = _infer_world_size(scenario)
    
    # Extract drones
    drone_positions = scenario.get("drone_positions", [])
    weapon_assignments = scenario.get("weapon_assignments", {})
    drones = []
    for i, pos in enumerate(drone_positions):
        drone_id = f"drone_{i}"
        drones.append({
            "id": drone_id,
            "position": tuple(pos),
            "weapon_type": weapon_assignments.get(drone_id, "unknown"),
        })
    
    # Extract targets
    target_positions = scenario.get("target_positions", [])
    target_classes = scenario.get("target_classes", [])
    targets = []
    for i, pos in enumerate(target_positions):
        target_id = f"target_{i}"
        class_type = target_classes[i] if i < len(target_classes) else "unknown"
        targets.append({
            "id": target_id,
            "position": tuple(pos),
            "class_type": class_type,
        })
    
    return {
        "world_size": world_size,
        "drones": drones,
        "targets": targets,
        "version": version,
    }


def _infer_world_size(scenario: Dict[str, Any]) -> Tuple[float, float]:
    """
    Infer world size from entity positions (fallback for v1.0 logs).
    
    Args:
        scenario: Scenario dict from episode log
        
    Returns:
        (width, height) tuple with 20% padding
    """
    all_positions = []
    
    for pos in scenario.get("drone_positions", []):
        all_positions.append(pos)
    for pos in scenario.get("target_positions", []):
        all_positions.append(pos)
    
    if not all_positions:
        return (1000.0, 1000.0)  # Default
    
    max_x = max(p[0] for p in all_positions)
    max_y = max(p[1] for p in all_positions)
    
    # Add 20% padding
    return (max_x * 1.2, max_y * 1.2)
