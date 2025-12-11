"""
Input/Output utilities for the FalconX snapshot viewer.

This module provides functions for loading and validating snapshot files.
"""

import json
from typing import Dict, Any, List, Optional
import os


class SnapshotValidationError(Exception):
    """Exception raised when snapshot validation fails."""
    pass


def validate_snapshot(data: Dict[str, Any]) -> None:
    """
    Validate that the snapshot data contains all required fields.
    
    Args:
        data: The snapshot data dictionary.
        
    Raises:
        SnapshotValidationError: If validation fails.
    """
    # Check for required top-level keys
    required_keys = ["meta", "world", "catalogs", "summary"]
    for key in required_keys:
        if key not in data:
            raise SnapshotValidationError(f"Missing required top-level field: {key}")
    
    # Check world bounds
    try:
        width = data["world"]["bounds_m"]["width"]
        height = data["world"]["bounds_m"]["height"]
        
        if width <= 0 or height <= 0:
            raise SnapshotValidationError(f"Invalid world bounds: width={width}, height={height}. Both must be > 0.")
    except KeyError as e:
        raise SnapshotValidationError(f"Missing required field: {e}")
    
    # Check catalogs
    if "catalogs" not in data:
        raise SnapshotValidationError("Missing required field: catalogs")
    
    catalogs = data["catalogs"]
    if "target_types" not in catalogs:
        raise SnapshotValidationError("Missing required field: catalogs.target_types")
    
    if "missile_types" not in catalogs:
        raise SnapshotValidationError("Missing required field: catalogs.missile_types")
    
    # Check summary
    if "summary" not in data:
        raise SnapshotValidationError("Missing required field: summary")
    
    summary = data["summary"]
    if "target_status_counts" not in summary:
        raise SnapshotValidationError("Missing required field: summary.target_status_counts")
    
    # Check target instances if they exist
    if "target_spawn_region" in data["world"] and "target_instances" in data["world"]["target_spawn_region"]:
        target_instances = data["world"]["target_spawn_region"]["target_instances"]
        
        if "count" not in target_instances:
            raise SnapshotValidationError("Missing required field: world.target_spawn_region.target_instances.count")
        
        if "items" in target_instances:
            targets = target_instances["items"]
            if not isinstance(targets, list):
                raise SnapshotValidationError("'world.target_spawn_region.target_instances.items' must be a list.")
            
            for i, target in enumerate(targets):
                for field in ["id", "x", "y", "status"]:
                    if field not in target:
                        raise SnapshotValidationError(f"Target at index {i} is missing required field: {field}")
                
                # Validate status values (optional warning)
                if target["status"] not in [0.0, 0.5, 1.0]:
                    print(f"Warning: Target {target['id']} has non-standard status value: {target['status']}")


def load_snapshot(path: str) -> Dict[str, Any]:
    """
    Load and validate a snapshot file.
    
    Args:
        path: Path to the snapshot JSON file.
        
    Returns:
        The validated snapshot data as a dictionary.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        SnapshotValidationError: If the snapshot data is invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    validate_snapshot(data)
    return data
