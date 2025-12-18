"""
Weapon assignment utilities for drone configurations.

Provides functions to randomly assign weapon types to drones based on
configurable weighted distributions.
"""

import random
import warnings
from typing import Dict, List, Any, Optional



def assign_weapons_to_drones(
    drones_config: List[Dict[str, Any]],
    distribution: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    valid_weapon_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Assign weapon types to drones based on weighted random distribution.
    
    Takes a list of drone configurations and adds 'weapon_type' field to each
    based on a weighted random selection. Returns new configuration dictionaries
    without modifying the originals.
    
    Args:
        drones_config: List of drone configuration dicts. Each must contain
            at minimum a 'position' field. Existing 'weapon_type' fields
            will be overwritten.
        distribution: Optional dict mapping weapon types to probability weights.
            Keys must be valid weapon types.
            Weights do not need to sum to 1.0 (will be normalized automatically).
            If None, uses uniform distribution across all valid weapon types.
            Example: {"light": 0.2, "medium": 0.5, "heavy": 0.3}
        seed: Optional random seed for reproducibility. If None, uses
            system randomness (non-reproducible).
        valid_weapon_types: List of valid weapon type names.
            Required - must be provided.
    
    Returns:
        New list of drone configuration dicts with 'weapon_type' added.
        Original configs are not modified.
    
    Raises:
        ValueError: If distribution contains invalid weapon types or
            if weights are not positive numbers.
    
    Examples:
        >>> # Uniform distribution (default)
        >>> configs = [{"position": (100, 100)}, {"position": (200, 200)}]
        >>> new_configs = assign_weapons_to_drones(configs, seed=42)
        >>> print(new_configs[0]['weapon_type'])
        'heavy'
        
        >>> # Weighted distribution (favor medium weapons)
        >>> distribution = {"light": 0.2, "medium": 0.6, "heavy": 0.2}
        >>> new_configs = assign_weapons_to_drones(configs, distribution, seed=42)
    """
    # Validate inputs
    if not isinstance(drones_config, list):
        raise TypeError(f"drones_config must be a list, got {type(drones_config).__name__}")
    
    if len(drones_config) == 0:
        return []
    
    # Validate valid_weapon_types is provided
    if valid_weapon_types is None:
        raise ValueError("valid_weapon_types is required")
    
    # Setup distribution
    if distribution is None:
        # Default: uniform distribution
        distribution = {wt: 1.0 for wt in valid_weapon_types}
    else:
        # Validate distribution keys
        invalid_types = set(distribution.keys()) - set(valid_weapon_types)
        if invalid_types:
            raise ValueError(
                f"Distribution contains invalid weapon types: {invalid_types}. "
                f"Valid types: {valid_weapon_types}"
            )
        
        # Validate distribution values
        for wt, weight in distribution.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"Distribution weight for '{wt}' must be a number, "
                    f"got {type(weight).__name__}"
                )
            if weight < 0:
                raise ValueError(
                    f"Distribution weight for '{wt}' must be non-negative, "
                    f"got {weight}"
                )
        
        # Check if all weights are zero
        total_weight = sum(distribution.values())
        if total_weight == 0:
            raise ValueError("All distribution weights are zero")
    
    # Normalize distribution weights
    total_weight = sum(distribution.values())
    normalized_dist = {wt: weight / total_weight for wt, weight in distribution.items()}
    
    # Warn if distribution was not normalized
    if abs(total_weight - 1.0) > 1e-6:
        warnings.warn(
            f"Distribution weights sum to {total_weight:.4f}, not 1.0. "
            f"Weights have been normalized automatically.",
            UserWarning
        )
    
    # Setup random number generator
    rng = random.Random(seed)
    
    # Prepare weapon types and weights for random.choices()
    weapon_types = list(normalized_dist.keys())
    weights = [normalized_dist[wt] for wt in weapon_types]
    
    # Create new configs with weapon assignments
    new_configs = []
    for drone_config in drones_config:
        # Create a copy of the config
        new_config = drone_config.copy()
        
        # Warn if overwriting existing weapon_type
        if "weapon_type" in new_config:
            warnings.warn(
                f"Overwriting existing weapon_type '{new_config['weapon_type']}' "
                f"in drone config at position {new_config.get('position', 'unknown')}",
                UserWarning
            )
        
        # Assign random weapon type
        weapon_type = rng.choices(weapon_types, weights=weights, k=1)[0]
        new_config["weapon_type"] = weapon_type
        
        new_configs.append(new_config)
    
    return new_configs
