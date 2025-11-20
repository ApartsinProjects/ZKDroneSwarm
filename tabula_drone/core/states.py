"""
Shared state representations for TabulaDrone environments.

This module defines dataclasses for drone state, target state, and world state
that are used across multiple environment implementations.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DroneState:
    """State representation for a single drone."""
    id: str
    position: Tuple[float, float]  # (x, y) in 2D space
    ammo: int  # Current ammunition count
    ammo_max: int  # Maximum ammunition capacity
    damage_per_shot: float  # Damage inflicted per shot


@dataclass
class TargetState:
    """State representation for a single target."""
    id: str
    position: Tuple[float, float]  # (x, y) in 2D space
    class_type: str  # Class label (e.g., "A", "B", "C")
    zone_id: str  # Zone identifier for future multi-target scenarios
    hp_initial: float  # Initial hit points
    hp_current: float  # Current hit points
    is_active: bool  # Active if hp_current > 0


@dataclass
class WorldState:
    """State representation for the 2D world environment."""
    world_size: Tuple[float, float]  # (width, height) bounds
    time_step: int  # Current step index
    max_steps: int  # Maximum allowed steps per episode
    scenario_id: str  # Scenario configuration identifier
    seed: Optional[int] = None  # Random seed for reproducibility


# Default class type to HP mapping
DEFAULT_CLASS_HP_MAPPING = {
    "A": 100.0,
    "B": 150.0,
    "C": 200.0,
}
