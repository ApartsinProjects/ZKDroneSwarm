"""
Shared state representations for TabulaDrone environments.

This module defines dataclasses for drone state, target state, and world state
that are used across multiple environment implementations.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class DroneState:
    """
    State representation for a drone.
    
    Attributes:
        id: Unique drone identifier
        position: (x, y) coordinates in 2D space
        ammo_used: Running count of shots fired (starts at 0)
        weapon_type: Weapon type ("light", "medium", or "heavy")
        damage_profile: Damage per attribute (hidden from agents in ZK environments)
    """
    id: str
    position: Tuple[float, float]
    ammo_used: int  # Running count of shots fired (starts at 0)
    weapon_type: str  # Weapon type: "light", "medium", or "heavy"
    damage_profile: Dict[str, float]  # Damage per attribute (hidden from agents)
    
    @property
    def damage_per_shot(self) -> float:
        """Backward compatibility: return total damage across all attributes."""
        return sum(self.damage_profile.values())


@dataclass
class WorldState:
    """State representation for the 2D world environment."""
    world_size: Tuple[float, float]  # (width, height) bounds
    time_step: int  # Current step index
    max_steps: int  # Maximum allowed steps per episode
    scenario_id: str  # Scenario configuration identifier
    seed: Optional[int] = None  # Random seed for reproducibility


@dataclass
class AttributeProfile:
    """
    Multi-attribute health/damage profile for targets.
    
    Encapsulates multiple named attributes (e.g., armor, shields) that can be
    independently damaged. A target is considered depleted when ALL attributes
    reach zero or below.
    
    Attributes:
        attributes: Current values for each attribute (mutable)
        initial_values: Original values at creation (immutable reference)
    """
    attributes: Dict[str, float]
    initial_values: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Store initial values if not provided."""
        if not self.initial_values:
            self.initial_values = dict(self.attributes)
    
    def apply_damage(self, damage_profile: Dict[str, float]) -> None:
        """
        Apply damage to each attribute based on the damage profile.
        
        Args:
            damage_profile: Dict mapping attribute names to damage values.
                           Attributes not in the profile are not affected.
        """
        for attr_name, damage in damage_profile.items():
            if attr_name in self.attributes:
                self.attributes[attr_name] -= damage
                if self.attributes[attr_name] < 0:
                    self.attributes[attr_name] = 0.0
    
    def is_depleted(self) -> bool:
        """
        Check if all attributes are depleted (all <= 0).
        
        Returns:
            True if ALL attributes are <= 0, False otherwise.
        """
        return all(value <= 0 for value in self.attributes.values())
    
    def get_total(self) -> float:
        """
        Get the sum of all current attribute values.
        
        Provides backward compatibility for code expecting a single HP value.
        
        Returns:
            Sum of all current attribute values.
        """
        return sum(self.attributes.values())


@dataclass
class TargetState:
    """State representation for a single target."""
    id: str
    position: Tuple[float, float]  # (x, y) in 2D space
    class_type: str  # Class label (e.g., "A", "B", "C")
    attributes: AttributeProfile  # Multi-attribute health profile
    is_active: bool = True  # Active until all attributes depleted
    
    @property
    def hp_current(self) -> float:
        """Backward compatibility: return total of all attributes."""
        return self.attributes.get_total()
    
    @property
    def hp_initial(self) -> float:
        """Backward compatibility: return total of initial attributes."""
        return sum(self.attributes.initial_values.values())


# Default class type to HP mapping (legacy single-dimensional)
DEFAULT_CLASS_HP_MAPPING = {
    "A": 100.0,
    "B": 150.0,
    "C": 200.0,
}

# Default class type to attribute mapping (multi-dimensional)
DEFAULT_CLASS_ATTRIBUTE_MAPPING = {
    "A": {"armor": 50.0, "shields": 50.0},
    "B": {"armor": 75.0, "shields": 75.0},
    "C": {"armor": 100.0, "shields": 100.0},
}
