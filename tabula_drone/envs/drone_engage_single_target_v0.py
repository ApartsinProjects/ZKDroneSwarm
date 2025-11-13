"""
DroneEngageSingleTarget-v0: Minimal Gymnasium environment for drone-target engagement.

A single static drone with limited ammunition engages a single static target in 2D space.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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


class DroneEngageSingleTargetV0(gym.Env):
    """
    Gymnasium environment for single drone engaging single target.
    
    Observation Space:
        Box(4) containing:
        - ammo_normalized: ammo / ammo_max ∈ [0, 1]
        - hp_normalized: hp_current / hp_initial ∈ [0, 1]
        - distance: Euclidean distance between drone and target
        - time_progress: time_step / max_steps ∈ [0, 1]
    
    Action Space:
        Discrete(2):
        - 0: Idle (do nothing)
        - 1: Fire (shoot at target)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (1000.0, 1000.0),
        max_steps: int = 100,
        drone_position: Tuple[float, float] = (100.0, 100.0),
        drone_ammo_max: int = 10,
        drone_damage_per_shot: float = 30.0,
        target_position: Tuple[float, float] = (500.0, 500.0),
        target_class_type: str = "A",
        target_zone_id: str = "zone_1",
        scenario_id: str = "default",
        class_hp_mapping: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the environment.
        
        Args:
            world_size: (width, height) bounds of the 2D world
            max_steps: Maximum steps allowed per episode
            drone_position: Fixed position of the drone
            drone_ammo_max: Maximum ammunition for the drone
            drone_damage_per_shot: Damage inflicted per shot
            target_position: Fixed position of the target
            target_class_type: Class type of the target (maps to HP)
            target_zone_id: Zone identifier for the target
            scenario_id: Identifier for this scenario configuration
            class_hp_mapping: Custom mapping of class types to HP values
        """
        super().__init__()
        
        # Configuration
        self.world_size = world_size
        self.max_steps = max_steps
        self.scenario_id = scenario_id
        self.class_hp_mapping = class_hp_mapping or DEFAULT_CLASS_HP_MAPPING
        
        # Drone configuration
        self.drone_position = drone_position
        self.drone_ammo_max = drone_ammo_max
        self.drone_damage_per_shot = drone_damage_per_shot
        
        # Target configuration
        self.target_position = target_position
        self.target_class_type = target_class_type
        self.target_zone_id = target_zone_id
        self.target_hp_initial = self.class_hp_mapping[target_class_type]
        
        # Define observation space: [ammo_norm, hp_norm, distance, time_progress]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )
        
        # Define action space: 0=Idle, 1=Fire
        self.action_space = spaces.Discrete(2)
        
        # State (will be initialized in reset)
        self.drone: Optional[DroneState] = None
        self.target: Optional[TargetState] = None
        self.world: Optional[WorldState] = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        
        Returns:
            observation: Initial observation vector
            info: Information dictionary
        """
        # Seed the environment
        super().reset(seed=seed)
        
        # Initialize world state
        self.world = WorldState(
            world_size=self.world_size,
            time_step=0,
            max_steps=self.max_steps,
            scenario_id=self.scenario_id,
            seed=seed,
        )
        
        # Initialize drone state
        self.drone = DroneState(
            id="drone_1",
            position=self.drone_position,
            ammo=self.drone_ammo_max,
            ammo_max=self.drone_ammo_max,
            damage_per_shot=self.drone_damage_per_shot,
        )
        
        # Initialize target state
        self.target = TargetState(
            id="target_1",
            position=self.target_position,
            class_type=self.target_class_type,
            zone_id=self.target_zone_id,
            hp_initial=self.target_hp_initial,
            hp_current=self.target_hp_initial,
            is_active=True,
        )
        
        # Compute initial observation
        observation = self._compute_observation()
        
        # Build info dictionary
        info = {
            "step_index": self.world.time_step,
            "scenario_id": self.scenario_id,
        }
        
        return observation, info
    
    def _compute_observation(self) -> np.ndarray:
        """
        Compute the observation vector from current state.
        
        Returns:
            observation: [ammo_norm, hp_norm, distance, time_progress]
        """
        # Ammo normalized
        ammo_normalized = self.drone.ammo / self.drone.ammo_max
        
        # HP normalized
        hp_normalized = self.target.hp_current / self.target.hp_initial
        
        # Euclidean distance between drone and target
        dx = self.drone.position[0] - self.target.position[0]
        dy = self.drone.position[1] - self.target.position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Time progress
        time_progress = self.world.time_step / self.world.max_steps
        
        return np.array(
            [ammo_normalized, hp_normalized, distance, time_progress],
            dtype=np.float32
        )
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=Idle, 1=Fire)
        
        Returns:
            observation: New observation vector
            reward: Reward for this step
            terminated: Whether episode ended due to task completion/failure
            truncated: Whether episode ended due to time limit
            info: Information dictionary
        """
        # Track target state before action
        was_active = self.target.is_active
        
        # Action handling
        if action == 0:
            # Idle: no state changes
            pass
        elif action == 1:
            # Fire: attempt to shoot at target
            if self.drone.ammo > 0 and self.target.is_active:
                # Decrement ammo
                self.drone.ammo -= 1
                
                # Reduce target HP
                self.target.hp_current -= self.drone.damage_per_shot
                
                # Clamp HP to 0 and update active status
                if self.target.hp_current <= 0:
                    self.target.hp_current = 0.0
                    self.target.is_active = False
            # If ammo == 0 or target already inactive, Fire has no effect
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0 (Idle) or 1 (Fire).")
        
        # Time progression
        self.world.time_step += 1
        
        # Compute observation
        observation = self._compute_observation()
        
        # Termination: check if episode should end
        # 1. Target neutralized
        target_neutralized = not self.target.is_active
        # 2. No ammo with active target
        no_ammo_with_active = (self.drone.ammo == 0) and self.target.is_active
        
        terminated = target_neutralized or no_ammo_with_active
        
        # Reward: +1.0 if target became neutralized at this step, 0.0 otherwise
        target_just_neutralized = was_active and not self.target.is_active
        reward = 1.0 if target_just_neutralized else 0.0
        
        # Truncation: check if max steps reached
        truncated = self.world.time_step >= self.world.max_steps
        
        # If both terminated and truncated, prioritize terminated when target neutralized
        if terminated and truncated and target_neutralized:
            truncated = False
        
        # Info dictionary
        info: Dict[str, Any] = {
            "step_index": self.world.time_step,
            "scenario_id": self.scenario_id,
        }
        
        # Add done_reason when episode ends
        if terminated or truncated:
            if target_neutralized:
                info["done_reason"] = "target_neutralized"
            elif no_ammo_with_active:
                info["done_reason"] = "no_ammo"
            elif truncated:
                info["done_reason"] = "max_steps"
        
        # Add optional debug fields
        info["ammo"] = self.drone.ammo
        info["hp_current"] = self.target.hp_current
        info["class_type"] = self.target.class_type
        info["zone_id"] = self.target.zone_id
        
        return observation, reward, terminated, truncated, info
