"""
DroneEngageMultiTarget-v0: Gymnasium environment for multi-target drone engagement.

A single static drone with limited ammunition engages multiple static targets in 2D space.
"""

from typing import Tuple, Optional, Dict, Any, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.states import (
    DroneState,
    TargetState,
    WorldState,
    DEFAULT_CLASS_HP_MAPPING,
)


class DroneEngageMultiTargetV0(gym.Env):
    """
    Gymnasium environment for single drone engaging multiple targets.
    
    This environment extends the single-target engagement pattern to support
    a configurable list of targets. The drone must manage ammunition and
    decision-making across multiple targets with varying priorities, positions,
    classes, and zones.
    
    Observation Space:
        Box containing drone features followed by per-target features.
        (Details to be documented as implementation progresses)
    
    Action Space:
        Discrete(N+1) where:
        - 0: Idle (do nothing)
        - 1-N: Fire at target index (1=first target, 2=second target, etc.)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (1000.0, 1000.0),
        max_steps: int = 100,
        drone_position: Tuple[float, float] = (100.0, 100.0),
        drone_ammo_max: int = 10,
        drone_damage_per_shot: float = 30.0,
        targets_config: List[Dict[str, Any]] = None,
        scenario_id: str = "default",
        class_hp_mapping: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the multi-target environment.
        
        Args:
            world_size: (width, height) bounds of the 2D world
            max_steps: Maximum steps allowed per episode
            drone_position: Fixed position of the drone
            drone_ammo_max: Maximum ammunition for the drone
            drone_damage_per_shot: Damage inflicted per shot
            targets_config: List of target configurations, each with keys:
                - position: (x, y) tuple
                - class_type: str (maps to HP via class_hp_mapping)
                - zone_id: str (zone identifier)
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
        
        # Targets configuration
        self.targets_config = targets_config or []
        
        # Validate targets configuration
        if not self.targets_config:
            raise ValueError(
                "targets_config must contain at least one target. "
                "Received empty list or None."
            )
        
        required_keys = {"position", "class_type", "zone_id"}
        for idx, target_cfg in enumerate(self.targets_config):
            if not isinstance(target_cfg, dict):
                raise ValueError(
                    f"Target at index {idx} must be a dict. "
                    f"Got {type(target_cfg).__name__}."
                )
            
            missing_keys = required_keys - set(target_cfg.keys())
            if missing_keys:
                raise ValueError(
                    f"Target at index {idx} is missing required keys: {missing_keys}. "
                    f"Required keys are: {required_keys}"
                )
        
        # Compute action space: 0=Idle, 1-N=Fire at target
        num_targets = len(self.targets_config)
        self.action_space = spaces.Discrete(num_targets + 1)
        
        # Compute observation space:
        # [ammo_norm, time_progress, reserved, reserved,
        #  target1_hp_norm, target1_distance, target1_active,
        #  target2_hp_norm, target2_distance, target2_active, ...]
        obs_dim = 4 + num_targets * 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State (will be initialized in reset)
        self.drone: Optional[DroneState] = None
        self.targets: Optional[List[TargetState]] = None
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
        
        # Initialize multiple targets from config
        self.targets = []
        for idx, target_cfg in enumerate(self.targets_config):
            target_id = f"target_{idx + 1}"
            target_class = target_cfg["class_type"]
            target_hp_initial = self.class_hp_mapping[target_class]
            
            target = TargetState(
                id=target_id,
                position=target_cfg["position"],
                class_type=target_class,
                zone_id=target_cfg["zone_id"],
                hp_initial=target_hp_initial,
                hp_current=target_hp_initial,
                is_active=True,
            )
            self.targets.append(target)
        
        # Compute initial observation (implementation in Step 4)
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
            observation: Flat array with structure:
                [ammo_norm, time_progress, reserved, reserved,
                 target1_hp_norm, target1_distance, target1_active,
                 target2_hp_norm, target2_distance, target2_active, ...]
        """
        # Drone features
        ammo_normalized = self.drone.ammo / self.drone.ammo_max
        time_progress = self.world.time_step / self.world.max_steps
        
        # Start with drone features (4 elements: ammo, time, 2 reserved)
        observation = [ammo_normalized, time_progress, 0.0, 0.0]
        
        # Add features for each target (3 elements per target)
        for target in self.targets:
            # HP normalized
            hp_normalized = target.hp_current / target.hp_initial
            
            # Euclidean distance between drone and target
            dx = self.drone.position[0] - target.position[0]
            dy = self.drone.position[1] - target.position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Active status as float (1.0 if active, 0.0 if not)
            is_active_float = 1.0 if target.is_active else 0.0
            
            # Append target features
            observation.extend([hp_normalized, distance, is_active_float])
        
        return np.array(observation, dtype=np.float32)
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=Idle, 1-N=Fire at target index)
        
        Returns:
            observation: New observation vector
            reward: Reward for this step
            terminated: Whether episode ended due to task completion/failure
            truncated: Whether episode ended due to time limit
            info: Information dictionary
        
        Implementation complete (all 14 baby steps).
        """
        # Track targets state before action (Step 6: for reward computation)
        was_active = [target.is_active for target in self.targets]
        
        # Action validation (Step 5b)
        num_targets = len(self.targets)
        if action < 0 or action > num_targets:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Must be 0 (Idle) or 1-{num_targets} (Fire at target)."
            )
        
        # Action handling
        selected_target = None
        
        if action == 0:
            # Idle: no state changes
            pass
        else:
            # Target selection (Step 5c)
            # Action 1 maps to targets[0], action 2 maps to targets[1], etc.
            target_idx = action - 1
            selected_target = self.targets[target_idx]
            
            # Fire logic (Step 5d)
            if self.drone.ammo > 0:
                # Decrement ammo (spent even if target inactive per user requirement)
                self.drone.ammo -= 1
                
                # Apply damage only if target is active
                if selected_target.is_active:
                    # Reduce target HP
                    selected_target.hp_current -= self.drone.damage_per_shot
                    
                    # Clamp HP to 0 and update active status
                    if selected_target.hp_current <= 0:
                        selected_target.hp_current = 0.0
                        selected_target.is_active = False
        
        # Time progression
        self.world.time_step += 1
        
        # Compute observation
        observation = self._compute_observation()
        
        # Reward computation (Step 6)
        # Count targets that transitioned from active to inactive this step
        neutralizations = 0
        for i, target in enumerate(self.targets):
            if was_active[i] and not target.is_active:
                neutralizations += 1
        reward = float(neutralizations)
        
        # Termination logic (Step 7)
        # Check if all targets neutralized
        all_targets_neutralized = all(not target.is_active for target in self.targets)
        # Check if no ammo remaining
        no_ammo = self.drone.ammo == 0
        
        terminated = all_targets_neutralized or no_ammo
        
        # Truncation: check if max steps reached
        truncated = self.world.time_step >= self.world.max_steps
        
        # If both terminated and truncated, prioritize terminated when all targets neutralized
        if terminated and truncated and all_targets_neutralized:
            truncated = False
        
        # Info dictionary
        info: Dict[str, Any] = {
            "step_index": self.world.time_step,
            "scenario_id": self.scenario_id,
        }
        
        # Add done_reason when episode ends (Step 7)
        if terminated or truncated:
            if all_targets_neutralized:
                info["done_reason"] = "all_targets_neutralized"
            elif no_ammo:
                info["done_reason"] = "no_ammo"
            elif truncated:
                info["done_reason"] = "max_steps"
        
        # Add per-target info arrays (Step 8)
        info["ammo"] = self.drone.ammo
        info["target_hps"] = [target.hp_current for target in self.targets]
        info["target_active"] = [target.is_active for target in self.targets]
        info["target_classes"] = [target.class_type for target in self.targets]
        info["target_zones"] = [target.zone_id for target in self.targets]
        
        return observation, reward, terminated, truncated, info
