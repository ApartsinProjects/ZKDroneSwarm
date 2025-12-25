"""
DroneEngageZKMRTA-v0: PettingZoo multi-agent environment for ZK-MRTA research.

Multiple static drones engage multiple static targets under Zero-Knowledge
Multi-Robot Task Allocation (ZK-MRTA) constraints:
- No prior knowledge of task attributes (HP, classes)
- No knowledge of agent capabilities (damage)
- No communication between agents
- Observations show only target positions and active states
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List, Union

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

# Import existing state classes
from ..core.states import (
    DroneState,
    TargetState,
    WorldState,
    AttributeProfile,
)


class DroneEngageZKMRTA(ParallelEnv):
    """
    PettingZoo ParallelEnv for Zero-Knowledge Multi-Robot Task Allocation.
    
    Multiple static drones engage multiple static targets with:
    - Zero-knowledge observations (target positions + active states only)
    - No communication between agents
    - Unlimited ammo (tracked for metrics)
    - Individual lethal contributor rewards (drone rewarded only if its shot
      would independently deplete all target attributes)
    
    Observation Space (per agent):
        Box containing [target_1_x, target_1_y, target_1_active, ...]
        Shape: (3 * num_targets,)
    
    Action Space (per agent):
        Discrete(num_targets + 1) where:
        - 0: NoOp (do nothing)
        - 1 to N: Fire at target index
    """
    
    metadata = {"name": "drone_engage_zk_mrta_v0"}
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (1000.0, 1000.0),
        max_steps: int = 100,
        drones_config: List[Dict[str, Any]] = None,
        targets_config: List[Dict[str, Any]] = None,
        scenario_id: str = "zk_mrta_baseline",
        class_attribute_mapping: Dict[str, Dict[str, float]] = None,
        weapon_damage_profile_mapping: Dict[str, Dict[str, float]] = None,
        policy_type: str = "random",
        observation_mode: str = "minimal",
        reward_noise: float = 0.0,
        observation_noise: float = 0.0,
    ):
        """
        Initialize ZK-MRTA environment.
        
        Args:
            world_size: (width, height) bounds of 2D world
            max_steps: Maximum steps per episode
            drones_config: List of drone configs, each with:
                - position: (x, y) tuple
                - weapon_type: str (must be key in weapon_damage_profile_mapping)
            targets_config: List of target configs, each with:
                - position: (x, y) tuple
                - class_type: str (must be key in class_attribute_mapping)
            scenario_id: Identifier for this scenario
            class_attribute_mapping: Dict mapping class types to attribute dicts.
                Required - must be provided.
            weapon_damage_profile_mapping: Dict mapping weapon types to damage profile dicts.
                Required - must be provided.
            observation_mode: Observation mode - "minimal" (default) or "collaborative".
                - minimal: Target positions + active status only
                - collaborative: Adds other agents' actions and rewards
            reward_noise: Gaussian noise σ added to actual rewards (default 0.0)
            observation_noise: Additional Gaussian noise σ when observing other
                agents' rewards in collaborative mode (default 0.0)
        """
        super().__init__()
        
        # Validate observation_mode
        valid_modes = ("minimal", "collaborative")
        if observation_mode not in valid_modes:
            raise ValueError(
                f"observation_mode must be one of {valid_modes}. "
                f"Got: '{observation_mode}'"
            )
        
        # Configuration
        self.world_size = world_size
        self.max_steps = max_steps
        self.scenario_id = scenario_id
        self.policy_type = policy_type
        self.observation_mode = observation_mode
        self.reward_noise = reward_noise
        self.observation_noise = observation_noise
        
        # Validate and store mappings (required)
        if class_attribute_mapping is None:
            raise ValueError("class_attribute_mapping is required")
        if weapon_damage_profile_mapping is None:
            raise ValueError("weapon_damage_profile_mapping is required")
        self.class_attribute_mapping = class_attribute_mapping
        self.weapon_damage_profile_mapping = weapon_damage_profile_mapping
        
        # Validate and store configs
        self.drones_config = drones_config or []
        self.targets_config = targets_config or []
        
        if not self.drones_config:
            raise ValueError(
                "drones_config must contain at least one drone. "
                "Received empty list or None."
            )
        
        if not self.targets_config:
            raise ValueError(
                "targets_config must contain at least one target. "
                "Received empty list or None."
            )
        
        # Validate drones_config structure
        for idx, drone_cfg in enumerate(self.drones_config):
            if not isinstance(drone_cfg, dict):
                raise ValueError(
                    f"Drone at index {idx} must be a dict. "
                    f"Got {type(drone_cfg).__name__}."
                )
            if "position" not in drone_cfg:
                raise ValueError(
                    f"Drone at index {idx} is missing required key 'position'."
                )
            if "weapon_type" not in drone_cfg:
                raise ValueError(
                    f"Drone at index {idx} is missing required key 'weapon_type'."
                )
            # Validate weapon_type is valid
            weapon_type = drone_cfg["weapon_type"]
            if weapon_type not in self.weapon_damage_profile_mapping:
                valid_types = list(self.weapon_damage_profile_mapping.keys())
                raise ValueError(
                    f"Drone at index {idx} has invalid weapon_type '{weapon_type}'. "
                    f"Valid types: {valid_types}"
                )
        
        # Validate targets_config structure
        required_target_keys = {"position", "class_type"}
        for idx, target_cfg in enumerate(self.targets_config):
            if not isinstance(target_cfg, dict):
                raise ValueError(
                    f"Target at index {idx} must be a dict. "
                    f"Got {type(target_cfg).__name__}."
                )
            missing_keys = required_target_keys - set(target_cfg.keys())
            if missing_keys:
                raise ValueError(
                    f"Target at index {idx} is missing required keys: {missing_keys}. "
                    f"Required keys are: {required_target_keys}"
                )
        
        # Agent identifiers (auto-generated)
        self.possible_agents = [f"drone_{i}" for i in range(len(self.drones_config))]
        self._agents = self.possible_agents[:]
        
        # Compute dimensions
        self.num_drones = len(self.drones_config)
        self.num_targets = len(self.targets_config)
        
        # Define action spaces: Discrete(num_targets + 1)
        # 0 = NoOp, 1 to num_targets = Fire at target index
        self.action_spaces = {
            agent_id: spaces.Discrete(self.num_targets + 1)
            for agent_id in self.possible_agents
        }
        
        # Define observation spaces based on mode
        if self.observation_mode == "minimal":
            # Minimal mode: Box(shape=(3 * num_targets,))
            # Each target contributes: [x, y, active] (3 values)
            obs_dim = 3 * self.num_targets
            self.observation_spaces = {
                agent_id: spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32
                )
                for agent_id in self.possible_agents
            }
        else:
            # Collaborative mode: Dict space with targets, actions, rewards
            obs_dim = 3 * self.num_targets
            self.observation_spaces = {
                agent_id: spaces.Dict({
                    "targets": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(obs_dim,),
                        dtype=np.float32
                    ),
                    "selected_targets": spaces.Box(
                        low=0,
                        high=self.num_targets,
                        shape=(self.num_drones,),
                        dtype=np.int32
                    ),
                    "observed_rewards": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.num_drones,),
                        dtype=np.float32
                    ),
                })
                for agent_id in self.possible_agents
            }
        
        # State (will be initialized in reset)
        self.drones: Optional[List[DroneState]] = None
        self.targets: Optional[List[TargetState]] = None
        self.world: Optional[WorldState] = None
        self.rng: Optional[np.random.RandomState] = None
        
        # Collaborative mode state tracking
        self.last_actions: Dict[str, int] = {}
        self.last_rewards: Dict[str, float] = {}
    
    @property
    def agents(self) -> List[str]:
        """Return list of active agent IDs."""
        return self._agents
    
    @property
    def num_agents(self) -> int:
        """Return number of agents."""
        return len(self.possible_agents)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Union[np.ndarray, Dict[str, Any]]], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        
        Returns:
            observations: Dict of {agent_id: observation_array}
            infos: Dict with metrics (shared across all agents)
        """
        # Initialize RNG for reproducible drone ordering
        self.rng = np.random.RandomState(seed)
        
        # Initialize world state
        self.world = WorldState(
            world_size=self.world_size,
            time_step=0,
            max_steps=self.max_steps,
            scenario_id=self.scenario_id,
            seed=seed,
        )
        
        # Initialize drones from config
        self.drones = []
        for idx, drone_cfg in enumerate(self.drones_config):
            drone_id = f"drone_{idx}"
            weapon_type = drone_cfg["weapon_type"]
            # Look up damage profile based on weapon type
            damage_profile = dict(self.weapon_damage_profile_mapping[weapon_type])
            drone = DroneState(
                id=drone_id,
                position=drone_cfg["position"],
                ammo_used=0,
                weapon_type=weapon_type,
                damage_profile=damage_profile,
            )
            self.drones.append(drone)
        
        # Initialize targets from config
        self.targets = []
        for idx, target_cfg in enumerate(self.targets_config):
            target_id = f"target_{idx}"
            target_class = target_cfg["class_type"]
            # Look up attribute values based on class type
            attr_values = dict(self.class_attribute_mapping[target_class])
            attributes = AttributeProfile(attributes=attr_values)
            
            target = TargetState(
                id=target_id,
                position=target_cfg["position"],
                class_type=target_class,
                attributes=attributes,
                is_active=True,
            )
            self.targets.append(target)
        
        # Reset agents list
        self._agents = self.possible_agents[:]
        
        # Initialize collaborative mode state tracking
        self.last_actions = {agent_id: 0 for agent_id in self.possible_agents}
        self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        
        # Compute initial observations (Step 6)
        observations = self._compute_observations()
        
        # Build info dict (Step 7)
        infos = self._build_info_dict(actions={})
        
        return observations, infos
    
    def _build_info_dict(self, actions: Dict[str, int]) -> Dict[str, Any]:
        """
        Build info dictionary with metrics for logging/analysis.
        
        This dict contains rich information for post-hoc metric computation
        but is NOT exposed to agent observations (ZK compliance).
        
        Args:
            actions: Dict of {agent_id: action} for this step
        
        Returns:
            Info dict with step index, scenario ID, and metrics
        """
        return {
            "step_index": self.world.time_step,
            "scenario_id": self.scenario_id,
            "actions": actions.copy(),
            "ammo_used": {drone.id: drone.ammo_used for drone in self.drones},
            "weapon_types": [drone.weapon_type for drone in self.drones],
            "target_hps": [target.hp_current for target in self.targets],
            "target_attributes": [dict(target.attributes.attributes) for target in self.targets],
            "target_classes": [target.class_type for target in self.targets],
            "target_active": [target.is_active for target in self.targets],
        }
    
    def _compute_observations(self) -> Dict[str, Any]:
        """
        Compute observations for all agents based on observation mode.
        
        Minimal mode (ZK-compliant):
        - Target positions (x, y) and active states only
        
        Collaborative mode:
        - Target positions and active states
        - Other agents' selected targets (last step)
        - Other agents' observed rewards (with noise)
        
        Returns:
            Dict of {agent_id: observation}
            - Minimal: np.ndarray of shape (3 * num_targets,)
            - Collaborative: Dict with 'targets', 'selected_targets', 'observed_rewards'
        """
        # Build target observation array (shared by both modes)
        target_obs = []
        for target in self.targets:
            x, y = target.position
            is_active_float = 1.0 if target.is_active else 0.0
            target_obs.extend([x, y, is_active_float])
        target_array = np.array(target_obs, dtype=np.float32)
        
        if self.observation_mode == "minimal":
            # All agents receive identical observations
            return {agent_id: target_array.copy() for agent_id in self.agents}
        
        # Collaborative mode: build Dict observations with noise
        observations = {}
        for agent_id in self.agents:
            # Build selected_targets array (actions from last step)
            selected_targets = np.array(
                [self.last_actions.get(aid, 0) for aid in self.possible_agents],
                dtype=np.int32
            )
            
            # Build observed_rewards array with noise
            observed_rewards = []
            for other_agent_id in self.possible_agents:
                base_reward = self.last_rewards.get(other_agent_id, 0.0)
                
                # Apply noise
                if other_agent_id == agent_id:
                    # Own reward: only reward_noise
                    noise = self.rng.normal(0, self.reward_noise) if self.reward_noise > 0 else 0.0
                else:
                    # Other's reward: reward_noise + observation_noise
                    total_noise_std = (self.reward_noise ** 2 + self.observation_noise ** 2) ** 0.5
                    noise = self.rng.normal(0, total_noise_std) if total_noise_std > 0 else 0.0
                
                observed_rewards.append(base_reward + noise)
            
            observed_rewards_array = np.array(observed_rewards, dtype=np.float32)
            
            observations[agent_id] = {
                "targets": target_array.copy(),
                "selected_targets": selected_targets,
                "observed_rewards": observed_rewards_array,
            }
        
        return observations
    
    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, Union[np.ndarray, Dict[str, Any]]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Execute one step in the environment with actions from all agents.
        
        Drones are processed sequentially in random order. Each drone receives
        a fresh observation before its action is evaluated. Actions targeting
        already-neutralized targets are wasted (ammo counted, no damage/reward).
        
        Args:
            actions: Dict of {agent_id: action} where action is:
                - 0: NoOp (do nothing)
                - 1 to num_targets: Fire at target index (1-indexed)
        
        Returns:
            observations: Dict of {agent_id: observation}
            rewards: Dict of {agent_id: reward}
            terminations: Dict of {agent_id: terminated}
            truncations: Dict of {agent_id: truncated}
            infos: Dict with metrics
        """
        # Validate actions
        self._validate_actions(actions)
        
        # Initialize rewards
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        
        # Shuffle drone processing order
        agent_ids = list(self.agents)
        self.rng.shuffle(agent_ids)
        processing_order = agent_ids.copy()
        
        # Track overkill across all drones
        overkill_map: Dict[int, float] = {}
        
        # Process each drone sequentially
        for agent_id in agent_ids:
            action = actions[agent_id]
            drone_idx = int(agent_id.split('_')[1])
            drone = self.drones[drone_idx]
            
            # NoOp - skip
            if action == 0:
                continue
            
            # Fire at target (action is 1-indexed)
            target_idx = action - 1
            target = self.targets[target_idx]
            
            # Always count ammo (even for wasted shots)
            drone.ammo_used += 1
            
            # Check if target is still active
            if not target.is_active:
                # Wasted shot - target already neutralized
                continue
            
            # Apply damage from this drone
            hp_before = target.hp_current
            damage_profile = drone.damage_profile
            target.attributes.apply_damage(damage_profile)
            
            # Check if target became inactive
            if target.attributes.is_depleted():
                target.is_active = False
                
                # Award reward to this drone (killing blow)
                rewards[agent_id] += 1.0
                
                # Calculate overkill
                total_damage = sum(damage_profile.values())
                overkill = max(0.0, total_damage - hp_before)
                if overkill > 0:
                    overkill_map[target_idx] = overkill
        
        # Time progression
        self.world.time_step += 1
        
        # Termination and truncation logic
        terminations, truncations, done_reason = self._check_termination()
        
        # Store actions and rewards for collaborative mode observations (before computing obs)
        self.last_actions = dict(actions)
        self.last_rewards = dict(rewards)
        
        # Compute final observations
        observations = self._compute_observations()
        
        # Build info dict
        infos = self._build_info_dict(actions)
        infos["processing_order"] = processing_order
        if overkill_map:
            infos["overkill"] = overkill_map
        if done_reason:
            infos["done_reason"] = done_reason
        
        return observations, rewards, terminations, truncations, infos
    
    def _check_termination(self) -> Tuple[Dict[str, bool], Dict[str, bool], Optional[str]]:
        """
        Check termination and truncation conditions.
        
        Terminated: All targets neutralized
        Truncated: Max steps reached
        
        Returns:
            terminations: Dict of {agent_id: bool}
            truncations: Dict of {agent_id: bool}
            done_reason: String reason if episode ended, else None
        """
        # Check if all targets neutralized
        all_targets_neutralized = all(not target.is_active for target in self.targets)
        
        # Check if max steps reached
        max_steps_reached = self.world.time_step >= self.world.max_steps
        
        # Determine done reason and flags
        if all_targets_neutralized:
            terminated = True
            truncated = False
            done_reason = "all_targets_neutralized"
        elif max_steps_reached:
            terminated = False
            truncated = True
            done_reason = "max_steps"
        else:
            terminated = False
            truncated = False
            done_reason = None
        
        # All agents share same termination state
        terminations = {agent_id: terminated for agent_id in self.agents}
        truncations = {agent_id: truncated for agent_id in self.agents}
        
        return terminations, truncations, done_reason
    
    def _compute_rewards(
        self,
        neutralizations: Dict[int, List[int]],
        pre_damage_attrs: Dict[int, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Compute killing blow rewards for neutralizations.
        
        All drones that fired at a target when it was neutralized receive +1.0 reward.
        This is the "shared credit" model - if multiple drones contributed to a kill,
        each one gets full credit.
        
        Args:
            neutralizations: Dict of {target_idx: [drone indices that fired at it]}
            pre_damage_attrs: Dict of {target_idx: {attr_name: value}} - attribute
                             snapshots before damage was applied (unused, kept for API)
        
        Returns:
            rewards: Dict of {agent_id: reward_value}
        """
        # Initialize all rewards to 0
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        
        # For each neutralized target, reward all drones that fired at it
        for target_idx, firing_drone_indices in neutralizations.items():
            for drone_idx in firing_drone_indices:
                agent_id = f"drone_{drone_idx}"
                rewards[agent_id] += 1.0
        
        return rewards
    
    def _apply_damage(self, firing_map: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, Dict[str, float]]]:
        """
        Apply aggregated damage to targets and track neutralizations + overkill.
        
        For each target that was fired upon:
        - Skip if target already inactive
        - Aggregate damage profiles from all firing drones
        - Apply damage to each attribute via AttributeProfile
        - If all attributes depleted: set inactive, record neutralization and overkill
        
        Args:
            firing_map: Dict of {target_idx: [list of drone indices]}
        
        Returns:
            neutralizations: Dict of {target_idx: [drone indices that fired at it]}
                            Only includes targets neutralized THIS step
            overkill_map: Dict of {target_idx: excess_damage}
                         Only includes targets with overkill > 0
            pre_damage_attrs: Dict of {target_idx: {attr_name: value}}
                             Attribute snapshots before damage for neutralized targets
        """
        neutralizations = {}
        overkill_map = {}
        pre_damage_attrs = {}
        
        for target_idx, firing_drone_indices in firing_map.items():
            target = self.targets[target_idx]
            
            # Skip if target already inactive
            if not target.is_active:
                continue
            
            # Track HP before damage (for overkill calculation)
            hp_before = target.hp_current
            
            # Snapshot attributes before damage (for lethal contributor check)
            attrs_before = dict(target.attributes.attributes)
            
            # Aggregate damage profiles from all firing drones
            aggregated_damage: Dict[str, float] = {}
            for drone_idx in firing_drone_indices:
                damage_profile = self.drones[drone_idx].damage_profile
                for attr_name, damage in damage_profile.items():
                    aggregated_damage[attr_name] = aggregated_damage.get(attr_name, 0.0) + damage
            
            # Apply aggregated damage to target attributes
            target.attributes.apply_damage(aggregated_damage)
            
            # Check if target became inactive (all attributes depleted)
            if target.attributes.is_depleted():
                # Calculate overkill (total damage beyond what was needed)
                total_damage = sum(aggregated_damage.values())
                overkill = max(0.0, total_damage - hp_before)
                if overkill > 0:
                    overkill_map[target_idx] = overkill
                
                # Set target inactive
                target.is_active = False
                
                # Record neutralization (target index and who fired at it)
                neutralizations[target_idx] = firing_drone_indices
                
                # Store pre-damage attributes for reward calculation
                pre_damage_attrs[target_idx] = attrs_before
        
        return neutralizations, overkill_map, pre_damage_attrs
    
    def _process_actions(self, actions: Dict[str, int]) -> Dict[int, List[int]]:
        """
        Process actions and build firing map.
        
        For each agent:
        - If action == 0: NoOp (do nothing)
        - If action > 0: Fire at target (increment ammo_used counter)
        
        Args:
            actions: Dict of {agent_id: action}
        
        Returns:
            firing_map: Dict of {target_idx: [list of drone indices that fired at it]}
        """
        firing_map = {}
        
        for agent_id, action in actions.items():
            # Get drone index from agent_id (e.g., "drone_0" -> 0)
            drone_idx = int(agent_id.split('_')[1])
            drone = self.drones[drone_idx]
            
            if action == 0:
                # NoOp - do nothing
                continue
            else:
                # Fire at target (action is 1-indexed, so target_idx = action - 1)
                target_idx = action - 1
                
                # Increment ammo usage counter (unlimited ammo, just tracking)
                drone.ammo_used += 1
                
                # Add to firing map
                if target_idx not in firing_map:
                    firing_map[target_idx] = []
                firing_map[target_idx].append(drone_idx)
        
        return firing_map
    
    def _validate_actions(self, actions: Dict[str, int]) -> None:
        """
        Validate that actions dict is properly formed.
        
        Args:
            actions: Dict of {agent_id: action}
        
        Raises:
            ValueError: If actions are invalid
        """
        # Check all agents provided actions
        if set(actions.keys()) != set(self.agents):
            raise ValueError(
                f"Actions must be provided for all agents. "
                f"Expected: {set(self.agents)}, Got: {set(actions.keys())}"
            )
        
        # Check each action is in valid range
        for agent_id, action in actions.items():
            if not isinstance(action, (int, np.integer)):
                raise ValueError(
                    f"Action for {agent_id} must be an integer. "
                    f"Got {type(action).__name__}: {action}"
                )
            
            if action < 0 or action > self.num_targets:
                raise ValueError(
                    f"Action for {agent_id} must be in range [0, {self.num_targets}]. "
                    f"Got: {action}"
                )
