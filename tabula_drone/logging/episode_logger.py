"""
EpisodeLogger for capturing and persisting episode data.

Captures initial scenario setup, per-step actions and state,
and episode summary for replay and offline analysis.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional


class EpisodeLogger:
    """
    Logger for capturing episode data to JSON files.
    
    Usage:
        logger = EpisodeLogger(output_dir="logs/")
        logger.start_episode(env, reset_info, seed=42)
        
        # In episode loop:
        logger.log_step(step_num, actions, rewards, terminated, truncated, info)
        
        # After episode ends:
        logger.end_episode(total_rewards, done_reason)
        filepath = logger.save()
    
    JSON Schema (version 1.1):
        {
            "version": "1.1",
            "episode_id": "<uuid>",
            "timestamp": "<ISO8601>",
            "rng_seed": <int|null>,
            "config": {
                "world_size": [<float>, <float>],
                "max_steps": <int>,
                "scenario_id": "<string>",
                "class_attribute_mapping": {"<class>": {"<attr>": <float>, ...}, ...},
                "weapon_damage_profile_mapping": {"<weapon>": {"<attr>": <float>, ...}, ...}
            },
            "scenario": {...},
            "steps": [...],
            "summary": {...}
        }
    """
    
    VERSION = "1.1"
    
    def __init__(self, output_dir: str = "logs/", policy_type: Optional[str] = None):
        """
        Initialize EpisodeLogger.
        
        Args:
            output_dir: Directory for output JSON files. Created if not exists.
            policy_type: Policy type identifier for filename (e.g., "oracle", "random").
        """
        self.output_dir = output_dir
        self.policy_type = policy_type
        self._episode_data: Optional[Dict[str, Any]] = None
        self._steps: List[Dict[str, Any]] = []
        self._episode_id: Optional[str] = None
        self._timestamp: Optional[str] = None
    
    def start_episode(
        self,
        env: Any,
        reset_info: Dict[str, Any],
        seed: Optional[int] = None,
        episode_num: Optional[int] = None,
        total_episodes: Optional[int] = None
    ) -> None:
        """
        Capture initial episode state after env.reset().
        
        Args:
            env: The environment instance (DroneEngageZKMRTA)
            reset_info: Info dict returned by env.reset()
            seed: Random seed used for this episode
            episode_num: Episode number (1-indexed)
        """
        self._episode_id = str(uuid.uuid4())[:8]
        self._timestamp = datetime.utcnow().isoformat() + "Z"
        self._steps = []
        
        scenario = self._build_scenario_snapshot(env, reset_info, seed)
        config = self._build_config_snapshot(env)
        
        self._episode_data = {
            "version": self.VERSION,
            "episode_id": self._episode_id,
            "scenario_id": config.get("scenario_id", ""),
            "episode_num": episode_num,
            "total_episodes": total_episodes,
            "timestamp": self._timestamp,
            "rng_seed": seed,
            "config": config,
            "scenario": scenario,
            "steps": self._steps,
            "summary": None,
        }
    
    def log_step(
        self,
        step_num: int,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> None:
        """
        Log a single step's data.
        
        Args:
            step_num: Current step number (1-indexed)
            actions: Dict of {agent_id: action}
            rewards: Dict of {agent_id: reward}
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict from env.step()
        """
        step_record = self._build_step_record(
            step_num, actions, rewards, terminated, truncated, info
        )
        self._steps.append(step_record)
    
    def end_episode(
        self,
        total_rewards: Dict[str, float],
        done_reason: Optional[str]
    ) -> None:
        """
        Finalize episode with summary data.
        
        Args:
            total_rewards: Dict of {agent_id: total_reward}
            done_reason: Reason for episode termination
        """
        summary = self._build_summary(total_rewards, done_reason, self._steps)
        self._episode_data["summary"] = summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return episode data as a dictionary without saving to disk.
        
        Returns:
            Copy of the episode data dictionary
        
        Raises:
            ValueError: If start_episode() was not called
        """
        if self._episode_data is None:
            raise ValueError("start_episode() must be called before to_dict()")
        return dict(self._episode_data)
    
    def set_learning_path(self, data: Dict[str, Any]) -> None:
        """
        Set learning path data for CF policies.
        
        Args:
            data: Learning path dict with agents and targets latent vectors
        """
        if self._episode_data is not None:
            self._episode_data["learning_path"] = data
    
    def save(self, is_best: bool = False, prefix: str = "") -> str:
        """
        Write episode data to JSON file.
        
        Args:
            is_best: If True, include "best_" prefix in filename after episode number
            prefix: Custom prefix to include in filename (overrides is_best if provided)
        
        Returns:
            Filepath of the saved JSON file
        
        Raises:
            ValueError: If start_episode() was not called
        """
        if self._episode_data is None:
            raise ValueError("start_episode() must be called before save()")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        policy_part = f"{self.policy_type}_" if self.policy_type else ""
        episode_num = self._episode_data.get("episode_num")
        episode_part = f"ep{episode_num:02d}_" if episode_num is not None else ""
        prefix_part = prefix if prefix else ("best_" if is_best else "")
        filename = f"episode_{policy_part}{episode_part}{prefix_part}{timestamp_str}_{self._episode_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(self._episode_data, f, indent=2)
        
        return filepath
    
    def _build_scenario_snapshot(
        self,
        env: Any,
        reset_info: Dict[str, Any],
        seed: Optional[int]
    ) -> Dict[str, Any]:
        """
        Build scenario snapshot from environment state after reset.
        
        Captures initial configuration:
        - num_drones, num_targets
        - drone_positions, target_positions
        - weapon_assignments, target_classes
        
        Args:
            env: The environment instance
            reset_info: Info dict from reset()
            seed: Random seed
            
        Returns:
            Scenario dict matching JSON schema
        """
        drone_positions = [list(drone.position) for drone in env.drones]
        target_positions = [list(target.position) for target in env.targets]
        weapon_assignments = {
            drone.id: drone.weapon_type for drone in env.drones
        }
        target_classes = [target.class_type for target in env.targets]
        
        return {
            "num_drones": env.num_drones,
            "num_targets": env.num_targets,
            "drone_positions": drone_positions,
            "target_positions": target_positions,
            "weapon_assignments": weapon_assignments,
            "target_classes": target_classes,
        }
    
    def _build_config_snapshot(self, env: Any) -> Dict[str, Any]:
        """
        Build config snapshot from environment configuration.
        
        Captures visualization-relevant configuration:
        - world_size, max_steps, scenario_id, class_attribute_mapping
        
        Args:
            env: The environment instance
            
        Returns:
            Config dict for visualization
        """
        return {
            "world_size": list(env.world_size),
            "max_steps": env.max_steps,
            "scenario_id": env.scenario_id,
            "policy_type": self.policy_type if self.policy_type else env.policy_type,
            "class_attribute_mapping": dict(env.class_attribute_mapping),
            "weapon_damage_profile_mapping": dict(env.weapon_damage_profile_mapping),
        }
    
    def _build_step_record(
        self,
        step_num: int,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a single step record from step data.
        
        Captures:
        - step_num, actions, rewards
        - terminated, truncated
        - info subset (target_hps, target_active, ammo_used, overkill)
        
        Args:
            step_num: Step number (1-indexed)
            actions: Agent actions
            rewards: Agent rewards
            terminated: Termination flag
            truncated: Truncation flag
            info: Info dict from step()
            
        Returns:
            Step record dict matching JSON schema
        """
        step_info = {
            "target_hps": info.get("target_hps", []),
            "target_attributes": info.get("target_attributes", []),
            "target_active": info.get("target_active", []),
            "ammo_used": info.get("ammo_used", {}),
        }
        if "overkill" in info:
            step_info["overkill"] = info["overkill"]
        
        actions_0indexed = {
            agent_id: action - 1 if action > 0 else action
            for agent_id, action in actions.items()
        }
        
        return {
            "step_num": step_num,
            "action": actions_0indexed,
            "reward": dict(rewards),
            "terminated": terminated,
            "truncated": truncated,
            "info": step_info,
        }
    
    def _build_summary(
        self,
        total_rewards: Dict[str, float],
        done_reason: Optional[str],
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build episode summary from accumulated data.
        
        Computes:
        - total_steps, total_reward
        - termination_reason, success
        - metrics (targets_destroyed, total_ammo_used)
        
        Args:
            total_rewards: Accumulated rewards per agent
            done_reason: Termination reason string
            steps: List of step records
            
        Returns:
            Summary dict matching JSON schema
        """
        total_steps = len(steps)
        
        last_info = steps[-1]["info"] if steps else {}
        target_active = last_info.get("target_active", [])
        targets_destroyed = sum(1 for active in target_active if not active)
        
        ammo_used = last_info.get("ammo_used", {})
        total_ammo_used = sum(ammo_used.values())
        
        success = done_reason == "all_targets_neutralized"
        
        return {
            "total_steps": total_steps,
            "total_reward": dict(total_rewards),
            "termination_reason": done_reason,
            "success": success,
            "metrics": {
                "targets_destroyed": targets_destroyed,
                "total_ammo_used": total_ammo_used,
            }
        }
