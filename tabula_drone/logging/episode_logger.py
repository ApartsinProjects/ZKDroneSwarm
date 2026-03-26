"""
EpisodeLogger for capturing and persisting episode data.

Captures initial scenario setup, per-step actions and state,
and episode summary for replay and offline analysis.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from .engagement_logger import EngagementLogger


class EpisodeLogger:
    """
    Data-builder for capturing episode data in memory.

    Accumulates steps, summary, and metrics data which can be read via to_dict()
    and get_analysis_data().  All file I/O is owned by EnvironmentLogger.

    Usage (via EnvironmentLogger):
        logger.start_episode(env, reset_info, seed=42)

        # In episode loop:
        logger.log_step(step_num, actions, rewards, terminated, truncated, info)

        # After episode ends:
        logger.end_episode(total_rewards, done_reason)
        data = logger.to_dict()
    """
    
    VERSION = "1.3"
    
    def __init__(self, output_dir: str = "logs/", policy_type: Optional[str] = None):
        """
        Initialize EpisodeLogger.
        
        Args:
            output_dir: Directory for output JSON files. Created if not exists.
            policy_type: Policy type identifier for filename (e.g., "oracle", "random").
        """
        self.policy_type = policy_type
        self._episode_data: Optional[Dict[str, Any]] = None
        self._steps: List[Dict[str, Any]] = []
        self._metrics: Optional[Dict[str, Any]] = None
        self._episode_id: Optional[str] = None
        self._timestamp: Optional[str] = None
        self._engagement_logger: EngagementLogger = EngagementLogger()
        self._env: Optional[Any] = None
        self.mode: str = "episodic"
        self._cumulative_steps: int = 0
        self._cumulative_neutralizations: int = 0
    
    def start_episode(
        self,
        env: Any,
        reset_info: Dict[str, Any],
        seed: Optional[int] = None,
        episode_num: Optional[int] = None,
        total_episodes: Optional[int] = None,
        environment_path: Optional[str] = None,
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
        self._cumulative_steps = 0
        self._cumulative_neutralizations = 0
        self._metrics = None
        self._env = env
        
        # Initialize engagement logger
        self._engagement_logger.start_episode(env, self._episode_id, self._timestamp)
        
        config = self._build_policy_config_snapshot(env)
        
        self._episode_data = {
            "version": self.VERSION,
            "episode_id": self._episode_id,
            "scenario_id": getattr(env, "scenario_id", ""),
            "episode_num": episode_num,
            "total_episodes": total_episodes,
            "timestamp": self._timestamp,
            "rng_seed": seed,
            "environment_path": environment_path,
            "config": config,
            "steps": self._steps,
            "summary": None,
            "metrics": None,
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
        
        # Forward to engagement logger
        self._engagement_logger.log_engagement(step_num, actions, rewards, info)
        
        # Track cumulative metrics
        self._cumulative_steps = step_num
        self._cumulative_neutralizations = info.get("cumulative_neutralizations", self._cumulative_neutralizations)
    
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
        if self._episode_data is None:
            return
            
        summary = self._build_summary(total_rewards, done_reason, self._steps)
        self._episode_data["summary"] = summary
        
        # Finalize engagement logger
        self._engagement_logger.end_episode()

    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Set calculated metrics for the episode.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self._metrics = metrics
        if self._episode_data is not None:
            self._episode_data["metrics"] = metrics

    def clear_buffers(self) -> None:
        """
        Clear accumulated step data and engagement buffers in memory.

        This is the pure in-memory counterpart of flush(); it does not
        perform any I/O.  EnvironmentLogger calls this after it has
        written the current chunk to disk.
        """
        self._steps.clear()
        self._engagement_logger.flush()

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

    def get_analysis_data(self) -> Dict[str, Any]:
        """
        Return engagement analysis data as a dictionary.

        Returns:
            Engagement analysis dictionary (drone_pov, target_pov, summary)

        Raises:
            ValueError: If start_episode() was not called
        """
        if self._episode_data is None:
            raise ValueError("start_episode() must be called before get_analysis_data()")
        return self._engagement_logger.to_dict()

    def build_scenario_snapshot(
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
    
    def build_shared_config_snapshot(self, env: Any) -> Dict[str, Any]:
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
            "class_attribute_mapping": dict(env.class_attribute_mapping),
            "weapon_damage_profile_mapping": dict(env.weapon_damage_profile_mapping),
        }

    def _build_policy_config_snapshot(self, env: Any) -> Dict[str, Any]:
        """Build the policy-local config snapshot stored in each episode."""
        return {
            "policy_type": self.policy_type if self.policy_type else env.policy_type,
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
            agent_id: action - 1 if action > 0 else -1
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

        Args:
            total_rewards: Accumulated rewards per agent
            done_reason: Termination reason string
            steps: List of step records

        Returns:
            Summary dict matching JSON schema
        """
        total_steps = self._cumulative_steps if self._cumulative_steps > 0 else len(steps)

        success = done_reason == "all_targets_neutralized"
        
        return {
            "total_steps": total_steps,
            "total_reward": dict(total_rewards),
            "termination_reason": done_reason,
            "success": success,
        }
