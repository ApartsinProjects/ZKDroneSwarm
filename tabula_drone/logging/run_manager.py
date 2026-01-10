"""
RunManager for orchestrating multi-policy episode runs.

Manages folder structure, episode accumulation, and selection of
representative episodes (first/best/mid) for each policy.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class RunManager:
    """
    Orchestrator for multi-policy episode runs.
    
    Creates a shared scenario folder at initialization, then manages
    per-policy subfolders for episodes and learning state.
    
    Usage:
        run_manager = RunManager(output_dir="logs/", scenario_id="test_scenario")
        
        # For each policy:
        run_manager.start_policy("max_damage_oracle", is_deterministic=True)
        # ... run episodes ...
        run_manager.finalize_policy()
    
    Folder Structure:
        logs/{scenario_id}_{timestamp}/
        ├── {policy_type}/
        │   ├── episodes/
        │   │   └── episode_{category}.json
        │   └── learning_state/
        │       └── learning_state_ep{N:02d}.json
    """
    
    def __init__(self, output_dir: str, scenario_id: str):
        """
        Initialize RunManager and create scenario folder.
        
        Args:
            output_dir: Base directory for logs (e.g., "logs/")
            scenario_id: Scenario identifier from config
        """
        self.output_dir = output_dir
        self.scenario_id = scenario_id
        self._scenario_folder = scenario_id
        self._scenario_path = os.path.join(output_dir, self._scenario_folder)
        
        # Create scenario folder
        os.makedirs(self._scenario_path, exist_ok=True)
        
        # Current policy state
        self._current_policy: Optional[str] = None
        self._current_policy_path: Optional[str] = None
        self._is_deterministic: bool = False
        
        # Episode accumulation state (reset per policy)
        # Each tuple: (episode_data, steps, episode_num)
        self._episodes: List[Tuple[Dict[str, Any], int, int]] = []
        self._first_data: Optional[Dict[str, Any]] = None
        self._first_steps: int = 0
        self._first_episode_num: int = 0
        self._best_data: Optional[Dict[str, Any]] = None
        self._best_steps: int = float('inf')
        self._best_episode_num: int = 0
        self._episode_counter: int = 0
    
    def start_policy(self, policy_type: str, is_deterministic: bool = False) -> None:
        """
        Start a new policy run, creating its subfolder structure.
        
        Args:
            policy_type: Policy type identifier (e.g., "max_damage_oracle")
            is_deterministic: If True, policy produces single episode (tagged "only")
        """
        self._current_policy = policy_type
        self._is_deterministic = is_deterministic
        self._current_policy_path = os.path.join(self._scenario_path, policy_type)
        
        # Create policy subfolders
        episodes_dir = os.path.join(self._current_policy_path, "episodes")
        os.makedirs(episodes_dir, exist_ok=True)
        
        # Always create analysis folder (for all policies)
        analysis_dir = os.path.join(self._current_policy_path, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        if not is_deterministic:
            learning_state_dir = os.path.join(self._current_policy_path, "learning_state")
            os.makedirs(learning_state_dir, exist_ok=True)
        
        # Reset episode accumulation state
        self._episodes = []
        self._first_data = None
        self._first_steps = 0
        self._first_episode_num = 0
        self._best_data = None
        self._best_steps = float('inf')
        self._best_episode_num = 0
        self._episode_counter = 0
    
    def record_episode(
        self,
        episode_data: Dict[str, Any],
        steps: int,
    ) -> None:
        """
        Record an episode for later selection.
        
        Accumulates episode data in memory and tracks first/best episodes.
        
        Args:
            episode_data: Episode data dictionary (from EpisodeLogger.to_dict())
            steps: Number of steps in the episode
        """
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before record_episode()")
        
        # Increment episode counter
        self._episode_counter += 1
        current_episode_num = self._episode_counter
        
        # Store episode with episode number
        self._episodes.append((episode_data, steps, current_episode_num))
        
        # Track first episode
        if self._first_data is None:
            self._first_data = episode_data
            self._first_steps = steps
            self._first_episode_num = current_episode_num
        
        # Track best episode (minimum steps)
        if steps < self._best_steps:
            self._best_data = episode_data
            self._best_steps = steps
            self._best_episode_num = current_episode_num
    
    def get_episodes_dir(self) -> str:
        """
        Get the episodes directory path for the current policy.
        
        Returns:
            Absolute path to episodes directory
        
        Raises:
            ValueError: If start_policy() was not called
        """
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_episodes_dir()")
        return os.path.join(self._current_policy_path, "episodes")
    
    def get_learning_state_dir(self) -> str:
        """
        Get the learning state directory path for the current policy.
        
        Returns:
            Absolute path to learning_state directory
        
        Raises:
            ValueError: If start_policy() was not called
        """
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_learning_state_dir()")
        return os.path.join(self._current_policy_path, "learning_state")
    
    def get_analysis_dir(self) -> str:
        """
        Get the analysis directory path for the current policy.
        
        Returns:
            Absolute path to analysis directory
        
        Raises:
            ValueError: If start_policy() was not called
        """
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_analysis_dir()")
        return os.path.join(self._current_policy_path, "analysis")
    
    def get_scenario_path(self) -> str:
        """
        Get the scenario folder path.
        
        Returns:
            Absolute path to scenario folder
        """
        return self._scenario_path
    
    def finalize_policy(self) -> Dict[str, Any]:
        """
        Finalize the current policy run by selecting and saving representative episodes.
        
        For deterministic policies: saves single episode as "episode_only.json"
        For learning policies: saves first/best/mid episodes
        
        Returns:
            Dict with 'files' (list of short paths) and 'steps' (dict of category: step_count)
        
        Raises:
            ValueError: If start_policy() was not called or no episodes recorded
        """
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before finalize_policy()")
        if not self._episodes:
            raise ValueError("No episodes recorded for current policy")
        
        episodes_dir = self.get_episodes_dir()
        saved_files = []
        steps_info = {}
        best_episode_num = self._first_episode_num
        
        if self._is_deterministic:
            # Single episode: save with episode number
            filepath = self._save_episode(
                self._first_data, "first", self._first_episode_num, episodes_dir
            )
            saved_files.append(f".../episode_first_ep{self._first_episode_num:02d}.json")
            steps_info["first"] = self._first_steps
        else:
            # Learning policy: save first, best, mid with episode numbers
            filepath = self._save_episode(
                self._first_data, "first", self._first_episode_num, episodes_dir
            )
            saved_files.append(f".../episode_first_ep{self._first_episode_num:02d}.json")
            steps_info["first"] = self._first_steps
            
            filepath = self._save_episode(
                self._best_data, "best", self._best_episode_num, episodes_dir
            )
            saved_files.append(f".../episode_best_ep{self._best_episode_num:02d}.json")
            steps_info["best"] = self._best_steps
            best_episode_num = self._best_episode_num
            
            mid_data, mid_steps, mid_episode_num = self._select_mid_episode()
            filepath = self._save_episode(
                mid_data, "mid", mid_episode_num, episodes_dir
            )
            saved_files.append(f".../episode_mid_ep{mid_episode_num:02d}.json")
            steps_info["mid"] = mid_steps
        
        return {"files": saved_files, "steps": steps_info, "best_episode_num": best_episode_num}
    
    def _select_mid_episode(self) -> tuple:
        """
        Select the episode closest to the average of first and best steps.
        
        Returns:
            Tuple of (episode_data, steps, episode_num) for the mid episode
        """
        target_steps = (self._first_steps + self._best_steps) / 2
        
        best_match = None
        best_match_steps = 0
        best_match_episode_num = 0
        best_distance = float('inf')
        
        for episode_data, steps, episode_num in self._episodes:
            distance = abs(steps - target_steps)
            if distance < best_distance:
                best_distance = distance
                best_match = episode_data
                best_match_steps = steps
                best_match_episode_num = episode_num
        
        return best_match, best_match_steps, best_match_episode_num
    
    def _save_episode(
        self,
        episode_data: Dict[str, Any],
        category: str,
        episode_num: int,
        episodes_dir: str
    ) -> str:
        """
        Save an episode to the episodes directory.
        
        Args:
            episode_data: Episode data dictionary
            category: Episode category ("first", "best", "mid")
            episode_num: Episode number (1-indexed)
            episodes_dir: Directory to save to
        
        Returns:
            Filepath of saved episode file
        """
        filename = f"episode_{category}_ep{episode_num:02d}.json"
        filepath = os.path.join(episodes_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(episode_data, f, indent=2)
        
        return filepath
    
    def save_analysis(
        self,
        analysis_data: Dict[str, Any],
        episode_num: int,
    ) -> str:
        """
        Save engagement analysis for an episode.
        
        Writes to: {policy_dir}/analysis/analysis_ep{N:02d}.json
        
        Args:
            analysis_data: Analysis data dict (from EpisodeLogger.get_analysis_data())
            episode_num: Episode number (1-indexed)
        
        Returns:
            Filepath of saved analysis file
        
        Raises:
            ValueError: If start_policy() was not called
        """
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before save_analysis()")
        
        analysis_dir = self.get_analysis_dir()
        filename = f"analysis_ep{episode_num:02d}.json"
        filepath = os.path.join(analysis_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(analysis_data, f, indent=2)
        
        return filepath
    
    def save_learning_state(
        self,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        episode_num: int,
        num_agents: int,
        num_targets: int,
        latent_dim: int,
    ) -> str:
        """
        Save learning state for a CF policy episode.
        
        Writes to: {policy_dir}/learning_state/learning_state_ep{N:02d}.json
        
        Args:
            pre_state: Dict with 'agents' list containing pre-episode latent vectors
            post_state: Dict with 'agents' list containing post-episode latent vectors
            episode_num: Episode number (1-indexed)
            num_agents: Number of agents
            num_targets: Number of targets
            latent_dim: Dimension of latent vectors
        
        Returns:
            Filepath of saved learning state file
        
        Raises:
            ValueError: If start_policy() was not called
        """
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before save_learning_state()")
        
        learning_state_dir = self.get_learning_state_dir()
        filename = f"learning_state_ep{episode_num:02d}.json"
        filepath = os.path.join(learning_state_dir, filename)
        
        learning_state = {
            "version": "1.0",
            "scenario_id": self.scenario_id,
            "episode_num": episode_num,
            "policy_type": self._current_policy,
            "num_agents": num_agents,
            "num_targets": num_targets,
            "latent_dim": latent_dim,
            "pre_episode": pre_state,
            "post_episode": post_state,
        }
        
        with open(filepath, "w") as f:
            json.dump(learning_state, f, indent=2)
        
        return filepath
