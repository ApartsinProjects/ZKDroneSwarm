"""
EnvironmentLogger facade for environment output orchestration.

Provides a single caller-facing entrypoint for run-level logging lifecycle
while reusing the existing EpisodeLogger for per-episode capture.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .episode_logger import EpisodeLogger


class EnvironmentLogger:
    """
    Run-level logging orchestrator over EpisodeLogger instances.

    Owns scenario/policy output directories, per-policy episode accumulation,
    representative-episode selection, and persistence of run-level artifacts.
    EpisodeLogger remains responsible for per-episode capture.
    """

    def __init__(self, output_dir: str, scenario_id: str, mode: str = "episodic"):
        self.output_dir = output_dir
        self.scenario_id = scenario_id
        self.mode = mode
        self._scenario_folder = scenario_id
        self._scenario_path = os.path.join(output_dir, self._scenario_folder)
        os.makedirs(self._scenario_path, exist_ok=True)

        self._current_policy: Optional[str] = None
        self._current_policy_path: Optional[str] = None
        self._is_deterministic = False

        self._episodes: List[Tuple[Dict[str, Any], int, int]] = []
        self._first_data: Optional[Dict[str, Any]] = None
        self._first_steps = 0
        self._first_episode_num = 0
        self._best_data: Optional[Dict[str, Any]] = None
        self._best_steps = float("inf")
        self._best_episode_num = 0
        self._episode_counter = 0

        self._active_episode_logger: Optional[EpisodeLogger] = None

    @property
    def active_episode_logger(self) -> EpisodeLogger:
        """
        Return the EpisodeLogger for the current policy.

        Raises:
            ValueError: If no policy has been started yet.
        """
        if self._active_episode_logger is None:
            raise ValueError("start_policy() must be called before accessing the active EpisodeLogger")
        return self._active_episode_logger

    def start_policy(self, policy_type: str, is_deterministic: bool = False) -> EpisodeLogger:
        """
        Start a policy run and create its active EpisodeLogger.

        Returns:
            The configured EpisodeLogger for the started policy.
        """
        self._current_policy = policy_type
        self._is_deterministic = is_deterministic
        self._current_policy_path = os.path.join(self._scenario_path, policy_type)

        os.makedirs(self.get_episodes_dir(), exist_ok=True)
        os.makedirs(self.get_analysis_dir(), exist_ok=True)
        if not is_deterministic:
            os.makedirs(self.get_learning_state_dir(), exist_ok=True)

        self._episodes = []
        self._first_data = None
        self._first_steps = 0
        self._first_episode_num = 0
        self._best_data = None
        self._best_steps = float("inf")
        self._best_episode_num = 0
        self._episode_counter = 0

        episode_logger = EpisodeLogger(
            output_dir=self.get_episodes_dir(),
            policy_type=policy_type,
        )
        episode_logger.analysis_dir = self.get_analysis_dir()
        episode_logger.mode = self.mode

        self._active_episode_logger = episode_logger
        return episode_logger

    def get_scenario_path(self) -> str:
        """Return the scenario output directory path."""
        return self._scenario_path

    def get_episodes_dir(self) -> str:
        """Return the current policy episodes directory."""
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_episodes_dir()")
        return os.path.join(self._current_policy_path, "episodes")

    def get_analysis_dir(self) -> str:
        """Return the current policy analysis directory."""
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_analysis_dir()")
        return os.path.join(self._current_policy_path, "analysis")

    def get_learning_state_dir(self) -> str:
        """Return the current policy learning state directory."""
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_learning_state_dir()")
        return os.path.join(self._current_policy_path, "learning_state")

    def record_episode(self, episode_data: Dict[str, Any], steps: int) -> None:
        """Record an episode for later representative-episode selection."""
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before record_episode()")

        self._episode_counter += 1
        current_episode_num = self._episode_counter
        self._episodes.append((episode_data, steps, current_episode_num))

        if self._first_data is None:
            self._first_data = episode_data
            self._first_steps = steps
            self._first_episode_num = current_episode_num

        if steps < self._best_steps:
            self._best_data = episode_data
            self._best_steps = steps
            self._best_episode_num = current_episode_num

    def save_analysis(self, analysis_data: Dict[str, Any], episode_num: int) -> str:
        """Persist engagement analysis for an episode."""
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before save_analysis()")

        if self.mode == "continuous" and episode_num == 1:
            filename = "analysis_continuous_final.json"
        else:
            filename = f"analysis_ep{episode_num:02d}.json"

        filepath = os.path.join(self.get_analysis_dir(), filename)
        with open(filepath, "w") as f:
            json.dump(analysis_data, f, indent=2)
        return filepath

    def persist_episode_outputs(self, episode_num: int, steps: int) -> None:
        """
        Persist the active episode's analysis and representative-episode inputs.

        This keeps the runner from coordinating direct handoff between the
        active EpisodeLogger and the run-level persistence logic.
        """
        logger = self.active_episode_logger
        self.save_analysis(logger.get_analysis_data(), episode_num)
        self.record_episode(logger.to_dict(), steps)

    def save_learning_state(
        self,
        pre_state: Optional[Dict[str, Any]],
        post_state: Optional[Dict[str, Any]],
        episode_num: int,
        num_agents: int,
        num_targets: int,
        latent_dim: Optional[int],
        entities: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
    ) -> str:
        """Persist learning-state snapshots for a policy episode."""
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before save_learning_state()")

        if self.mode == "continuous":
            filename = f"learning_state_{tag}.json" if tag else "learning_state_continuous_final.json"
        else:
            tag_str = f"_{tag}" if tag else ""
            filename = f"learning_state_ep{episode_num:02d}{tag_str}.json"

        filepath = os.path.join(self.get_learning_state_dir(), filename)
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
        if entities is not None:
            learning_state["entities"] = entities

        with open(filepath, "w") as f:
            json.dump(learning_state, f, indent=2)
        return filepath

    def finalize_policy(self) -> Dict[str, Any]:
        """
        Finalize the current policy run and clear the active episode logger.

        Returns:
            The policy-finalization result.
        """
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before finalize_policy()")
        if not self._episodes:
            raise ValueError("No episodes recorded for current policy")

        episodes_dir = self.get_episodes_dir()
        saved_files = []
        steps_info = {}
        best_episode_num = self._first_episode_num
        mid_episode_num = self._first_episode_num

        if self.mode == "continuous":
            self._save_episode(self._first_data, "continuous", "final", episodes_dir)
            saved_files.append(".../episode_continuous_final.json")
            steps_info["final"] = self._first_steps
            result = {
                "files": saved_files,
                "steps": steps_info,
                "best_episode_num": 1,
                "milestones": {"final": 1},
            }
            self._active_episode_logger = None
            return result

        if self._is_deterministic:
            self._save_episode(self._first_data, "first", self._first_episode_num, episodes_dir)
            saved_files.append(f".../episode_first_ep{self._first_episode_num:02d}.json")
            steps_info["first"] = self._first_steps
        else:
            self._save_episode(self._first_data, "first", self._first_episode_num, episodes_dir)
            saved_files.append(f".../episode_first_ep{self._first_episode_num:02d}.json")
            steps_info["first"] = self._first_steps

            self._save_episode(self._best_data, "best", self._best_episode_num, episodes_dir)
            saved_files.append(f".../episode_best_ep{self._best_episode_num:02d}.json")
            steps_info["best"] = self._best_steps
            best_episode_num = self._best_episode_num

            mid_data, mid_steps, mid_episode_num = self._select_mid_episode()
            self._save_episode(mid_data, "mid", mid_episode_num, episodes_dir)
            saved_files.append(f".../episode_mid_ep{mid_episode_num:02d}.json")
            steps_info["mid"] = mid_steps

        result = {
            "files": saved_files,
            "steps": steps_info,
            "best_episode_num": best_episode_num,
            "milestones": {
                "first": self._first_episode_num,
                "best": best_episode_num,
                "mid": mid_episode_num,
            },
        }
        self._active_episode_logger = None
        return result

    def _select_mid_episode(self) -> Tuple[Dict[str, Any], int, int]:
        """Select the episode closest to the average of first and best steps."""
        target_steps = (self._first_steps + self._best_steps) / 2

        best_match: Optional[Dict[str, Any]] = None
        best_match_steps = 0
        best_match_episode_num = 0
        best_distance = float("inf")

        for episode_data, steps, episode_num in self._episodes:
            distance = abs(steps - target_steps)
            if distance < best_distance:
                best_distance = distance
                best_match = episode_data
                best_match_steps = steps
                best_match_episode_num = episode_num

        if best_match is None:
            raise ValueError("No episodes available for mid-episode selection")

        return best_match, best_match_steps, best_match_episode_num

    def _save_episode(
        self,
        episode_data: Optional[Dict[str, Any]],
        category: str,
        episode_val: Any,
        episodes_dir: str,
    ) -> str:
        """Persist a selected episode into the episodes directory."""
        if episode_data is None:
            raise ValueError(f"No episode data available for category '{category}'")

        if isinstance(episode_val, int):
            filename = f"episode_{category}_ep{episode_val:02d}.json"
        else:
            filename = f"episode_{category}_{episode_val}.json"

        filepath = os.path.join(episodes_dir, filename)
        with open(filepath, "w") as f:
            json.dump(episode_data, f, indent=2)
        return filepath
