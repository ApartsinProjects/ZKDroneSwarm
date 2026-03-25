"""
EnvironmentLogger facade for environment output orchestration.

Provides a single caller-facing entrypoint for run-level logging lifecycle
while reusing the existing EpisodeLogger for per-episode capture.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        self._environment_data: Optional[Dict[str, Any]] = None

        self._current_policy: Optional[str] = None
        self._current_policy_path: Optional[str] = None
        self._is_deterministic = False

        self._episodes: List[Tuple[Dict[str, Any], int, int]] = []
        self._first_data: Optional[Dict[str, Any]] = None
        self._first_steps = 0
        self._first_episode_num = 0
        self._best_steps = float("inf")
        self._best_episode_num = 0
        self._episode_counter = 0

        self._active_episode_logger: Optional[EpisodeLogger] = None
        self._continuous_learning_state_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._continuous_episode_num: Optional[int] = None
        self._continuous_num_agents: Optional[int] = None
        self._continuous_num_targets: Optional[int] = None
        self._continuous_latent_dim: Optional[int] = None

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
        self._best_steps = float("inf")
        self._best_episode_num = 0
        self._episode_counter = 0
        self._clear_continuous_flush_context()

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

    def get_environment_path(self) -> str:
        """Return the run-level shared environment artifact path."""
        return os.path.join(self._scenario_path, "environment.json")

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

    def get_episode_summary_path(self) -> str:
        """Return the current policy episode summary artifact path."""
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_episode_summary_path()")
        return os.path.join(self._current_policy_path, "episodes_summary.json")

    def get_learning_state_dir(self) -> str:
        """Return the current policy learning state directory."""
        if self._current_policy_path is None:
            raise ValueError("start_policy() must be called before get_learning_state_dir()")
        return os.path.join(self._current_policy_path, "learning_state")

    def record_episode_data(self, episode_data: Dict[str, Any], steps: int) -> None:
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
            self._best_steps = steps
            self._best_episode_num = current_episode_num

    def record_episode(self, steps: int) -> None:
        """Record the active episode for later representative-episode selection."""
        self.record_episode_data(self.active_episode_logger.to_dict(), steps)

    def save_analysis_data(self, analysis_data: Dict[str, Any], episode_num: int) -> str:
        """Persist engagement analysis for an episode."""
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before save_analysis()")

        filename = f"analysis_ep{episode_num:02d}.json"

        filepath = os.path.join(self.get_analysis_dir(), filename)
        with open(filepath, "w") as f:
            json.dump(analysis_data, f, indent=2)
        return filepath

    def save_analysis(self, episode_num: int) -> str:
        """Persist engagement analysis for the active episode."""
        return self.save_analysis_data(self.active_episode_logger.get_analysis_data(), episode_num)



    def configure_continuous_flush(
        self,
        episode_num: int,
        learning_state_provider: Optional[Callable[[], Dict[str, Any]]],
        num_agents: Optional[int] = None,
        num_targets: Optional[int] = None,
        latent_dim: Optional[int] = None,
    ) -> None:
        """
        Register the current episode's learning-state provider for flush checkpoints.

        The provider remains external so EnvironmentLogger does not need to know
        policy internals; it only owns when and how checkpoint artifacts are saved.
        """
        self._continuous_episode_num = episode_num
        self._continuous_learning_state_provider = learning_state_provider
        self._continuous_num_agents = num_agents
        self._continuous_num_targets = num_targets
        self._continuous_latent_dim = latent_dim

    def start_episode(
        self,
        env: Any,
        reset_info: Dict[str, Any],
        seed: Optional[int] = None,
        episode_num: Optional[int] = None,
        total_episodes: Optional[int] = None,
    ) -> None:
        """Start the active episode logger for the current episode."""
        self._persist_environment_data(
            {
                "version": EpisodeLogger.VERSION,
                "scenario_id": getattr(env, "scenario_id", self.scenario_id),
                "config": self.active_episode_logger._build_shared_config_snapshot(env),
                "scenario": self.active_episode_logger._build_scenario_snapshot(
                    env, reset_info, seed
                ),
            }
        )
        self.active_episode_logger.start_episode(
            env=env,
            reset_info=reset_info,
            seed=seed,
            episode_num=episode_num,
            total_episodes=total_episodes,
            environment_path=os.path.relpath(
                self.get_environment_path(),
                self.get_episodes_dir(),
            ),
        )

    def log_step(
        self,
        step_num: int,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Record a step on the active episode logger."""
        self.active_episode_logger.log_step(
            step_num=step_num,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def flush_episode(self, step_num: int) -> str:
        """Flush the active episode logger's buffered step chunk."""
        return self.active_episode_logger.flush(step_num)

    def end_episode(
        self,
        total_rewards: Dict[str, float],
        done_reason: Optional[str],
    ) -> None:
        """Finalize the active episode logger for the current episode."""
        self.active_episode_logger.end_episode(
            total_rewards=total_rewards,
            done_reason=done_reason,
        )

    def handle_flush(self, step: int) -> None:
        """
        Persist any run-level artifacts that should accompany an episode flush.

        In continuous mode, this currently means saving a learning-state snapshot
        when a learning-state provider was registered for the active episode.
        """
        if self.mode != "continuous":
            return
        if self._continuous_learning_state_provider is None:
            return
        if self._continuous_episode_num is None:
            raise ValueError("configure_continuous_flush() must be called before handle_flush()")

        current_episode_state = self._continuous_learning_state_provider()
        self.save_learning_state(
            episode_state=current_episode_state,
            episode_num=self._continuous_episode_num,
            num_agents=self._continuous_num_agents or 0,
            num_targets=self._continuous_num_targets or 0,
            latent_dim=self._continuous_latent_dim,
            tag=f"step_{step:05d}",
        )

    def save_learning_state(
        self,
        episode_state: Optional[Dict[str, Any]],
        episode_num: int,
        num_agents: int,
        num_targets: int,
        latent_dim: Optional[int],
        tag: Optional[str] = None,
    ) -> str:
        """Persist learning-state snapshots for a policy episode."""
        if self._current_policy is None:
            raise ValueError("start_policy() must be called before save_learning_state()")

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
            "episode_state": episode_state,
        }

        with open(filepath, "w") as f:
            json.dump(learning_state, f, indent=2)
        return filepath

    def save_policy_episodes(self) -> Dict[str, Any]:
        """
        Save the current policy's episodes to disk and clear the active logger.

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
        milestones = {"first": self._first_episode_num}

        # 1. Continuous Mode: Capture the one long episode
        if self.mode == "continuous":
            self._save_episode(self._first_data, 1, episodes_dir)
            saved_files.append(".../episode_ep01.json")
            steps_info["final"] = self._first_steps
            milestones = {"final": 1}
        # 2. Deterministic/Baseline (Oracles, Random): Only save first attempt
        elif self._is_deterministic:
            self._save_episode(self._first_data, self._first_episode_num, episodes_dir)
            saved_files.append(f".../episode_ep{self._first_episode_num:02d}.json")
            steps_info["first"] = self._first_steps
        # 3. Non-deterministic (Learning): Save full history for analysis/t-SNE
        else:
            saved_paths = []
            for ep_data, ep_steps, ep_num in self._episodes:
                saved_paths.append(self._save_episode(ep_data, ep_num, episodes_dir))
            
            saved_files.extend(f".../{os.path.basename(p)}" for p in saved_paths)
            self._save_episode_summary(saved_paths)
            steps_info["first"] = self._first_steps
            steps_info["final"] = self._episodes[-1][1]
            milestones["best"] = self._best_episode_num

        result = {
            "files": saved_files,
            "steps": steps_info,
            "best_episode_num": self._best_episode_num,
            "milestones": milestones,
        }
        self._active_episode_logger = None
        self._clear_continuous_flush_context()
        return result

    def _save_episode(
        self,
        episode_data: Optional[Dict[str, Any]],
        episode_num: int,
        episodes_dir: str,
    ) -> str:
        """Persist an episode artifact into the episodes directory."""
        if episode_data is None:
            raise ValueError("No episode data available to persist")

        filename = f"episode_ep{episode_num:02d}.json"
        filepath = os.path.join(episodes_dir, filename)
        with open(filepath, "w") as f:
            json.dump(episode_data, f, indent=2)
        return filepath

    def _save_episode_summary(self, saved_episode_paths: List[str]) -> str:
        """Persist the per-policy episode summary artifact."""
        total_steps = 0
        total_steps_to_best = 0

        for _episode_data, steps, episode_num in self._episodes:
            total_steps += steps
            if episode_num <= self._best_episode_num:
                total_steps_to_best += steps

        best_episode_filename = os.path.basename(
            saved_episode_paths[self._best_episode_num - 1]
        )
        summary = {
            "total_episodes": len(self._episodes),
            "total_steps": total_steps,
            "total_steps_to_best": total_steps_to_best,
            "best_episode_path": f"episodes/{best_episode_filename}",
        }

        summary_path = self.get_episode_summary_path()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path

    def _persist_environment_data(self, environment_data: Dict[str, Any]) -> None:
        """Persist the run-level shared environment artifact once per scenario."""
        if self._environment_data is not None:
            return

        self._environment_data = environment_data
        with open(self.get_environment_path(), "w") as f:
            json.dump(environment_data, f, indent=2)

    def _clear_continuous_flush_context(self) -> None:
        self._continuous_learning_state_provider = None
        self._continuous_episode_num = None
        self._continuous_num_agents = None
        self._continuous_num_targets = None
        self._continuous_latent_dim = None
