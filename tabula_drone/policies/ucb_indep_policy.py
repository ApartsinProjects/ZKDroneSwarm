"""
UCB1-Independent Policy for ZK-MRTA Environment.

Per-(drone, target) arm UCB1 bandit. ZK-compliant: uses only the public
observation stream (selected_targets + observed_rewards). No latent structure
or HP information is used.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import IPolicy, EnvInfos


class UCBIndepPolicy(IPolicy):
    """
    UCB1 bandit over independent (drone, target) arms.

    Each drone maintains its own UCB1 statistics over all targets.
    Statistics are updated from the shared public observation stream, so every
    drone updates all (drone, target) arm estimates, not just its own action.

    ZK-Compliant:
      - No latent vectors, no HP, no agent identities used.
      - Purely outcome-based: only observed_rewards and selected_targets.

    is_deterministic = False (exploration is non-trivial across episodes).
    Statistics persist across episodes via soft_reset (no-op).
    """

    is_deterministic: bool = False

    def __init__(
        self,
        num_agents: int,
        num_targets: int,
        c: float = 2.0,
        seed: Optional[int] = None,
        allow_noop: bool = False,
    ):
        """
        Args:
            num_agents: Number of drones.
            num_targets: Number of targets.
            c: UCB1 exploration constant (higher = more exploration).
            seed: RNG seed for tie-breaking.
            allow_noop: If True, return action 0 when no active targets exist.
        """
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.c = c
        self.allow_noop = allow_noop
        self.rng = np.random.RandomState(seed)

        # Per-(drone, target) statistics; shape (num_agents, num_targets)
        self._counts = np.zeros((num_agents, num_targets), dtype=np.int64)
        self._reward_sums = np.zeros((num_agents, num_targets), dtype=np.float64)
        # Per-drone total pull count (denominator for UCB log term)
        self._total_pulls = np.zeros(num_agents, dtype=np.int64)

    def _ucb_scores(self, drone_idx: int, active_mask: np.ndarray) -> np.ndarray:
        """Compute UCB1 scores for drone_idx over all targets."""
        counts = self._counts[drone_idx]
        sums = self._reward_sums[drone_idx]
        total = self._total_pulls[drone_idx]

        scores = np.full(self.num_targets, -np.inf)
        for t_idx in range(self.num_targets):
            if not active_mask[t_idx]:
                continue
            if counts[t_idx] == 0:
                scores[t_idx] = np.inf  # Untried arm: force exploration
            else:
                mean_r = sums[t_idx] / counts[t_idx]
                bonus = self.c * np.sqrt(np.log(max(total, 1)) / counts[t_idx])
                scores[t_idx] = mean_r + bonus
        return scores

    def select_actions(
        self,
        obs: Dict[str, Any],
        infos: EnvInfos,
        env: Any = None,
    ) -> Dict[str, int]:
        agent_ids = sorted(obs.keys())

        # Parse active mask from first observation's targets array
        first_obs = next(iter(obs.values()))
        target_array = first_obs["targets"]
        active_mask = np.array(
            [target_array[t * 3 + 2] > 0.5 for t in range(self.num_targets)],
            dtype=bool,
        )
        active_indices = np.where(active_mask)[0]

        actions: Dict[str, int] = {}
        for drone_idx, agent_id in enumerate(agent_ids):
            if len(active_indices) == 0:
                actions[agent_id] = 0
                continue

            scores = self._ucb_scores(drone_idx, active_mask)
            active_scores = scores[active_indices]
            max_score = float(np.max(active_scores))

            if np.isinf(max_score):
                # Multiple untried arms: pick uniformly among them
                untried = active_indices[np.isinf(active_scores)]
                chosen_target = int(self.rng.choice(untried))
            else:
                # Highest UCB score; break ties randomly
                best_mask = active_scores == max_score
                candidates = active_indices[best_mask]
                chosen_target = int(self.rng.choice(candidates))

            actions[agent_id] = chosen_target + 1  # Convert to 1-indexed

        return actions

    def update(self, obs: Dict[str, Any]) -> None:
        """Update arm statistics from the public observation stream."""
        # The public stream is the same for all agents; use any agent's obs.
        first_obs = next(iter(obs.values()))
        selected_targets = first_obs["selected_targets"]  # (num_agents,), 1-indexed
        observed_rewards = first_obs["observed_rewards"]  # (num_agents,), cosine + noise

        for drone_idx in range(self.num_agents):
            target_action = int(selected_targets[drone_idx])
            if target_action <= 0:
                continue
            target_idx = target_action - 1
            if target_idx >= self.num_targets:
                continue
            reward = float(observed_rewards[drone_idx])
            self._counts[drone_idx, target_idx] += 1
            self._reward_sums[drone_idx, target_idx] += reward
            self._total_pulls[drone_idx] += 1

    def soft_reset(self) -> None:
        """Keep statistics across episodes (persistent bandit learning)."""
        pass

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return {
            "counts": self._counts.tolist(),
            "reward_sums": self._reward_sums.tolist(),
        }
