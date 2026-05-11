"""
tabula_drone/policies/ucb_homogeneous_policy.py

UCB1-Homogeneous Policy: a single shared arm table across ALL drones.

This is the "naive pooling" baseline -- the policy assumes all drones are
identical and treats arm(j) as a single target-quality estimate regardless
of which drone fires at it.

In a heterogeneous environment (drones have different latent modes), pooling
rewards from all drones at target j produces confounded estimates: a mode-A
target gives reward ~1.0 to mode-A drones but ~0.1 to all others.  The pooled
mean collapses to ~1/num_modes for all targets, making targets indistinguishable
and driving near-random selection.

This is a deliberate stress-test of what happens when the learning policy
ignores latent heterogeneity.  It contrasts with:
  - UCBIndepPolicy : per-(drone, target) arm -- learns correct heterogeneity
                     but has no generalisation across arms.
  - MF-CF          : low-rank factorization -- learns the structure that
                     *explains* the heterogeneity and generalises across arms.

ZK-Compliant: uses only the public observation stream (no latent vectors,
no HP).  Statistics persist across episodes.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base import IPolicy, EnvInfos


class UCBHomogeneousPolicy(IPolicy):
    """
    UCB1 with a single shared target-quality arm table.

    Every drone observes every engagement outcome via the broadcast stream.
    All observed rewards for target j (from any drone) update the single
    arm(j) estimate.  Every drone then selects greedily from the same shared
    arm table.

    This makes the policy completely drone-agnostic: drone 0 and drone 8 make
    the same target selection given the same arm table state.

    Failure mode in heterogeneous environments:
      arm(j).mean converges to avg_reward_over_all_drone_modes(j)
          = sum_modes( p(mode) * compatibility(mode, mode_of_j) )
          ~= 1 / num_modes    (uniform mode distribution, one-hot structure)
      All targets converge to the same pooled mean --> indistinguishable.
      The exploration bonus resolves ties randomly --> near-random assignment.
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
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.c = c
        self.allow_noop = allow_noop
        self.rng = np.random.RandomState(seed)

        # Single shared arm table -- one estimate per target
        self._counts = np.zeros(num_targets, dtype=np.int64)
        self._reward_sums = np.zeros(num_targets, dtype=np.float64)
        self._total_observations = 0  # total arm pulls seen across all drones

    def _ucb_scores(self, active_mask: np.ndarray) -> np.ndarray:
        """Compute shared UCB1 scores (same for every drone)."""
        scores = np.full(self.num_targets, -np.inf)
        total = max(self._total_observations, 1)
        for t_idx in range(self.num_targets):
            if not active_mask[t_idx]:
                continue
            if self._counts[t_idx] == 0:
                scores[t_idx] = np.inf
            else:
                mean_r = self._reward_sums[t_idx] / self._counts[t_idx]
                bonus = self.c * np.sqrt(np.log(total) / self._counts[t_idx])
                scores[t_idx] = mean_r + bonus
        return scores

    def select_actions(
        self,
        obs: Dict[str, Any],
        infos: EnvInfos,
        env: Any = None,
    ) -> Dict[str, int]:
        agent_ids = sorted(obs.keys())
        first_obs = next(iter(obs.values()))
        target_array = first_obs["targets"]
        active_mask = np.array(
            [target_array[t * 3 + 2] > 0.5 for t in range(self.num_targets)],
            dtype=bool,
        )
        active_indices = np.where(active_mask)[0]

        # All drones share the same UCB scores
        scores = self._ucb_scores(active_mask)

        actions: Dict[str, int] = {}
        for agent_id in agent_ids:
            if len(active_indices) == 0:
                actions[agent_id] = 0
                continue
            active_scores = scores[active_indices]
            max_score = float(np.max(active_scores))
            if np.isinf(max_score):
                untried = active_indices[np.isinf(active_scores)]
                chosen = int(self.rng.choice(untried))
            else:
                best_mask = active_scores == max_score
                candidates = active_indices[best_mask]
                chosen = int(self.rng.choice(candidates))
            actions[agent_id] = chosen + 1  # 1-indexed
        return actions

    def update(self, obs: Dict[str, Any]) -> None:
        """Update shared arm table from ALL drones' observed engagements."""
        first_obs = next(iter(obs.values()))
        selected_targets = first_obs["selected_targets"]  # (num_agents,), 1-indexed
        observed_rewards = first_obs["observed_rewards"]  # (num_agents,)

        for drone_idx in range(self.num_agents):
            target_action = int(selected_targets[drone_idx])
            if target_action <= 0:
                continue
            target_idx = target_action - 1
            if target_idx >= self.num_targets:
                continue
            reward = float(observed_rewards[drone_idx])
            self._counts[target_idx] += 1
            self._reward_sums[target_idx] += reward
            self._total_observations += 1

    def soft_reset(self) -> None:
        """Statistics persist across episodes."""
        pass

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return {
            "counts": self._counts.tolist(),
            "reward_sums": self._reward_sums.tolist(),
        }
