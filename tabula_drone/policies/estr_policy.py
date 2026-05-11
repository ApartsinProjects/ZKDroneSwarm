"""
Explore-Then-Spectral-Refit (ESTR) Policy: a centralised low-rank bandit
baseline adapted from Kang, Hsieh, Lee 2022 ("Efficient frameworks for
generalised low-rank matrix bandit problems"), restricted to the public
broadcast information available in ZK-MRTA.

Algorithm structure:
  Phase 1 (Exploration), steps 1..T_explore:
      Pick actions uniformly at random; accumulate a single shared
      empirical reward matrix R_hat from the broadcast.
  Transition: compute rank-d SVD of R_hat, store the resulting factor
      matrices P_hat, U_hat.
  Phase 2 (Exploitation), steps T_explore+1..T_max:
      For each drone i, pick a = argmax_j (P_hat[i] @ U_hat[:, j])
      among active targets, with epsilon-greedy random tie-break.

This is a CENTRALISED estimator (one shared R_hat) although the
information it consumes is only the public broadcast (no latent vectors,
no HP, no parameter sharing across distinct learners since there is
only one learner). It is included as a baseline to compare against the
DECENTRALISED MF-CF / PTF policies of this paper.

Reference: Kang, Hsieh, Lee 2022 (NeurIPS 35). Section 4: ESTR for
generalised low-rank matrix bandits.
"""

from typing import Any, Dict, Optional
import numpy as np


class ESTRPolicy:
    """
    Explore-Then-Spectral-Refit, centralised, broadcast-only.

    Phase 1: T_explore steps of uniform-random arm pulls.
    Phase 2: Argmax over rank-d SVD-derived utility estimates.
    """

    is_deterministic: bool = False

    def __init__(
        self,
        num_agents: int,
        num_targets: int,
        latent_dim: int = 3,
        explore_steps: int = 350,  # ~5 episodes at s_ep=70 for n=27
        epsilon_explore: float = 0.05,  # post-transition exploration floor
        seed: Optional[int] = None,
        allow_noop: bool = False,
    ):
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.latent_dim = latent_dim
        self.explore_steps = explore_steps
        self.epsilon_explore = epsilon_explore
        self.allow_noop = allow_noop
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        # Centralised empirical reward matrix (sum of rewards / count)
        self._sum = np.zeros((num_agents, num_targets), dtype=np.float64)
        self._cnt = np.zeros((num_agents, num_targets), dtype=np.float64)

        # Cached factor estimates after transition
        self._P_hat: Optional[np.ndarray] = None
        self._U_hat: Optional[np.ndarray] = None
        self._transitioned = False

        # Step counter
        self._step = 0

    # ------------------------------------------------------------------

    def select_actions(self, obs: Dict[str, Any], infos: Any = None, env: Any = None) -> Dict[str, int]:
        agent_ids = sorted(obs.keys())
        first_obs = obs[agent_ids[0]]
        target_array = first_obs["targets"]
        active = np.array(
            [target_array[t * 3 + 2] > 0.5 for t in range(self.num_targets)],
            dtype=bool,
        )
        active_idx = np.where(active)[0]
        actions: Dict[str, int] = {}

        for drone_idx, agent_id in enumerate(agent_ids):
            if len(active_idx) == 0:
                actions[agent_id] = 0
                continue
            # Phase decision
            if self._step < self.explore_steps or not self._transitioned:
                # Exploration: uniform random among active
                target_idx = int(self.rng.choice(active_idx))
            else:
                # Exploitation with small epsilon-explore floor
                if self.rng.random() < self.epsilon_explore:
                    target_idx = int(self.rng.choice(active_idx))
                else:
                    # Argmax of rank-d utility
                    utility = self._P_hat[drone_idx] @ self._U_hat[:, active_idx]
                    best = int(np.argmax(utility))
                    target_idx = int(active_idx[best])
            actions[agent_id] = target_idx + 1

        self._step += 1
        return actions

    def update(self, obs: Dict[str, Any]) -> None:
        """Update shared empirical matrix from broadcast."""
        first_obs = next(iter(obs.values()))
        selected = first_obs["selected_targets"]
        rewards = first_obs["observed_rewards"]
        for i in range(self.num_agents):
            target_action = int(selected[i])
            if target_action <= 0:
                continue
            j = target_action - 1
            if j >= self.num_targets:
                continue
            self._sum[i, j] += float(rewards[i])
            self._cnt[i, j] += 1.0

        # Transition: at the end of the exploration phase, compute SVD
        if (not self._transitioned) and self._step >= self.explore_steps:
            self._compute_factors()
            self._transitioned = True

    def _compute_factors(self):
        """Truncated SVD of the empirical matrix to get rank-d factors."""
        R_hat = np.where(self._cnt > 0, self._sum / np.maximum(self._cnt, 1.0), 0.0)
        try:
            U_svd, S_svd, Vt_svd = np.linalg.svd(R_hat, full_matrices=False)
            rank = min(self.latent_dim, len(S_svd))
            sqrt_S = np.sqrt(np.maximum(S_svd[:rank], 0.0))
            self._P_hat = U_svd[:, :rank] * sqrt_S[np.newaxis, :]
            self._U_hat = (Vt_svd[:rank, :].T * sqrt_S[np.newaxis, :]).T
            # Pad if rank < latent_dim (R_hat nearly zero)
            if rank < self.latent_dim:
                pad_P = self.rng.normal(0.0, 0.01, (self.num_agents, self.latent_dim - rank))
                pad_U = self.rng.normal(0.0, 0.01, (self.latent_dim - rank, self.num_targets))
                self._P_hat = np.hstack([self._P_hat, pad_P])
                self._U_hat = np.vstack([self._U_hat, pad_U])
        except np.linalg.LinAlgError:
            # SVD failed: fall back to small random
            self._P_hat = self.rng.normal(0.0, 0.01, (self.num_agents, self.latent_dim))
            self._U_hat = self.rng.normal(0.0, 0.01, (self.latent_dim, self.num_targets))

    def soft_reset(self, episode_count: Optional[int] = None) -> None:
        pass

    def reset(self) -> None:
        self.rng = np.random.RandomState(self.seed)
        self._sum[:, :] = 0.0
        self._cnt[:, :] = 0.0
        self._P_hat = None
        self._U_hat = None
        self._transitioned = False
        self._step = 0

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return {
            "policy": "estr",
            "step": self._step,
            "transitioned": self._transitioned,
            "explore_steps": self.explore_steps,
        }
