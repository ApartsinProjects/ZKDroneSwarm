"""
Zero-Knowledge Independent Q-Learning (IQL-ZK) Policy for ZK-MRTA.

A standard ZK-compliant Independent Q-Learning baseline adapted from the
multi-agent reinforcement learning literature (Tan 1993; Tampuu et al. 2017)
to satisfy the strict ZK-MRTA observation model.

Each drone i maintains a private tabular Q-function Q_i : [n] -> R over
target identities. Action selection uses epsilon-greedy with multiplicative
decay across steps. Updates are TD-style with a learning-rate alpha:

    Q_i[a_t] <- (1 - alpha) * Q_i[a_t] + alpha * r_{t,i}

(equivalent to running-average estimation with rate alpha when gamma=0 and
rewards are immediate per-step).

ZK-compliance:
  - Each drone updates Q_i[j] only from its OWN observed reward r_{t,i}
    when it pulled target j at step t. The shared public broadcast is
    observed (since it is public) but only the drone's own (action,
    reward) entry is consumed for Q-table updates.
  - No latent vectors are read; no HP is read; no parameter sharing.
  - Differs from UCB-Indep in two ways: (i) TD update rather than
    sample-mean; (ii) epsilon-greedy exploration rather than UCB
    confidence bonus.

Reference: Tampuu et al. 2017 (independent DQN), adapted to the tabular
discrete-action ZK-MRTA setting.
"""

from typing import Any, Dict, Optional
import numpy as np


class IQLZKPolicy:
    """
    Tabular Independent Q-Learning, ZK-compliant.
    """

    is_deterministic: bool = False

    def __init__(
        self,
        num_agents: int,
        num_targets: int,
        alpha: float = 0.1,
        gamma: float = 0.0,
        epsilon: float = 0.30,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        q_init: float = 0.5,
        seed: Optional[int] = None,
        allow_noop: bool = False,
    ):
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_init = q_init
        self.seed = seed
        self.allow_noop = allow_noop

        self.rng = np.random.RandomState(seed)
        # Q[i, j] = estimated value of drone i picking target j
        self.Q = np.full((num_agents, num_targets), q_init, dtype=np.float64)
        self._step_count = 0

    # ------------------------------------------------------------------
    # IPolicy interface
    # ------------------------------------------------------------------

    def select_actions(self, obs: Dict[str, Any], infos: Any = None, env: Any = None) -> Dict[str, int]:
        agent_ids = sorted(obs.keys())
        first_obs = obs[agent_ids[0]]
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
            if self.rng.random() < self.epsilon:
                target_idx = int(self.rng.choice(active_indices))
            else:
                q_vals = self.Q[drone_idx, active_indices]
                max_q = float(np.max(q_vals))
                # Break ties randomly
                best_mask = q_vals == max_q
                candidates = active_indices[best_mask]
                target_idx = int(self.rng.choice(candidates))
            actions[agent_id] = target_idx + 1  # 1-indexed action

        self._step_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return actions

    def update(self, obs: Dict[str, Any]) -> None:
        """
        Tabular TD update from each drone's own observation only.
        ZK-compliant: drone i consumes only its own (action, reward) entry
        from the broadcast, ignoring others' entries.
        """
        first_obs = next(iter(obs.values()))
        selected_targets = first_obs["selected_targets"]  # 1-indexed per drone
        observed_rewards = first_obs["observed_rewards"]

        for drone_idx in range(self.num_agents):
            target_action = int(selected_targets[drone_idx])
            if target_action <= 0:
                continue
            target_idx = target_action - 1
            if target_idx >= self.num_targets:
                continue
            r = float(observed_rewards[drone_idx])
            # TD update with gamma=0: Q <- (1-alpha) Q + alpha r
            self.Q[drone_idx, target_idx] = (
                (1.0 - self.alpha) * self.Q[drone_idx, target_idx]
                + self.alpha * r
            )

    def soft_reset(self, episode_count: Optional[int] = None) -> None:
        """End-of-episode reset. Q-table is preserved across episodes."""
        pass

    def reset(self) -> None:
        """Full reset."""
        self.rng = np.random.RandomState(self.seed)
        self.Q[:, :] = self.q_init
        self.epsilon = self.initial_epsilon
        self._step_count = 0

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return {
            "policy": "iql_zk",
            "step_count": self._step_count,
            "epsilon": float(self.epsilon),
            "q_mean": float(np.mean(self.Q)),
            "q_std": float(np.std(self.Q)),
        }
