"""
Decentralized Matrix Factorization Policy for ZK-MRTA Environment.

Implements classical collaborative-filtering matrix factorization adapted
for decentralized online ZK-MRTA. Each drone maintains a unified local
latent model (P and U matrices) learned via SGD with L2 regularization
from public swarm interaction events.

This is a standalone policy that does NOT inherit from BaseCFAgentPolicy.
It satisfies MultiAgentPolicy's duck-typing contract.

Reference: docs/policies/classic-collaborative-filtering/matrix-factorization-policy.md
"""

from typing import Dict, Optional, Any

import numpy as np


class MatrixFactorizationPolicy:
    """
    Decentralized Collaborative Matrix-Factorization policy for ZK-MRTA.

    Each drone maintains its own local latent model:
    - P: drone latent matrix (num_agents x latent_dim) — rows represent drones
    - U: target latent matrix (latent_dim x num_targets) — columns represent targets

    Predicted utility of drone i engaging target t:
        r_hat = P[i].T @ U[:, t]

    Learning uses SGD with L2 regularization on public swarm interaction events.
    Action selection uses the drone's own row (P[agent_idx]) with ε-greedy
    exploration using a linear decay schedule.

    ZK-MRTA Compliant: No shared state between drone instances.
    Designed for use with MultiAgentPolicy wrapper.
    """

    is_deterministic: bool = False

    def __init__(
        self,
        num_targets: int,
        agent_idx: int,
        num_agents: int,
        latent_dim: int = 8,
        learning_rate: float = 0.01,
        lambda_reg: float = 0.02,
        epsilon_start: float = 0.20,
        epsilon_min: float = 0.02,
        decay_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize MF policy for a single drone.

        Args:
            num_targets: Number of targets in the environment
            agent_idx: This drone's index (0-based)
            num_agents: Total number of drones
            latent_dim: Dimension of latent vectors (default 8)
            learning_rate: SGD learning rate η (default 0.01)
            lambda_reg: L2 regularization coefficient λ (default 0.02)
            epsilon_start: Initial exploration rate (default 0.20)
            epsilon_min: Minimum exploration rate (default 0.02)
            decay_steps: Steps over which ε decays linearly (default: None,
                         computed as half of expected training horizon if not set)
            seed: Random seed for reproducibility
        """
        self.num_targets = num_targets
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_steps = decay_steps
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        # Local latent matrices
        self.P: np.ndarray = None  # (num_agents, latent_dim)
        self.U: np.ndarray = None  # (latent_dim, num_targets)
        self._init_matrices()

        # Step counter for linear ε-decay
        self.step_count = 0

    def _init_matrices(self) -> None:
        """Initialize P and U with small Gaussian random values N(0, 0.01^2)."""
        self.P = self.rng.normal(0.0, 0.01, (self.num_agents, self.latent_dim)).astype(
            np.float64
        )
        self.U = self.rng.normal(0.0, 0.01, (self.latent_dim, self.num_targets)).astype(
            np.float64
        )

    @property
    def agent_lv(self) -> np.ndarray:
        """Visualization alias: this drone's latent row in P. Shape: (latent_dim,)."""
        return self.P[self.agent_idx]

    @property
    def target_lv(self) -> np.ndarray:
        """Visualization alias: target latent vectors transposed. Shape: (num_targets, latent_dim)."""
        return self.U.T

    @property
    def epsilon(self) -> float:
        """Current exploration rate based on linear decay schedule."""
        if self.decay_steps is None or self.decay_steps <= 0:
            return self.epsilon_start
        return max(
            self.epsilon_min,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_min) * self.step_count / self.decay_steps,
        )

    def predict_reward(self, target_idx: int) -> float:
        """
        Predict utility of this drone engaging a target.

        Uses this drone's own row in P and the target column in U:
            r_hat = P[agent_idx].T @ U[:, target_idx]

        Args:
            target_idx: Target index (0-based)

        Returns:
            Predicted utility score (unbounded, not clipped to [0,1])
        """
        return float(self.P[self.agent_idx] @ self.U[:, target_idx])

    def _predict_for_drone(self, drone_idx: int, target_idx: int) -> float:
        """
        Predict utility of any drone engaging a target using local model.

        Args:
            drone_idx: Drone index (0-based)
            target_idx: Target index (0-based)

        Returns:
            Predicted utility score
        """
        return float(self.P[drone_idx] @ self.U[:, target_idx])

    def select_action(
        self,
        observation: Dict[str, Any],
        allow_noop: bool = False,
    ) -> int:
        """
        Select action using ε-greedy with linear decay over predicted scores.

        Args:
            observation: Dict observation from collaborative mode with 'targets' key
            allow_noop: If True, include NoOp (0) as valid action

        Returns:
            Action: 0 for NoOp, 1-N for fire at target
        """
        targets_obs = observation["targets"]
        num_targets = len(targets_obs) // 3

        # Identify active targets
        active_targets = []
        for t_idx in range(num_targets):
            is_active = targets_obs[t_idx * 3 + 2] > 0.5
            if is_active:
                active_targets.append(t_idx)

        if not active_targets:
            return 0

        current_epsilon = self.epsilon

        # ε-greedy: Explore
        if self.rng.random() < current_epsilon:
            valid_actions = [t + 1 for t in active_targets]
            if allow_noop:
                valid_actions = [0] + valid_actions
            action = self.rng.choice(valid_actions)
            self.step_count += 1
            return int(action)

        # ε-greedy: Exploit — choose target with highest predicted score
        best_action = active_targets[0] + 1
        best_score = -np.inf

        for t_idx in active_targets:
            score = self.predict_reward(t_idx)
            if score > best_score:
                best_score = score
                best_action = t_idx + 1

        self.step_count += 1
        return int(best_action)

    def update_from_observation(self, observation: Dict[str, Any]) -> None:
        """
        Update local P and U matrices from collaborative mode observation.

        Processes all public interaction events from the swarm.
        For each event (drone i, target t, reward r):
            - Compute prediction error: e = P[i].T @ U[:, t] - r
            - Update: P[i] -= η * (2*e*U[:, t] + λ*P[i])
            - Update: U[:, t] -= η * (2*e*P[i] + λ*U[:, t])

        Args:
            observation: Dict with 'selected_targets' and 'observed_rewards' arrays
        """
        selected_targets = observation["selected_targets"]
        observed_rewards = observation["observed_rewards"]

        for drone_idx in range(self.num_agents):
            target_action = selected_targets[drone_idx]
            reward = observed_rewards[drone_idx]

            # Skip NoOp actions
            if target_action <= 0:
                continue

            target_idx = int(target_action) - 1

            # Skip negative rewards (wasted shots — timing issue, not
            # reflective of true compatibility)
            if reward < 0:
                continue

            # Compute prediction and error
            predicted = self._predict_for_drone(drone_idx, target_idx)
            error = predicted - float(reward)

            # Snapshot vectors before update (for simultaneous update)
            p_i = self.P[drone_idx].copy()
            u_t = self.U[:, target_idx].copy()

            # SGD update with L2 regularization
            self.P[drone_idx] -= self.learning_rate * (
                2.0 * error * u_t + self.lambda_reg * p_i
            )
            self.U[:, target_idx] -= self.learning_rate * (
                2.0 * error * p_i + self.lambda_reg * u_t
            )

    def soft_reset(self, episode_count: Optional[int] = None) -> None:
        """
        Reset for new episode, preserving learned knowledge.

        The MF model retains P and U across episodes.
        Only the step counter is NOT reset — ε continues decaying
        across episodes as per the spec's training horizon concept.
        """
        pass

    def reset(self) -> None:
        """Full reset: reinitialize all latent matrices and step counter."""
        self.rng = np.random.RandomState(self.seed)
        self._init_matrices()
        self.step_count = 0

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        """
        Return learning state for logging/visualization.

        Returns:
            Dict with this drone's latent model state
        """
        predicted_rewards = [
            float(self.predict_reward(t)) for t in range(self.num_targets)
        ]
        ranked_targets = [
            int(x)
            for x in np.argsort(-np.asarray(predicted_rewards, dtype=np.float64)).tolist()
        ]
        best_target = int(ranked_targets[0]) if ranked_targets else None

        return {
            "agent_idx": self.agent_idx,
            "agent_lv": self.agent_lv.tolist(),
            "target_lv": self.target_lv.tolist(),
            "P": self.P.tolist(),
            "U": self.U.tolist(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "match": {
                "predicted_rewards": predicted_rewards,
                "ranked_targets": ranked_targets,
                "best_target": best_target,
            },
        }
