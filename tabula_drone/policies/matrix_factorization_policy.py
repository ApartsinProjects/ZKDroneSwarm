"""
Decentralized Matrix Factorization Policy for ZK-MRTA Environment.

Implements classical collaborative-filtering matrix factorization adapted
for decentralized online ZK-MRTA. Each drone maintains a unified local
embedding model (P and U matrices) learned via SGD with L2 regularization
from public swarm interaction events.

This is a standalone policy that does NOT inherit from BaseCFAgentPolicy.
It satisfies MultiAgentPolicy's duck-typing contract.

Reference: docs/policies/classic-collaborative-filtering/matrix-factorization-policy.md
"""

from typing import Dict, List, Optional, Any

import numpy as np


class MatrixFactorizationPolicy:
    """
    Decentralized Collaborative Matrix-Factorization policy for ZK-MRTA.

    Each drone maintains its own local embedding model:
    - P: drone embedding matrix (num_agents x latent_dim) — rows represent drones
    - U: target embedding matrix (latent_dim x num_targets) — columns represent targets

    Predicted utility of drone i engaging target t:
        r_hat = P[i].T @ U[:, t]

    Learning uses SGD with L2 regularization on public swarm interaction events.
    Action selection uses the drone's own row (P[agent_idx]) with ε-greedy
    exploration using a multiplicative decay schedule.

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
        epsilon: float = 0.20,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.02,
        anti_signal_weight: float = 0.1,
        use_integration_matrix: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize MF policy for a single drone.

        Args:
            num_targets: Number of targets in the environment
            agent_idx: This drone's index (0-based)
            num_agents: Total number of drones
            latent_dim: Dimension of embedding vectors (default 8)
            learning_rate: SGD learning rate η (default 0.01)
            lambda_reg: L2 regularization coefficient λ (default 0.02)
            epsilon: Initial exploration rate (default 0.20)
            epsilon_decay: Multiplicative decay factor for ε (default 1.0)
            epsilon_min: Minimum exploration rate (default 0.02)
            anti_signal_weight: Weight applied to negative reward updates (default 0.1)
            seed: Random seed for reproducibility
        """
        self.num_targets = num_targets
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.anti_signal_weight = anti_signal_weight
        self.use_integration_matrix = use_integration_matrix
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        # Local embedding matrices
        self.P: np.ndarray = None  # (num_agents, latent_dim)
        self.U: np.ndarray = None  # (latent_dim, num_targets)
        self._init_matrices()

        # Integration-matrix state (only when mode is active)
        if self.use_integration_matrix:
            self.M_sum = np.zeros((self.num_agents, self.num_targets), dtype=np.float64)
            self.M_count = np.zeros((self.num_agents, self.num_targets), dtype=np.float64)

        # Action counter for logging
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
    def agent_emb(self) -> np.ndarray:
        """Visualization alias: this drone's embedding row in P. Shape: (latent_dim,)."""
        return self.P[self.agent_idx]

    @property
    def target_emb(self) -> np.ndarray:
        """Visualization alias: target embedding vectors transposed. Shape: (num_targets, latent_dim)."""
        return self.U.T

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
        Select action using ε-greedy with multiplicative decay over predicted scores.

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

        # ε-greedy: Explore
        if self.rng.random() < self.epsilon:
            valid_actions = [t + 1 for t in active_targets]
            if allow_noop:
                valid_actions = [0] + valid_actions
            action = int(self.rng.choice(valid_actions))
        else:
            # ε-greedy: Exploit — choose target based on predicted scores
            scores = np.array([self.predict_reward(t_idx) for t_idx in active_targets])
            
            # Pure greedy max
            best_idx = np.argmax(scores)
            action = active_targets[best_idx] + 1

        # Apply multiplicative decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        
        return action

    def _update_integration_matrix(
        self, drone_idx: int, target_idx: int, reward: float
    ) -> float:
        """
        Accumulate an eligible reward into the supervised interaction matrix.

        The integration matrix serves as the MF supervision target. The environment
        decides which public events are eligible for accumulation, allowing the
        policy to learn a running mean over active-target rewards while excluding
        dead-target penalties.

        Args:
            drone_idx: Drone index (0-based)
            target_idx: Target index (0-based)
            reward: Eligible observed reward

        Returns:
            M_avg for this (drone, target) pair after accumulation
        """
        self.M_sum[drone_idx, target_idx] += reward
        self.M_count[drone_idx, target_idx] += 1.0
        return self.M_sum[drone_idx, target_idx] / self.M_count[drone_idx, target_idx]

    def update_from_observation(self, observation: Dict[str, Any]) -> None:
        """
        Update local P and U matrices from collaborative mode observation.

        Processes all public interaction events from the swarm.
        For each event (drone i, target t, reward r):
            - Compute prediction error: e = P[i].T @ U[:, t] - r
            - If r < 0, error is weighted by anti_signal_weight
            - Update: P[i] -= η * (2*e*U[:, t] + λ*P[i])
            - Update: U[:, t] -= η * (2*e*P[i] + λ*U[:, t])

        Args:
            observation: Dict with 'selected_targets' and 'observed_rewards' arrays.
                When integration-matrix mode is enabled, the environment may also
                provide a 'target_was_active_at_engagement' mask aligned with
                those arrays.
        """
        selected_targets = observation["selected_targets"]
        observed_rewards = observation["observed_rewards"]
        target_was_active_at_engagement = observation.get(
            "target_was_active_at_engagement"
        )

        for drone_idx in range(self.num_agents):
            target_action = selected_targets[drone_idx]
            reward = observed_rewards[drone_idx]

            # Skip NoOp actions
            if target_action <= 0:
                continue

            target_idx = int(target_action) - 1

            # Compute prediction and error
            predicted = self._predict_for_drone(drone_idx, target_idx)
            if self.use_integration_matrix:
                use_for_supervision = True
                if target_was_active_at_engagement is not None:
                    use_for_supervision = bool(
                        target_was_active_at_engagement[drone_idx]
                    )

                if not use_for_supervision:
                    continue

                m_avg = self._update_integration_matrix(
                    drone_idx, target_idx, float(reward)
                )
                error = predicted - m_avg
            else:
                error = predicted - float(reward)

            # Weight negative reward events in direct mode only.
            # In integration-matrix mode, dead-target shots are already excluded
            # by the target_was_active_at_engagement guard above, so this path
            # is only reached for active-target interactions whose raw reward
            # happened to go negative due to reward noise. anti_signal_weight=1.0
            # (the configured default) disables this weighting intentionally.
            if not self.use_integration_matrix and reward < 0:
                error *= self.anti_signal_weight

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
        """Full reset: reinitialize all latent matrices, step counter, and exploration state."""
        self.rng = np.random.RandomState(self.seed)
        self._init_matrices()
        if self.use_integration_matrix:
            self.M_sum[:] = 0.0
            self.M_count[:] = 0.0
        self.step_count = 0
        self.epsilon = self.initial_epsilon

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        """
        Return learning state for logging/visualization.

        Returns:
            Dict with this drone's embedding model state.
            P and U contain the full embedding matrices for offline enrichment.
        """
        predicted_rewards = [
            float(self.predict_reward(t)) for t in range(self.num_targets)
        ]
        ranked_targets = [
            int(x)
            for x in np.argsort(-np.asarray(predicted_rewards, dtype=np.float64)).tolist()
        ]
        best_target = int(ranked_targets[0]) if ranked_targets else None

        state = {
            "agent_idx": self.agent_idx,
            "P": self.P.tolist(),
            "U": self.U.tolist(),
            "epsilon": self.epsilon,
            "anti_signal_weight": self.anti_signal_weight,
            "step_count": self.step_count,
            "match": {
                "predicted_rewards": predicted_rewards,
                "ranked_targets": ranked_targets,
                "best_target": best_target,
            },
        }

        if self.use_integration_matrix:
            m_pred = self.P @ self.U
            m_avg = np.divide(
                self.M_sum, self.M_count,
                out=np.zeros_like(self.M_sum),
                where=self.M_count > 0,
            )
            state["integration_matrix"] = {
                "M_avg": m_avg.tolist(),
                "M_pred": m_pred.tolist(),
                "M_count": self.M_count.tolist(),
            }

        return state
