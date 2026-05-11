"""
No-Broadcast Matrix Factorization Policy for ZK-MRTA.

Identical to MatrixFactorizationPolicy except that update_from_observation()
only consumes this drone's own engagement outcome from the broadcast tuple,
ignoring the other m-1 entries. This implements the "no broadcast" ablation
in which each drone learns purely from its own private observations.

Used in the empirical verification of Theorem 6 (broadcast multiplier):
the prediction is that without broadcast, MF-CF's cold-start is m times
longer, well beyond the 35-episode evaluation budget at m=9, d=3.
"""

from typing import Any, Dict

from .matrix_factorization_policy import MatrixFactorizationPolicy


class NoBroadcastMatrixFactorizationPolicy(MatrixFactorizationPolicy):
    """
    Drop-in replacement for MatrixFactorizationPolicy that uses only the
    drone's own observation per step. Action selection is identical; only
    the SGD update step differs.
    """

    def update_from_observation(self, observation: Dict[str, Any]) -> None:
        """
        Update local P and U matrices using ONLY this drone's own engagement.

        Mirrors the parent class's update, but the for-loop is restricted
        to drone_idx == self.agent_idx, ignoring the broadcast.
        """
        selected_targets = observation["selected_targets"]
        observed_rewards = observation["observed_rewards"]

        # Only consume this drone's own data
        drone_idx = self.agent_idx
        target_action = selected_targets[drone_idx]
        reward = observed_rewards[drone_idx]

        if target_action <= 0:
            return

        target_idx = int(target_action) - 1

        predicted = self._predict_for_drone(drone_idx, target_idx)
        if self.use_integration_matrix:
            m_avg = self._update_integration_matrix(
                drone_idx, target_idx, float(reward)
            )
            error = predicted - m_avg
        else:
            error = predicted - float(reward)

        if not self.use_integration_matrix and reward < 0:
            error *= self.anti_signal_weight

        p_i = self.P[drone_idx].copy()
        u_t = self.U[:, target_idx].copy()

        self.P[drone_idx] -= self.learning_rate * (
            2.0 * error * u_t + self.lambda_reg * p_i
        )
        self.U[:, target_idx] -= self.learning_rate * (
            2.0 * error * p_i + self.lambda_reg * u_t
        )
