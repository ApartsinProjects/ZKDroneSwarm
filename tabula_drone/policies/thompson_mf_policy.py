"""
Thompson-Sampling Matrix Factorization (TS-MF) Policy for ZK-MRTA.

A Bayesian variant of the MF-CF policy that replaces epsilon-greedy
exploration with Thompson sampling over the latent embedding factors.
Inspired by Kawale et al. (2015) "Efficient Thompson Sampling for Online
Matrix-Factorization Recommendation" and adapted to the ZK-MRTA online
setting.

Mechanism:
  - Maintains MF factors (P, U) updated by SGD identically to MF-CF.
  - At action-selection time, samples noisy factor copies and selects
    argmax_j tilde_P[i] @ tilde_U[:, j] for drone i.
  - Variance sigma_t = sigma_0 / sqrt(t) decays as the policy accumulates
    observations, mirroring posterior contraction of a Bayesian
    linear-Gaussian model.

The SGD update is identical to MatrixFactorizationPolicy; only the
single-drone select_action() method differs.
"""

from typing import Any, Dict, Optional
import numpy as np

from .matrix_factorization_policy import MatrixFactorizationPolicy


class ThompsonMFPolicy(MatrixFactorizationPolicy):
    """MF-CF with Thompson-sampling action selection."""

    is_deterministic: bool = False

    def __init__(self, *args, ts_sigma_0: float = 0.30, ts_sigma_min: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.ts_sigma_0 = ts_sigma_0
        self.ts_sigma_min = ts_sigma_min

    def _current_sigma(self) -> float:
        t = max(self.step_count, 1)
        sigma = self.ts_sigma_0 / np.sqrt(t)
        return float(max(sigma, self.ts_sigma_min))

    def select_action(self, observation: Dict[str, Any], allow_noop: bool = False) -> int:
        """
        Thompson-sampling action selection for a single drone.

        Replaces the parent's epsilon-greedy exploration with a noisy
        posterior sample over the latent factors. The SGD update remains
        identical; we only modify action selection.
        """
        targets_obs = observation["targets"]
        num_targets = len(targets_obs) // 3

        active_targets = []
        for t_idx in range(num_targets):
            is_active = targets_obs[t_idx * 3 + 2] > 0.5
            if is_active:
                active_targets.append(t_idx)

        if not active_targets:
            return 0

        sigma = self._current_sigma()
        # Noisy posterior samples of the relevant factors
        p_tilde = self.P[self.agent_idx] + self.rng.normal(0.0, sigma, self.latent_dim)
        u_tilde_active = self.U[:, active_targets] + self.rng.normal(
            0.0, sigma, (self.latent_dim, len(active_targets))
        )
        scores = p_tilde @ u_tilde_active
        best_idx = int(np.argmax(scores))
        action = active_targets[best_idx] + 1

        # Preserve epsilon-decay accounting from parent (no functional effect
        # on TS, but maintains parity with MF-CF instrumentation).
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        return action

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        state = super().get_learning_state() or {}
        state["policy"] = "ts_mf"
        state["ts_sigma"] = self._current_sigma()
        return state
