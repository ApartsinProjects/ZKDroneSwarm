"""
Oracle-L Policy for ZK-MRTA Environment.

Privileged baseline that accesses true drone and target latent vectors directly
from the environment. Selects the target with highest cosine similarity to each
drone's latent vector. No HP knowledge is used (pure latent matching).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import IPolicy, EnvInfos


class OracleLPolicy(IPolicy):
    """
    Oracle policy using true latent vectors (cosine-similarity greedy).

    Privileged:
      - Reads env.drones[i].latent_vector (drone weapon profile).
      - Reads env.targets[j].latent_vector and env.targets[j].is_active.
      - Does NOT use HP (no planning, just greedy cosine matching).

    is_deterministic = True: given same env state -> same actions.
    Runs 1 episode in the standard harness (deterministic flag).
    """

    is_deterministic: bool = True

    def __init__(self, allow_noop: bool = False):
        """
        Args:
            allow_noop: If True, return action 0 when no targets are active.
                        If False, same behaviour (noop is forced anyway when
                        no active targets remain).
        """
        self.allow_noop = allow_noop
        self._agent_ids: Optional[List[str]] = None

    def select_actions(
        self,
        obs: Dict[str, Any],
        infos: EnvInfos,
        env: Any = None,
    ) -> Dict[str, int]:
        if env is None:
            raise RuntimeError(
                "OracleLPolicy requires the environment to be passed as "
                "select_actions(..., env=env)"
            )

        if self._agent_ids is None:
            self._agent_ids = sorted(obs.keys())

        # Extract drone latent vectors: shape (n_drones, latent_dim)
        drone_vectors = np.array(
            [d.latent_vector for d in env.drones], dtype=np.float64
        )

        # Extract target latent vectors: shape (n_targets, latent_dim)
        target_vectors = np.array(
            [t.latent_vector for t in env.targets], dtype=np.float64
        )
        active_mask = np.array([t.is_active for t in env.targets], dtype=bool)

        n_drones, n_targets = drone_vectors.shape[0], target_vectors.shape[0]

        # Normalise for cosine similarity
        d_norms = np.linalg.norm(drone_vectors, axis=1, keepdims=True)
        t_norms = np.linalg.norm(target_vectors, axis=1, keepdims=True)
        d_norms = np.where(d_norms == 0.0, 1.0, d_norms)
        t_norms = np.where(t_norms == 0.0, 1.0, t_norms)

        drone_unit = drone_vectors / d_norms   # (n_drones, latent_dim)
        target_unit = target_vectors / t_norms  # (n_targets, latent_dim)

        sim_matrix = drone_unit @ target_unit.T  # (n_drones, n_targets)

        # Mask inactive targets
        sim_matrix[:, ~active_mask] = -np.inf

        active_indices = np.where(active_mask)[0]

        actions: Dict[str, int] = {}
        for drone_idx, agent_id in enumerate(self._agent_ids):
            if len(active_indices) == 0:
                actions[agent_id] = 0
                continue
            # Greedy: pick target with highest cosine similarity
            best_target = int(np.argmax(sim_matrix[drone_idx]))
            actions[agent_id] = best_target + 1  # Convert to 1-indexed

        return actions

    def update(self, obs: Dict[str, Any]) -> None:
        """No-op: OracleLPolicy does not learn."""
        pass

    def soft_reset(self) -> None:
        """No-op: OracleLPolicy has no episode-level state to reset."""
        pass

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return None
