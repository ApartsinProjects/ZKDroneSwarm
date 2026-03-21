"""
Optimal Assignment Oracle for ZK-MRTA Environment.

Uses SciPy's linear sum assignment to compute globally optimal drone-to-target
assignments that maximize total dot-product score while enforcing collision-free
constraints (at most one drone per target).

Privileged baseline (not ZK-compliant): uses true target attribute values.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import IPolicy, EnvInfos, extract_shared_info


class OptimalAssignmentOracle(IPolicy):
    """
    Oracle policy that computes globally optimal drone-to-target assignments.
    
    Uses linear sum assignment to maximize total score where:
    - Score matrix S[i,j] = dot(agent_vector[i], target_vector[j])
    - Agent vectors are derived from weapon damage profiles
    - Target vectors are derived from remaining attribute values
    
    Enforces collision-free assignments: at most one drone assigned per target.
    
    Privileged:
    - Uses true remaining attribute values (not ZK-compliant)
    - Requires targets_state parameter at action-selection time
    
    Action Space:
    - 0: NoOp (do nothing) - only when allow_noop=True and no assignment
    - 1 to N: Fire at target index (1-indexed)
    - -1: Unassigned (when more drones than active targets)
    """
    
    is_deterministic: bool = True
    
    def __init__(
        self,
        agent_weapon_profiles: Dict[str, Dict[str, float]],
        seed: Optional[int] = None,
        allow_noop: bool = True,
    ):
        """
        Initialize optimal assignment oracle.
        
        Args:
            agent_weapon_profiles: Dict mapping agent_id to weapon damage profile.
                                   e.g., {"drone_0": {"armor": 10.0, "shields": 5.0}, ...}
            seed: Random seed for reproducibility (unused, kept for interface consistency)
            allow_noop: If True, unassigned drones get action 0. If False, they get -1.
        """
        self.agent_weapon_profiles = {
            agent_id: {k: float(v) for k, v in profile.items()}
            for agent_id, profile in agent_weapon_profiles.items()
        }
        self.rng = np.random.RandomState(seed)
        self.allow_noop = allow_noop
        
        self.agent_ids = sorted(self.agent_weapon_profiles.keys())
        self.num_agents = len(self.agent_ids)
        
        self._attribute_names: Optional[List[str]] = None
    
    def _get_attribute_names(self, targets_state: List[Dict[str, float]]) -> List[str]:
        """Extract and cache attribute names from targets_state."""
        if self._attribute_names is None and targets_state:
            self._attribute_names = sorted(targets_state[0].keys())
        return self._attribute_names or []
    
    def _build_agent_vectors(self, attribute_names: List[str]) -> np.ndarray:
        """
        Build agent vectors from weapon damage profiles.
        
        Args:
            attribute_names: Ordered list of attribute names
        
        Returns:
            Agent vectors array of shape (num_agents, num_attributes)
        """
        vectors = []
        for agent_id in self.agent_ids:
            profile = self.agent_weapon_profiles[agent_id]
            vec = [profile.get(attr, 0.0) for attr in attribute_names]
            vectors.append(vec)
        return np.array(vectors, dtype=np.float64)
    
    def _build_target_vectors(
        self,
        targets_state: List[Dict[str, float]],
        attribute_names: List[str],
    ) -> np.ndarray:
        """
        Build target vectors from remaining attribute values.
        
        Args:
            targets_state: List of target attribute dicts
            attribute_names: Ordered list of attribute names
        
        Returns:
            Target vectors array of shape (num_targets, num_attributes)
        """
        vectors = []
        for attrs in targets_state:
            vec = [attrs.get(attr, 0.0) for attr in attribute_names]
            vectors.append(vec)
        return np.array(vectors, dtype=np.float64)
    
    def _parse_active_mask(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        num_targets: int,
    ) -> np.ndarray:
        """
        Parse observation to extract active mask for targets.
        
        Args:
            observation: ZK observation Dict with 'targets' key containing
                        array of shape (3 * num_targets,)
            num_targets: Total number of targets
        
        Returns:
            Boolean array of shape (num_targets,), True = active
        """
        # Extract targets array from dict observation
        if isinstance(observation, dict):
            target_array = observation["targets"]
        else:
            target_array = observation

        active = np.zeros(num_targets, dtype=bool)
        for target_idx in range(num_targets):
            obs_idx = target_idx * 3 + 2
            active[target_idx] = target_array[obs_idx] > 0.5
        return active
    
    def _solve_assignment(
        self,
        agent_vectors: np.ndarray,
        target_vectors: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple:
        """
        Solve optimal assignment using linear sum assignment.
        
        Args:
            agent_vectors: Shape (N_agents, d)
            target_vectors: Shape (N_targets, d)
            active_mask: Shape (N_targets,) boolean
        
        Returns:
            Tuple of (actions, assigned_mask, total_score) where:
            - actions: np.ndarray of shape (N_agents,) with target indices (0-indexed)
                      or -1 for unassigned
            - assigned_mask: np.ndarray of shape (N_agents,) boolean
            - total_score: float, sum of selected score matrix entries
        """
        n_agents = agent_vectors.shape[0]
        n_targets = target_vectors.shape[0]
        n_active = np.sum(active_mask)
        
        actions = np.full(n_agents, -1, dtype=np.int64)
        assigned_mask = np.zeros(n_agents, dtype=bool)
        
        if n_active == 0:
            return actions, assigned_mask, 0.0
        
        active_indices = np.where(active_mask)[0]
        
        S_full = agent_vectors @ target_vectors.T
        S_active = S_full[:, active_indices]
        
        C_active = -S_active
        
        row_ind, col_ind = linear_sum_assignment(C_active)
        
        total_score = 0.0
        for r, c in zip(row_ind, col_ind):
            original_target_idx = active_indices[c]
            actions[r] = original_target_idx
            assigned_mask[r] = True
            total_score += S_active[r, c]
        
        return actions, assigned_mask, total_score
    
    def select_actions(
        self,
        obs: Dict[str, Any],
        infos: EnvInfos,
    ) -> Dict[str, int]:
        """
        Select globally optimal actions for all agents.
        
        Computes one-to-one assignment maximizing total dot-product score.
        
        Args:
            obs: Dict of {agent_id: observation_array}
            infos: Environment infos dict keyed by agent_id
        
        Returns:
            actions: Dict of {agent_id: action}
                    0 = NoOp (if allow_noop and unassigned)
                    1 to num_targets = Fire at target (1-indexed)
                    Note: Unassigned drones get 0 if allow_noop, else -1
        """
        shared_info = extract_shared_info(infos)
        num_targets = len(shared_info.get("target_active", []))
        targets_state = shared_info.get("target_attributes", [])
        
        first_obs = next(iter(obs.values()))
        active_mask = self._parse_active_mask(first_obs, num_targets)
        
        attribute_names = self._get_attribute_names(targets_state)
        
        if not attribute_names:
            return {agent_id: 0 if self.allow_noop else -1 for agent_id in self.agent_ids}
        
        agent_vectors = self._build_agent_vectors(attribute_names)
        target_vectors = self._build_target_vectors(targets_state, attribute_names)
        active_indices = np.where(active_mask)[0]
        has_active_targets = active_indices.size > 0
        score_matrix = agent_vectors @ target_vectors.T
        
        actions_arr, assigned_mask, total_score = self._solve_assignment(
            agent_vectors, target_vectors, active_mask
        )
        
        result = {}
        for i, agent_id in enumerate(self.agent_ids):
            if assigned_mask[i]:
                result[agent_id] = int(actions_arr[i]) + 1
            else:
                if self.allow_noop or not has_active_targets:
                    result[agent_id] = 0
                else:
                    agent_scores = score_matrix[i, active_indices]
                    best_target_idx = int(active_indices[int(np.argmax(agent_scores))])
                    result[agent_id] = best_target_idx + 1
        
        return result

    def update(self, obs: Dict[str, Any]) -> None:
        """No-op: OptimalAssignmentOracle does not learn."""
        pass

    def soft_reset(self) -> None:
        """No-op: OptimalAssignmentOracle has no episode-level state."""
        pass

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return None
