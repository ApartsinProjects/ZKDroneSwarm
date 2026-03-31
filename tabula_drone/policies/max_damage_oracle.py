"""
Optimal Assignment Oracle for ZK-MRTA Environment.

HP-aware marginal-allocation planner:
  - Allows multiple drones per target (focus-fire)
  - Penalizes overkill
  - Prioritizes bottleneck targets that need many drones
  - Approximates multi-step value via bottleneck weighting

Privileged baseline (not ZK-compliant): uses true target latent vectors and HP.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import IPolicy, EnvInfos, DiagnosticsProvider, extract_shared_info


class OptimalAssignmentOracle(IPolicy):
    """
    Oracle policy using HP-aware marginal-allocation focus-fire planning.
    
    Single greedy pass that assigns one drone at a time to the globally
    best (drone, target) pair based on marginal value.  Allows multiple
    drones per target when focus-fire is beneficial.  Scoring includes:
    
    - Effective damage (capped at remaining HP)
    - Bottleneck weighting (harder targets get priority)
    - Finish bonus for killing a target this step
    - Overkill penalty to avoid wasting damage
    - Future-value term for starting work on hard targets early
    
    Privileged:
    - Uses true target latent vectors and HP (not ZK-compliant)
    - Requires env with target state at action-selection time
    
    Action Space:
    - 0: NoOp (do nothing) - only when allow_noop=True and no assignment
    - 1 to N: Fire at target index (1-indexed)
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
        self._diagnostics_provider: Optional[DiagnosticsProvider] = None

    def set_diagnostics_provider(self, provider: DiagnosticsProvider) -> None:
        """Bind an env-owned diagnostics provider for privileged state lookup."""
        self._diagnostics_provider = provider

    def _get_shared_info(self, infos: EnvInfos) -> Dict[str, Any]:
        if self._diagnostics_provider is not None:
            return self._diagnostics_provider()
        return extract_shared_info(infos)
    
    def _extract_target_state_from_env(self, env: Any) -> List[Dict[str, float]]:
        """Extract target state directly from environment.
        
        Uses raw latent vectors (no HP scaling) because the environment's
        reward function is cosine similarity, which is scale-invariant.
        HP is only used for the active mask (dead targets are skipped).
        """
        targets_state = []
        if not hasattr(env, 'targets'):
            return targets_state
        
        for target in env.targets:
            if hasattr(target, 'attributes') and hasattr(target.attributes, 'attributes'):
                # ZK world: explicit attributes
                targets_state.append(dict(target.attributes.attributes))
            elif hasattr(target, 'latent_vector'):
                # Latent world: raw latent vectors (no HP scaling)
                targets_state.append({
                    f"d{i}": v
                    for i, v in enumerate(target.latent_vector)
                })
            else:
                targets_state.append({})
        
        return targets_state
    
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
        target_hps: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple:
        """
        HP-aware focus-fire oracle.

        Single marginal-allocation planner that:
        - allows multiple drones per target
        - penalizes overkill
        - prioritizes bottleneck targets
        - approximates multi-step value

        Returns:
            (actions, assigned_mask, total_score)
        """
        n_agents = agent_vectors.shape[0]
        n_targets = target_vectors.shape[0]

        actions = np.full(n_agents, -1, dtype=np.int64)
        assigned_mask = np.zeros(n_agents, dtype=bool)

        active_targets = np.where(active_mask)[0]
        if len(active_targets) == 0:
            return actions, assigned_mask, 0.0

        # Damage model aligned with env step():
        # damage = max(0, dot(drone_vec, target_vec))
        damage_matrix = np.maximum(0.0, agent_vectors @ target_vectors.T)

        remaining_hp = np.array(target_hps, dtype=np.float64).copy()

        # Estimate bottleneck level for each target:
        # minimum number of best available drones needed to kill it.
        # Harder targets get extra priority so they are not postponed forever.
        bottleneck_weight = np.ones(n_targets, dtype=np.float64)
        for t_idx in active_targets:
            hp = remaining_hp[t_idx]
            if hp <= 0:
                continue

            drone_damages = np.sort(damage_matrix[:, t_idx])[::-1]
            cumulative = 0.0
            drones_needed = 0
            for dmg in drone_damages:
                if dmg <= 0:
                    break
                cumulative += dmg
                drones_needed += 1
                if cumulative >= hp:
                    break

            if cumulative >= hp and drones_needed > 0:
                bottleneck_weight[t_idx] = 1.0 + 0.35 * (drones_needed - 1)
            else:
                bottleneck_weight[t_idx] = 1.0 + 0.35 * max(
                    1, len(active_targets) // 2
                )

        FINISH_BONUS = 6.0
        OVERKILL_PENALTY = 1.25
        WASTE_PENALTY = 0.15
        FUTURE_VALUE_WEIGHT = 0.75

        # Greedily assign one drone at a time using marginal value
        for _ in range(n_agents):
            best_score = -np.inf
            best_pair = None

            unassigned = np.where(~assigned_mask)[0]
            if len(unassigned) == 0:
                break

            for drone_idx in unassigned:
                for t_idx in active_targets:
                    hp = remaining_hp[t_idx]
                    if hp <= 0:
                        continue

                    dmg = damage_matrix[drone_idx, t_idx]
                    if dmg <= 0:
                        continue

                    effective = min(dmg, hp)
                    overkill = max(0.0, dmg - hp)
                    will_finish = dmg >= hp

                    future_value = (
                        FUTURE_VALUE_WEIGHT * bottleneck_weight[t_idx] * effective
                    )

                    waste_term = WASTE_PENALTY * dmg

                    score = (
                        effective
                        + future_value
                        + (
                            FINISH_BONUS * bottleneck_weight[t_idx]
                            if will_finish
                            else 0.0
                        )
                        - OVERKILL_PENALTY * overkill
                        - waste_term
                    )

                    # Tiny tie-breaker toward higher raw damage
                    score += 1e-6 * dmg

                    if score > best_score:
                        best_score = score
                        best_pair = (drone_idx, t_idx)

            if best_pair is None:
                break

            drone_idx, t_idx = best_pair
            actions[drone_idx] = t_idx
            assigned_mask[drone_idx] = True
            remaining_hp[t_idx] -= damage_matrix[drone_idx, t_idx]

        total_score = float(
            sum(
                damage_matrix[i, actions[i]]
                for i in range(n_agents)
                if actions[i] >= 0
            )
        )
        return actions, assigned_mask, total_score
    
    def select_actions(
        self,
        obs: Dict[str, Any],
        infos: EnvInfos,
        env: Any = None,
    ) -> Dict[str, int]:
        """
        Select actions using privileged HP-aware focus-fire planning.
        """
        if env is not None:
            targets_state = self._extract_target_state_from_env(env)
            num_targets = len(env.targets) if hasattr(env, "targets") else 0
        else:
            shared_info = self._get_shared_info(infos)
            num_targets = len(shared_info.get("target_active", []))
            targets_state = shared_info.get("target_attributes", [])

        first_obs = next(iter(obs.values()))
        active_mask = self._parse_active_mask(first_obs, num_targets)

        attribute_names = self._get_attribute_names(targets_state)
        if not attribute_names:
            return {
                agent_id: 0 if self.allow_noop else -1
                for agent_id in self.agent_ids
            }

        agent_vectors = self._build_agent_vectors(attribute_names)
        target_vectors = self._build_target_vectors(targets_state, attribute_names)

        if env is not None and hasattr(env, "targets"):
            target_hps = np.array([t.hp for t in env.targets], dtype=np.float64)
        else:
            shared_info = self._get_shared_info(infos)
            target_hps = np.array(
                shared_info.get("target_hps", [0.0] * num_targets),
                dtype=np.float64,
            )

        has_active_targets = np.any(active_mask)

        actions_arr, assigned_mask, _ = self._solve_assignment(
            agent_vectors, target_vectors, target_hps, active_mask
        )

        result = {}
        for i, agent_id in enumerate(self.agent_ids):
            if assigned_mask[i] and actions_arr[i] >= 0:
                result[agent_id] = int(actions_arr[i]) + 1
            elif self.allow_noop or not has_active_targets:
                result[agent_id] = 0
            else:
                # Force-fire: pick best-damage active target
                damage_matrix = np.maximum(0.0, agent_vectors @ target_vectors.T)
                active_indices = np.where(active_mask)[0]
                best_t = int(active_indices[np.argmax(damage_matrix[i, active_indices])])
                result[agent_id] = best_t + 1

        return result

    def update(self, obs: Dict[str, Any]) -> None:
        """No-op: OptimalAssignmentOracle does not learn."""
        pass

    def soft_reset(self) -> None:
        """No-op: OptimalAssignmentOracle has no episode-level state."""
        pass

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return None
