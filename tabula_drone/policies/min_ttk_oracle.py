"""
Oracle Policy (Time-to-Kill) for ZK-MRTA Environment (Privileged Baseline).

Selects the active target with minimal estimated hits-to-kill given a fixed
weapon damage profile. Not ZK-compliant by design (uses privileged state).
"""

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import IPolicy


class OracleTimeToKillPolicy(IPolicy):
    """
    Oracle baseline (upper bound): choose the target that can be eliminated
    in the fewest estimated hits, given the agent's weapon profile.
    
    Assumes per-hit attribute damage is constant and applied independently
    to each attribute with clamping at 0.
    
    Privileged:
    - Uses true remaining attribute values (not ZK-compliant)
    - Requires targets_state parameter at action-selection time
    
    Action Space:
    - 0: NoOp (do nothing) - only when allow_noop=True
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
        Initialize oracle policy with per-agent weapon damage profiles.
        
        Args:
            agent_weapon_profiles: Dict mapping agent_id to weapon damage profile.
                                   e.g., {"drone_0": {"armor": 10.0, "shields": 5.0}, ...}
            seed: Random seed for reproducibility (tie-breaking only)
            allow_noop: If True, action 0 (NoOp) is included in valid actions.
                       If False, agents must always fire at an active target.
        """
        self.agent_weapon_profiles = {
            agent_id: {k: float(v) for k, v in profile.items()}
            for agent_id, profile in agent_weapon_profiles.items()
        }
        self.rng = np.random.RandomState(seed)
        self.allow_noop = allow_noop
    
    def _parse_active_targets_from_obs(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        num_targets: int,
    ) -> List[bool]:
        """
        Parse observation to extract active status for each target.
        
        Args:
            observation: ZK observation Dict with 'targets' key containing
                        array of shape (3 * num_targets,)
                        Format: [target_0_x, target_0_y, target_0_active, ...]
            num_targets: Total number of targets in environment
        
        Returns:
            List of booleans indicating active status per target
        """
        # Extract targets array from dict observation
        if isinstance(observation, dict):
            target_array = observation["targets"]
        else:
            target_array = observation
            
        active = []
        for target_idx in range(num_targets):
            obs_idx = target_idx * 3 + 2  # Index of active field
            active.append(target_array[obs_idx] > 0.5)
        return active
    
    def _estimated_hits_to_kill(
        self,
        target_attributes: Dict[str, float],
        weapon_damage_profile: Dict[str, float],
    ) -> float:
        """
        Compute exact hits-to-kill under constant per-hit damage.
        
        Formula: hits = max_a ceil(rem_a / dmg_a)
        Returns infinity if dmg_a == 0 and rem_a > 0 (unkillable).
        
        Args:
            target_attributes: Dict mapping attribute names to remaining values.
                              e.g., {"armor": 25.0, "shields": 10.0}
            weapon_damage_profile: Dict mapping attribute names to damage per hit.
        
        Returns:
            Estimated number of hits to neutralize target, or inf if unkillable.
        """
        hits = 0.0
        
        for attr_name, remaining in target_attributes.items():
            rem = float(remaining)
            if rem <= 0.0:
                continue
            
            dmg = float(weapon_damage_profile.get(attr_name, 0.0))
            if dmg <= 0.0:
                return float("inf")  # Can't finish this attribute -> can't kill target
            
            hits = max(hits, math.ceil(rem / dmg))
        
        return hits
    
    def select_action(
        self,
        agent_id: str,
        observation: np.ndarray,
        num_targets: int,
        targets_state: List[Dict[str, float]],
    ) -> int:
        """
        Select action for a single agent using privileged target state.
        
        Selects the active target with minimum estimated hits-to-kill.
        Uses random tie-breaking among equal candidates.
        
        Args:
            agent_id: ID of the agent (e.g., "drone_0")
            observation: ZK observation array with shape (3 * num_targets,)
                        Format: [target_0_x, target_0_y, target_0_active, ...]
            num_targets: Total number of targets in environment
            targets_state: Privileged list of target attribute dicts.
                          Each dict maps attribute names to remaining values.
                          e.g., [{"armor": 25.0, "shields": 10.0}, ...]
        
        Returns:
            action: Integer in [0, num_targets] if allow_noop=True,
                   or [1, num_targets] if allow_noop=False
                   0 = NoOp (only when allow_noop=True)
                   1 to num_targets = Fire at target (1-indexed)
        """
        active_from_obs = self._parse_active_targets_from_obs(observation, num_targets)
        active_targets = [i for i in range(num_targets) if active_from_obs[i]]
        
        # No active targets: safe fallback to NoOp
        if not active_targets:
            return 0
        
        # Get this agent's weapon profile
        weapon_profile = self.agent_weapon_profiles[agent_id]
        
        best_hits = float("inf")
        best_targets: List[int] = []
        
        for t in active_targets:
            attrs = targets_state[t]  # Privileged true remaining attributes
            hits = self._estimated_hits_to_kill(attrs, weapon_profile)
            
            if hits < best_hits:
                best_hits = hits
                best_targets = [t]
            elif hits == best_hits:
                best_targets.append(t)
        
        # If all are "unkillable" with this weapon profile:
        if best_hits == float("inf"):
            if self.allow_noop:
                return 0
            # Must act -> pick first active target (deterministic)
            return int(min(active_targets)) + 1
        
        # Deterministic tie-break: pick lowest index target
        chosen_target_idx = int(min(best_targets))
        return chosen_target_idx + 1  # 1-indexed action
    
    def select_actions(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Select actions for all agents using privileged target state.
        
        Each agent independently selects the target with minimum hits-to-kill
        based on their own weapon profile.
        Actions are independent - no coordination between agents.
        
        Args:
            obs: Dict of {agent_id: observation_array}
            info: Environment info dict containing 'target_attributes'
        
        Returns:
            actions: Dict of {agent_id: action}
        """
        num_targets = len(info.get("target_active", []))
        targets_state = info.get("target_attributes", [])
        return {
            agent_id: self.select_action(agent_id, observation, num_targets, targets_state)
            for agent_id, observation in obs.items()
        }

    def update(self, obs: Dict[str, Any]) -> None:
        """No-op: OracleTimeToKillPolicy does not learn."""
        pass

    def soft_reset(self) -> None:
        """No-op: OracleTimeToKillPolicy has no episode-level state."""
        pass

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        return None
