"""
Decentralized Policy Wrapper for ZK-MRTA Environment.

Wraps a dict of per-agent policies into a single Policy-compliant object,
enabling uniform interaction from orchestration code.
"""

from typing import Any, Dict, Optional

from .base_cf_policy import BaseCFPolicy


class DecentralizedPolicyWrapper:
    """
    Wrapper that aggregates per-agent policies into a single Policy interface.
    
    Takes a dict of {agent_id: policy} where each policy has:
    - select_action(observation) -> int
    - update_from_observation(observation) -> None
    - soft_reset() -> None
    - get_learning_state() -> Optional[Dict]
    
    Provides the unified Policy interface:
    - select_actions(obs, info) -> Dict[str, int]
    - update(obs) -> None
    - soft_reset() -> None
    - get_learning_state() -> Optional[Dict]
    """
    
    def __init__(self, policies: Dict[str, BaseCFPolicy]):
        """
        Initialize wrapper with per-agent policies.
        
        Args:
            policies: Dict mapping agent_id to policy instance
        """
        self.policies = policies
        self.agent_ids = sorted(policies.keys())
        self.num_agents = len(policies)
        # Expose first policy's config for logging compatibility
        first_policy = next(iter(policies.values()))
        self.num_targets = first_policy.num_targets
        self.latent_dim = first_policy.latent_dim
        # Expose metadata attributes from wrapped policies
        self.is_deterministic = first_policy.is_deterministic
        self.is_ep_greedy_cf = first_policy.is_ep_greedy_cf
    
    def select_actions(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Select actions for all agents by delegating to individual policies.
        
        Args:
            obs: Dict of {agent_id: observation}
            info: Environment info dict (unused, passed for interface compliance)
        
        Returns:
            Dict of {agent_id: action}
        """
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = self.policies[agent_id].select_action(agent_obs)
        return actions
    
    def update(self, obs: Dict[str, Any]) -> None:
        """
        Update all agent policies from their observations.
        
        Args:
            obs: Dict of {agent_id: observation}
        """
        for agent_id, agent_obs in obs.items():
            self.policies[agent_id].update_from_observation(agent_obs)
    
    def soft_reset(self) -> None:
        """Reset all agent policies for new episode."""
        for policy in self.policies.values():
            policy.soft_reset()
    
    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        """
        Return aggregated learning state from all agent policies.
        
        Returns:
            Dict with 'agents' list containing each agent's learning state
        """
        return {
            "agents": [
                self.policies[agent_id].get_learning_state()
                for agent_id in self.agent_ids
            ]
        }
