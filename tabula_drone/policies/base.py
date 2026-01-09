"""Base protocol for all policy implementations."""

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    """
    Protocol defining the interface all policies must implement.
    
    This enables uniform interaction with any policy type from orchestration
    code, eliminating type-checking branches.
    
    Class Attributes:
        is_deterministic: True if policy produces deterministic actions (no randomness)
    """
    
    is_deterministic: bool

    def select_actions(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Select actions for all agents.
        
        Args:
            obs: Observations dict keyed by agent_id
            info: Environment info dict (policies may ignore if not needed)
        
        Returns:
            Actions dict keyed by agent_id
        """
        ...

    def update(self, obs: Dict[str, Any]) -> None:
        """
        Update policy state from observations (e.g., learning).
        
        Default: no-op for non-learning policies.
        
        Args:
            obs: Observations dict keyed by agent_id
        """
        ...

    def soft_reset(self) -> None:
        """
        Reset episode-level state while preserving learned parameters.
        
        Default: no-op for stateless policies.
        """
        ...

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        """
        Return learning state for logging/visualization.
        
        Returns:
            Dict with learning state (e.g., latent vectors), or None if not a learning policy.
        """
        ...
