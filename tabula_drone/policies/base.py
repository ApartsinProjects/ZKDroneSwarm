"""Base protocol and helpers for all policy implementations."""

from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

EnvInfos = Dict[str, Dict[str, Any]]
DiagnosticsProvider = Callable[[], Dict[str, Any]]


def extract_shared_info(infos: EnvInfos) -> Dict[str, Any]:
    """
    Extract a representative shared payload from legacy agent-keyed infos.

    The current env exposes shared telemetry through `env.diagnostics`, not
    PettingZoo `infos`. This helper remains only as a compatibility fallback for
    callers that still pass older infos payloads.
    """
    if not infos:
        return {}

    first_agent_id = next(iter(infos))
    return dict(infos[first_agent_id])


def bind_diagnostics_provider(policy: Any, provider: DiagnosticsProvider) -> None:
    """Attach a diagnostics provider to a policy tree when supported."""
    setter = getattr(policy, "set_diagnostics_provider", None)
    if callable(setter):
        setter(provider)

    child_policies = getattr(policy, "policies", None)
    if isinstance(child_policies, dict):
        for child_policy in child_policies.values():
            bind_diagnostics_provider(child_policy, provider)


@runtime_checkable
class IPolicy(Protocol):
    """
    Protocol defining the interface all policies must implement.
    
    This enables uniform interaction with any policy type from orchestration
    code, eliminating type-checking branches.
    
    Class Attributes:
        is_deterministic: True if policy produces deterministic actions (no randomness)
    """
    
    is_deterministic: bool

    def select_actions(
        self, obs: Dict[str, Any], infos: EnvInfos
    ) -> Dict[str, int]:
        """
        Select actions for all agents.
        
        Args:
            obs: Observations dict keyed by agent_id
            infos: Environment infos dict keyed by agent_id
        
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
        return None
