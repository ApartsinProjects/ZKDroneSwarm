"""
Gymnasium environments for drone engagement scenarios.
"""

from typing import Any

from .diagnostics import EnvDiagnosticsSnapshot
from .drone_engage_latent_mrta import DroneEngageLatentMRTA


def parallel_env(**kwargs: Any) -> Any:
    """Canonical factory for the PettingZoo parallel environment."""
    return DroneEngageLatentMRTA(**kwargs)


def make_env(**kwargs: Any) -> Any:
    """Repo-local alias for the canonical environment factory."""
    return parallel_env(**kwargs)


__all__ = [
    "DroneEngageLatentMRTA",
    "EnvDiagnosticsSnapshot",
    "parallel_env",
    "make_env",
]
