"""
Gymnasium environments for drone engagement scenarios.
"""

from typing import Any

from .diagnostics import EnvDiagnosticsSnapshot
from .drone_engage_latent_mrta import DroneEngageLatentMRTA
from .drone_engage_zk_mrta_v0 import DroneEngageZKMRTA


def parallel_env(**kwargs: Any) -> Any:
    """Canonical factory for the PettingZoo parallel environment."""
    world_model = kwargs.pop("world_model", "custom")
    if world_model == "legacy":
        world_model = "custom"
    if world_model == "latent":
        return DroneEngageLatentMRTA(**kwargs)
    return DroneEngageZKMRTA(**kwargs)


def make_env(**kwargs: Any) -> Any:
    """Repo-local alias for the canonical environment factory."""
    return parallel_env(**kwargs)


__all__ = [
    "DroneEngageZKMRTA",
    "DroneEngageLatentMRTA",
    "EnvDiagnosticsSnapshot",
    "parallel_env",
    "make_env",
]
