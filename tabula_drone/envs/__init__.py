"""
Gymnasium environments for drone engagement scenarios.
"""

from typing import Any

from .diagnostics import EnvDiagnosticsSnapshot
from .drone_engage_zk_mrta_v0 import DroneEngageZKMRTA


def parallel_env(**kwargs: Any) -> DroneEngageZKMRTA:
    """Canonical factory for the PettingZoo parallel environment."""
    return DroneEngageZKMRTA(**kwargs)


def make_env(**kwargs: Any) -> DroneEngageZKMRTA:
    """Repo-local alias for the canonical environment factory."""
    return parallel_env(**kwargs)


__all__ = [
    "DroneEngageZKMRTA",
    "EnvDiagnosticsSnapshot",
    "parallel_env",
    "make_env",
]
