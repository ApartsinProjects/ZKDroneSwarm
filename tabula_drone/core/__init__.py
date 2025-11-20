"""
Core domain models and data structures for TabulaDrone environments.

This module contains shared state representations used across multiple
environment implementations.
"""

from .states import (
    DroneState,
    TargetState,
    WorldState,
    DEFAULT_CLASS_HP_MAPPING,
)

__all__ = [
    "DroneState",
    "TargetState",
    "WorldState",
    "DEFAULT_CLASS_HP_MAPPING",
]
