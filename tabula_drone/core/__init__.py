"""
Core domain models and data structures for TabulaDrone environments.

This module contains shared state representations used across multiple
environment implementations.
"""

from .states import (
    DroneState,
    TargetState,
    WorldState,
    AttributeProfile,
    DEFAULT_CLASS_HP_MAPPING,
    DEFAULT_CLASS_ATTRIBUTE_MAPPING,
)

__all__ = [
    "DroneState",
    "TargetState",
    "WorldState",
    "AttributeProfile",
    "DEFAULT_CLASS_HP_MAPPING",
    "DEFAULT_CLASS_ATTRIBUTE_MAPPING",
]
