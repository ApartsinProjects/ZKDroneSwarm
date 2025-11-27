"""
Scenario generation utilities for TabulaDrone environments.

This module provides utilities for generating randomized configurations
for drones, targets, and other scenario elements.
"""

from .weapon_assignment import assign_weapons_to_drones

__all__ = ["assign_weapons_to_drones"]
