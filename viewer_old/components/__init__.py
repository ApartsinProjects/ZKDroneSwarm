"""
Component-based architecture for the FalconX viewer.

This package contains the components used in the viewer, organized by their roles:
- base: Foundation components that other components inherit from
- containers: Components that contain and manage other components
- panels: Visual components that display specific types of information
"""

# This file will re-export components from their new locations
# as they are moved from the original components.py file

from viewer.components.base import BaseComponent
from viewer.components.containers import TabContainer
from viewer.components.panels import MapComponent, OverviewComponent, DroneDecisionsComponent

__all__ = ['BaseComponent', 'TabContainer', 'MapComponent', 'OverviewComponent', 'DroneDecisionsComponent']
