"""
Component-based architecture for the TabulaDrone viewer.

This package contains the components used in the viewer, organized by their roles:
- base: Foundation components that other components inherit from
- containers: Components that contain and manage other components
- panels: Visual components that display specific types of information
"""

from viewer.components.base import BaseComponent
from viewer.components.containers import TabContainer, MapPanel
from viewer.components.panels import EmptyPanel, InfoPanel, ResultsPanel, SummaryPanel

__all__ = ['BaseComponent', 'TabContainer', 'MapPanel', 'EmptyPanel', 'InfoPanel', 'ResultsPanel', 'SummaryPanel']
