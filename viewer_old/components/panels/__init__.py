"""
Panel components for the FalconX viewer.

This package contains visual components that display specific types of information.
"""

# Panel components will be imported and re-exported here as they are moved
# from the original components.py file

from viewer.components.panels.map_component import MapComponent
from viewer.components.panels.overview_component import OverviewComponent
from viewer.components.panels.drone_decisions_component import DroneDecisionsComponent

__all__ = ['MapComponent', 'OverviewComponent', 'DroneDecisionsComponent']
