"""
Map component for the FalconX viewer.

This module provides a component for rendering the map view of the simulation.
"""

from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from viewer.components.base.base_component import BaseComponent


class MapComponent(BaseComponent):
    """
    Component for rendering the map view.

    This component handles the visualization of the simulation world map.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the map component.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        super().__init__(fig, ax)
        self.show_ranges = True
        self.ranges_origin = (0.0, 0.0)
        self.show_spawn_regions = False
        self.show_drone_detection = False
        self.snapshot_data = None

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process the snapshot data without rendering.

        Args:
            data: The snapshot data to process.
        """
        # Store the snapshot data for later rendering
        self.snapshot_data = data

    def render_display(self) -> None:
        """
        Render the component's visual elements based on its current state.
        """
        if self.snapshot_data:
            from viewer.draw import plot_map
            
            # Clear the axes before rendering
            self.ax.clear()
            
            # Call the plot_map function with the component's settings
            plot_map(
                self.ax,
                self.snapshot_data,
                show_ranges=self.show_ranges,
                ranges_origin=self.ranges_origin,
                show_spawn_regions=self.show_spawn_regions,
                show_drone_detection=self.show_drone_detection
            )

    def render(self, data: Dict[str, Any]) -> None:
        """
        Process data and render the component if visible.

        Args:
            data: The snapshot data to render.
        """
        # Always process data regardless of visibility
        self.process_data(data)
        
        # Only render display if component is visible
        if self.is_visible():
            self.render_display()
            
    def clear(self) -> None:
        """
        Clear the component's rendering and reset its state.
        """
        # Clear the axes
        self.ax.clear()
        
        # Reset the snapshot data
        self.snapshot_data = None
