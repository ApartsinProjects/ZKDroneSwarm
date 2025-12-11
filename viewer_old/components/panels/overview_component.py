"""
Overview component for the FalconX viewer.

This module provides a component for rendering the overview panel of the simulation.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
from viewer.components.base.base_component import BaseComponent


class OverviewComponent(BaseComponent):
    """
    Component for rendering the overview panel.

    This component displays summary information about the simulation.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the overview component.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        super().__init__(fig, ax)
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
            from viewer.draw import plot_overview
            
            # Clear the axes before rendering
            self.ax.clear()
            
            # Call the plot_overview function
            plot_overview(self.ax, self.snapshot_data)

    def clear(self) -> None:
        """
        Clear the component's rendering and reset its state.
        """
        # Clear the axes
        self.ax.clear()
        
        # Reset the snapshot data
        self.snapshot_data = None

    def should_preserve_state(self) -> bool:
        """
        Override to indicate that this component should preserve its state when hidden.
        
        Returns:
            True to preserve state across tab switches.
        """
        return True
