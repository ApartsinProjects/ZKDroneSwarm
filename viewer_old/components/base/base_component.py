"""
Base component for the FalconX viewer.

This module provides the foundation class for all visualization components.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt


class BaseComponent:
    """
    Base class for all visualization components.

    This class defines the common interface for all components in the viewer.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the component.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        self.fig = fig
        self.ax = ax

    def is_visible(self) -> bool:
        """
        Check if the component is currently visible.

        Returns:
            True if the component's axes are visible, False otherwise.
        """
        return self.ax.get_visible()

    def should_preserve_state(self) -> bool:
        """
        Check if the component should preserve its state when hidden.
        
        Components that need to maintain their state (like history) across tab switches
        should override this method to return True.
        
        Returns:
            True if the component should preserve its state, False otherwise.
        """
        return False

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process data without rendering. This method should be used to update
        internal state based on new data, regardless of visibility.

        Args:
            data: The data to process.
        """
        pass

    def render_display(self) -> None:
        """
        Render the component's visual elements based on its current state.
        This should only be called when the component is visible.
        """
        pass

    def render(self, data: Dict[str, Any]) -> None:
        """
        Process data and render the component if visible.

        Args:
            data: The data to render.
        """
        # Always process data regardless of visibility
        self.process_data(data)
        
        # Only render display if component is visible
        if self.is_visible():
            self.render_display()

    def clear(self) -> None:
        """
        Clear the component's rendering.
        """
        self.ax.clear()
