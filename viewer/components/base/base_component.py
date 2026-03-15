"""
Base component for the TabulaDrone viewer.

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
        self.chart_ax: plt.Axes | None = None

    def is_visible(self) -> bool:
        """
        Check if the component is currently visible.

        Returns:
            True if the component's axes are visible, False otherwise.
        """
        return self.ax.get_visible()

    def set_visible(self, visible: bool) -> None:
        """
        Set the visibility of the component and its subsidiary axes.

        Args:
            visible: Whether the component should be visible.
        """
        self.ax.set_visible(visible)
        if hasattr(self, 'chart_ax') and self.chart_ax is not None:
            self.chart_ax.set_visible(visible)
        if hasattr(self, 'chart_ax_left') and self.chart_ax_left is not None:
            self.chart_ax_left.set_visible(visible)
        if hasattr(self, 'chart_ax_right') and self.chart_ax_right is not None:
            self.chart_ax_right.set_visible(visible)

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
        self.process_data(data)
        
        if self.is_visible():
            self.render_display()

    def clear(self) -> None:
        """
        Clear the component's rendering.
        """
        self.ax.clear()
