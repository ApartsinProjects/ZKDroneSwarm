"""
Empty panel component for the TabulaDrone viewer.

This module provides a placeholder panel that displays "Empty" text.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
from viewer.components.base import BaseComponent


class EmptyPanel(BaseComponent):
    """
    Placeholder panel that displays "Empty" text.

    This component serves as a template for future content panels.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes, text: str = "Empty"):
        """
        Initialize the empty panel.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
            text: The text to display in the panel.
        """
        super().__init__(fig, ax)
        self.text = text

    def render_display(self) -> None:
        """
        Render the "Empty" placeholder text centered in the axes.
        """
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.text(
            0.0, 1, self.text,
            ha='left', va='top',
            fontsize=16, color='gray',
            transform=self.ax.transAxes
        )
        self.ax.axis('off')

    def clear(self) -> None:
        """
        Clear the panel.
        """
        self.ax.clear()
