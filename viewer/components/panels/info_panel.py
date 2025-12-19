"""
Info panel component for the TabulaDrone viewer.

This module provides a panel that displays episode metadata including
drone counts, target counts, and breakdown tables.
"""

from typing import Dict, Any
from collections import Counter
import matplotlib.pyplot as plt
from viewer.components.base import BaseComponent


class InfoPanel(BaseComponent):
    """
    Panel that displays episode metadata.

    Shows drone count, target count, drone types table, and target classes table.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the info panel.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        super().__init__(fig, ax)
        self.num_drones = 0
        self.num_targets = 0
        self.drone_types = Counter()
        self.target_classes = Counter()
        self.class_attribute_mapping = {}

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process state data to extract counts and breakdowns.

        Args:
            data: The state dict containing drones and targets.
        """
        drones = data.get("drones", [])
        targets = data.get("targets", [])

        self.num_drones = len(drones)
        self.num_targets = len(targets)

        self.drone_types = Counter(d.get("weapon_type", "unknown") for d in drones)
        self.target_classes = Counter(t.get("class_type", "unknown") for t in targets)
        self.class_attribute_mapping = data.get("class_attribute_mapping", {})

    def render_display(self) -> None:
        """
        Render the info panel with counts and tables.
        """
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        y_pos = 0.95
        line_height = 0.06

        self.ax.text(
            0.0, y_pos, f"Drones: {self.num_drones}  |  Targets: {self.num_targets}",
            ha='left', va='top', fontsize=12, fontweight='bold',
            transform=self.ax.transAxes
        )
        y_pos -= line_height * 1.5

        y_pos = self._render_table(
            "Drone Types", self.drone_types, y_pos, line_height
        )

        y_pos -= line_height * 0.5

        self._render_target_classes(y_pos, line_height)

    def _render_table(
        self, title: str, data: Counter, y_start: float, line_height: float
    ) -> float:
        """
        Render a simple table with title and key-value rows.

        Args:
            title: Table title.
            data: Counter with items to display.
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, title,
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        for key, count in sorted(data.items()):
            self.ax.text(
                0.02, y_pos, f"{key}:",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            self.ax.text(
                0.25, y_pos, str(count),
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            y_pos -= line_height

        return y_pos

    def _render_target_classes(self, y_start: float, line_height: float) -> float:
        """
        Render target classes with their attribute profiles.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Target Classes",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        for class_type in sorted(self.class_attribute_mapping.keys()):
            attrs = self.class_attribute_mapping[class_type]
            attr_str = ", ".join(
                f"{k}={v}" for k, v in sorted(attrs.items())
            )
            self.ax.text(
                0.02, y_pos, f"{class_type}: {attr_str}",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            y_pos -= line_height

        return y_pos

    def clear(self) -> None:
        """
        Clear the panel.
        """
        self.ax.clear()
