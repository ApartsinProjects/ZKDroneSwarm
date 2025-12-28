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
        self.total_episodes = None
        self.drone_types = Counter()
        self.target_classes = Counter()
        self.class_attribute_mapping = {}
        self.weapon_damage_profile_mapping = {}

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
        self.weapon_damage_profile_mapping = data.get("weapon_damage_profile_mapping", {})
        self.total_episodes = data.get("total_episodes", None)

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

        episodes_str = f"  |  Episodes: {self.total_episodes}" if self.total_episodes else ""
        self.ax.text(
            0.0, y_pos, f"Drones: {self.num_drones}  |  Targets: {self.num_targets}{episodes_str}",
            ha='left', va='top', fontsize=12, fontweight='bold',
            transform=self.ax.transAxes
        )
        y_pos -= line_height * 1.5

        y_pos = self._render_drone_types(y_pos, line_height)

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

    def _render_drone_types(self, y_start: float, line_height: float) -> float:
        """
        Render drone types with their damage profiles in a table format.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Drone Weapon Types",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        if not self.weapon_damage_profile_mapping:
            return y_pos

        # Get weapon types for column headers and attribute names for rows
        weapon_types = sorted(self.weapon_damage_profile_mapping.keys())
        first_weapon = next(iter(self.weapon_damage_profile_mapping.values()))
        attr_names = sorted(first_weapon.keys())

        # Render header row: Attribute | weapon1 | weapon2 | ...
        col_positions = [0.02 + i * 0.16 for i in range(len(weapon_types) + 1)]
        self.ax.text(
            col_positions[0], y_pos, "Attribute",
            ha='left', va='top', fontsize=8, fontweight='bold', color='#444444',
            transform=self.ax.transAxes
        )
        for i, weapon in enumerate(weapon_types):
            self.ax.text(
                col_positions[i + 1], y_pos, weapon,
                ha='left', va='top', fontsize=8, fontweight='bold', color='#444444',
                transform=self.ax.transAxes
            )
        y_pos -= line_height

        # Render data rows: one row per attribute
        for attr in attr_names:
            abbrev = self._abbreviate_attr(attr)
            self.ax.text(
                col_positions[0], y_pos, abbrev,
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            for i, weapon in enumerate(weapon_types):
                value = self.weapon_damage_profile_mapping[weapon].get(attr, 0.0)
                self.ax.text(
                    col_positions[i + 1], y_pos, f"{value:.0f}",
                    ha='left', va='top', fontsize=9, color='#555555',
                    transform=self.ax.transAxes
                )
            y_pos -= line_height

        # Render count row: how many drones of each weapon type
        self.ax.text(
            col_positions[0], y_pos, "count",
            ha='left', va='top', fontsize=9, fontweight='bold', color='#555555',
            transform=self.ax.transAxes
        )
        for i, weapon in enumerate(weapon_types):
            count = self.drone_types.get(weapon, 0)
            self.ax.text(
                col_positions[i + 1], y_pos, str(count),
                ha='left', va='top', fontsize=9, fontweight='bold', color='#555555',
                transform=self.ax.transAxes
            )
        y_pos -= line_height

        return y_pos

    def _abbreviate_attr(self, attr_name: str) -> str:
        """
        Abbreviate attribute names for table column headers.

        Args:
            attr_name: Full attribute name.

        Returns:
            Abbreviated name.
        """
        abbreviations = {
            "structural_integrity": "struct",
            "envelope_integrity": "envelope",
            "utilities_lifesafety": "util/life",
        }
        return abbreviations.get(attr_name, attr_name[:8])

    def _render_target_classes(self, y_start: float, line_height: float) -> float:
        """
        Render target classes with their attribute profiles in a table format.

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

        if not self.class_attribute_mapping:
            return y_pos

        # Get class types for column headers and attribute names for rows
        class_types = sorted(self.class_attribute_mapping.keys())
        first_class = next(iter(self.class_attribute_mapping.values()))
        attr_names = sorted(first_class.keys())

        # Render header row: Attribute | A | B | C | ...
        col_positions = [0.02 + i * 0.16 for i in range(len(class_types) + 1)]
        self.ax.text(
            col_positions[0], y_pos, "Attribute",
            ha='left', va='top', fontsize=8, fontweight='bold', color='#444444',
            transform=self.ax.transAxes
        )
        for i, class_type in enumerate(class_types):
            self.ax.text(
                col_positions[i + 1], y_pos, class_type,
                ha='left', va='top', fontsize=8, fontweight='bold', color='#444444',
                transform=self.ax.transAxes
            )
        y_pos -= line_height

        # Render data rows: one row per attribute
        for attr in attr_names:
            abbrev = self._abbreviate_attr(attr)
            self.ax.text(
                col_positions[0], y_pos, abbrev,
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            for i, class_type in enumerate(class_types):
                value = self.class_attribute_mapping[class_type].get(attr, 0.0)
                self.ax.text(
                    col_positions[i + 1], y_pos, f"{value:.0f}",
                    ha='left', va='top', fontsize=9, color='#555555',
                    transform=self.ax.transAxes
                )
            y_pos -= line_height

        # Render count row: how many targets of each class type
        self.ax.text(
            col_positions[0], y_pos, "count",
            ha='left', va='top', fontsize=9, fontweight='bold', color='#555555',
            transform=self.ax.transAxes
        )
        for i, class_type in enumerate(class_types):
            count = self.target_classes.get(class_type, 0)
            self.ax.text(
                col_positions[i + 1], y_pos, str(count),
                ha='left', va='top', fontsize=9, fontweight='bold', color='#555555',
                transform=self.ax.transAxes
            )
        y_pos -= line_height

        return y_pos

    def clear(self) -> None:
        """
        Clear the panel.
        """
        self.ax.clear()
