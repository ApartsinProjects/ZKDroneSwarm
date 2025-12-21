"""
Summary panel component for the TabulaDrone viewer.

This module provides a panel that displays episode summary metrics including
efficiency (steps, ammo), overkill stats, fleet composition, and target distribution.
"""

from typing import Dict, Any, List
from collections import Counter
import matplotlib.pyplot as plt
from viewer.components.base import BaseComponent


class SummaryPanel(BaseComponent):
    """
    Panel that displays episode summary metrics.

    Shows efficiency metrics, overkill analysis, fleet composition,
    and target distribution breakdown.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the summary panel.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        super().__init__(fig, ax)
        self.policy_type: str | None = None
        self.seed: int | None = None
        self.summary: Dict[str, Any] = {}
        self.steps: List[Dict[str, Any]] = []
        self.drones: List[Dict[str, Any]] = []
        self.targets: List[Dict[str, Any]] = []

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process state data to extract summary information.

        Args:
            data: The state dict containing summary, steps, drones, targets.
        """
        self.policy_type = data.get("policy_type")
        self.seed = data.get("seed")
        self.summary = data.get("summary", {})
        self.steps = data.get("steps", [])
        self.drones = data.get("drones", [])
        self.targets = data.get("targets", [])

    def render_display(self) -> None:
        """
        Render the summary panel with metrics sections.
        """
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        y_pos = 0.95
        line_height = 0.06

        y_pos = self._render_efficiency(y_pos, line_height)
        y_pos = self._render_overkill(y_pos, line_height)
        y_pos = self._render_fleet(y_pos, line_height)
        y_pos = self._render_targets(y_pos, line_height)

    def _render_efficiency(self, y_start: float, line_height: float) -> float:
        """
        Render the efficiency section.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Efficiency",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        policy_str = self.policy_type or "unknown"
        seed_str = str(self.seed) if self.seed is not None else "N/A"
        self.ax.text(
            0.02, y_pos, f"Policy: {policy_str}  |  Seed: {seed_str}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        total_steps = self.summary.get("total_steps", 0)
        self.ax.text(
            0.02, y_pos, f"Steps: {total_steps}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        metrics = self.summary.get("metrics", {})
        total_ammo = metrics.get("total_ammo_used", 0)
        targets_destroyed = metrics.get("targets_destroyed", 0)
        ammo_per_target = total_ammo / targets_destroyed if targets_destroyed > 0 else 0
        self.ax.text(
            0.02, y_pos, f"Ammo: {total_ammo} ({ammo_per_target:.2f} per target)",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        return y_pos

    def _render_overkill(self, y_start: float, line_height: float) -> float:
        """
        Render the overkill section.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Overkill",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        overkill = {}
        if self.steps:
            last_info = self.steps[-1].get("info", {})
            overkill = last_info.get("overkill", {})

        if not overkill:
            self.ax.text(
                0.02, y_pos, "None",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            y_pos -= line_height
        else:
            total_wasted = sum(overkill.values())
            num_targets = len(overkill)
            self.ax.text(
                0.02, y_pos, f"Wasted damage: {total_wasted:.1f}",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            y_pos -= line_height
            self.ax.text(
                0.02, y_pos, f"Affected targets: {num_targets}",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            y_pos -= line_height

        return y_pos

    def _render_fleet(self, y_start: float, line_height: float) -> float:
        """
        Render the fleet section.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Fleet",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        num_drones = len(self.drones)
        self.ax.text(
            0.02, y_pos, f"Drones: {num_drones}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        weapon_counts = Counter(d.get("weapon_type", "unknown") for d in self.drones)
        weapon_str = ", ".join(f"{count}× {wtype}" for wtype, count in sorted(weapon_counts.items()))
        self.ax.text(
            0.02, y_pos, f"Weapons: {weapon_str}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        return y_pos

    def _render_targets(self, y_start: float, line_height: float) -> float:
        """
        Render the targets section.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Targets",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        metrics = self.summary.get("metrics", {})
        targets_destroyed = metrics.get("targets_destroyed", 0)
        total_targets = len(self.targets)
        self.ax.text(
            0.02, y_pos, f"Destroyed: {targets_destroyed} / {total_targets}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        class_counts = Counter(t.get("class_type", "unknown") for t in self.targets)
        class_str = ", ".join(f"{count}× {ctype}" for ctype, count in sorted(class_counts.items()))
        self.ax.text(
            0.02, y_pos, f"Classes: {class_str}",
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
