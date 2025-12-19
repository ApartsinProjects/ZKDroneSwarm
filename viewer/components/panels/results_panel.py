"""
Results panel component for the TabulaDrone viewer.

This module provides a panel that displays episode summary results including
total steps, success status, targets destroyed, ammo used, and per-drone rewards.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from viewer.components.base import BaseComponent


class ResultsPanel(BaseComponent):
    """
    Panel that displays episode summary results.

    Shows total steps, success status, targets destroyed, ammo used,
    and per-drone reward breakdown.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the results panel.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        super().__init__(fig, ax)
        self.summary: Optional[Dict[str, Any]] = None

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process state data to extract summary information.

        Args:
            data: The state dict containing the summary section.
        """
        self.summary = data.get("summary", None)

    def render_display(self) -> None:
        """
        Render the results panel with summary metrics.
        """
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        if self.summary is None:
            self._render_no_results()
            return

        y_pos = 0.95
        line_height = 0.06

        y_pos = self._render_outcome(y_pos, line_height)
        y_pos -= line_height * 0.5
        y_pos = self._render_metrics(y_pos, line_height)
        y_pos -= line_height * 0.5
        self._render_rewards(y_pos, line_height)

    def _render_no_results(self) -> None:
        """
        Render placeholder when no summary data is available.
        """
        self.ax.text(
            0.0, 0.95, "No results available",
            ha='left', va='top', fontsize=12, color='gray',
            transform=self.ax.transAxes
        )

    def _render_outcome(self, y_start: float, line_height: float) -> float:
        """
        Render episode outcome (success/failure and termination reason).

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        success = self.summary.get("success", False)
        status_text = "SUCCESS" if success else "FAILURE"
        status_color = "#27ae60" if success else "#e74c3c"

        self.ax.text(
            0.0, y_pos, f"Outcome: {status_text}",
            ha='left', va='top', fontsize=12, fontweight='bold', color=status_color,
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        reason = self.summary.get("termination_reason", "unknown")
        self.ax.text(
            0.02, y_pos, f"Reason: {reason}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        return y_pos

    def _render_metrics(self, y_start: float, line_height: float) -> float:
        """
        Render episode metrics (steps, targets destroyed, ammo used).

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Metrics",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        total_steps = self.summary.get("total_steps", 0)
        self.ax.text(
            0.02, y_pos, f"Total Steps: {total_steps}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        metrics = self.summary.get("metrics", {})
        targets_destroyed = metrics.get("targets_destroyed", 0)
        self.ax.text(
            0.02, y_pos, f"Targets Destroyed: {targets_destroyed}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        total_ammo = metrics.get("total_ammo_used", 0)
        self.ax.text(
            0.02, y_pos, f"Total Ammo Used: {total_ammo}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        return y_pos

    def _render_rewards(self, y_start: float, line_height: float) -> float:
        """
        Render per-drone reward breakdown.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start

        self.ax.text(
            0.0, y_pos, "Rewards",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        total_reward = self.summary.get("total_reward", {})
        total = 0.0
        for drone_id, reward in sorted(total_reward.items()):
            self.ax.text(
                0.02, y_pos, f"{drone_id}:",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            self.ax.text(
                0.25, y_pos, f"{reward:.1f}",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=self.ax.transAxes
            )
            total += reward
            y_pos -= line_height

        self.ax.text(
            0.02, y_pos, "Total:",
            ha='left', va='top', fontsize=9, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        self.ax.text(
            0.25, y_pos, f"{total:.1f}",
            ha='left', va='top', fontsize=9, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        return y_pos

    def clear(self) -> None:
        """
        Clear the panel.
        """
        self.ax.clear()
