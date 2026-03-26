"""
Results panel component for the TabulaDrone viewer.

This module provides a panel that displays episode summary results including
total steps, success status, targets destroyed, ammo used, and per-drone rewards.
"""

from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.text import Text
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
        self.metrics: Dict[str, Any] = {}
        self.hp_history: List[float] = []
        self.active_targets_history: List[int] = []
        self.chart_ax: Optional[plt.Axes] = None
        self._reward_items: List[Tuple[str, float]] = []
        self._pick_cid: Optional[int] = None
        self._more_text: Optional[Text] = None

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process state data to extract summary information.

        Args:
            data: The state dict containing the summary section.
        """
        self.summary = data.get("summary", None)
        self.metrics = data.get("metrics", {})
        self.hp_history = data.get("hp_history", [])
        self.active_targets_history = data.get("active_targets_history", [])

    def render_display(self) -> None:
        """
        Render the results panel with summary metrics and HP chart.
        
        Layout: Upper ~45% for text metrics, lower ~55% for HP chart.
        """
        if self._pick_cid is not None:
            self.fig.canvas.mpl_disconnect(self._pick_cid)
            self._pick_cid = None
        
        if self.chart_ax is not None:
            self.chart_ax.remove()
            self.chart_ax = None
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
        self._render_metrics(y_pos, line_height)
        
        self._render_hp_chart()
        
        self._pick_cid = self.fig.canvas.mpl_connect('pick_event', self._on_rewards_click)

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
        Render episode metrics (steps, targets destroyed, ammo used) and rewards in two columns.

        Args:
            y_start: Starting y position.
            line_height: Height per line.

        Returns:
            The y position after rendering.
        """
        y_pos = y_start
        col1_x = 0.0
        col2_x = 0.5

        self.ax.text(
            col1_x, y_pos, "Metrics",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        self.ax.text(
            col2_x, y_pos, "Rewards",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        total_steps = self.summary.get("total_steps", 0)
        self.ax.text(
            col1_x + 0.02, y_pos, f"Total Steps: {total_steps}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )

        total_reward = self.summary.get("total_reward", {})
        self._reward_items = sorted(total_reward.items())
        self._render_rewards_column(col2_x, y_pos, line_height)
        y_pos -= line_height

        targets_neutralized = self.metrics.get("targets_neutralized", 0)
        self.ax.text(
            col1_x + 0.02, y_pos, f"Targets Destroyed: {targets_neutralized}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        total_ammo = self.metrics.get("total_ammo_used", 0)
        self.ax.text(
            col1_x + 0.02, y_pos, f"Total Ammo Used: {total_ammo}",
            ha='left', va='top', fontsize=9, color='#555555',
            transform=self.ax.transAxes
        )
        y_pos -= line_height

        return y_pos

    def _render_rewards_column(self, col_x: float, y_start: float, line_height: float) -> None:
        """
        Render the rewards column with drone rewards in compact horizontal grid.

        Shows up to MAX_VISIBLE_DRONES in grid format. If more exist, shows
        a clickable "+ X more" text that opens a popup with all rewards.

        Args:
            col_x: X position for the column.
            y_start: Starting y position.
            line_height: Height per line.
        """
        MAX_VISIBLE_DRONES = 8
        ITEMS_PER_LINE = 4
        
        y_pos = y_start
        reward_items = self._reward_items
        total_reward_sum = sum(r for _, r in reward_items)
        
        visible_items = reward_items[:MAX_VISIBLE_DRONES]
        remaining_count = len(reward_items) - MAX_VISIBLE_DRONES
        
        for i, (drone_id, reward) in enumerate(visible_items):
            col_offset = (i % ITEMS_PER_LINE) * 0.12
            if i > 0 and i % ITEMS_PER_LINE == 0:
                y_pos -= line_height * 0.8
            
            x_pos = col_x + 0.02 + col_offset
            short_id = self._abbreviate_drone_id(drone_id)
            self.ax.text(
                x_pos, y_pos, short_id,
                ha='left', va='top', fontsize=8, fontweight='normal', color='#333333',
                transform=self.ax.transAxes
            )
            self.ax.text(
                x_pos + 0.04, y_pos, f"{reward:.0f}",
                ha='left', va='top', fontsize=8, fontweight='bold', color='#2980b9',
                transform=self.ax.transAxes
            )
        
        if len(visible_items) > 0:
            y_pos -= line_height * 0.8
        
        if remaining_count > 0:
            more_text = f"+ {remaining_count} more ▶"
            self._more_text = self.ax.text(
                col_x + 0.02, y_pos, more_text,
                ha='left', va='top', fontsize=8, color='#2980b9',
                transform=self.ax.transAxes,
                picker=True,
                bbox=dict(facecolor='#ecf0f1', edgecolor='none', pad=1)
            )
            y_pos -= line_height * 0.8
        else:
            self._more_text = None
        
        self.ax.text(
            col_x + 0.02, y_pos, f"Total: {total_reward_sum:.1f}",
            ha='left', va='top', fontsize=9, fontweight='bold', color='#333333',
            transform=self.ax.transAxes
        )

    def _abbreviate_drone_id(self, drone_id: str) -> str:
        """
        Abbreviate drone ID for compact display.

        Converts 'drone_0' to 'D0', 'drone_10' to 'D10', etc.
        Returns original ID if pattern doesn't match.

        Args:
            drone_id: The full drone identifier.

        Returns:
            Abbreviated identifier.
        """
        if drone_id.startswith("drone_"):
            return f"D{drone_id[6:]}"
        return drone_id

    def _on_rewards_click(self, event) -> None:
        """
        Handle pick event on the rewards "more" text.

        Args:
            event: The matplotlib pick event.
        """
        if event.artist is not self._more_text:
            return
        if self._more_text is None:
            return
        self._show_rewards_popup()

    def _show_rewards_popup(self) -> None:
        """
        Show a popup window with the full rewards list.
        
        Creates a new figure window displaying all drone rewards in a table format.
        """
        if not self._reward_items:
            return
        
        popup_fig = plt.figure(figsize=(4, 6))
        popup_fig.canvas.manager.set_window_title("All Drone Rewards")
        popup_ax = popup_fig.add_subplot(111)
        popup_ax.axis('off')
        
        popup_ax.text(
            0.5, 0.95, "All Drone Rewards",
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=popup_ax.transAxes
        )
        
        y_pos = 0.88
        line_height = 0.04
        
        popup_ax.text(
            0.2, y_pos, "Drone", ha='left', va='top', fontsize=10, fontweight='bold',
            transform=popup_ax.transAxes
        )
        popup_ax.text(
            0.6, y_pos, "Reward", ha='left', va='top', fontsize=10, fontweight='bold',
            transform=popup_ax.transAxes
        )
        y_pos -= line_height * 1.5
        
        for drone_id, reward in self._reward_items:
            popup_ax.text(
                0.2, y_pos, str(drone_id),
                ha='left', va='top', fontsize=9, color='#555555',
                transform=popup_ax.transAxes
            )
            popup_ax.text(
                0.6, y_pos, f"{reward:.1f}",
                ha='left', va='top', fontsize=9, color='#555555',
                transform=popup_ax.transAxes
            )
            y_pos -= line_height
        
        y_pos -= line_height * 0.5
        total = sum(r for _, r in self._reward_items)
        popup_ax.text(
            0.2, y_pos, "Total",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=popup_ax.transAxes
        )
        popup_ax.text(
            0.6, y_pos, f"{total:.1f}",
            ha='left', va='top', fontsize=10, fontweight='bold', color='#333333',
            transform=popup_ax.transAxes
        )
        
        popup_fig.tight_layout()
        popup_fig.show()

    def _render_hp_chart(self) -> None:
        """
        Render the HP and active targets chart in the lower portion of the panel.
        
        Creates a sub-axes for the chart if HP history data is available.
        Both metrics are normalized to percentage of initial value.
        """
        if not self.hp_history:
            return
        
        if self.chart_ax is not None:
            self.chart_ax.remove()
            self.chart_ax = None
        
        parent_pos = self.ax.get_position()
        chart_height = parent_pos.height * 0.50
        chart_bottom = parent_pos.y0
        chart_left = parent_pos.x0 + parent_pos.width * 0.08
        chart_width = parent_pos.width * 0.84
        
        self.chart_ax = self.fig.add_axes((
            chart_left, chart_bottom, chart_width, chart_height
        ))
        
        steps = list(range(1, len(self.hp_history) + 1))
        
        initial_hp = self.hp_history[0] if self.hp_history[0] > 0 else 1
        hp_percent = [hp / initial_hp * 100 for hp in self.hp_history]
        self.chart_ax.plot(steps, hp_percent, color='#3498db', linewidth=1.5, label='Total HP')
        
        if self.active_targets_history:
            initial_targets = self.active_targets_history[0] if self.active_targets_history[0] > 0 else 1
            targets_percent = [t / initial_targets * 100 for t in self.active_targets_history]
            self.chart_ax.plot(steps[:len(targets_percent)], targets_percent, 
                              color='#e67e22', linewidth=1.5, label='Active Targets')
        
        self.chart_ax.set_title('HP & Active Targets Over Time', fontsize=9, fontweight='bold')
        self.chart_ax.set_xlabel('Step', fontsize=8)
        self.chart_ax.set_ylabel('% of Initial', fontsize=8)
        self.chart_ax.tick_params(axis='both', labelsize=7)
        self.chart_ax.grid(True, linestyle='--', alpha=0.3)
        
        self.chart_ax.set_xlim(1, len(steps))
        self.chart_ax.set_ylim(0, 105)
        self.chart_ax.legend(loc='upper right', fontsize=7)

    def clear(self) -> None:
        """
        Clear the panel and remove chart axes if present.
        """
        if self._pick_cid is not None:
            self.fig.canvas.mpl_disconnect(self._pick_cid)
            self._pick_cid = None
        self._more_text = None
        self._reward_items = []
        
        self.ax.clear()
        if self.chart_ax is not None:
            self.chart_ax.remove()
            self.chart_ax = None
