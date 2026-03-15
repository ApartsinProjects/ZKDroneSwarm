"""
Radar chart panel component for the TabulaDrone viewer.

Displays a Top-K radar chart of predicted rewards for the selected agent,
helping to visualize specialization and target class preference.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from viewer.components.base import BaseComponent

WEAPON_COLORS = {
    "breach": "#1a5276",
    "structural": "#1e8449",
    "systems": "#922b21",
    "unknown": "#9370DB",
}

TARGET_COLORS = {
    "A": "#e74c3c",
    "B": "#2eaa71",
    "C": "#3498db",
    "unknown": "#5d6d7e",
}

class RadarChartPanel(BaseComponent):
    """
    Panel that displays a Top-K radar chart.

    Shows the top K predicted rewards for a selected agent, where each
    spoke represents a target. The color of labels and points indicates
    the target class.
    """

    def __init__(self, fig: plt.Figure, ax: plt.Axes, k: int = 50):
        super().__init__(fig, ax)
        self.k = k
        self.decentralized_learning_state: Optional[Dict[str, Any]] = None
        self.drones: List[Dict[str, Any]] = []
        self.targets: List[Dict[str, Any]] = []
        self.target_classes: List[str] = []
        self.selected_agent: int = 0
        self.agent_buttons: List[plt.Text] = []
        self.radar_ax: Optional[plt.Axes] = None
        self.current_episode_num: int = 1
        self._cid: Optional[int] = None # Event connection ID

    def process_data(self, data: Dict[str, Any]) -> None:
        self.decentralized_learning_state = data.get("decentralized_learning_state", None)
        self.drones = data.get("drones", [])
        self.targets = data.get("targets", [])
        self.target_classes = [t.get("class_type", "unknown") for t in self.targets]
        if self.decentralized_learning_state:
            self.current_episode_num = self.decentralized_learning_state.get("episode_num", 1)

    def render_display(self) -> None:
        self._clear_extra_axes()
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")

        if self.decentralized_learning_state is None:
            self._render_no_data()
            return

        pre = self.decentralized_learning_state.get("pre_episode", {})
        agents = pre.get("agents", [])
        if not agents or self.selected_agent >= len(agents):
            self._render_no_data()
            return

        # Title
        self.ax.text(
            0.0, 0.98, f"Top-{self.k} Predicted Rewards (Episode {self.current_episode_num})",
            ha="left", va="top", fontsize=12, fontweight="bold",
            transform=self.ax.transAxes,
        )

        # Agent selector
        self._render_agent_grid(agents)

        # Selected agent data
        agent_data = agents[self.selected_agent]
        match = agent_data.get("match", {})
        preds = match.get("predicted_rewards", None)
        
        if preds:
            self._render_radar(preds)

    def _render_agent_grid(self, agents: List[Dict[str, Any]]) -> None:
        self.agent_buttons = []
        num_agents = len(agents)
        cols = min(6, num_agents)
        col_width = 1.0 / cols
        start_y = 0.88

        for i, agent in enumerate(agents):
            col = i % cols
            row = i // cols
            x = col * col_width + col_width / 2
            y = start_y - row * 0.06

            drone_idx = i
            agent_id = agent.get("id", f"drone_{i}")
            if "_" in agent_id:
                drone_idx = int(agent_id.split("_")[1])

            # Use gray for all agent buttons
            button_color = "gray"

            if i == self.selected_agent:
                bbox = dict(boxstyle="round,pad=0.2", facecolor=button_color, alpha=0.8)
                text_color = "white"
            else:
                bbox = dict(boxstyle="round,pad=0.2", facecolor="white",
                            edgecolor=button_color, alpha=0.8)
                text_color = button_color

            txt = self.ax.text(
                x, y, f"A{i}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=text_color, transform=self.ax.transAxes,
                bbox=bbox, picker=True,
            )
            txt.agent_index = i
            txt.panel_type = "radar" # Unique identifier for this panel's buttons
            self.agent_buttons.append(txt)

        if self._cid is None:
            self._cid = self.fig.canvas.mpl_connect("pick_event", self._on_agent_click)

    def _on_agent_click(self, event) -> None:
        # Only handle if this panel is visible and the click is on our own buttons
        if not self.is_visible():
            return
            
        if hasattr(event.artist, "agent_index") and getattr(event.artist, "panel_type", None) == "radar":
            new_agent = event.artist.agent_index
            if new_agent != self.selected_agent:
                self.selected_agent = new_agent
                self.render_display()
                self.fig.canvas.draw_idle()

    def _render_radar(self, rewards: List[float]) -> None:
        # Get Top-K targets
        ranked_indices = np.argsort(-np.array(rewards))[:self.k]
        top_rewards = [rewards[i] for i in ranked_indices]
        
        num_vars = len(ranked_indices)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Close the loop
        top_rewards += top_rewards[:1]
        angles += angles[:1]

        parent_pos = self.ax.get_position()
        # Offset chart below the agent grid
        chart_top = parent_pos.y0 + parent_pos.height * 0.70
        chart_bottom = parent_pos.y0 + parent_pos.height * 0.05
        chart_left = parent_pos.x0 + parent_pos.width * 0.1
        chart_width = parent_pos.width * 0.8
        chart_height = chart_top - chart_bottom

        self.radar_ax = self.fig.add_axes(
            [chart_left, chart_bottom, chart_width, chart_height],
            polar=True
        )

        # Plot values
        self.radar_ax.plot(angles, top_rewards, color='gray', linewidth=1, linestyle='solid', alpha=0.3)
        self.radar_ax.fill(angles, top_rewards, color='gray', alpha=0.05)

        # Weapon type for the selected agent
        weapon_type = "unknown"
        drone_idx = self.selected_agent
        if drone_idx < len(self.drones):
            weapon_type = self.drones[drone_idx].get("weapon_type", "unknown")
        
        # Use gray for the title
        title_color = "gray"

        # Set radial limit
        rmax = max(top_rewards) * 1.25 if top_rewards else 1.0
        self.radar_ax.set_ylim(0, rmax)
        self.radar_ax.set_yticklabels([]) # Hide radial numbers to reduce clutter

        # Set labels and colors for each spoke
        labels = []
        for i, idx in enumerate(ranked_indices):
            ctype = self.target_classes[idx] if idx < len(self.target_classes) else "?"
            label = f"T{idx}\n({ctype})"
            labels.append(label)
            
            # Draw individual dots for rewards with target class color
            color = TARGET_COLORS.get(ctype, TARGET_COLORS["unknown"])
            self.radar_ax.scatter(angles[i], top_rewards[i], color=color, s=40, zorder=10)
            
            # Add reward value annotation above the point
            self.radar_ax.text(
                angles[i], top_rewards[i] + (0.02 * rmax), f"{top_rewards[i]:.2f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=color, zorder=11
            )

        self.radar_ax.set_xticks(angles[:-1])
        self.radar_ax.set_xticklabels(labels, fontsize=8)
        
        # Color the x-labels (target labels) by class
        for i, label in enumerate(self.radar_ax.get_xticklabels()):
            idx = ranked_indices[i]
            ctype = self.target_classes[idx] if idx < len(self.target_classes) else "?"
            label.set_color(TARGET_COLORS.get(ctype, TARGET_COLORS["unknown"]))
            label.set_fontweight('bold')
        
        # Add a subtle grid
        self.radar_ax.grid(True, alpha=0.2)
        self.radar_ax.set_title(f"Agent {self.selected_agent} ({weapon_type})", 
                                fontsize=10, pad=20, color=title_color, fontweight='bold')

    def _render_no_data(self) -> None:
        self.ax.text(
            0.5, 0.5, "No preference data available",
            ha="center", va="center", fontsize=12, color="gray",
            transform=self.ax.transAxes,
        )

    def _clear_extra_axes(self) -> None:
        if self.radar_ax is not None:
            self.radar_ax.remove()
            self.radar_ax = None

    def set_visible(self, visible: bool) -> None:
        super().set_visible(visible)
        if self.radar_ax is not None:
            self.radar_ax.set_visible(visible)

    def clear(self) -> None:
        self.ax.clear()
        self._clear_extra_axes()
