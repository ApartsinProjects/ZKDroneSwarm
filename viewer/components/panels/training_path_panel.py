"""
Training path panel component for the TabulaDrone viewer.

This module provides a panel that displays the latent space scatter chart
for collaborative filtering policies, showing agent and target latent vectors.
"""

from typing import Dict, Any, Optional, List
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


class TrainingPathPanel(BaseComponent):
    """
    Panel that displays the latent space scatter chart.

    Shows agent latent vectors (triangles) and target latent vectors (circles)
    colored by weapon type and class type respectively.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the training path panel.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        super().__init__(fig, ax)
        self.learning_path: Optional[Dict[str, Any]] = None
        self.chart_ax: Optional[plt.Axes] = None

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process state data to extract learning path information.

        Args:
            data: The state dict containing the learning_path section.
        """
        self.learning_path = data.get("learning_path", None)

    def render_display(self) -> None:
        """
        Render the training path panel with latent space scatter chart.
        """
        if self.chart_ax is not None:
            self.chart_ax.remove()
            self.chart_ax = None
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        if self.learning_path is None:
            self._render_no_data()
            return

        agents = self.learning_path.get("agents", [])
        targets = self.learning_path.get("targets", [])

        if not agents and not targets:
            self._render_no_data()
            return

        if not self._validate_latent_dim(agents, targets):
            self._render_unsupported_dim()
            return

        self._render_title()
        self._render_agent_vectors(agents)
        self._render_scatter_chart(agents, targets)

    def _render_no_data(self) -> None:
        """
        Render placeholder when no learning path data is available.
        """
        self.ax.text(
            0.5, 0.5, "No learning path data available",
            ha='center', va='center', fontsize=12, color='gray',
            transform=self.ax.transAxes
        )

    def _render_unsupported_dim(self) -> None:
        """
        Render message when latent dimensions > 2.
        """
        self.ax.text(
            0.5, 0.5, "Latent dimensions > 2 not currently supported",
            ha='center', va='center', fontsize=12, color='gray',
            transform=self.ax.transAxes
        )

    def _validate_latent_dim(
        self, agents: List[Dict[str, Any]], targets: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate that all latent vectors have dimension <= 2.

        Args:
            agents: List of agent dicts with latent_vector.
            targets: List of target dicts with latent_vector.

        Returns:
            True if all vectors have dim <= 2, False otherwise.
        """
        for agent in agents:
            vec = agent.get("latent_vector", [])
            if len(vec) > 2:
                return False
        for target in targets:
            vec = target.get("latent_vector", [])
            if len(vec) > 2:
                return False
        return True

    def _render_title(self) -> None:
        """
        Render the panel title.
        """
        self.ax.text(
            0.0, 0.98, "Latent Space",
            ha='left', va='top', fontsize=12, fontweight='bold',
            transform=self.ax.transAxes
        )

    def _render_agent_vectors(self, agents: List[Dict[str, Any]]) -> None:
        """
        Render agent latent vectors as text with black background in 3-column grid.

        Args:
            agents: List of agent dicts with id, weapon_type, latent_vector.
        """
        if not agents:
            return

        items = []
        for agent in agents:
            agent_id = agent.get("id", "?")
            vec = agent.get("latent_vector", [0, 0])
            x = vec[0] if len(vec) > 0 else 0
            y = vec[1] if len(vec) > 1 else 0
            items.append(f"{agent_id}: [{x:.2f}, {y:.2f}]")

        num_cols = 3
        rows = []
        for i in range(0, len(items), num_cols):
            row_items = items[i:i + num_cols]
            rows.append("  ".join(f"{item:<18}" for item in row_items))

        text = "\n\n".join(rows)
        self.ax.text(
            0.0, 0.90, text,
            ha='left', va='top', fontsize=8, color='white',
            transform=self.ax.transAxes,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.9)
        )

    def _render_scatter_chart(
        self, agents: List[Dict[str, Any]], targets: List[Dict[str, Any]]
    ) -> None:
        """
        Render the scatter chart with agents and targets.

        Args:
            agents: List of agent dicts with id, weapon_type, latent_vector.
            targets: List of target dicts with id, class_type, latent_vector.
        """
        if self.chart_ax is not None:
            self.chart_ax.remove()
            self.chart_ax = None

        parent_pos = self.ax.get_position()
        chart_height = parent_pos.height * 0.72
        chart_bottom = parent_pos.y0
        chart_left = parent_pos.x0 + parent_pos.width * 0.08
        chart_width = parent_pos.width * 0.84

        self.chart_ax = self.fig.add_axes((
            chart_left, chart_bottom, chart_width, chart_height
        ))

        weapon_types_plotted = set()
        class_types_plotted = set()

        for agent in agents:
            vec = agent.get("latent_vector", [0, 0])
            x = vec[0] if len(vec) > 0 else 0
            y = vec[1] if len(vec) > 1 else 0
            weapon_type = agent.get("weapon_type", "unknown")
            color = WEAPON_COLORS.get(weapon_type, WEAPON_COLORS["unknown"])
            agent_id = agent.get("id", "?")

            self.chart_ax.scatter(
                x, y, s=35, c=color, marker='^', zorder=10,
                label=f"W:{weapon_type}" if weapon_type not in weapon_types_plotted else None
            )
            weapon_types_plotted.add(weapon_type)

            self.chart_ax.annotate(
                agent_id, (x, y), textcoords="offset points",
                xytext=(5, 5), fontsize=7, color=color
            )

        for target in targets:
            vec = target.get("latent_vector", [0, 0])
            x = vec[0] if len(vec) > 0 else 0
            y = vec[1] if len(vec) > 1 else 0
            class_type = target.get("class_type", "unknown")
            color = TARGET_COLORS.get(class_type, TARGET_COLORS["unknown"])
            target_id = target.get("id", "?")

            self.chart_ax.scatter(
                x, y, s=40, c=color, marker='o', zorder=9,
                label=f"C:{class_type}" if class_type not in class_types_plotted else None
            )
            class_types_plotted.add(class_type)

            self.chart_ax.annotate(
                target_id, (x, y), textcoords="offset points",
                xytext=(5, -10), fontsize=6, color=color, alpha=0.8
            )

        self.chart_ax.set_xlabel('Latent Dim 0', fontsize=8)
        self.chart_ax.set_ylabel('Latent Dim 1', fontsize=8)
        self.chart_ax.tick_params(axis='both', labelsize=7)
        self.chart_ax.grid(True, linestyle='--', alpha=0.3)

        self.chart_ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        self.chart_ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)

        self.chart_ax.set_xlim(-1.2, 1.2)
        self.chart_ax.set_ylim(-1.2, 1.2)
        self.chart_ax.set_aspect('equal', adjustable='box')

        
    def clear(self) -> None:
        """
        Clear the panel and remove chart axes if present.
        """
        self.ax.clear()
        if self.chart_ax is not None:
            self.chart_ax.remove()
            self.chart_ax = None
