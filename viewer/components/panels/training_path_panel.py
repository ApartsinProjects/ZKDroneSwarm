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
    
    For decentralized policies, shows clickable agent grid and dual pre/post charts.
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
        self.decentralized_learning_state: Optional[Dict[str, Any]] = None
        self.decentralized_learning_state_ep1: Optional[Dict[str, Any]] = None
        self.chart_ax: Optional[plt.Axes] = None
        self.chart_ax_left: Optional[plt.Axes] = None
        self.chart_ax_right: Optional[plt.Axes] = None
        self.selected_agent: int = 0
        self.agent_buttons: List[plt.Text] = []
        self.drones: List[Dict[str, Any]] = []
        self.current_episode_num: int = 1

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process state data to extract learning path information.

        Args:
            data: The state dict containing the learning_path or decentralized_learning_state.
        """
        self.learning_path = data.get("learning_path", None)
        self.decentralized_learning_state = data.get("decentralized_learning_state", None)
        self.decentralized_learning_state_ep1 = data.get("decentralized_learning_state_ep1", None)
        self.drones = data.get("drones", [])
        # Get current episode number from decentralized learning state
        if self.decentralized_learning_state:
            self.current_episode_num = self.decentralized_learning_state.get("episode_num", 1)

    def render_display(self) -> None:
        """
        Render the training path panel with latent space scatter chart.
        """
        self._clear_chart_axes()
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        # Check for decentralized learning state first
        if self.decentralized_learning_state is not None:
            self._render_decentralized()
            return

        # Fall back to centralized learning path
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
    
    def _clear_chart_axes(self) -> None:
        """Clear all chart axes."""
        if self.chart_ax is not None:
            self.chart_ax.remove()
            self.chart_ax = None
        if self.chart_ax_left is not None:
            self.chart_ax_left.remove()
            self.chart_ax_left = None
        if self.chart_ax_right is not None:
            self.chart_ax_right.remove()
            self.chart_ax_right = None

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

    def _render_decentralized(self) -> None:
        """
        Render decentralized policy learning state with agent grid and dual charts.
        Shows Episode 1 (random init) vs Current Episode pre-state.
        """
        # Current episode pre-state
        current_pre = self.decentralized_learning_state.get("pre_episode", {})
        current_agents = current_pre.get("agents", [])
        
        # Episode 1 pre-state (random initialization)
        ep1_agents = []
        if self.decentralized_learning_state_ep1:
            ep1_pre = self.decentralized_learning_state_ep1.get("pre_episode", {})
            ep1_agents = ep1_pre.get("agents", [])
        
        if not current_agents:
            self._render_no_data()
            return
        
        # Render title with decentralized indicator
        self.ax.text(
            0.0, 0.98, "Latent Space (Decentralized)",
            ha='left', va='top', fontsize=12, fontweight='bold',
            transform=self.ax.transAxes
        )
        self.ax.text(
            1.0, 0.98, f"Selected: A{self.selected_agent}",
            ha='right', va='top', fontsize=10, color='#1a5276',
            transform=self.ax.transAxes
        )
        
        # Render clickable agent grid
        self._render_agent_grid(current_agents)
        
        # Render dual charts: Episode 1 (left) vs Current Episode (right)
        self._render_dual_charts(ep1_agents, current_agents)
    
    def _render_agent_grid(self, agents: List[Dict[str, Any]]) -> None:
        """
        Render clickable agent grid for selection.
        
        Args:
            agents: List of agent dicts from decentralized learning state.
        """
        self.agent_buttons = []
        num_agents = len(agents)
        
        # Calculate grid layout
        cols = min(6, num_agents)
        rows = (num_agents + cols - 1) // cols
        
        start_y = 0.88
        row_height = 0.06
        col_width = 1.0 / cols
        
        for i, agent in enumerate(agents):
            row = i // cols
            col = i % cols
            
            x = col * col_width + col_width / 2
            y = start_y - row * row_height
            
            agent_id = agent.get("id", f"drone_{i}")
            drone_idx = int(agent_id.split("_")[1]) if "_" in agent_id else i
            
            # Get weapon type from drones list
            weapon_type = "unknown"
            if drone_idx < len(self.drones):
                weapon_type = self.drones[drone_idx].get("weapon_type", "unknown")
            
            color = WEAPON_COLORS.get(weapon_type, WEAPON_COLORS["unknown"])
            
            # Highlight selected agent
            if i == self.selected_agent:
                bbox = dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
                text_color = 'white'
            else:
                bbox = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8)
                text_color = color
            
            text = self.ax.text(
                x, y, f"A{i}",
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=text_color, transform=self.ax.transAxes,
                bbox=bbox, picker=True
            )
            text.agent_index = i
            self.agent_buttons.append(text)
        
        # Connect click event
        self.fig.canvas.mpl_connect('pick_event', self._on_agent_click)
    
    def _on_agent_click(self, event) -> None:
        """Handle click on agent button."""
        if hasattr(event.artist, 'agent_index'):
            new_agent = event.artist.agent_index
            if new_agent != self.selected_agent:
                self.selected_agent = new_agent
                self.render_display()
                self.fig.canvas.draw_idle()
    
    def _render_dual_charts(
        self, ep1_agents: List[Dict[str, Any]], current_agents: List[Dict[str, Any]]
    ) -> None:
        """
        Render side-by-side Episode 1 vs Current Episode charts for selected agent.
        
        Args:
            ep1_agents: List of agent dicts from episode 1 pre-state (random init).
            current_agents: List of agent dicts from current episode pre-state.
        """
        parent_pos = self.ax.get_position()
        chart_height = parent_pos.height * 0.58
        chart_bottom = parent_pos.y0
        chart_width = parent_pos.width * 0.42
        gap = parent_pos.width * 0.04
        
        # Left chart (Episode 1 - random initialization)
        chart_left_x = parent_pos.x0 + parent_pos.width * 0.04
        self.chart_ax_left = self.fig.add_axes((
            chart_left_x, chart_bottom, chart_width, chart_height
        ))
        
        # Right chart (Current Episode)
        chart_right_x = chart_left_x + chart_width + gap
        self.chart_ax_right = self.fig.add_axes((
            chart_right_x, chart_bottom, chart_width, chart_height
        ))
        
        # Get selected agent data
        if ep1_agents and self.selected_agent < len(ep1_agents):
            ep1_agent = ep1_agents[self.selected_agent]
            self._render_agent_chart(self.chart_ax_left, ep1_agent, "Episode 1 (Random)")
        else:
            self.chart_ax_left.text(
                0.5, 0.5, "Episode 1 data\nnot available",
                ha='center', va='center', fontsize=10, color='gray',
                transform=self.chart_ax_left.transAxes
            )
            self.chart_ax_left.set_title("Episode 1 (Random)", fontsize=9, fontweight='bold')
        
        if self.selected_agent < len(current_agents):
            current_agent = current_agents[self.selected_agent]
            self._render_agent_chart(self.chart_ax_right, current_agent, f"Episode {self.current_episode_num}")
    
    def _render_agent_chart(
        self, ax: plt.Axes, agent_data: Dict[str, Any], title: str
    ) -> None:
        """
        Render a single agent's latent space chart.
        
        Args:
            ax: The axes to render on.
            agent_data: Agent dict with agent_lv, target_lv.
            title: Chart title.
        """
        agent_lv = agent_data.get("agent_lv", [0, 0])
        target_lv = agent_data.get("target_lv", [])
        
        # Plot agent position (triangle)
        ax.scatter(
            agent_lv[0] if len(agent_lv) > 0 else 0,
            agent_lv[1] if len(agent_lv) > 1 else 0,
            s=80, c='#1a5276', marker='^', zorder=10, label='Agent'
        )
        
        # Plot target positions (circles)
        class_types_plotted = set()
        for i, target_vec in enumerate(target_lv):
            x = target_vec[0] if len(target_vec) > 0 else 0
            y = target_vec[1] if len(target_vec) > 1 else 0
            
            # Determine class type based on index pattern (A, B, C cycling)
            class_idx = i % 3
            class_type = ["A", "B", "C"][class_idx] if class_idx < 3 else "unknown"
            color = TARGET_COLORS.get(class_type, TARGET_COLORS["unknown"])
            
            ax.scatter(
                x, y, s=30, c=color, marker='o', zorder=9, alpha=0.7,
                label=f"C:{class_type}" if class_type not in class_types_plotted else None
            )
            class_types_plotted.add(class_type)
            
            ax.annotate(
                f"T{i}", (x, y), textcoords="offset points",
                xytext=(3, -8), fontsize=5, color=color, alpha=0.6
            )
        
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Dim 0', fontsize=7)
        ax.set_ylabel('Dim 1', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal', adjustable='box')
        
    def clear(self) -> None:
        """
        Clear the panel and remove chart axes if present.
        """
        self.ax.clear()
        self._clear_chart_axes()
