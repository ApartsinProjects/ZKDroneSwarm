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
        self.decentralized_learning_state: Optional[Dict[str, Any]] = None
        self.decentralized_learning_state_ep1: Optional[Dict[str, Any]] = None
        self.chart_ax_left: Optional[plt.Axes] = None
        self.chart_ax_right: Optional[plt.Axes] = None
        self.selected_agent: int = 0
        self.agent_buttons: List[plt.Text] = []
        self.drones: List[Dict[str, Any]] = []
        self.target_classes: List[str] = []
        self.current_episode_num: int = 1
        self._cid: Optional[int] = None # Event connection ID

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
        
        # Extract target classes for correct coloring
        targets = data.get("targets", [])
        self.target_classes = [t.get("class_type", "unknown") for t in targets]
        
        # Get current episode number from decentralized learning state
        if self.decentralized_learning_state:
            self.current_episode_num = self.decentralized_learning_state.get("episode_num", 1)

    def _compute_topk_private_predictions(
        self,
        agent_lv: List[float],
        target_lv_private: List[List[float]],
        k: int,
        match_best_target: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not agent_lv or not target_lv_private or k <= 0:
            return {"topk": [], "best_target": match_best_target}

        def _dot(a: List[float], b: List[float]) -> float:
            n = min(len(a), len(b))
            return float(sum(float(a[i]) * float(b[i]) for i in range(n)))

        scored = []
        for idx, t_vec in enumerate(target_lv_private):
            dot = _dot(agent_lv, t_vec)
            pred = (1.0 + dot) / 2.0
            scored.append((idx, float(pred)))

        scored.sort(key=lambda x: x[1], reverse=True)
        topk = scored[: min(k, len(scored))]

        best_target = match_best_target
        if best_target is None and topk:
            best_target = int(topk[0][0])

        return {"topk": topk, "best_target": best_target}

    def _compute_topk_from_match(
        self,
        predicted_rewards: Optional[List[float]],
        k: int,
        match_best_target: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not predicted_rewards or k <= 0:
            return {"topk": [], "best_target": match_best_target}

        scored = [(int(i), float(r)) for i, r in enumerate(predicted_rewards)]
        scored.sort(key=lambda x: x[1], reverse=True)
        topk = scored[: min(k, len(scored))]

        best_target = match_best_target
        if best_target is None and topk:
            best_target = int(topk[0][0])

        return {"topk": topk, "best_target": best_target}

    def render_display(self) -> None:
        """
        Render the training path panel.
        """
        self._clear_chart_axes()
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        if self.decentralized_learning_state is None:
            self._render_no_data()
            return

        self._render_decentralized()
    
    def _clear_chart_axes(self) -> None:
        """Clear all chart axes."""
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

    def _render_decentralized(self) -> None:
        """
        Render decentralized policy learning state with agent grid and dual charts.
        Shows an Episode 1 reference snapshot vs the current episode snapshot.
        """
        current_agents = self.decentralized_learning_state.get("episode_state_agents", [])

        ep1_agents = []
        if self.decentralized_learning_state_ep1:
            ep1_agents = self.decentralized_learning_state_ep1.get("episode_state_agents", [])
        
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
            ha='right', va='top', fontsize=10, color='gray',
            transform=self.ax.transAxes
        )
        
        # Render clickable agent grid
        self._render_agent_grid(current_agents)
        
        # Render dual charts: Episode 1 reference (left) vs Current Episode snapshot (right)
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
            
            # Use gray for all agent buttons
            button_color = "gray"
            
            # Highlight selected agent
            if i == self.selected_agent:
                bbox = dict(boxstyle='round,pad=0.2', facecolor=button_color, alpha=0.8)
                text_color = 'white'
            else:
                bbox = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=button_color, alpha=0.8)
                text_color = button_color
            
            text = self.ax.text(
                x, y, f"A{i}",
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=text_color, transform=self.ax.transAxes,
                bbox=bbox, picker=True
            )
            text.agent_index = i
            text.panel_type = "training_path" # Unique identifier
            self.agent_buttons.append(text)
        
        # Connect click event once
        if self._cid is None:
            self._cid = self.fig.canvas.mpl_connect('pick_event', self._on_agent_click)
    
    def _on_agent_click(self, event) -> None:
        """Handle click on agent button."""
        # Only handle if this panel is visible and the click is on our own buttons
        if not self.is_visible():
            return
            
        if hasattr(event.artist, 'agent_index') and getattr(event.artist, 'panel_type', None) == "training_path":
            new_agent = event.artist.agent_index
            if new_agent != self.selected_agent:
                self.selected_agent = new_agent
                self.render_display()
                self.fig.canvas.draw_idle()
    
    def _render_dual_charts(
        self, ep1_agents: List[Dict[str, Any]], current_agents: List[Dict[str, Any]]
    ) -> None:
        """
        Render side-by-side Episode 1 reference vs current episode snapshot charts.
        
        Args:
            ep1_agents: List of agent dicts from the Episode 1 reference snapshot.
            current_agents: List of agent dicts from the current episode snapshot.
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
            self._render_agent_chart(self.chart_ax_left, ep1_agent, "Episode 1 Reference")
        else:
            self.chart_ax_left.text(
                0.5, 0.5, "Episode 1 data\nnot available",
                ha='center', va='center', fontsize=10, color='gray',
                transform=self.chart_ax_left.transAxes
            )
            self.chart_ax_left.set_title("Episode 1 Reference", fontsize=9, fontweight='bold')
        
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
        target_lv_private = agent_data.get("target_lv_private", None)
        target_lv = target_lv_private if target_lv_private is not None else agent_data.get("target_lv", [])
        
        # Plot agent position (triangle)
        ax.scatter(
            agent_lv[0] if len(agent_lv) > 0 else 0,
            agent_lv[1] if len(agent_lv) > 1 else 0,
            s=80, c='#1a5276', marker='^', zorder=10, label='Agent'
        )
        
        match = agent_data.get("match", {})
        match_best_target = match.get("best_target", None) if isinstance(match, dict) else None
        predicted_rewards = match.get("predicted_rewards", None) if isinstance(match, dict) else None

        overlay = self._compute_topk_from_match(
            predicted_rewards=predicted_rewards,
            k=6,
            match_best_target=match_best_target,
        )
        best_target = overlay.get("best_target", None)

        # Plot target positions (circles)
        class_types_plotted = set()
        for i, target_vec in enumerate(target_lv):
            x = target_vec[0] if len(target_vec) > 0 else 0
            y = target_vec[1] if len(target_vec) > 1 else 0
            
            # Determine class type from actual target data
            class_type = "unknown"
            if i < len(self.target_classes):
                class_type = self.target_classes[i]
            
            color = TARGET_COLORS.get(class_type, TARGET_COLORS["unknown"])
            
            ax.scatter(
                x, y, s=30, c=color, marker='o', zorder=9, alpha=0.7,
                label=f"C:{class_type}" if class_type not in class_types_plotted else None
            )
            class_types_plotted.add(class_type)

            if predicted_rewards is not None and best_target is not None and i == int(best_target):
                ax.scatter(
                    x, y, s=60, facecolors='none', edgecolors='#f1c40f', linewidths=1.5,
                    marker='o', zorder=11, alpha=0.95
                )
            
            ax.annotate(
                f"T{i}", (x, y), textcoords="offset points",
                xytext=(3, -8), fontsize=6, color="#000000", alpha=1
            )

        agent_x = agent_lv[0] if len(agent_lv) > 0 else 0
        agent_y = agent_lv[1] if len(agent_lv) > 1 else 0
        for target_idx, pred in overlay.get("topk", []):
            if target_idx >= len(target_lv):
                continue
            t_vec = target_lv[target_idx]
            tx = t_vec[0] if len(t_vec) > 0 else 0
            ty = t_vec[1] if len(t_vec) > 1 else 0

            is_best = (
                predicted_rewards is not None
                and best_target is not None
                and int(target_idx) == int(best_target)
            )
            line_color = '#f1c40f' if is_best else '#7f8c8d'
            line_width = 1.6 if is_best else 0.8
            line_alpha = 0.85 if is_best else 0.35

            ax.plot(
                [agent_x, tx], [agent_y, ty],
                color=line_color, linewidth=line_width, alpha=line_alpha, zorder=8
            )

            mx = (agent_x + tx) / 2
            my = (agent_y + ty) / 2
            ax.text(
                mx, my, f"{pred:.2f}",
                fontsize=7, color="#000000", alpha=min(1.0, line_alpha + 0.1),
                ha='center', va='center', zorder=12
            )
        
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Dim 0', fontsize=7)
        ax.set_ylabel('Dim 1', fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.autoscale()
        ax.margins(0.1)
        ax.set_aspect('equal', adjustable='datalim')
        
    def clear(self) -> None:
        """
        Clear the panel and remove chart axes if present.
        """
        self.ax.clear()
        self._clear_chart_axes()
