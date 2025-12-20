"""
Drawing functions for TabulaDrone Episode Viewer.
"""

import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from typing import Dict, Any

from viewer.components import TabContainer, EmptyPanel, InfoPanel, ResultsPanel
from viewer.state_adapter import load_episode, extract_initial_state


# Color schemes
WEAPON_COLORS = {
    "light": "#3498db",    # Blue
    "medium": "#2ecc71",   # Green
    "heavy": "#e74c3c",    # Red
    "unknown": "#95a5a6",  # Gray
}

TARGET_COLOR = "#9370DB"  # Medium Purple (all targets same color)


def render_map(ax: plt.Axes, state: Dict[str, Any]) -> None:
    """
    Render the map view on the given axes.
    
    Args:
        ax: The matplotlib axes to draw on
        state: Initial state dict from extract_initial_state()
    """
    world_size = state["world_size"]
    drones = state["drones"]
    targets = state["targets"]
    
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_aspect("equal")
    
    ax.grid(True, linestyle="--", alpha=0.3)
    
    border = plt.Rectangle(
        (0, 0), world_size[0], world_size[1],
        fill=False, edgecolor="black", linewidth=2
    )
    ax.add_patch(border)
    
    for target in targets:
        x, y = target["position"]
        ax.scatter(x, y, s=40, c=TARGET_COLOR, marker="o", zorder=10)
        class_type = target.get("class_type", "")
        ax.text(x, y + 12, class_type, fontsize=7, ha='center', va='bottom', zorder=12)
    
    for drone in drones:
        x, y = drone["position"]
        weapon_type = drone["weapon_type"]
        color = WEAPON_COLORS.get(weapon_type, WEAPON_COLORS["unknown"])
        ax.scatter(x, y, s=35, c=color, marker="^", zorder=11)
        ax.text(x, y + 12, weapon_type, fontsize=7, ha='center', va='bottom', zorder=12)
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    seed = state.get("seed")
    title = "TabulaDrone - Initial World State"
    if seed is not None:
        title += f" (Seed: {seed})"
    ax.set_title(title)


def display_viewer(
    state: Dict[str, Any],
    episode_files: list[str] | None = None,
    current_index: int = 0
) -> None:
    """
    Display the split-panel viewer with map on left and info panel on right.
    
    Args:
        state: Initial state dict from extract_initial_state()
        episode_files: Optional list of episode file paths for navigation (sorted descending)
        current_index: Index of current episode in episode_files (0 = newest)
    """
    fig = plt.figure(figsize=(12, 7))
    
    gs = GridSpec(1, 2, figure=fig, width_ratios=[70, 30], wspace=0.25,
                  left=0.06, right=0.98, top=0.95, bottom=0.18)
    
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])
    
    render_map(ax_left, state)
    
    tab_region = (0.62, 0.95, 0.35, 0.05)
    tab_container = TabContainer(fig, ax_right, tab_region)
    
    info_ax = fig.add_axes([0.62, 0.08, 0.35, 0.84])
    info_panel = InfoPanel(fig, info_ax)
    info_panel.render(state)
    
    actions_ax = fig.add_axes([0.62, 0.08, 0.35, 0.84])
    actions_panel = EmptyPanel(fig, actions_ax, text="Actions")
    
    results_ax = fig.add_axes([0.62, 0.08, 0.35, 0.84])
    results_panel = ResultsPanel(fig, results_ax)
    results_panel.render(state)
    
    tab_container.add_tab("Info", info_panel)
    tab_container.add_tab("Actions", actions_panel)
    tab_container.add_tab("Results", results_panel)
    
    if episode_files is not None:
        index_state = [current_index]
        
        def _refresh_viewer():
            try:
                episode_data = load_episode(episode_files[index_state[0]])
                new_state = extract_initial_state(episode_data)
                
                ax_left.clear()
                render_map(ax_left, new_state)
                
                info_panel.render(new_state)
                actions_panel.render(new_state)
                results_panel.render(new_state)
                
                info_text.set_text(f"Episode {index_state[0] + 1} of {len(episode_files)}")
                
                fig.canvas.draw_idle()
            except Exception as e:
                print(f"Error loading episode: {e}", file=sys.stderr)
        
        def _update_button_states():
            if index_state[0] == 0:
                prev_button.color = '0.85'
                prev_button.hovercolor = '0.85'
            else:
                prev_button.color = 'white'
                prev_button.hovercolor = '0.95'
            
            if index_state[0] >= len(episode_files) - 1:
                next_button.color = '0.85'
                next_button.hovercolor = '0.85'
            else:
                next_button.color = 'white'
                next_button.hovercolor = '0.95'
            
            fig.canvas.draw_idle()
        
        def on_prev(event):
            if index_state[0] > 0:
                index_state[0] -= 1
                _refresh_viewer()
                _update_button_states()
        
        def on_next(event):
            if index_state[0] < len(episode_files) - 1:
                index_state[0] += 1
                _refresh_viewer()
                _update_button_states()
        
        button_width = 0.06
        button_height = 0.04
        button_spacing = 0.02
        button_y = 0.03
        
        left_panel_center = (0.06 + 0.62) / 2
        total_width = button_width * 2 + button_spacing
        left_button_x = left_panel_center - total_width / 2
        right_button_x = left_button_x + button_width + button_spacing
        
        prev_ax = fig.add_axes([left_button_x, button_y, button_width, button_height])
        prev_button = Button(prev_ax, "Previous")
        prev_button.on_clicked(on_prev)
        
        next_ax = fig.add_axes([right_button_x, button_y, button_width, button_height])
        next_button = Button(next_ax, "Next")
        next_button.on_clicked(on_next)
        
        info_text_y = button_y + button_height + 0.005
        info_text = fig.text(
            left_panel_center, info_text_y,
            f"Episode {current_index + 1} of {len(episode_files)}",
            ha='center', va='bottom', fontsize=10
        )
        
        _update_button_states()
    
    plt.show()
