"""
Drawing functions for TabulaDrone Episode Viewer.
"""

import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from typing import Dict, Any

from viewer.components import TabContainer, MapPanel, EmptyPanel, InfoPanel, ResultsPanel
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
    policy_type = state.get("policy_type", "unknown")
    title = f"World State (Seed: {seed} | Policy: {policy_type})"
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
    
    map_width_inches = 6.5
    gap_inches = 0.05
    
    left_margin = 0.01
    right_margin = 0.95
    bottom_margin = 0.18
    top_margin = 0.95
    
    def _layout_values() -> tuple[float, float, float, float]:
        fig_width = fig.get_figwidth()
        map_width = map_width_inches / fig_width
        gap_fig = gap_inches / fig_width
        map_left = left_margin
        right_panel_left = map_left + map_width + gap_fig
        right_panel_width = right_margin - right_panel_left
        return map_left, map_width, right_panel_left, right_panel_width
    
    map_left, map_width, right_panel_left, right_panel_width = _layout_values()
    
    ax_right = fig.add_axes([right_panel_left, bottom_margin, right_panel_width, top_margin - bottom_margin])
    
    map_panel: MapPanel | None = None
    
    tab_region = (right_panel_left, 0.95, right_panel_width, 0.05)
    tab_container = TabContainer(fig, ax_right, tab_region)
    
    info_ax = fig.add_axes([right_panel_left, 0.08, right_panel_width, 0.84])
    info_panel = InfoPanel(fig, info_ax)
    info_panel.render(state)
    
    actions_ax = fig.add_axes([right_panel_left, 0.08, right_panel_width, 0.84])
    actions_panel = EmptyPanel(fig, actions_ax, text="Actions")
    
    results_ax = fig.add_axes([right_panel_left, 0.08, right_panel_width, 0.84])
    results_panel = ResultsPanel(fig, results_ax)
    results_panel.render(state)
    
    tab_container.add_tab("Info", info_panel)
    tab_container.add_tab("Results", results_panel)
    tab_container.add_tab("Actions", actions_panel)

    
    def _apply_layout(event=None):
        map_left_inner, map_width_inner, right_panel_left_inner, right_panel_width_inner = _layout_values()
        
        ax_right.set_position([right_panel_left_inner, bottom_margin, right_panel_width_inner, top_margin - bottom_margin])
        
        for panel_ax in (info_ax, actions_ax, results_ax):
            panel_ax.set_position([right_panel_left_inner, 0.08, right_panel_width_inner, 0.84])
        
        tab_container.tab_region = (right_panel_left_inner, 0.95, right_panel_width_inner, 0.05)
        if tab_container.tab_button_axes:
            button_width = 0.08
            button_height = 0.03
            button_spacing = 0.01
            region_y = tab_container.tab_button_axes[next(iter(tab_container.tab_button_axes))].get_position().y0
            for idx, button_ax in enumerate(tab_container.tab_button_axes.values()):
                new_x = right_panel_left_inner + (button_width + button_spacing) * idx
                button_ax.set_position([new_x, region_y, button_width, button_height])
        
        if map_panel is not None:
            map_panel.update_position((map_left_inner, bottom_margin, map_width_inner, top_margin - bottom_margin))
        
        fig.canvas.draw_idle()
    
    def _on_episode_change(new_index: int) -> None:
        try:
            episode_data = load_episode(episode_files[new_index])
            new_state = extract_initial_state(episode_data)
            
            map_panel.refresh(new_state)
            info_panel.render(new_state)
            actions_panel.render(new_state)
            results_panel.render(new_state)
        except Exception as e:
            print(f"Error loading episode: {e}", file=sys.stderr)
    
    left_region = (map_left, bottom_margin, map_width, top_margin - bottom_margin)
    map_panel = MapPanel(
        fig,
        left_region,
        state,
        episode_files=episode_files,
        current_index=current_index,
        on_episode_change=_on_episode_change if episode_files else None
    )
    
    _apply_layout()
    fig.canvas.mpl_connect('resize_event', _apply_layout)
    
    plt.show()
