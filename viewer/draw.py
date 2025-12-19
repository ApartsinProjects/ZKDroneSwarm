"""
Drawing functions for TabulaDrone Episode Viewer.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any

from viewer.components import TabContainer, EmptyPanel, InfoPanel, ResultsPanel


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
    ax.set_title("TabulaDrone - Initial World State")


def display_viewer(state: Dict[str, Any]) -> None:
    """
    Display the split-panel viewer with map on left and info panel on right.
    
    Args:
        state: Initial state dict from extract_initial_state()
    """
    fig = plt.figure(figsize=(12, 7))
    
    gs = GridSpec(1, 2, figure=fig, width_ratios=[70, 30], wspace=0.25,
                  left=0.06, right=0.98, top=0.95, bottom=0.08)
    
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
    
    plt.show()
