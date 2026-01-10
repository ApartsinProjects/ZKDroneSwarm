"""
Drawing functions for TabulaDrone Episode Viewer.
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import Dict, Any

_BACKGROUND_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "background.png")
_BACKGROUND_IMAGE = None

_DRONE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "drone.png")
_DRONE_IMAGE = None

_TARGET_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "target_1.png")
_TARGET_IMAGE = None

_TARGET_FLAME_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "target_with_flame.png")
_TARGET_FLAME_IMAGE = None

_TARGET_DESTROYED_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "target_destroyed.png")
_TARGET_DESTROYED_IMAGE = None

from viewer.components import TabContainer, MapPanel, EmptyPanel, InfoPanel, ResultsPanel, SummaryPanel, TrainingPathPanel
from viewer.state_adapter import load_episode, extract_initial_state


# Color schemes
WEAPON_COLORS = {
    "breach": "#1a5276",    # Dark Blue
    "structural": "#1e8449",   # Dark Green
    "systems": "#922b21",    # Dark Red
    "unknown": "#9370DB",  # Medium Purple (fallback)
}

TARGET_COLORS = {
    "A": "#e74c3c",    # Red
    "B": "#2eaa71",    # Green
    "C": "#3498db",    # Blue
    "unknown": "#5d6d7e",  # Dark Gray (fallback)
    "destroyed": "#808080",  # Gray (destroyed target)
}


ENGAGEMENT_LINE_COLOR = "#808080"  # Gray
ENGAGEMENT_CIRCLE_COLOR = "#ff8c00"  # Orange

TARGET_IMAGE_ZOOM = 0.08

def draw_engagements(
    ax: plt.Axes,
    drones: list[Dict[str, Any]],
    targets: list[Dict[str, Any]],
    actions: Dict[str, int]
) -> None:
    """
    Draw engagement visuals: dashed line from drone to target + orange circle at target.
    
    Args:
        ax: The matplotlib axes to draw on
        drones: List of drone dicts with 'id' and 'position'
        targets: List of target dicts with 'position'
        actions: Dict mapping drone_id to target_index
    """
    for drone_id, target_index in actions.items():
        drone_idx = int(drone_id.split("_")[1])
        if drone_idx >= len(drones) or target_index < 0 or target_index >= len(targets):
            continue
        
        drone_pos = drones[drone_idx]["position"]
        target_pos = targets[target_index]["position"]
        
        ax.plot(
            [drone_pos[0], target_pos[0]],
            [drone_pos[1], target_pos[1]],
            linestyle="--",
            color=ENGAGEMENT_LINE_COLOR,
            linewidth=.7,
            alpha=0.4,
            zorder=5
        )
        
        circle = plt.Circle(
            target_pos,
            linestyle="--",
            radius=20,
            fill=False,
            edgecolor=ENGAGEMENT_CIRCLE_COLOR,
            linewidth=1.5,
            alpha=0.9,
            zorder=6
        )
        # ax.add_patch(circle)
        
        target = targets[target_index]
        if target.get("hp", 0) > 0:
            global _TARGET_FLAME_IMAGE
            if _TARGET_FLAME_IMAGE is None and os.path.exists(_TARGET_FLAME_IMAGE_PATH):
                _TARGET_FLAME_IMAGE = plt.imread(_TARGET_FLAME_IMAGE_PATH)
            if _TARGET_FLAME_IMAGE is not None:
                im = OffsetImage(_TARGET_FLAME_IMAGE, zoom=TARGET_IMAGE_ZOOM, alpha=1)
                ab = AnnotationBbox(im, target_pos, frameon=False, zorder=7)
                ax.add_artist(ab)


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
    class_attribute_mapping = state.get("class_attribute_mapping", {})
    
    global _BACKGROUND_IMAGE
    if _BACKGROUND_IMAGE is None and os.path.exists(_BACKGROUND_IMAGE_PATH):
        _BACKGROUND_IMAGE = plt.imread(_BACKGROUND_IMAGE_PATH)
    if _BACKGROUND_IMAGE is not None:
        ax.imshow(_BACKGROUND_IMAGE, extent=[0, world_size[0], 0, world_size[1]], zorder=0)
    
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_aspect("equal")
    
    for spine in ax.spines.values():
        spine.set_color("lightgray")
    
    border = plt.Rectangle(
        (0, 0), world_size[0], world_size[1],
        fill=False, edgecolor="yellow", linewidth=1
    )
    ax.add_patch(border)
    
    global _TARGET_IMAGE
    if _TARGET_IMAGE is None and os.path.exists(_TARGET_IMAGE_PATH):
        _TARGET_IMAGE = plt.imread(_TARGET_IMAGE_PATH)
    
    for target in targets:
        x, y = target["position"]
        class_type = target.get("class_type", "unknown")
        hp = target.get("hp", 0)
        
        if class_type == "destroyed" or hp <= 0:
            global _TARGET_DESTROYED_IMAGE
            if _TARGET_DESTROYED_IMAGE is None and os.path.exists(_TARGET_DESTROYED_IMAGE_PATH):
                _TARGET_DESTROYED_IMAGE = plt.imread(_TARGET_DESTROYED_IMAGE_PATH)
            if _TARGET_DESTROYED_IMAGE is not None:
                im = OffsetImage(_TARGET_DESTROYED_IMAGE, zoom=TARGET_IMAGE_ZOOM, alpha=0.8)
                ab = AnnotationBbox(im, (x, y), frameon=False, zorder=10)
                ax.add_artist(ab)
            else:
                circle = plt.Circle((x, y), radius=8, fill=False, edgecolor=TARGET_COLORS["destroyed"], linestyle="--", linewidth=1, zorder=10)
                ax.add_patch(circle)
        else:
            class_attrs = class_attribute_mapping.get(class_type, {})
            max_hp = sum(class_attrs.values()) if class_attrs else hp
            alpha = max(0.3, hp / max_hp) if max_hp > 0 else 1.0
            if _TARGET_IMAGE is not None:
                im = OffsetImage(_TARGET_IMAGE, zoom=TARGET_IMAGE_ZOOM, alpha=alpha)
                ab = AnnotationBbox(im, (x, y), frameon=False, zorder=10)
                ax.add_artist(ab)
            else:
                color = TARGET_COLORS.get(class_type, TARGET_COLORS["unknown"])
                ax.scatter(x, y, s=40, c=color, marker="o", alpha=alpha, zorder=10)
            ax.text(x, y - 40, f"{hp:.0f}", fontsize=7, ha='center', va='top', alpha=0.9, zorder=12)
    
    global _DRONE_IMAGE
    if _DRONE_IMAGE is None and os.path.exists(_DRONE_IMAGE_PATH):
        _DRONE_IMAGE = plt.imread(_DRONE_IMAGE_PATH)
    
    for drone in drones:
        x, y = drone["position"]
        if _DRONE_IMAGE is not None:
            ax.imshow(_DRONE_IMAGE, extent=[x-30, x+30, y-30, y+30], zorder=11)
        else:
            weapon_type = drone["weapon_type"]
            color = WEAPON_COLORS.get(weapon_type, WEAPON_COLORS["unknown"])
            ax.scatter(x, y, s=35, c=color, marker="^", zorder=11)
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    policy_type = state.get("policy_type", "unknown")
    title = f"World State Map ( Policy: {policy_type} )"
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
    
    summary_ax = fig.add_axes([right_panel_left, 0.08, right_panel_width, 0.84])
    summary_panel = SummaryPanel(fig, summary_ax)
    summary_panel.render(state)
    
    results_ax = fig.add_axes([right_panel_left, 0.08, right_panel_width, 0.84])
    results_panel = ResultsPanel(fig, results_ax)
    results_panel.render(state)
    
    training_path_ax = fig.add_axes([right_panel_left, 0.08, right_panel_width, 0.84])
    training_path_panel = TrainingPathPanel(fig, training_path_ax)
    training_path_panel.render(state)
    
    tab_container.add_tab("Info", info_panel)
    tab_container.add_tab("Results", results_panel)
    tab_container.add_tab("Summary", summary_panel)
    tab_container.add_tab("Training Path", training_path_panel)

    
    def _apply_layout(event=None):
        map_left_inner, map_width_inner, right_panel_left_inner, right_panel_width_inner = _layout_values()
        
        ax_right.set_position([right_panel_left_inner, bottom_margin, right_panel_width_inner, top_margin - bottom_margin])
        
        for panel_ax in (info_ax, summary_ax, results_ax, training_path_ax):
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
            episode_path = episode_files[new_index]
            episode_data = load_episode(episode_path)
            new_state = extract_initial_state(episode_data, episode_path=episode_path)
            
            map_panel.refresh(new_state)
            info_panel.render(new_state)
            summary_panel.render(new_state)
            results_panel.render(new_state)
            training_path_panel.render(new_state)
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
