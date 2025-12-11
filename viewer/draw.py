"""
Drawing functions for TabulaDrone Episode Viewer.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


# Color schemes
WEAPON_COLORS = {
    "light": "#3498db",    # Blue
    "medium": "#2ecc71",   # Green
    "heavy": "#e74c3c",    # Red
    "unknown": "#95a5a6",  # Gray
}

TARGET_COLOR = "#9370DB"  # Medium Purple (all targets same color)


def plot_initial_world(state: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plot the initial world state.
    
    Args:
        state: Initial state dict from extract_initial_state()
        save_path: If provided, save figure to this path instead of displaying
    """
    world_size = state["world_size"]
    drones = state["drones"]
    targets = state["targets"]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Set world bounds
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_aspect("equal")
    
    # Draw grid
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Draw world border
    border = plt.Rectangle(
        (0, 0), world_size[0], world_size[1],
        fill=False, edgecolor="black", linewidth=2
    )
    ax.add_patch(border)
    
    # Draw targets (circles)
    for target in targets:
        x, y = target["position"]
        
        ax.scatter(
            x, y,
            s=40,
            c=TARGET_COLOR,
            marker="o",
            zorder=10,
        )
    
    # Draw drones (triangles)
    for drone in drones:
        x, y = drone["position"]
        weapon_type = drone["weapon_type"]
        color = WEAPON_COLORS.get(weapon_type, WEAPON_COLORS["unknown"])
        
        ax.scatter(
            x, y,
            s=35,
            c=color,
            marker="^",
            zorder=11,
        )
    
    # Labels
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("TabulaDrone - Initial World State")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()
