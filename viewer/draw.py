"""
Drawing functions for TabulaDrone Episode Viewer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, Optional


# Color schemes
WEAPON_COLORS = {
    "light": "#3498db",    # Blue
    "medium": "#2ecc71",   # Green
    "heavy": "#e74c3c",    # Red
    "unknown": "#95a5a6",  # Gray
}

TARGET_COLORS = {
    "A": "#9b59b6",  # Purple
    "B": "#e67e22",  # Orange
    "C": "#7f8c8d",  # Gray
    "unknown": "#bdc3c7",  # Light gray
}


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
        class_type = target["class_type"]
        color = TARGET_COLORS.get(class_type, TARGET_COLORS["unknown"])
        
        ax.scatter(
            x, y,
            s=200,
            c=color,
            marker="o",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )
        ax.annotate(
            target["id"],
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
        )
    
    # Draw drones (triangles)
    for drone in drones:
        x, y = drone["position"]
        weapon_type = drone["weapon_type"]
        color = WEAPON_COLORS.get(weapon_type, WEAPON_COLORS["unknown"])
        
        ax.scatter(
            x, y,
            s=250,
            c=color,
            marker="^",
            edgecolors="black",
            linewidths=1,
            zorder=11,
        )
        ax.annotate(
            drone["id"],
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
        )
    
    # Labels
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("TabulaDrone - Initial World State")
    
    # Build legend
    legend_handles = []
    
    # Drone legend entries
    for weapon_type, color in WEAPON_COLORS.items():
        if weapon_type == "unknown":
            continue
        if any(d["weapon_type"] == weapon_type for d in drones):
            handle = mpatches.Patch(color=color, label=f"Drone ({weapon_type})")
            legend_handles.append(handle)
    
    # Target legend entries
    for class_type, color in TARGET_COLORS.items():
        if class_type == "unknown":
            continue
        if any(t["class_type"] == class_type for t in targets):
            handle = mpatches.Patch(color=color, label=f"Target (class {class_type})")
            legend_handles.append(handle)
    
    ax.legend(handles=legend_handles, loc="upper right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()
