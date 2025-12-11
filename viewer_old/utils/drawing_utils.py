"""
Drawing utilities for the viewer.

This module provides functions for drawing various elements of the simulation on a matplotlib plot.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Polygon
from typing import Dict, Any, List, Tuple, Optional
from targets.state import TargetState
from drone.entity import Drone
import logging
from sim.events import EventType

def setup_plot_axes(ax: plt.Axes, bounds: Dict[str, float]) -> None:
    """
    Configure axes with proper limits, labels, and grid.
    
    Args:
        ax: The matplotlib axes to configure.
        bounds: Dictionary with width and height of the world.
    """
    width_m = bounds["width"]
    height_m = bounds["height"]
    
    # Set plot limits
    ax.set_xlim(0, width_m)
    ax.set_ylim(0, height_m)
    ax.set_aspect('equal')
    
    # Set labels and grid
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, linestyle='--', alpha=0.7)


def draw_targets(ax: plt.Axes, targets: List[TargetState]) -> None:
    """
    Draw targets with appropriate styling.
    
    Args:
        ax: The matplotlib axes to draw on.
        targets: List of TargetState objects.
        detailed: Whether to use detailed styling (for static view) or simple styling (for animation).
    """
    if not targets:
        return
    
    # Define colors for different target statuses
    target_colors = {
        'intact': '#9370DB',    # green
        'damaged': '#ff7f0e',   # orange
        'destroyed': '#d62728', # red
        'Default': '#7f7f7f'    # gray
    }
    
    # Group targets by status for legend
    target_groups = {}
    
    for target in targets:
        # Convert numeric status to text
        if target.destroyed:
            status_text = "destroyed"
        elif target.status > 0.0:
            status_text = "damaged"
        else:
            status_text = "intact"
        
        # Add to groups
        if status_text not in target_groups:
            target_groups[status_text] = []
        
        target_groups[status_text].append(target)
    
    # Plot targets by group
    for status, group_targets in target_groups.items():
        color = target_colors.get(status, target_colors["Default"])
        
        # Extract coordinates
        x_coords = [t.pos_m[0] for t in group_targets]
        y_coords = [t.pos_m[1] for t in group_targets]
        
        # Plot targets
        scatter_size = 50   # Slightly smaller for animation
        alpha = 0.9
        
        ax.scatter(
            x_coords, y_coords,
            c=color,
            marker='o',  # Always use circle for targets
            s=scatter_size,
            alpha=alpha,
            edgecolors='none',
            linewidths=1,
            label=f"Target ({status})"
        )
        
        # Add target IDs as labels
        for target in group_targets:
            label_offset = 37
            font_size = 7

            current_status = f"{round(target.status)}/{round(target.resilience_score)}"
            ax.text(
                target.pos_m[0], target.pos_m[1] + label_offset,
                f"{current_status}",
                fontsize=font_size,
                ha='center',
                va='bottom',
                fontweight='normal'
            )


def draw_drones(ax: plt.Axes, drones: List[Drone], 
                show_detection: bool = False) -> None:
    """
    Draw drones with appropriate styling.
    
    Args:
        ax: The matplotlib axes to draw on.
        drones: List of Drone objects.
        show_detection: Whether to show detection range circles.
    """
    if not drones:
        return
    
    # Define colors for different drone types
    drone_colors = {
        'scout': '#1f77b4',     # blue
        'striker_light': '#d62728',   # red
        'striker_heavy': '#2ca02c',  # green
        'Default': '#7f7f7f'      # gray
    }
    
    # Define markers for different drone statuses
    drone_status_markers = {
        'active': '^',      # triangle up for active
        'inactive': 'v',    # triangle down for inactive
        'Default': 's'      # square for default
    }
    
    # Group drones by type and status for legend
    drone_groups = {}
    
    for drone in drones:
        drone_type = drone.type
        status = drone.status
        
        # Create group key
        group_key = (drone_type, status)
        
        # Add to groups
        if group_key not in drone_groups:
            drone_groups[group_key] = []
        
        drone_groups[group_key].append(drone)
    
    # Plot drones by group
    for (drone_type, status), group_drones in drone_groups.items():
        color = drone_colors.get(drone_type, drone_colors["Default"])
        marker = drone_status_markers.get(status, drone_status_markers["Default"])
        
        # Extract coordinates
        x_coords = [d.pos_m[0] for d in group_drones]
        y_coords = [d.pos_m[1] for d in group_drones]
        
        # Plot drones
        scatter_size = 60
        alpha = 0.9
        
        ax.scatter(
            x_coords, y_coords,
            c=color,
            marker=marker,
            s=scatter_size,
            alpha=alpha,
            edgecolors='none',
            linewidths=1,
            label=f"{drone_type.capitalize()} ({status})"
        )
        
        # Add drone IDs as labels
        for drone in group_drones:
            label_offset = 30
            font_size = 5
            
            ax.text(
                drone.pos_m[0], drone.pos_m[1] + label_offset,
                str(drone.id.replace("drone-", "d")),
                fontsize=font_size,
                ha='center',
                va='bottom',
                fontweight='normal'
            )
            
            # Draw detection range circles if requested
            if show_detection and hasattr(drone, "detection_range_m"):
                detection_range = drone.__dict__["detection_range_m"]
                circle = Circle(
                    (drone.pos_m[0], drone.pos_m[1]),
                    radius=detection_range,
                    fill=False,
                    edgecolor=color,
                    linestyle=':',  # dotted line
                    linewidth=1.0,
                    alpha=0.4
                )
                ax.add_patch(circle)


def draw_resupply_station(ax: plt.Axes, resupply_station: Dict[str, Any], detailed: bool = True) -> None:
    """
    Draw the resupply station with appropriate styling and show available missiles.
    
    Args:
        ax: The matplotlib axes to draw on.
        resupply_station: Dictionary with resupply station data.
        detailed: Whether to use detailed styling (for static view) or simple styling (for animation).
    """
    if not resupply_station:
        return
    
    # Extract coordinates
    x = resupply_station["x"]
    y = resupply_station["y"]
    
    # Define marker size and style based on detail level
    marker_size = 200
    alpha = 0.7
    
    # Plot resupply station
    ax.scatter(
        x, y,
        c='#FFD700',  # gold
        marker='*',   # star for resupply station
        s=marker_size,
        alpha=alpha,
        edgecolors='#9370DB',
        linewidths=.5,
        label="Resupply Station" if detailed else None
    )
    
    # Add label
    font_size = 6
    label_offset = 85
    
    ax.text(
        x, y + label_offset,
        "Resupply Station",
        fontsize=font_size,
        ha='center',
        va='bottom',
        fontweight='normal'
    )
    
    # Display missiles if available
    if "missiles" in resupply_station and resupply_station["missiles"]:
        missiles = resupply_station["missiles"]
        
        # Create missile info text
        missile_text = "Missiles: "
        missile_items = []
        
        for missile_type, count in missiles.items():
            missile_items.append(f"{missile_type}: {count if count > 0 else '0'}")
        
        if missile_items:
            missile_text += ", ".join(missile_items)
            
            # Add missile info below the resupply station
            missile_label_offset = 70
            missile_font_size = 6
            
            # Draw small missile icons next to the station
            icon_offset = 130
            icon_size = 90
            
            for i, (missile_type, count) in enumerate(missiles.items()):
                # Calculate position (arrange in a circle around the station)
                angle = 2 * np.pi * i / len(missiles)
                icon_x = x + icon_offset * np.cos(angle)
                icon_y = y + icon_offset * np.sin(angle) - 150
                
                # Draw missile icon
                missile_color = '#ffb366' if missile_type == 'A' else '#66b2ff'  # orange for A, blue for B
                
                ax.scatter(
                    icon_x, icon_y,
                    c=missile_color,
                    marker='H',
                    s=icon_size,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                # Add count as text
                ax.text(
                    icon_x, icon_y - 75,
                    str(count if count > 0 else '0'),
                    fontsize=7,
                    ha='center',
                    va='top',
                    fontweight='bold',
                    color='black'
                )


def draw_spawn_regions(ax: plt.Axes, regions: Dict[str, Any], bounds: Dict[str, float]) -> None:
    """
    Draw spawn regions if requested.
    
    Args:
        ax: The matplotlib axes to draw on.
        regions: Dictionary with spawn region data.
        bounds: Dictionary with width and height of the world.
    """
    if not regions:
        return
    
    width_m = bounds["width"]
    height_m = bounds["height"]
    
    if "y_fraction" in regions:
        y_fractions = regions["y_fraction"]
        if isinstance(y_fractions, list) and len(y_fractions) == 2:
            y_min = y_fractions[0] * height_m
            y_max = y_fractions[1] * height_m
            
            # Draw spawn region as a rectangle
            rect = Rectangle(
                (0, y_min),
                width_m,
                y_max - y_min,
                fill=True,
                color='lightblue',
                alpha=0.2,
                label="Target Spawn Region"
            )
            ax.add_patch(rect)
            
            # Add text label
            ax.text(
                width_m / 2,
                (y_min + y_max) / 2,
                "Target Spawn Region",
                ha='center',
                va='center',
                fontsize=10,
                alpha=0.7
            )


def draw_range_circles(ax: plt.Axes, origin: Tuple[float, float], missile_types: Dict[str, Dict[str, Any]]) -> None:
    """
    Draw missile range circles.
    
    Args:
        ax: The matplotlib axes to draw on.
        origin: Origin coordinates for range circles (x, y).
        missile_types: Dictionary of missile types with their properties.
    """
    if not missile_types:
        return
    
    # Colors for different missile types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (missile_type, properties) in enumerate(missile_types.items()):
        if "range_m" in properties:
            range_m = properties["range_m"]
            color = colors[i % len(colors)]
            
            circle = Circle(
                origin,
                radius=range_m,
                fill=False,
                edgecolor=color,
                linestyle='-',
                linewidth=1.0,
                alpha=0.6,
                label=f"{missile_type} Range ({range_m}m)"
            )
            ax.add_patch(circle)


def draw_event_range_circles(ax: plt.Axes, drones: List[Drone], missile_types: Dict[str, Dict[str, Any]]) -> None:
    """
    Draw missile range circles for drones that have MISSILE_FIRED_HIT events.
    
    Args:
        ax: The matplotlib axes to draw on.
        drones: List of Drone objects.
        missile_types: Dictionary of missile types with their properties.
    """
    if not drones:
        return
    
    for drone in drones:
        # Check if drone has events attribute
        if not hasattr(drone, "events"):
            continue
            
        # Get events list
        events = drone.__dict__.get("events", [])
        
        # Process list-based events
        for event in events:
            event_type = event.get("event_type")
            
            # Check if this is a missile fired event (hit or miss)
            if event_type in [EventType.MISSILE_FIRED_HIT.name, EventType.MISSILE_FIRED_MISS.name]:
                # Get missile type from event if available
                missile_type = None
                if "missile_type" in event:
                    missile_type = event["missile_type"]
                
                # Get drone position
                drone_pos = (drone.pos_m[0], drone.pos_m[1])
                
                # If position is in the event data, use that instead
                if "position" in event:
                    pos = event["position"]
                    if isinstance(pos, (list, tuple)) and len(pos) == 2:
                        drone_pos = pos
                
                # Determine event-specific settings
                is_hit = event_type == EventType.MISSILE_FIRED_HIT.name
                event_label = "Hit" if is_hit else "Miss"
                circle_color = '#ff7f0e' if is_hit else '#808080'  # orange for hit, gray for miss
                circle_style = 'dashdot' if is_hit else 'dotted'
                circle_alpha = 0.7 if is_hit else 0.5
                
                # Log warning if missile type is None, otherwise draw circle
                if missile_type is None:
                    logging.warning(f"Missile {event_label.lower()} event without missile type for drone {drone.id}")
                else:
                    # Draw circle for specific missile type
                    if missile_type in missile_types and "range_m" in missile_types[missile_type]:
                        range_m = missile_types[missile_type]["range_m"]

                        circle = Circle(
                            drone_pos,
                            radius=range_m,
                            fill=False,
                            edgecolor=circle_color,
                            linestyle=circle_style,
                            linewidth=1.0,
                            alpha=circle_alpha,
                            label=f"{missile_type} {event_label} ({range_m}m)"
                        )
                        ax.add_patch(circle)


def set_simulation_time_title(ax: plt.Axes, time_s: float, wall_time_ms: float = 0) -> None:
    """
    Set the title with simulation time.
    
    Args:
        ax: The matplotlib axes to update.
        time_s: Current simulation time in seconds.
    """
    ax.set_title(f"FalconX Simulation - Time: {time_s:.2f}s, Wall Time: {wall_time_ms:.2f}ms")
