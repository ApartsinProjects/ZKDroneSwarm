"""
Drawing utilities for the FalconX snapshot viewer.

This module provides functions for visualizing snapshot data using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.table import Table
from matplotlib.widgets import Button
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math
import os

from viewer.animation import SnapshotAnimator
from viewer.utils import drawing_utils, simulator_helper
from viewer.components.base import BaseComponent
from viewer.components.containers import TabContainer
from viewer.components.panels import MapComponent, OverviewComponent, DroneDecisionsComponent

# Global variable to store animation
_global_animation = None


def plot_overview(ax: plt.Axes, snap: Dict[str, Any]) -> None:
    """
    Plot an overview panel with world, targets, and missiles information.
    
    Args:
        ax: The matplotlib axes to draw on.
        snap: The snapshot data dictionary.
    """
    # Extract data
    world = snap["world"]
    catalogs = snap["catalogs"]
    summary = snap["summary"]
    status_counts = summary["target_status_counts"]
    target_type_counts = summary.get("target_type_counts", {})
    drone_type_counts = summary.get("drone_type_counts", {})
    
    # Get target count
    target_count = 0
    if "target_spawn_region" in world and "target_instances" in world["target_spawn_region"]:
        target_instances = world["target_spawn_region"]["target_instances"]
        target_count = target_instances["count"]
    
    # Get drone count
    drone_count = 0
    if "drones" in world:
        drone_count = world["drones"].get("count", 0)
    
    # Set title
    # ax.set_title("Initial Overview", fontweight='bold')
    ax.axis('off')  # Hide axes
    
    # World and targets info
    world_text = (
        f"World:\n"
        f"  Size: {world['bounds_m']['width']:.1f} × {world['bounds_m']['height']:.1f} m\n"
    )
    
    # Add dt_s and mission_time_limit_s if available
    if "dt_s" in world:
        world_text += f"  Time step: {world['dt_s']:.2f} s\n"
    if "mission_time_limit_s" in world:
        world_text += f"  Mission limit: {world['mission_time_limit_s']:.1f} s\n"
    
    # Targets section in the requested format
    world_text += f"\nTargets:\n"
    world_text += f"  Total: {target_count}\n"
    world_text += f"  Details:\n"
    
    # Add target type counts
    for target_type in sorted(target_type_counts.keys()):
        world_text += f"    {target_type}: {target_type_counts.get(target_type, 0)}\n"
    
    # Add Drones section
    world_text += f"\nDrones:\n"
    world_text += f"  Total: {drone_count}\n"
    world_text += f"  Details:\n"
    
    # Add drone type counts
    for drone_type in sorted(drone_type_counts.keys()):
        world_text += f"    {drone_type}: {drone_type_counts.get(drone_type, 0)}\n"
    
    # Add Resupply Station info if available
    if "resupply_station" in world:
        station = world["resupply_station"]
        if "missiles" in station:
            stock = station["missiles"]
            total_missiles = sum(stock.values())
            
            world_text += f"\nResupply Station:\n"
            world_text += f"  Missiles:\n"
            world_text += f"    Total: {total_missiles}\n"
            
            for missile_type, count in sorted(stock.items()):
                world_text += f"    {missile_type}: {count}\n"
    
    # Move text slightly inside the axes so it won't be clipped
    ax.text(0.0, 0.99, world_text, va='top', family='monospace', fontsize=9, transform=ax.transAxes)

    # Missiles table from catalogs
    missile_types = catalogs["missile_types"]
    missile_data = []
    for missile_type in sorted(missile_types.keys()):
        missile = missile_types[missile_type]
        row_data = [
            missile_type,
            f"{missile['range_m']:.1f}",
            f"{missile['accuracy']:.2f}",
            f"{missile['weight_kg']:.2f}"
        ]
        
        # Add damage score if available
        if 'damage_score' in missile:
            row_data.append(f"{missile['damage_score']:.1f}")
        else:
            row_data.append("N/A")
            
        missile_data.append(row_data)

    # Create column headers including damage score
    col_labels = ["Type", "Range", "Accuracy", "Weight", "Damage"]

    # Create a wider table that fills the overview pane - adjust position to reduce gap
    table_missile_y_pos = 0.3  # vertical placement within the right pane - moved up to reduce gap
    table = ax.table(
        cellText=missile_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0.00, table_missile_y_pos, 1, 0.1]  # <-- full width of the right pane
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.8, 2.0)  # slightly less than before so it balances with the wider bbox

    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')

    # Add table title
    ax.text(0.00, table_missile_y_pos + 0.11, "Missiles:", va='bottom', fontweight='bold', fontsize=9, transform=ax.transAxes)
    
    # Add Target Table - adjust position to reduce gap
    target_data = []
    target_types = catalogs["target_types"]
    
    for target_type in sorted(target_types.keys()):
        target = target_types[target_type]
        count = target_type_counts.get(target_type, 0)
        
        row_data = [
            target_type,
            target.get("resilience", "N/A"),
            f"{target.get('resilience_score', 0):.1f}",
            # f"{count}"
        ]
        
        target_data.append(row_data)
    
    # Create column headers for target table
    target_col_labels = ["Type", "Resilience", "Score", "Count"]
    
    # Create target table - adjust position to reduce gap
    target_table_y_pos = table_missile_y_pos - 0.16 # vertical placement below missiles table - moved up to reduce gap
    target_table = ax.table(
        cellText=target_data,
        colLabels=target_col_labels,
        loc='center',
        cellLoc='center',
        bbox=[0.00, target_table_y_pos, 1.00, 0.1]  # full width of the right pane
    )
    
    # Style the target table
    target_table.auto_set_font_size(False)
    target_table.set_fontsize(7)
    target_table.scale(1.8, 2.0)
    
    for key, cell in target_table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')
    
    # Add target table title
    ax.text(0.00, target_table_y_pos + 0.11, "Targets:", va='bottom', fontweight='bold', fontsize=9, transform=ax.transAxes)
    
    # Add Drone Table
    drone_data = []
    
    # Get drone profiles from world if available
    if "drones" in world and "items" in world["drones"]:
        drone_items = world["drones"]["items"]
        drone_types = {}
        
        # Group drones by type to get representative data
        for drone in drone_items:
            drone_type = drone.get("type", "")
            if drone_type not in drone_types:
                drone_types[drone_type] = {
                    "speed_kmh": drone.get("speed_kmh", 0),
                    "detection_range_m": drone.get("detection_range_m", 0),
                    "max_payload_kg": drone.get("max_payload_kg", 0),
                    "battery_capacity_max": drone.get("battery_capacity_max", 0),
                    "count": 1
                }
            else:
                drone_types[drone_type]["count"] += 1
        
        # Create table data
        for drone_type, data in sorted(drone_types.items()):
            row_data = [
                drone_type,
                f"{data['speed_kmh']:.0f}",
                f"{data['battery_capacity_max']:.0f}",
                f"{data['max_payload_kg']:.0f}",
                f"{data['count']}"
            ]
            drone_data.append(row_data)
    
    # Create column headers for drone table
    drone_col_labels = ["Type", "Speed", "Battery", "Max Weight", "Count"]
    
    # Create drone table - position below target table
    drone_table_y_pos = target_table_y_pos - 0.16  # vertical placement below target table
    if drone_data:  # Only create table if we have drone data
        drone_table = ax.table(
            cellText=drone_data,
            colLabels=drone_col_labels,
            loc='center',
            cellLoc='center',
            bbox=[0.00, drone_table_y_pos, 1.25, 0.1]  # full width of the right pane
        )
        
        # Style the drone table
        drone_table.auto_set_font_size(False)
        drone_table.set_fontsize(7)
        drone_table.scale(1.8, 2.0)
        
        for key, cell in drone_table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0')
        
        # Add drone table title
        ax.text(0.00, drone_table_y_pos + 0.11, "Drones:", va='bottom', fontweight='bold', fontsize=9, transform=ax.transAxes)


def plot_map(
    ax: plt.Axes,
    snap: Dict[str, Any],
    show_ranges: bool = False,
    ranges_origin: Tuple[float, float] = (0.0, 0.0),
    show_spawn_regions: bool = False,
    show_drone_detection: bool = False
) -> None:
    """
    Plot just the map portion of the visualization.
    
    Args:
        ax: The matplotlib axes to draw on.
        snap: The snapshot data dictionary.
        show_ranges: Whether to show missile range circles.
        ranges_origin: Origin coordinates for range circles.
        show_spawn_regions: Whether to show target spawn regions.
        show_drone_detection: Whether to show drone detection ranges.
    """
    # Get data from snapshot using simulator_helper
    bounds = simulator_helper.get_world_bounds(snap)
    targets = simulator_helper.get_targets(snap)
    drones = simulator_helper.get_drones(snap)
    resupply_station = simulator_helper.get_resupply_station(snap)
    spawn_regions = simulator_helper.get_spawn_regions(snap)
    simulation_time = simulator_helper.get_simulation_time(snap)
    missile_types = simulator_helper.get_missile_types(snap)
    
    # Setup plot axes
    drawing_utils.setup_plot_axes(ax, bounds)
    
    # Set title
    meta = snap["meta"]
    title = f"World Snapshot - Seed: {meta['seed']}"
    ax.set_title(title)
    
    # Draw spawn regions if requested
    if show_spawn_regions and spawn_regions:
        drawing_utils.draw_spawn_regions(ax, spawn_regions, bounds)
    
    # Draw targets
    drawing_utils.draw_targets(ax, targets)
    
    # Draw drones
    drawing_utils.draw_drones(ax, drones, show_detection=show_drone_detection)
    
    # Draw resupply station
    if resupply_station:
        drawing_utils.draw_resupply_station(ax, resupply_station)
    
    # Draw missile range circles if requested
    # if show_ranges and missile_types:
    drawing_utils.draw_event_range_circles(ax, drones, missile_types)
    
    # Add legend
    legend = False
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc='upper right',
            fontsize=8,
            framealpha=0.7
        )


def plot_world(
    snap: Dict[str, Any],
    save_path: Optional[str] = None,
    dpi: int = 150,
    show_ranges: bool = False,
    ranges_origin: Tuple[float, float] = (0.0, 0.0),
    show_spawn_regions: bool = False,
    show_drone_detection: bool = False
) -> None:
    """
    Plot a visualization of the world snapshot with a main map view and tabbed secondary views.
    
    Args:
        snap: The snapshot data dictionary.
        save_path: Optional path to save the plot as an image.
        dpi: DPI for the saved image (if save_path is provided).
        show_ranges: Whether to show missile range circles.
        ranges_origin: Origin coordinates for range circles.
        show_spawn_regions: Whether to show target spawn regions.
        show_drone_detection: Whether to show drone detection ranges.
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=(13, 8))
    
    # Store the snapshot in the figure's user_data for later use by the animation
    if not hasattr(fig, 'user_data'):
        fig.user_data = {}
    fig.user_data['snapshot'] = snap

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.2)  # Reduced wspace to minimize gap between panels

    # Set the window title to FalconX
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title('FalconX')

    # Create main plot and info panel
    ax_main = fig.add_subplot(gs[0])
    ax_tabs = fig.add_subplot(gs[1])
    
    # Create map component and render it
    map_component = MapComponent(fig, ax_main)
    map_component.show_ranges = show_ranges
    map_component.ranges_origin = ranges_origin
    map_component.show_spawn_regions = show_spawn_regions
    map_component.show_drone_detection = show_drone_detection
    map_component.render(snap)
    
    # Create tab container for secondary views
    tab_container = TabContainer(fig, ax_tabs)
    
    # Create overview component and add it as a tab
    overview_ax = fig.add_subplot(gs[1])
    overview_ax.set_visible(False)  # Initially hidden, TabContainer will show it
    overview_component = OverviewComponent(fig, overview_ax)
    
    # Create drone decisions component and add it as a tab
    drone_decisions_ax = fig.add_subplot(gs[1])
    drone_decisions_ax.set_visible(False)  # Initially hidden, TabContainer will show it
    drone_decisions_component = DroneDecisionsComponent(fig, drone_decisions_ax)
    
    # Store components in figure's user_data for access during animation
    fig.user_data['overview_component'] = overview_component
    fig.user_data['drone_decisions_component'] = drone_decisions_component

    # Add tabs to the container
    tab_container.add_tab("Mission Setup", overview_component)
    tab_container.add_tab("Drone Decisions", drone_decisions_component)

    # Process data for all components
    overview_component.process_data(snap)
    drone_decisions_component.process_data(snap)

    # Render only the visible components
    if overview_component.is_visible():
        overview_component.render_display()
    if drone_decisions_component.is_visible():
        drone_decisions_component.render_display()

    # Add play button for animation
    if not save_path:  # Only add button when displaying interactively
        # Create a new axes for the button - position it at the bottom of the figure
        play_button_ax = fig.add_axes((0.25, 0.01, 0.1, 0.04))  # [left, bottom, width, height]
        play_button = Button(play_button_ax, 'Play')
        play_button.on_clicked(play_animation)
        
        # Add pause button between play and close buttons
        pause_button_ax = fig.add_axes((0.37, 0.01, 0.1, 0.04))  # [left, bottom, width, height]
        pause_button = Button(pause_button_ax, 'Pause')
        pause_button.on_clicked(pause_animation)
        
        # Store pause button in figure's user_data for access in pause_animation function
        if not hasattr(fig, 'user_data'):
            fig.user_data = {}
        fig.user_data['pause_button'] = pause_button
        
        # Add close button (moved to the right to make room for pause button)
        close_button_ax = fig.add_axes((0.49, 0.01, 0.1, 0.04))  # [left, bottom, width, height]
        close_button = Button(close_button_ax, 'Close')
        close_button.on_clicked(close_viewer)
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def play_animation(event):
    """
    Function to start animation playback when the play button is clicked.
    
    This function creates a SnapshotAnimator instance and attempts to load
    snapshots from the 'out/snapshots' directory to animate them.
    
    Args:
        event: The button click event.
    """

    print("Play button clicked!")
    
    # Get the current figure and axes
    fig = plt.gcf()
    axes = fig.get_axes()
    
    # Find the main plot axes (the first one)
    if not axes:
        print("Error: No axes found in the current figure")
        return
    
    print(f"Found {len(axes)} axes in the figure")
    main_ax = axes[0]
    
    # Create animator instance
    animator = SnapshotAnimator(fig, main_ax)
    
    # Store animator in figure's user_data for access by pause button
    if not hasattr(fig, 'user_data'):
        fig.user_data = {}
    fig.user_data['animator'] = animator
    
    # Only try to load snapshots from the out/snapshots directory
    print("Checking for snapshots in out/snapshots directory...")
    if os.path.exists('out/snapshots'):
        print("out/snapshots directory exists")
        success = animator.load_snapshots('out/snapshots', 'snapshot_*.json')
        print(f"Load snapshots result: {success}, found {len(animator.snapshots)} snapshots")
        if success:
            print("Starting animation with snapshots from 'out/snapshots' directory")
            
            # Enable interactive mode
            plt.ion()
            
            # Create a custom update function that uses our component-based structure
            def update_function(snap):
                # Clear the current axes
                main_ax.clear()
                
                # Create a map component for the animation frame and use the new methods
                map_component = MapComponent(fig, main_ax)
                map_component.process_data(snap)
                map_component.render_display()
                
                # Get simulation time for the title
                simulation_time = simulator_helper.get_simulation_time(snap)
                simulation_wall_clock = simulator_helper.get_wall_clock(snap)
                
                # Set title with simulation time
                drawing_utils.set_simulation_time_title(ax=main_ax, time_s=simulation_time, wall_time_ms=simulation_wall_clock)
                
                # Always process data for all components regardless of visibility
                if hasattr(fig, 'user_data'):
                    # Process data for overview component
                    if 'overview_component' in fig.user_data:
                        fig.user_data['overview_component'].process_data(snap)
                        # Only render if visible
                        if fig.user_data['overview_component'].is_visible():
                            fig.user_data['overview_component'].render_display()
                    
                    # Process data for drone decisions component
                    if 'drone_decisions_component' in fig.user_data:
                        fig.user_data['drone_decisions_component'].process_data(snap)
                        # Only render if visible
                        if fig.user_data['drone_decisions_component'].is_visible():
                            fig.user_data['drone_decisions_component'].render_display()
                
                # Force redraw
                main_ax.figure.canvas.draw_idle()
            
            # Start the animation
            animator.start_animation(update_function)
            
            # Store animation in global variable to prevent garbage collection
            global _global_animation
            _global_animation = animator.animation
            
            # Force a draw to make sure the animation is visible
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            return
    else:
        print("out/snapshots directory does not exist")

    print("No snapshots found in out/snapshots directory.\nPlease generate snapshots first.")


def pause_animation(event):
    """
    Function to pause/resume the animation when the pause button is clicked.
    
    This function toggles the animation between paused and playing states.
    When paused, the animation will stop at the current frame.
    When resumed, the animation will continue from where it left off.
    
    Args:
        event: The button click event.
    """
    print("Pause button clicked!")
    
    # Get the current figure
    fig = plt.gcf()
    
    # Access the global animation variable
    global _global_animation
    
    # Check if animation exists
    if _global_animation and hasattr(_global_animation, 'event_source'):
        # Get the animator instance from the figure's user_data
        if hasattr(fig, 'user_data') and 'animator' in fig.user_data:
            animator = fig.user_data['animator']
            
            # Toggle pause state
            is_paused = animator.toggle_pause()
            
            # Update button text based on state
            if hasattr(fig, 'user_data') and 'pause_button' in fig.user_data:
                pause_button = fig.user_data['pause_button']
                pause_button.label.set_text('Resume' if is_paused else 'Pause')
                fig.canvas.draw_idle()
        else:
            print("No animator found in figure's user_data")
    else:
        print("No active animation to pause")


def close_viewer(event):
    """
    Function to close the viewer when the close button is clicked.
    
    Args:
        event: The button click event.
    """
    print("Close button clicked!")
    plt.close()


def draw_range_circles_niu(ax: plt.Axes, origin_xy: Tuple[float, float], missile_types: Dict[str, Dict[str, Any]]) -> None:
    """
    Draw missile range circles on the map.

    Args:
        ax: The matplotlib axes to draw on.
        origin_xy: The (x, y) coordinates of the origin point.
        missile_types: Dictionary of missile types with range information.
    """
    # Define colors for different missile types
    colors = {
        'A': '#ff7f0e',  # orange
        'B': '#1f77b4',  # blue
        'C': '#2ca02c',  # green
        'D': '#d62728',  # red
        'E': '#9467bd',  # purple
    }

    # Draw range circles for each missile type
    for missile_type, missile in missile_types.items():
        range_m = missile['range_m']
        color = colors.get(missile_type, '#7f7f7f')  # default to gray if type not in colors

        # Draw circle
        circle = Circle(
            origin_xy,
            radius=range_m,
            fill=False,
            edgecolor=color,
            linestyle='--',
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(circle)

        # Add label at the edge of the circle
        angle = np.random.uniform(0, 2 * np.pi)  # Random angle for label placement
        label_x = origin_xy[0] + range_m * np.cos(angle)
        label_y = origin_xy[1] + range_m * np.sin(angle)

        ax.text(
            label_x, label_y,
            f"{missile_type}: {range_m:.0f}m",
            color=color,
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
        )


def create_target_table_niu(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create a table of target data from a snapshot.

    Args:
        snapshot: Dictionary with snapshot data.

    Returns:
        List of dictionaries with target data for tabular display or export.
    """
    targets = snapshot["targets"]
    table_data = []

    for target_id, target in sorted(targets.items()):
        # Get status text
        if target.get("destroyed", False):
            status = "Destroyed"
        elif target.get("status", 0.0) > 0.5:
            status = "Damaged"
        else:
            status = "Intact"

        # Create table row
        row = {
            "ID": target_id,
            "X (m)": f"{target['pos_m'][0]:.1f}",
            "Y (m)": f"{target['pos_m'][1]:.1f}",
            "Type": target.get("type", "unknown"),
            "Status": status
        }

        # Add resilience information if available
        if "resilience" in target:
            row["Resilience"] = target["resilience"]

        if "resilience_score" in target:
            row["Resilience Score"] = f"{target['resilience_score']:.1f}"

        table_data.append(row)

    return table_data


def extract_target_data_niu(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract target data from a snapshot for tabular display or export.

    Args:
        snapshot: The snapshot data dictionary.

    Returns:
        List of dictionaries with target data for tabular display or export.
    """
    target_items = []
    if "world" in snapshot and "target_spawn_region" in snapshot["world"]:
        target_spawn_region = snapshot["world"]["target_spawn_region"]
        if "target_instances" in target_spawn_region and "items" in target_spawn_region["target_instances"]:
            target_items = target_spawn_region["target_instances"]["items"]

    table_data = []

    for target in target_items:
        target_id = target.get("id", "N/A")
        target_type = target.get("type", "N/A")
        x = target.get("x", 0.0)
        y = target.get("y", 0.0)
        status = target.get("status", 0.0)

        # Convert status to text
        status_text = "intact"
        if status == 1.0:
            status_text = "destroyed"
        elif 0.0 < status < 1.0:
            status_text = "damaged"

        # Create data row
        row_data = {
            "id": target_id,
            "type": target_type,
            "x": x,
            "y": y,
            "status": status_text,
            "status_value": status
        }

        # Add additional fields if present
        if "resilience" in target:
            row_data["resilience"] = target["resilience"]

        if "resilience_score" in target:
            row_data["resilience_score"] = target["resilience_score"]

        table_data.append(row_data)

    return table_data
