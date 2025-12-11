"""
Utility modules for the FalconX viewer.

This package contains utility functions and classes used by the viewer module.
"""

from viewer.utils.drawing_utils import (
    setup_plot_axes,
    draw_targets,
    draw_drones,
    draw_resupply_station,
    draw_spawn_regions,
    draw_range_circles,
    draw_event_range_circles,
    set_simulation_time_title
)

from viewer.utils.simulator_helper import (
    get_world_bounds,
    get_targets,
    get_drones,
    get_resupply_station,
    get_spawn_regions,
    get_simulation_time,
    get_wall_clock,
    get_missile_types,
    get_target_types
)

__all__ = [
    # Drawing utilities
    'setup_plot_axes',
    'draw_targets',
    'draw_drones',
    'draw_resupply_station',
    'draw_spawn_regions',
    'draw_range_circles',
    'draw_event_range_circles',
    'set_simulation_time_title',
    
    # Simulator helper utilities
    'get_world_bounds',
    'get_targets',
    'get_drones',
    'get_resupply_station',
    'get_spawn_regions',
    'get_simulation_time',
    'get_wall_clock',
    'get_missile_types',
    'get_target_types'
]
