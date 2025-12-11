"""
Animation module for the FalconX snapshot viewer.

This module provides functionality for animating simulation snapshots.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import json
import os

from viewer.utils import drawing_utils, simulator_helper


class SnapshotAnimator:
    """
    Class for animating a series of snapshots.
    
    This class handles loading multiple snapshots and animating transitions between them.
    """
    
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the animator.
        
        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        self.fig = fig
        self.ax = ax
        self.snapshots = []
        self.animation = None
        self.frame_interval = 200  # milliseconds
        self.update_func = None  # Store the update function for reuse
        self.paused = False  # Track whether animation is paused
        self.current_frame = 0  # Track current frame index
    
    def load_snapshots(self, snapshot_dir: str, pattern: str = "snapshot_*.json") -> bool:
        """
        Load snapshots from a directory.
        
        Args:
            snapshot_dir: Directory containing snapshot files.
            pattern: Glob pattern to match snapshot files.
            
        Returns:
            True if snapshots were loaded successfully, False otherwise.
        """
        import glob
        
        # Find all matching snapshot files - don't sort here
        snapshot_files = glob.glob(os.path.join(snapshot_dir, pattern))
        
        if not snapshot_files:
            print(f"No snapshots found matching pattern {pattern} in {snapshot_dir}")
            return False
        
        # Load each snapshot
        self.snapshots = []
        valid_snapshots = []
        
        for snapshot_file in snapshot_files:
            try:
                with open(snapshot_file, 'r') as f:
                    snapshot = json.load(f)
                    
                    # Validate snapshot structure
                    if self._validate_snapshot(snapshot, snapshot_file):
                        valid_snapshots.append(snapshot)
                    else:
                        print(f"Skipping invalid snapshot: {snapshot_file}")
            except Exception as e:
                print(f"Error loading snapshot {snapshot_file}: {e}")
        
        if not valid_snapshots:
            print("No valid snapshots found")
            return False
            
        # Sort snapshots by simulation time (meta.t_s)
        print(f"Sorting {len(valid_snapshots)} snapshots by simulation time")
        valid_snapshots.sort(key=lambda snap: snap.get("meta", {}).get("t_s", 0.0))
        
        # Print simulation times for debugging
        times = [snap.get("meta", {}).get("t_s", 0.0) for snap in valid_snapshots]
        # print(f"Simulation times: {times}")

        self.snapshots = valid_snapshots
        return len(self.snapshots) > 0
        
    def _validate_snapshot(self, snapshot: Dict[str, Any], filename: str) -> bool:
        """
        Validate that a snapshot has the required structure.
        
        Args:
            snapshot: The snapshot data to validate.
            filename: The filename for error reporting.
            
        Returns:
            True if the snapshot is valid, False otherwise.
        """
        # Check for meta.t_s (simulation time)
        if "meta" not in snapshot or "t_s" not in snapshot.get("meta", {}):
            print(f"Warning: Snapshot {filename} is missing meta.t_s (simulation time)")
            # Add a default meta.t_s if missing
            if "meta" not in snapshot:
                snapshot["meta"] = {}
            if "t_s" not in snapshot["meta"]:
                # Try to extract time from filename (e.g., snapshot_5.json -> 5.0)
                try:
                    import re
                    time_match = re.search(r'snapshot_(\d+)', filename)
                    if time_match:
                        snapshot["meta"]["t_s"] = float(time_match.group(1))
                        print(f"  Extracted time {snapshot['meta']['t_s']} from filename")
                    else:
                        snapshot["meta"]["t_s"] = 0.0
                except:
                    snapshot["meta"]["t_s"] = 0.0
        
        # Check for world structure
        if "world" not in snapshot:
            print(f"Error: Snapshot {filename} is missing 'world' data")
            snapshot["world"] = {}
            
        # Ensure drones structure exists
        if "drones" not in snapshot["world"]:
            print(f"Warning: Snapshot {filename} is missing 'world.drones'")
            snapshot["world"]["drones"] = {"items": []}
        elif "items" not in snapshot["world"]["drones"]:
            print(f"Warning: Snapshot {filename} is missing 'world.drones.items'")
            snapshot["world"]["drones"]["items"] = []
            
        # Ensure other required structures exist
        if "bounds_m" not in snapshot["world"]:
            snapshot["world"]["bounds_m"] = {"width": 1000.0, "height": 1000.0}
            
        return True  # Return True even if we had to fix things

    def start_animation(self, update_func: Callable[[Dict[str, Any]], None]) -> None:
        """
        Start the animation.
        
        Args:
            update_func: Function to update the plot with a snapshot.
        """
        print("Starting animation...")
        if not self.snapshots:
            print("No snapshots to animate!")
            return
        
        print(f"Animation will use {len(self.snapshots)} snapshots")
        self.update_func = update_func  # Store the update function for reuse
        
        def animate(i):
            if i < len(self.snapshots):
                # Use i directly as the frame index
                snapshot = self.snapshots[i]
                self.update_func(snapshot)
            else:
                # This should not be reached with the new stopping logic,
                # but keeping as a fallback
                print("Animation complete")
                self.stop_animation()
            return []
        
        print("Creating FuncAnimation...")
        self.animation = animation.FuncAnimation(
            self.fig,
            animate,
            interval=self.frame_interval,
            blit=False,
            cache_frame_data=False,
            save_count=max(1000, len(self.snapshots))
        )
        print("Animation started!")

    def stop_animation(self) -> None:
        """Stop the animation."""
        if self.animation:
            # Stop the animation event source
            self.animation.event_source.stop()
    
    def toggle_pause(self) -> bool:
        """
        Toggle the animation between paused and playing states.
        
        Returns:
            bool: True if the animation is now paused, False if it's playing
        """
        if not self.animation:
            return False
            
        self.paused = not self.paused
        
        if self.paused:
            # Pause the animation
            self.animation.event_source.stop()
        else:
            # Resume the animation
            self.animation.event_source.start()
            
        return self.paused
    
    def set_frame_rate(self, frames_per_second: float) -> None:
        """
        Set the animation frame rate.
        
        Args:
            frames_per_second: Number of frames to show per second.
        """
        self.frame_interval = int(1000 / frames_per_second)
        if self.animation:
            self.animation.event_source.interval = self.frame_interval
