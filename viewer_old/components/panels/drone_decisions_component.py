"""
Drone decisions component for the FalconX viewer.

This module provides a component for rendering drone events and decisions in a scrollable text area.
"""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
from sim.events import EventType
from viewer.ui.scrollable_text_area import ScrollableTextArea
from viewer.components.base.base_component import BaseComponent


class DroneDecisionsComponent(BaseComponent):
    """
    Component for rendering the drone decisions tab.

    This component displays drone events and decisions in a scrollable text area.
    """
    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        """
        Initialize the drone decisions component.

        Args:
            fig: The matplotlib figure to draw on.
            ax: The matplotlib axes to draw on.
        """
        super().__init__(fig, ax)
        
        # Create a scrollable text area for displaying events
        self.text_area = ScrollableTextArea(
            ax, 
            x=0.02,  # X position (left margin)
            y=0.02,  # Y position (bottom margin)
            w=1,  # Width as proportion of axes
            h=0.96,  # Height as proportion of axes
            fontsize=8
        )
        
        # Buffer for events
        self.event_buffer = []  # Store all formatted events
        
        # Event type whitelist - only these events will be displayed
        self.event_whitelist = [
            EventType.MISSION_START.name,
            EventType.GO_TO_RESUPPLY.name,
            EventType.ARRIVED_AT_RESUPPLY.name,
            EventType.PICK_TARGET.name,
            EventType.MISSILE_FIRED_HIT.name,
            EventType.MISSILE_FIRED_MISS.name,
            EventType.BATTERY_CRITICAL.name,
            EventType.LOADOUT.name,
            EventType.BATTERY_DEPLETED.name,
            EventType.NO_MISSILES_AVAILABLE.name,
            EventType.TARGET_IN_MISSILE_RANGE.name,
            EventType.MISSION_COMPLETED.name,
            # EventType.MOVED.name
        ]
        
        # Flag to track if there are new events to display
        self.has_new_events = False
        
        # Track current visibility state
        self._is_visible = self.is_visible()
        
        # Set initial visibility of the text area based on axes visibility
        self.text_area.set_visible(self._is_visible)

    def should_preserve_state(self) -> bool:
        """
        Override to indicate that this component should preserve its state when hidden.
        
        Returns:
            True to preserve event history across tab switches.
        """
        return True
        
    def is_visible(self) -> bool:
        """
        Override to check if the component is currently visible.
        
        Returns:
            True if the component's axes are visible, False otherwise.
        """
        return self.ax.get_visible()

    def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process the snapshot data to extract and buffer drone events.
        This method is always called regardless of component visibility.
        
        Args:
            data: The snapshot data to process.
        """
        # Extract events from the snapshot data
        new_events = self._extract_drone_events(data)
        
        # Add new events to the buffer (if any)
        if new_events:
            self.event_buffer.extend(new_events)
            self.has_new_events = True

    def render_display(self) -> None:
        """
        Update the visual display based on the current event buffer.
        This method is only called when the component is visible.
        """
        # Check if visibility has changed
        current_visibility = self.is_visible()
        
        # If becoming visible
        if current_visibility and not self._is_visible:
            # Create a new text area if needed
            if not hasattr(self, 'text_area') or self.text_area is None:
                self.text_area = ScrollableTextArea(
                    self.ax, 
                    x=0.02,
                    y=0.02,
                    w=1,
                    h=0.96,
                    fontsize=8
                )
            
            # Set the text area to visible
            self.text_area.set_visible(True)
            
            # Mark that we have events to display
            self.has_new_events = True
        
        # If becoming invisible
        elif not current_visibility and self._is_visible:
            # Completely remove the text area
            if hasattr(self, 'text_area') and self.text_area is not None:
                self.text_area.remove()
        
        # Update visibility state
        self._is_visible = current_visibility
        
        # Only update the display if the component is visible and we have new events
        if self._is_visible and self.has_new_events and hasattr(self, 'text_area') and self.text_area is not None:
            # Join all events into a single string and set to text area
            events_text = "\n".join(self.event_buffer)
            self.text_area.set_text(events_text)
            self.has_new_events = False

    def render(self, data: Dict[str, Any]) -> None:
        """
        Process data and render the component if visible.
        
        Args:
            data: The snapshot data to render.
        """
        # Always process data to extract and buffer events
        self.process_data(data)
        
        # Check if visibility has changed and update display accordingly
        self.render_display()

    def clear(self) -> None:
        """
        Clear only the visual elements but preserve the event buffer.
        This allows the component to maintain its history across tab switches.
        """
        # Only clear the axes, not the event buffer
        self.ax.clear()
        
        # Completely remove the text area
        if hasattr(self, 'text_area') and self.text_area is not None:
            self.text_area.remove()
        
        # Update visibility state
        self._is_visible = False
        
        # Mark that we have events to display when the component becomes visible again
        if self.event_buffer:
            self.has_new_events = True
            
    def reset(self) -> None:
        """
        Fully reset the component, clearing both visual elements and event buffer.
        Use this method when you want to completely reset the component's state.
        """
        # Clear the axes
        self.ax.clear()
        
        # Clear the event buffer
        self.event_buffer = []
        self.has_new_events = False
        
        # Completely remove the text area
        if hasattr(self, 'text_area') and self.text_area is not None:
            self.text_area.remove()
        
        # Update visibility state
        self._is_visible = False

    def _extract_drone_events(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract drone events from the snapshot data.
        
        Args:
            data: The snapshot data to extract events from.
            
        Returns:
            A list of formatted event strings.
        """
        formatted_events = []
        
        # Check if world exists in the data
        if "world" not in data:
            return formatted_events
            
        # Check if drones exist in the world data
        if "drones" not in data["world"]:
            return formatted_events
            
        # Get drones data
        drones = data["world"]["drones"]
        
        # Check if items exist in drones
        if "items" not in drones:
            return formatted_events

        # Process each drone
        for i, drone in enumerate(drones["items"]):
            drone_id = drone.get("id", "unknown")
            
            # Check if events exist for this drone
            if "events" not in drone:
                continue

            # Process each event - events are now stored as an array
            for event in drone["events"]:
                # Get event type from the event data
                event_type = event.get("event_type", "UNKNOWN")
                
                # Skip events not in the whitelist
                if event_type not in self.event_whitelist:
                    continue
                    
                event_tick = event.get("tick", 0)
                event_time = event_tick  # Tick is equivalent to simulation time in seconds
                
                # Generic format for events
                event_str = f"[{event_time:.1f}s] {drone_id}: {event_type}"

                formatted_events.append(event_str)
        
        # Sort events by time
        formatted_events.sort(key=lambda x: float(x.split(']')[0][1:].split('s')[0]))
        
        return formatted_events
