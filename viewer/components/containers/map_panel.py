"""
Map panel container for the TabulaDrone viewer.

This module provides a container that encapsulates the world map display.
"""

from typing import Dict, Any, Tuple, List, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from matplotlib.text import Text


class MapPanel:
    """
    Container for the map display panel.

    This class manages the world map visualization within a defined region.
    """
    def __init__(
        self,
        fig: plt.Figure,
        region: Tuple[float, float, float, float],
        state: Dict[str, Any],
        episode_files: Optional[List[str]] = None,
        current_index: int = 0,
        on_episode_change: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize the map panel.

        Args:
            fig: The matplotlib figure to draw on.
            region: Tuple (x, y, width, height) defining the panel region in figure coords.
            state: Initial state dict from extract_initial_state().
            episode_files: Optional list of episode file paths for navigation.
            current_index: Index of current episode in episode_files.
            on_episode_change: Optional callback when episode changes, receives new index.
        """
        self.fig = fig
        self.region = region
        self.state = state
        self.episode_files = episode_files
        self.current_index = current_index
        self.on_episode_change = on_episode_change
        
        self.map_ax: plt.Axes = self._create_map_axes()

        # self.border: Rectangle = self._draw_border()

        self._render_map(state)
        
        self.prev_ax: Optional[plt.Axes] = None
        self.next_ax: Optional[plt.Axes] = None
        self.prev_button: Optional[Button] = None
        self.next_button: Optional[Button] = None
        self.info_text: Optional[Text] = None
        
        self.steps: List[Dict[str, Any]] = state.get("steps", [])
        self.current_step: int = 0
        self.is_playing: bool = False
        self.timer: Optional[Any] = None
        self.play_ax: Optional[plt.Axes] = None
        self.pause_ax: Optional[plt.Axes] = None
        self.play_button: Optional[Button] = None
        self.pause_button: Optional[Button] = None
        self.step_text: Optional[Text] = None
        
        if episode_files is not None:
            self._create_nav_buttons()
        
        if self.steps:
            self._create_playback_buttons()
    
    def _create_map_axes(self) -> plt.Axes:
        """
        Create the map axes within the panel region.

        Returns:
            The matplotlib axes for the map.
        """
        x, y, width, height = self.region
        return self.fig.add_axes([x, y, width, height])
    
    def _draw_border(self) -> Rectangle:
        """
        Draw a light border around the panel region.

        Returns:
            The Rectangle patch representing the border.
        """
        x, y, width, height = self.region
        border = Rectangle(
            (x, y), width, height,
            fill=False,
            edgecolor='lightgray',
            linewidth=0.5,
            transform=self.fig.transFigure,
            zorder=0
        )
        self.fig.patches.append(border)
        return border
    
    def _create_nav_buttons(self) -> None:
        """
        Create navigation buttons (Previous/Next) below the map.
        """
        x, y, width, height = self.region
        
        button_width = 0.06
        button_height = 0.04
        button_spacing = 0.02
        button_y = 0.03
        
        panel_center = x + width / 2
        total_width = button_width * 2 + button_spacing
        left_button_x = panel_center - total_width / 2
        right_button_x = left_button_x + button_width + button_spacing
        
        self.prev_ax = self.fig.add_axes([left_button_x, button_y, button_width, button_height])
        self.prev_button = Button(self.prev_ax, "Previous")
        self.prev_button.on_clicked(self._on_prev)
        
        self.next_ax = self.fig.add_axes([right_button_x, button_y, button_width, button_height])
        self.next_button = Button(self.next_ax, "Next")
        self.next_button.on_clicked(self._on_next)
        
        info_text_y = button_y + button_height + 0.005
        self.info_text = self.fig.text(
            panel_center, info_text_y,
            f"Episode {self.current_index + 1} of {len(self.episode_files)}",
            ha='center', va='bottom', fontsize=10
        )
        
        self._update_button_states()
    
    def _create_playback_buttons(self) -> None:
        """
        Create playback buttons (Play/Pause) below the map.
        """
        x, y, width, height = self.region
        
        button_width = 0.05
        button_height = 0.04
        button_spacing = 0.01
        button_y = 0.03
        
        play_x = x + 0.02
        pause_x = play_x + button_width + button_spacing
        
        self.play_ax = self.fig.add_axes([play_x, button_y, button_width, button_height])
        self.play_button = Button(self.play_ax, "Play")
        self.play_button.on_clicked(self._on_play)
        
        self.pause_ax = self.fig.add_axes([pause_x, button_y, button_width, button_height])
        self.pause_button = Button(self.pause_ax, "Pause")
        self.pause_button.on_clicked(self._on_pause)
        
        step_text_x = play_x + button_width + button_spacing / 2
        step_text_y = button_y + button_height + 0.005
        self.step_text = self.fig.text(
            step_text_x, step_text_y,
            f"Step 0 of {len(self.steps)}",
            ha='center', va='bottom', fontsize=9
        )
    
    def _update_step_text(self) -> None:
        """
        Update the step indicator text.
        """
        if self.step_text and self.steps:
            self.step_text.set_text(f"Step {self.current_step} of {len(self.steps)}")
    
    def _on_play(self, event) -> None:
        """
        Handle Play button click. Starts timer-based playback.
        """
        if self.is_playing:
            return
        
        if self.current_step >= len(self.steps):
            self.current_step = 0
        
        self.is_playing = True
        self.timer = self.fig.canvas.new_timer(interval=300)
        self.timer.add_callback(self._timer_callback)
        self.timer.start()
    
    def _on_pause(self, event) -> None:
        """
        Handle Pause button click. Stops timer-based playback.
        """
        if not self.is_playing:
            return
        
        self.is_playing = False
        if self.timer:
            self.timer.stop()
            self.timer = None
    
    def _timer_callback(self) -> None:
        """
        Timer callback that advances playback by one step.
        """
        if self.current_step >= len(self.steps):
            self._on_pause(None)
            return
        
        self._advance_step()
    
    def _advance_step(self) -> None:
        """
        Advance to the next step and update the map.
        
        Reads target_hps from current step, updates target HP values,
        and refreshes the map. Destroyed targets (HP <= 0) get gray color.
        """
        if not self.steps or self.current_step >= len(self.steps):
            return
        
        step_data = self.steps[self.current_step]
        info = step_data.get("info", {})
        target_hps = info.get("target_hps", [])
        
        updated_targets = []
        for i, target in enumerate(self.state["targets"]):
            new_target = dict(target)
            if i < len(target_hps):
                new_target["hp"] = target_hps[i]
                if target_hps[i] <= 0:
                    new_target["class_type"] = "destroyed"
            updated_targets.append(new_target)
        
        updated_state = dict(self.state)
        updated_state["targets"] = updated_targets
        
        self.map_ax.clear()
        self._render_map(updated_state)
        self.fig.canvas.draw_idle()
        
        self.current_step += 1
        self._update_step_text()
    
    def _on_prev(self, event) -> None:
        """
        Handle Previous button click.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self._update_button_states()
            self._update_info_text()
            if self.on_episode_change:
                self.on_episode_change(self.current_index)
    
    def _on_next(self, event) -> None:
        """
        Handle Next button click.
        """
        if self.episode_files and self.current_index < len(self.episode_files) - 1:
            self.current_index += 1
            self._update_button_states()
            self._update_info_text()
            if self.on_episode_change:
                self.on_episode_change(self.current_index)
    
    def _update_button_states(self) -> None:
        """
        Update button visual states based on current index.
        """
        if not self.prev_button or not self.next_button or not self.episode_files:
            return
        
        if self.current_index == 0:
            self.prev_button.color = '0.85'
            self.prev_button.hovercolor = '0.85'
        else:
            self.prev_button.color = 'white'
            self.prev_button.hovercolor = '0.95'
        
        if self.current_index >= len(self.episode_files) - 1:
            self.next_button.color = '0.85'
            self.next_button.hovercolor = '0.85'
        else:
            self.next_button.color = 'white'
            self.next_button.hovercolor = '0.95'
        
        self.fig.canvas.draw_idle()
    
    def _update_info_text(self) -> None:
        """
        Update the episode info text.
        """
        if self.info_text and self.episode_files:
            self.info_text.set_text(f"Episode {self.current_index + 1} of {len(self.episode_files)}")
    
    def _render_map(self, state: Dict[str, Any]) -> None:
        """
        Render the map on the map axes.
        
        Uses local import to avoid circular dependency.

        Args:
            state: State dict to render.
        """
        from viewer.draw import render_map
        render_map(self.map_ax, state)
    
    def refresh(self, new_state: Dict[str, Any]) -> None:
        """
        Refresh the map with new state data.

        Args:
            new_state: New state dict from extract_initial_state().
        """
        if self.is_playing:
            self._on_pause(None)
        
        self.current_step = 0
        self.steps = new_state.get("steps", [])
        
        self.state = new_state
        self.map_ax.clear()
        self._render_map(new_state)
        self.fig.canvas.draw_idle()
    
    def update_position(self, new_region: Tuple[float, float, float, float]) -> None:
        """
        Update panel position for resize handling.

        Args:
            new_region: New (x, y, width, height) tuple in figure coords.
        """
        self.region = new_region
        x, y, width, height = new_region
        
        self.map_ax.set_position([x, y, width, height])
        
        # self.border.set_xy((x, y))
        # self.border.set_width(width)
        # self.border.set_height(height)
        
        if self.prev_ax and self.next_ax:
            button_width = 0.06
            button_height = 0.04
            button_spacing = 0.02
            button_y = 0.03
            
            panel_center = x + width / 2
            total_width = button_width * 2 + button_spacing
            left_button_x = panel_center - total_width / 2
            right_button_x = left_button_x + button_width + button_spacing
            
            self.prev_ax.set_position([left_button_x, button_y, button_width, button_height])
            self.next_ax.set_position([right_button_x, button_y, button_width, button_height])
            
            if self.info_text:
                info_text_y = button_y + button_height + 0.005
                self.info_text.set_position((panel_center, info_text_y))
