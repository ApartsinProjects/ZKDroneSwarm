"""
Tab container component for the FalconX viewer.

This module provides a container for managing tabbed views in the viewer.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from viewer.components.base import BaseComponent


class TabContainer:
    """
    Container for managing tabbed views.

    This class provides functionality for creating and managing tabs
    that can contain different components.
    """
    def __init__(self, fig: plt.Figure, parent_ax: plt.Axes):
        """
        Initialize the tab container.

        Args:
            fig: The matplotlib figure to draw on.
            parent_ax: The parent axes that will contain the tabs.
        """
        self.fig = fig
        self.parent_ax = parent_ax
        self.tabs = {}  # Dictionary of tab name -> component
        self.active_tab = None
        self.tab_buttons = {}  # Dictionary of tab name -> button
        self.tab_button_axes = {}  # Dictionary of tab name -> button axes

        # Hide the parent axes - we'll use it only as a container
        self.parent_ax.axis('off')

    def add_tab(self, name: str, component: BaseComponent) -> None:
        """
        Add a new tab with the given component.

        Args:
            name: The name of the tab.
            component: The component to display in the tab.
        """
        self.tabs[name] = component

        # Create a button for the tab
        button_width = 0.1
        button_height = 0.04
        button_spacing = 0.02

        # Position tabs at the top of the secondary panel
        # Change these values to adjust tab positions
        button_y = 0.90  # Move tabs down slightly from the top

        # Calculate button position based on number of existing tabs
        # Start from the center of the parent axes
        num_tabs = len(self.tab_buttons)
        button_x = 0.65 + (button_width + button_spacing) * num_tabs

        # Create button axes
        button_ax = self.fig.add_axes([button_x, button_y, button_width, button_height])
        button = Button(button_ax, name)

        # Remove button borders by setting properties
        button.ax.spines['top'].set_visible(False)
        button.ax.spines['right'].set_visible(False)
        button.ax.spines['bottom'].set_color('gray')  # or any gray color you prefer
        button.ax.spines['left'].set_visible(False)

        button.on_clicked(lambda event, tab_name=name: self.switch_to(tab_name))
        
        self.tab_buttons[name] = button
        self.tab_button_axes[name] = button_ax
        
        # If this is the first tab, make it active
        if self.active_tab is None:
            self.switch_to(name)
        else:
            # Hide the component initially if it's not the active tab
            component.ax.set_visible(False)
    
    def switch_to(self, tab_name: str) -> None:
        """
        Switch to the specified tab.
        
        Args:
            tab_name: The name of the tab to switch to.
        """
        if tab_name in self.tabs:
            # Update button styles
            for name, button in self.tab_buttons.items():
                if name == tab_name:
                    button.color = 'lightblue'  # Highlight active tab
                else:
                    button.color = '0.85'  # Default color
            
            # Show the selected component
            if self.active_tab:
                # Get the current active component
                current_component = self.tabs[self.active_tab]
                
                # Only clear the component if it doesn't need to preserve state
                if not hasattr(current_component, 'should_preserve_state') or not current_component.should_preserve_state():
                    current_component.clear()
                
                # Hide current tab's component
                current_component.ax.set_visible(False)
                
                # Force a redraw to ensure the component is properly hidden
                self.fig.canvas.draw_idle()
            
            # Get the new component to show
            new_component = self.tabs[tab_name]
            
            # Show new tab's component
            self.active_tab = tab_name
            new_component.ax.set_visible(True)
            
            # If the component has a render_display method, call it to ensure
            # it's properly rendered when becoming visible
            if hasattr(new_component, 'render_display'):
                new_component.render_display()
            
            # Force a redraw to ensure the component is properly shown
            self.fig.canvas.draw_idle()
