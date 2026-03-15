"""
Tab container component for the TabulaDrone viewer.

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
    def __init__(self, fig: plt.Figure, parent_ax: plt.Axes, tab_region: tuple):
        """
        Initialize the tab container.

        Args:
            fig: The matplotlib figure to draw on.
            parent_ax: The parent axes that will contain the tabs.
            tab_region: Tuple (x, y, width, height) defining the tab button region in figure coords.
        """
        self.fig = fig
        self.parent_ax = parent_ax
        self.tab_region = tab_region
        self.tabs = {}
        self.active_tab = None
        self.tab_buttons = {}
        self.tab_button_axes = {}

        self.parent_ax.axis('off')

    def add_tab(self, name: str, component: BaseComponent) -> None:
        """
        Add a new tab with the given component.

        Args:
            name: The name of the tab.
            component: The component to display in the tab.
        """
        self.tabs[name] = component

        button_width = 0.08
        button_height = 0.03
        button_spacing = 0.01

        region_x, region_y, region_w, region_h = self.tab_region
        
        num_tabs = len(self.tab_buttons)
        button_x = region_x + (button_width + button_spacing) * num_tabs
        button_y = region_y

        button_ax = self.fig.add_axes([button_x, button_y, button_width, button_height])
        button = Button(button_ax, name)

        button.ax.spines['top'].set_visible(False)
        button.ax.spines['right'].set_visible(False)
        button.ax.spines['bottom'].set_color('gray')
        button.ax.spines['left'].set_visible(False)

        button.on_clicked(lambda event, tab_name=name: self.switch_to(tab_name))
        
        self.tab_buttons[name] = button
        self.tab_button_axes[name] = button_ax
        
        if self.active_tab is None:
            self.switch_to(name)
        else:
            component.set_visible(False)
    
    def switch_to(self, tab_name: str) -> None:
        """
        Switch to the specified tab.
        
        Args:
            tab_name: The name of the tab to switch to.
        """
        if tab_name not in self.tabs:
            return
            
        for name, button in self.tab_buttons.items():
            if name == tab_name:
                button.color = 'lightblue'
            else:
                button.color = '0.85'
        
        if self.active_tab:
            current_component = self.tabs[self.active_tab]
            current_component.set_visible(False)
            self.fig.canvas.draw_idle()
        
        new_component = self.tabs[tab_name]
        self.active_tab = tab_name
        new_component.set_visible(True)
        
        if hasattr(new_component, 'render_display'):
            new_component.render_display()
        
        self.fig.canvas.draw_idle()
