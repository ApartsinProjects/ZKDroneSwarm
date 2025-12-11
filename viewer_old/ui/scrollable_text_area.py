import textwrap
from typing import Optional, Tuple
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class ScrollableTextArea:
    def __init__(self, parent_ax, *, x=0.02, y=0.02, w=0.45, h=0.40, fontsize=10):
        """
        Create a fixed text area inside parent_ax (in axes coordinates),
        with wheel scrolling always enabled.
        """
        self.parent_ax = parent_ax
        self.fig = parent_ax.figure
        self.fontsize = fontsize
        self.fp = FontProperties(family='monospace', size=fontsize)
        
        # Store position and size parameters for recreation
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
        # State variables to preserve during removal/recreation
        self._lines = []        # logical (unwrapped) lines
        self._wrapped = []      # wrapped for viewport width
        self._scroll = 0        # how many lines up from bottom (0 = stick to bottom)
        self._is_visible = True # visibility state
        
        # Create the initial inset axes
        self._create_inset_axes()
        
        # Connect wheel for scrolling
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _create_inset_axes(self):
        """
        Create the inset axes and text object.
        """
        # A fixed-position child axes that acts as the "text viewport".
        self.ax = inset_axes(self.parent_ax, width=f"{self.w*100:.1f}%", height=f"{self.h*100:.1f}%",
                             bbox_to_anchor=(self.x, self.y, self.w, self.h), 
                             bbox_transform=self.parent_ax.transAxes,
                             borderpad=0)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

        # Background box (optional)
        self.bg = Rectangle((0, 0), 1, 1, transform=self.ax.transAxes, 
                           facecolor='white', alpha=0.1)
        self.ax.add_patch(self.bg)

        # The actual text artist; clipped to this axes
        self.text_obj = self.ax.text(
            0.02, 0.98, "", va='top', ha='left',
            fontsize=self.fontsize, family='monospace', wrap=False,
            transform=self.ax.transAxes, clip_on=True
        )
        
        # Set initial visibility
        self.ax.set_visible(self._is_visible)
        self.text_obj.set_visible(self._is_visible)
        self.bg.set_visible(self._is_visible)
        
        # If we have content, redraw it
        if self._lines:
            self._reflow_wrap()
            self._draw()

    def _remove_inset_axes(self):
        """
        Remove the inset axes and text object from the figure.
        """
        if hasattr(self, 'ax') and self.ax in self.fig.axes:
            # Remove the inset axes from the figure
            self.fig.delaxes(self.ax)
            
            # Force a redraw to ensure the axes is removed
            if self.fig.canvas:
                self.fig.canvas.draw_idle()
            
            # Remove references to the axes and text object
            self.ax = None
            self.text_obj = None
            self.bg = None

    # --- Public API ---------------------------------------------------------

    def append(self, text: str):
        """Append new text (may contain newlines)."""
        new_lines = text.splitlines()
        self._lines.extend(new_lines if new_lines else [""])
        
        # Only reflow and draw if we have an axes
        if hasattr(self, 'ax') and self.ax is not None:
            self._reflow_wrap()
            self._draw()

    def set_text(self, text: str):
        """Replace entire buffer."""
        self._lines = text.splitlines()
        
        # Only reflow and draw if we have an axes
        if hasattr(self, 'ax') and self.ax is not None:
            self._reflow_wrap()
            self._draw()
        
    def set_visible(self, visible: bool):
        """
        Set the visibility of the text area and its contents.
        
        This method either removes or recreates the inset axes based on the visibility state.
        
        Args:
            visible: True to make the text area visible, False to hide it.
        """
        # Store the visibility state
        self._is_visible = visible
        
        if visible:
            # If becoming visible and we don't have an axes, recreate it
            if not hasattr(self, 'ax') or self.ax is None:
                self._create_inset_axes()
        else:
            # If becoming invisible, remove the axes completely
            self._remove_inset_axes()
    
    def remove(self):
        """
        Completely remove the text area from the figure.
        This is more thorough than just setting visibility to False.
        """
        # Remove the inset axes
        self._remove_inset_axes()
        
        # Disconnect the scroll event handler
        if hasattr(self, 'cid_scroll') and self.cid_scroll is not None:
            self.fig.canvas.mpl_disconnect(self.cid_scroll)
            self.cid_scroll = None
        
        # Clear all state
        self._lines = []
        self._wrapped = []
        self._scroll = 0
        self._is_visible = False

    # --- Internals ----------------------------------------------------------

    def _viewport_metrics(self):
        """Compute how many rows/columns fit, based on font size and axes size."""
        # If we don't have an axes, return default values
        if not hasattr(self, 'ax') or self.ax is None:
            return 10, 40
            
        renderer = self.fig.canvas.get_renderer()
        ax_px = self.ax.get_window_extent(renderer=renderer)
        # Approximate line height and char width in pixels
        # 1 pt = 1/72 inch; pixels = inches * dpi
        px_per_pt = self.fig.dpi / 72.0
        line_height = self.fp.get_size_in_points() * px_per_pt * 1.2
        char_width = self.fp.get_size_in_points() * px_per_pt * 0.6  # monospace approx

        max_rows = max(1, int(ax_px.height / line_height))
        max_cols = max(10, int(ax_px.width / char_width))
        return max_rows, max_cols

    def _reflow_wrap(self):
        """Wrap logical lines to viewport width (so scrolling is by visual lines)."""
        # If we don't have an axes, skip this
        if not hasattr(self, 'ax') or self.ax is None:
            return
            
        max_rows, max_cols = self._viewport_metrics()
        wrapped = []
        for ln in self._lines:
            pieces = textwrap.wrap(ln, width=max_cols, drop_whitespace=False,
                                   replace_whitespace=False) or [""]
            wrapped.extend(pieces)
        self._wrapped = wrapped
        # Keep scroll within bounds
        max_scroll = max(0, len(self._wrapped) - max_rows)
        self._scroll = min(self._scroll, max_scroll)

    def _visible_slice(self):
        # If we don't have an axes, return empty list
        if not hasattr(self, 'ax') or self.ax is None:
            return []
            
        max_rows, _ = self._viewport_metrics()
        total = len(self._wrapped)
        if total <= max_rows:
            return self._wrapped
        start = max(0, total - max_rows - self._scroll)
        end = start + max_rows
        return self._wrapped[start:end]

    def _draw(self):
        # If we don't have an axes or text object, skip this
        if not hasattr(self, 'ax') or self.ax is None or not hasattr(self, 'text_obj') or self.text_obj is None:
            return
            
        vis = self._visible_slice()
        self.text_obj.set_text("\n".join(vis))
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        # If we don't have an axes, skip this
        if not hasattr(self, 'ax') or self.ax is None:
            return
            
        # Only scroll when the mouse is over the text viewport
        if event.inaxes is not self.ax:
            return
        max_rows, _ = self._viewport_metrics()
        max_scroll = max(0, len(self._wrapped) - max_rows)
        # Matplotlib: 'up' means wheel away from user -> scroll up the history
        delta = 1 if event.button == 'up' else -1
        new_scroll = min(max_scroll, max(0, self._scroll + delta))
        if new_scroll != self._scroll:
            self._scroll = new_scroll
            self._draw()
