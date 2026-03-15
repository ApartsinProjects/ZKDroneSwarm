"""
Panel components for the TabulaDrone viewer.

This package contains visual components that display specific types of information.
"""

from viewer.components.panels.empty_panel import EmptyPanel
from viewer.components.panels.info_panel import InfoPanel
from viewer.components.panels.results_panel import ResultsPanel
from viewer.components.panels.summary_panel import SummaryPanel
from viewer.components.panels.training_path_panel import TrainingPathPanel
from viewer.components.panels.radar_chart_panel import RadarChartPanel

__all__ = ['EmptyPanel', 'InfoPanel', 'ResultsPanel', 'SummaryPanel', 'TrainingPathPanel', 'RadarChartPanel']
