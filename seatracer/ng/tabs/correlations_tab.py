"""Correlations tab: how often rowers are boated together (design-matrix corr)."""
from nicegui import ui

from ..ui_common import df_to_grid


def build(dash):
    ui.label("Correlation matrix").classes("text-base font-semibold")
    ui.label("Shows how often rowers are boated with others. When correlation is too "
             "high (>= 0.5 or <= -0.5) the model cannot separate their performances.") \
        .classes("text-xs text-gray-500 italic mb-2")
    corr = dash.state.analysis.final_results["corr"].round(2)
    df_to_grid(corr, index_label="Parameter", round_floats=2, height=520, auto_height=False)
