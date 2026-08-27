"""Debug tab: raw model outputs and correlated-athlete groupings."""
from __future__ import annotations

import pandas as pd
from nicegui import ui

from seatracer.utils.grouping import group_highly_correlated_parameters
from ..ui_common import df_to_grid


def build(dash):
    s = dash.state
    analysis = s.analysis
    fr = analysis.final_results

    ui.label("Shell classes").classes("text-base font-semibold")
    df_to_grid(fr["shell_classes"], index_label="Shell Class")

    ui.label("Athletes").classes("text-base font-semibold mt-3")
    df_to_grid(fr["athletes"], index_label="Rower", height=360, auto_height=False)

    if s.athlete_ergs_df is not None and not s.athlete_ergs_df.empty:
        ui.label("Athlete ergs").classes("text-base font-semibold mt-3")
        df_to_grid(s.athlete_ergs_df, index_label="Athlete")

    ui.label("Piece weights").classes("text-base font-semibold mt-3")
    weights = fr["weights"]
    wdf = weights.to_frame("Weight") if isinstance(weights, pd.Series) else pd.DataFrame(weights)
    df_to_grid(wdf, index_label="Row", height=300, auto_height=False)

    ui.label("Correlated groups").classes("text-base font-semibold mt-3")
    groups = group_highly_correlated_parameters(fr["corr"], threshold=s.max_correlation)
    if not groups:
        ui.label("No correlated groups above the current threshold.").classes("text-gray-500")
    for i, group in enumerate(groups, 1):
        ui.label(f"Group {i}: {', '.join(sorted(group))}").classes("text-sm")
