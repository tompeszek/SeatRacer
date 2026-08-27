"""New Lineup tab: athletes ranked by side, derived from the fitted model.

The old Streamlit section only instantiated ``LineupOptimizer2`` without rendering
anything; here we surface its ranked port/starboard/sculling lists, which is the
useful information that optimizer exposes.
"""
from __future__ import annotations

import pandas as pd
from nicegui import ui

from seatracer.optimize.lineup_optimizer2 import LineupOptimizer2
from ..ui_common import df_to_grid


def _ranked_frame(optimizer, athletes):
    rows = [{"Rower": a, "Coefficient": round(optimizer.athlete_coefficients.get(a, float("nan")), 2)}
            for a in athletes]
    return pd.DataFrame(rows)


def build(dash):
    analysis = dash.state.analysis
    try:
        opt = LineupOptimizer2(analysis)
    except ValueError as exc:
        ui.label(str(exc)).classes("text-gray-500")
        return

    ui.label("Athletes ranked by side").classes("text-base font-semibold")
    ui.label("Lower coefficient = faster. Ranked from the current model fit.") \
        .classes("text-xs text-gray-500 italic mb-2")

    sides = [
        ("Starboard", opt.starboard_athletes),
        ("Port", opt.port_athletes),
        ("Scull", opt.sculling_athletes),
    ]
    with ui.row().classes("w-full items-start gap-6"):
        for title, athletes in sides:
            if not athletes:
                continue
            with ui.column().classes("gap-1"):
                ui.label(title).classes("font-medium")
                df_to_grid(_ranked_frame(opt, athletes), height=380, auto_height=False)
