"""Synergies tab: athlete pairs that consistently row faster/slower than predicted."""
from __future__ import annotations

import pandas as pd
from nicegui import ui

from ..ui_common import df_to_grid

MIN_RACES = 5
P_VALUE_OPTIONS = {
    "All": {"value": 1.0, "text": "Showing all pairs, regardless of statistical significance"},
    "0.05": {"value": 0.05, "text": "Showing pairs with p-value <= 0.05 (statistically significant)"},
    "0.01": {"value": 0.01, "text": "Showing pairs with p-value <= 0.01 (highly significant)"},
    "0.001": {"value": 0.001, "text": "Showing pairs with p-value <= 0.001 (extremely significant)"},
}


def build(dash):
    pairs = dash.state.analysis.final_results["pairs"]
    if pairs.empty:
        ui.label("No athlete pairs found with sufficient data.").classes("text-gray-500")
        return

    base = pairs[pairs["Races"] >= MIN_RACES].copy()
    if base.empty:
        ui.label(f"No athlete pairs found with at least {MIN_RACES} races together.") \
            .classes("text-gray-500")
        return

    # p-value selector first (above the table), then the caption, then the table.
    with ui.row().classes("items-center gap-3"):
        ui.label("p-value threshold").classes("text-sm")
        ui.radio(list(P_VALUE_OPTIONS.keys()), value="0.05",
                 on_change=lambda e: (caption.set_text(P_VALUE_OPTIONS[e.value]["text"]),
                                      _render(e.value))) \
            .props("inline dense")
    caption = ui.label(P_VALUE_OPTIONS["0.05"]["text"]).classes("text-xs text-gray-500 italic")

    box = ui.column().classes("w-full max-w-[760px]")

    def _render(choice):
        box.clear()
        threshold = P_VALUE_OPTIONS[choice]["value"]
        df = base[base["p_value"] <= threshold].copy()
        df["Pair"] = df.apply(lambda r: f"{r['Athlete1']} + {r['Athlete2']}", axis=1)
        df["Performance"] = df["AvgDelta"].apply(
            lambda x: f"{x:.2f}s " + ("(faster)" if x < 0 else "(slower)"))
        df["Significance"] = df["p_value"].apply(
            lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
        df = df.sort_values("AvgDelta", ascending=True)
        with box:
            ui.label("Pairs showing synergy or discord").classes("text-base font-semibold")
            if df.empty:
                ui.label("No pairs meet this significance threshold.").classes("text-gray-500")
            else:
                df_to_grid(df[["Pair", "Performance", "Races", "Significance"]]
                           .reset_index(drop=True), height=460, auto_height=False)

    _render("0.05")
