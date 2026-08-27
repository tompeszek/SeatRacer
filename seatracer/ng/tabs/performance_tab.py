"""Performance tab: speed coefficients, dropped rowers, confidence intervals and
one-on-one probabilities, broken out by side."""
from __future__ import annotations

import pandas as pd
from nicegui import ui

from ..ui_common import (
    df_to_grid, echart_confidence_bars, echart_probability_heatmap,
    build_side_confidence_table, PORT_RED, STAR_GREEN, BLUE,
)

BOAT_CLASSES = ["8+", "4x/-", "2x/-", "1x"]
DISTANCES = [500, 1000, 1500, 2000, 4000, 6000]
SIDE_COLORS = {"starboard": STAR_GREEN, "port": PORT_RED,
               "coxswain": "#6366F1", "scull": "#0EA5E9"}


def _standardize_speed(speed, boat_class, meters, include_coxswains):
    n = int(boat_class[0])
    if boat_class[0] == "8" and include_coxswains:
        n = 9
    return (speed / 2000.0 * 4.0) * meters / n


def _adjust_metrics(df, boat_class, distance, include_coxswains):
    df = df.copy()
    for col in ["Speed", "Lower", "Upper", "Coefficient"]:
        df[f"{col}_Adjusted"] = df[col].apply(
            lambda v: _standardize_speed(v, boat_class, distance, include_coxswains))
    df["Behind_Adjusted"] = df["Speed_Adjusted"].apply(
        lambda x: f"+{round(x, 1)}" if x > 0 else f"{round(x, 1)}")
    return df


def _avg_behind(df, bests):
    df = df.copy()
    df["Port Count"] = df["Group Members"].str.count("ᵖ")
    df["Starboard Count"] = df["Group Members"].str.count("ˢ")
    df["Sculler Count"] = df["Group Members"].str.count("ˣ")
    df["Coxswain Count"] = df["Group Members"].str.count("ᶜ")
    denom = (df["Starboard Count"] + df["Port Count"] + df["Sculler Count"] + df["Coxswain Count"])
    numer = (df["Group Coefficient Sum"]
             - df["Starboard Count"] * bests["starboard"]
             - df["Port Count"] * bests["port"]
             - df["Sculler Count"] * bests["scull"]
             - df["Coxswain Count"] * bests["coxswain"])
    df["Average Behind"] = (numer / denom).round(1).apply(
        lambda x: f"+{round(x, 1)}" if x > 0 else f"{round(x, 1)}")
    return df


def build(dash):
    s = dash.state
    analysis = s.analysis
    athletes_df = analysis.final_results["athletes"]
    dropped_df = analysis.final_results["dropped_athletes"]
    sides_count = s.sides_count

    if s.correlation_auto_adjusted:
        ui.label("Note: max correlation automatically raised to show all athletes.") \
            .classes("text-xs text-amber-700 italic")

    groups = {
        "starboard": [r for r, v in sides_count.items() if v["Starboard"] > 0],
        "port": [r for r, v in sides_count.items() if v["Port"] > 0],
        "coxswain": [r for r, v in sides_count.items() if v["Coxswain"] > 0],
        "scull": [r for r, v in sides_count.items() if v["Scull"] > 0],
    }

    # --- controls (each selector on its own line) ---
    ui.label("Speed coefficients").classes("text-base font-semibold")

    def _set_view(attr, value):
        setattr(s, attr, value)
        dash.rebuild("Performance")

    with ui.row().classes("items-center gap-2"):
        ui.label("Boat class").classes("text-sm w-24")
        ui.radio(BOAT_CLASSES, value=s.perf_boat_class,
                 on_change=lambda e: _set_view("perf_boat_class", e.value)).props("inline dense")
    with ui.row().classes("items-center gap-2"):
        ui.label("Distance").classes("text-sm w-24")
        ui.radio(DISTANCES, value=s.perf_distance,
                 on_change=lambda e: _set_view("perf_distance", e.value)).props("inline dense")

    ui.label(f"Seconds, over {s.perf_distance}m in a {s.perf_boat_class}, slower than the "
             f"best rower on the same side").classes("text-xs text-gray-500 italic")

    dfs = {g: _adjust_metrics(athletes_df.loc[athletes_df.index.isin(idx)], s.perf_boat_class,
                              s.perf_distance, s.include_coxswains)
           for g, idx in groups.items()}

    visible_sides = [("Starboard", "starboard"), ("Port", "port")]
    if s.include_coxswains and len(dfs["coxswain"]) > 1:
        visible_sides.append(("Coxswains", "coxswain"))
    if len(dfs["scull"]) > 0:
        visible_sides.append(("Scull", "scull"))

    # --- speed coefficients: per-rower confidence distribution ---
    ui.label("Each bar chart shows the probability of a rower's true performance across a "
             "range of levels on a shared scale - a tall narrow shape means the model is "
             "confident, a low broad shape means it is uncertain.") \
        .classes("text-xs text-gray-500 italic")
    with ui.row().classes("w-full items-start gap-6"):
        for title, key in visible_sides:
            df = dfs[key]
            if df.empty:
                continue
            with ui.column().classes("flex-1 min-w-[340px] max-w-[480px] gap-1"):
                ui.label(title).classes("font-medium")
                res = build_side_confidence_table(df, color=SIDE_COLORS.get(key, BLUE))
                if res:
                    ui.html(res["html"]).classes("w-full")
                    ui.label(f"Adjusted seconds:  {res['x_min']:.1f} (faster)  →  "
                             f"{res['x_max']:.1f} (slower)").classes("text-xs text-gray-400 mt-1")
                else:
                    ui.label("(no athletes)").classes("text-xs text-gray-400")

    # --- dropped rowers ---
    if len(dropped_df) > 0:
        bests = {k: (dfs[k]["Coefficient"].min() if not dfs[k].empty else 0)
                 for k in ("starboard", "port", "coxswain", "scull")}
        ui.label("Dropped rowers").classes("text-base font-semibold mt-4")
        ui.label("Rowers with high uncertainty due to colinearity (always boated together)") \
            .classes("text-xs text-gray-500 italic")
        with ui.row().classes("w-full items-start gap-4"):
            for title, key in visible_sides:
                rowers = groups[key]
                dd = dropped_df.loc[dropped_df.index.isin(rowers)].sort_index().copy()
                if dd.empty:
                    continue
                dd = _avg_behind(dd, bests)
                with ui.column().classes("flex-1 min-w-[260px] gap-1"):
                    ui.label(title).classes("font-medium")
                    df_to_grid(dd[["Group Members", "Average Behind"]], index_label="Rower")

    # --- confidence intervals ---
    ui.label("Confidence intervals").classes("text-base font-semibold mt-4")
    with ui.row().classes("w-full items-start gap-4"):
        for title, key in visible_sides:
            df = dfs[key]
            if df.empty:
                continue
            with ui.column().classes("flex-1 min-w-[280px] gap-1"):
                ui.label(title).classes("font-medium")
                chart_box = ui.column().classes("w-full")

                def _draw(k=key, box=chart_box):
                    box.clear()
                    with box:
                        echart_confidence_bars(dfs[k], s.conf[k])

                def _on_conf(e, k=key, box=chart_box):
                    s.conf[k] = int(e.value)

                ui.label().bind_text_from(s.conf, key, lambda v: f"Confidence: {v}%") \
                    .classes("text-xs text-gray-500")
                slider = ui.slider(min=0, max=99, step=1, value=s.conf[key]) \
                    .props("label").classes("w-full")
                slider.on_value_change(lambda e, k=key: _on_conf(e, k))
                slider.on("change", lambda _, k=key, box=chart_box: _draw(k, box))
                _draw()

    # --- one-on-one probabilities (heatmap) ---
    ui.label("One-on-one probabilities").classes("text-base font-semibold mt-4")
    ui.label("Each cell is the probability the rower on the left is faster than the rower "
             "along the top (green = more likely, red = less likely).") \
        .classes("text-xs text-gray-500 italic")
    with ui.row().classes("w-full items-start gap-6"):
        for title, key in visible_sides:
            df = dfs[key].sort_values("Speed")
            if len(df) < 2:
                continue
            with ui.column().classes("flex-1 min-w-[320px] max-w-[560px] gap-1"):
                ui.label(title).classes("font-medium")
                echart_probability_heatmap(df)
