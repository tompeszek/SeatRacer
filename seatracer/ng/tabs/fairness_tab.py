"""Fairness tab: how consistently the model evaluates each athlete (prediction bias)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from nicegui import ui

from ..ui_common import df_to_grid

CAT_COLORS = {
    "Consistently Underperforms Model": "#E4572E",
    "No Significant Bias": "#9CA3AF",
    "Consistently Outperforms Model": "#3B82F6",
}


def _analyze_prediction_bias(analysis) -> pd.DataFrame:
    fr = analysis.final_results
    if fr is None or "comparison" not in fr:
        return pd.DataFrame()
    comparison = fr["comparison"]
    errors: dict[str, list] = {}
    for _, row in comparison.iterrows():
        for athlete in str(row["Crew"]).split("/"):
            errors.setdefault(athlete, []).append(row["Delta"])

    data = []
    for athlete, errs in errors.items():
        if len(errs) < 2:
            continue
        avg, std, n = float(np.mean(errs)), float(np.std(errs)), len(errs)
        t_stat = avg / (std / np.sqrt(n)) if std > 0 else float("inf")
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if std > 0 else 0.0
        data.append({
            "Athlete": athlete, "Average Error": round(avg, 2),
            "Standard Deviation": round(std, 2), "Races": n,
            "Bias Direction": "Underperformed" if avg < 0 else "Overperformed",
            "P-Value": round(p_value, 3), "Significant": p_value < 0.05,
        })
    return pd.DataFrame(data)


def _echart_bias(df: pd.DataFrame, height: int = 360):
    df = df.copy()
    data = [{"value": round(r["Average Error"], 2),
             "itemStyle": {"color": CAT_COLORS.get(r["Performance vs Model"], "#9CA3AF")}}
            for _, r in df.iterrows()]
    option = {
        "grid": {"left": 8, "right": 16, "top": 20, "bottom": 70, "containLabel": True},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": df["Athlete"].tolist(),
                  "axisLabel": {"rotate": 45, "interval": 0, "fontSize": 10}},
        "yAxis": {"type": "value", "name": "Avg error (s)"},
        "series": [{"type": "bar", "data": data}],
    }
    return ui.echart(option).style(f"height: {height}px").classes("w-full")


def build(dash):
    s = dash.state
    analysis = s.analysis
    sides_count = s.sides_count

    ui.label("Performance bias analysis").classes("text-base font-semibold")
    ui.label("How consistently the model evaluates each athlete across events. Positive = "
             "boats with this athlete are faster than predicted; negative = slower.") \
        .classes("text-xs text-gray-500 italic mb-2")

    bias_df = _analyze_prediction_bias(analysis)
    if bias_df.empty:
        ui.label("Insufficient data: athletes need at least 2 races with comparison data.") \
            .classes("text-amber-700")
        return

    bias_df = bias_df.sort_values(["Significant", "Average Error"], ascending=[False, True])
    bias_df["Performance vs Model"] = bias_df.apply(
        lambda x: "Consistently Underperforms Model" if x["Average Error"] < -0.5 and x["Significant"]
        else "Consistently Outperforms Model" if x["Average Error"] > 0.5 and x["Significant"]
        else "No Significant Bias", axis=1)
    bias_df["Position"] = bias_df["Athlete"].str.extract(r"([ᵖˢᶜˣ])$")[0].map(
        {"ᵖ": "Port", "ˢ": "Starboard", "ᶜ": "Coxswain", "ˣ": "Scull"})

    position_groups = {
        "Starboard": [r for r, v in sides_count.items() if v["Starboard"] > 0],
        "Port": [r for r, v in sides_count.items() if v["Port"] > 0],
        "Coxswain": [r for r, v in sides_count.items() if v["Coxswain"] > 0],
        "Scull": [r for r, v in sides_count.items() if v["Scull"] > 0],
    }

    with ui.tabs().classes("w-full").props("dense") as subtabs:
        ui.tab("Overview")
        ui.tab("By Position")
    with ui.tab_panels(subtabs, value="Overview").classes("w-full"):
        with ui.tab_panel("Overview"):
            significant = bias_df[bias_df["Significant"]]
            if not significant.empty:
                ui.label("Statistically significant bias by athlete").classes("font-medium")
                _echart_bias(significant)
            else:
                ui.label("No statistically significant performance bias detected.") \
                    .classes("text-gray-600")
                _echart_bias(bias_df)
            ui.label("Complete performance bias metrics").classes("font-medium mt-3")
            df_to_grid(bias_df[["Athlete", "Position", "Average Error", "Standard Deviation",
                                "Races", "P-Value", "Significant", "Performance vs Model"]]
                       .reset_index(drop=True), height=420, auto_height=False)

        with ui.tab_panel("By Position"):
            available = [p for p, ath in position_groups.items()
                         if any(a in bias_df["Athlete"].values for a in ath)]
            if not available:
                ui.label("No position data available.").classes("text-gray-500")
            else:
                with ui.row().classes("w-full items-start gap-4"):
                    for position in available:
                        pos_df = bias_df[bias_df["Athlete"].isin(position_groups[position])]
                        if pos_df.empty:
                            continue
                        with ui.column().classes("flex-1 min-w-[260px] gap-1"):
                            ui.label(position).classes("font-medium")
                            _echart_bias(pos_df, height=240)
                            df_to_grid(pos_df[["Athlete", "Average Error", "Races",
                                               "P-Value", "Significant"]].reset_index(drop=True))
