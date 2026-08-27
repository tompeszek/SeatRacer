"""Streamlit-free Plotly builder for the 'Over Time' tab.

Ported from ``TemporalVisualizer.plot_position_trends`` (which imports Streamlit
and is therefore not importable in the NiceGUI app). The maths is identical: for
each time point, speed is recomputed relative to the fastest athlete per side.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Note: we resolve position athletes directly from the temporal stats frame.
# Analysis.get_position_athletes is shadowed by a later, suffix-based override on
# the engine class (it returns a DataFrame, not the list this view needs), so we
# replicate the intended list logic here instead.
POSITION_SUFFIX = {"Starboard": "ˢ", "Port": "ᵖ", "Sculling": "ˣ", "Coxswain": "ᶜ"}


def build_position_figure(analysis, position: str, top_n: int = 8, default_visible=None):
    """Return a Plotly figure of side-aware speed trends for ``position``.

    ``position`` is one of 'Starboard', 'Port', 'Sculling', 'Coxswain'.
    Returns ``None`` if there is no data for that position.
    """
    temporal = analysis.get_temporal_data()
    if temporal.get("stats_df") is None:
        return None
    suffix = POSITION_SUFFIX.get(position)
    position_athletes = [a for a in temporal["stats_df"]["Rower"]
                         if suffix and str(a).endswith(suffix)]
    if not position_athletes:
        return None

    time_series_df = temporal["time_series_df"]
    stats_df = temporal["stats_df"]
    by_piece = temporal.get("by_piece", False)

    position_stats = stats_df[stats_df["Rower"].isin(position_athletes)].sort_values("Mean")
    if default_visible is None:
        default_visible = position_stats.head(top_n)["Rower"].tolist()

    fig = go.Figure()
    new_time_series = pd.DataFrame()

    for time_point in time_series_df["point"].unique():
        point_data = time_series_df[time_series_df["point"] == time_point].copy()
        point_athletes = {
            a: point_data[a].iloc[0]
            for a in position_athletes
            if a in point_data.columns and not pd.isna(point_data[a].iloc[0])
        }
        if not point_athletes:
            continue

        athlete_df = pd.DataFrame({"Coefficient": point_athletes})
        athlete_df.index.name = "Rower"
        athlete_df["Suffix"] = athlete_df.index.to_series().str.extract(r"([ᵖˢᶜˣ])$")[0]
        fastest = athlete_df.groupby("Suffix")["Coefficient"].transform("min")
        athlete_df["Speed"] = athlete_df["Coefficient"] - fastest

        for a in athlete_df.index:
            if a in point_data.columns:
                point_data.loc[point_data.index[0], f"{a}_Speed"] = athlete_df.loc[a, "Speed"]
        new_time_series = pd.concat([new_time_series, point_data])

    if new_time_series.empty:
        return None

    for athlete in position_stats["Rower"]:
        speed_col = f"{athlete}_Speed"
        if speed_col not in new_time_series.columns:
            continue
        adata = new_time_series[["point", "date", athlete, speed_col]].dropna(subset=[speed_col])
        if adata.empty:
            continue
        visible = True if athlete in default_visible else "legendonly"
        x_values = adata["point"] if by_piece else adata["date"]
        y_values = -adata[speed_col]  # higher is better
        hover = [f"{athlete}: Leader" if v == 0 else f'{athlete}: +{-v:.1f}"/500m'
                 for v in -adata[speed_col]]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="lines+markers", name=athlete,
            visible=visible, line=dict(width=2), marker=dict(size=6),
            text=hover, hovertemplate="%{text}<extra></extra>",
        ))

    x_title = "Race Piece" if by_piece else "Date"
    title = (f"Performance Trends by {'Race' if by_piece else 'Date'} "
             f"for {position.capitalize()} Position")
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title="Speed (higher is better)",
        hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    if by_piece:
        unique_points = new_time_series.sort_values("date")["point"].unique()
        fig.update_xaxes(categoryorder="array", categoryarray=unique_points)
    return fig
