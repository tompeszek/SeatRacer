"""Shared UI helpers for the NiceGUI frontend: AG Grid builders, ECharts option
builders, and a few pure stats helpers copied from the old ``charts.py`` (which
itself imports Streamlit and is therefore not importable here)."""
from __future__ import annotations

import html
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm

from nicegui import ui

# Colours roughly matching the old palette.
PORT_RED = "#E4572E"
STAR_GREEN = "#2E8B57"
BLUE = "#3B82F6"
GREY = "#9CA3AF"


# --------------------------------------------------------------------------- #
# AG Grid helpers
# --------------------------------------------------------------------------- #
def _clean_value(v):
    """Make a single cell value JSON-serialisable for AG Grid."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def df_to_grid(
    df: pd.DataFrame,
    *,
    index_label: Optional[str] = None,
    round_floats: int = 2,
    editable_columns: Optional[Iterable[str]] = None,
    column_order: Optional[list] = None,
    height: Optional[int] = None,
    auto_height: bool = True,
    pinned_first: bool = False,
) -> ui.aggrid:
    """Render a DataFrame as an AG Grid.

    ``index_label`` (when given) promotes the DataFrame index into a leading,
    optionally pinned column – used for athlete/shell-class indexed frames.
    """
    df = df.copy()
    editable_columns = set(editable_columns or [])

    if index_label is not None:
        df.insert(0, index_label, df.index.astype(str))

    if column_order is not None:
        cols = ([index_label] if index_label is not None else [])
        cols += [c for c in column_order if c in df.columns]
        df = df[cols]

    # Round float columns for display (kept numeric so sorting still works).
    if round_floats is not None:
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(round_floats)

    column_defs = []
    for c in df.columns:
        col = {"headerName": str(c), "field": str(c), "sortable": True, "resizable": True}
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            col["filter"] = "agNumberColumnFilter"
            col["type"] = "numericColumn"
        else:
            col["filter"] = "agTextColumnFilter"
        if str(c) in editable_columns:
            col["editable"] = True
        if pinned_first and ((index_label is not None and c == index_label) or
                             (index_label is None and c == df.columns[0])):
            col["pinned"] = "left"
        column_defs.append(col)

    row_data = [{str(k): _clean_value(v) for k, v in rec.items()}
                for rec in df.to_dict(orient="records")]

    options = {
        "columnDefs": column_defs,
        "rowData": row_data,
        "defaultColDef": {"resizable": True, "sortable": True, "minWidth": 90},
        "animateRows": False,
    }
    if height is not None:
        grid = ui.aggrid(options).style(f"height: {height}px").classes("w-full")
    elif auto_height:
        options["domLayout"] = "autoHeight"
        grid = ui.aggrid(options).classes("w-full")
    else:
        grid = ui.aggrid(options).classes("w-full").style("height: 400px")
    return grid


# --------------------------------------------------------------------------- #
# Pure stats helpers (copied from the old visualization/charts.py)
# --------------------------------------------------------------------------- #
def compute_probability_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Probability that the row athlete is faster (lower coefficient) than the
    column athlete, using a normal approximation of each confidence interval."""
    df = df.copy()
    rowers = df.index.tolist()
    df.loc[:, "StdDev"] = (df["Upper"] - df["Lower"]) / 3.92

    prob = pd.DataFrame(index=rowers, columns=rowers, dtype=object)
    for a in rowers:
        for b in rowers:
            if a == b:
                prob.loc[a, b] = "-"
                continue
            mu_a, mu_b = df.loc[a, "Coefficient"], df.loc[b, "Coefficient"]
            sa, sb = df.loc[a, "StdDev"], df.loc[b, "StdDev"]
            var = sa ** 2 + sb ** 2
            if var < 1e-10:
                prob.loc[a, b] = "100%" if mu_a < mu_b else ("0%" if mu_a > mu_b else "50%")
            else:
                z = (mu_b - mu_a) / np.sqrt(var)
                prob.loc[a, b] = f"{norm.cdf(z) * 100:.0f}%"
    return prob


def likelihood_curve(mean: float, lower: float, upper: float, x_vals: np.ndarray) -> list:
    """Normal PDF for an athlete across a shared x-axis (for distribution charts)."""
    std = (upper - lower) / 3.92
    if std < 1e-10:
        return [1.0 if abs(x - mean) < 1e-10 else 0.0 for x in x_vals]
    return scipy.stats.norm.pdf(x_vals, mean, std).tolist()


# --------------------------------------------------------------------------- #
# ECharts option builders
# --------------------------------------------------------------------------- #
def echart_confidence_bars(side_df: pd.DataFrame, confidence: int = 50,
                           value_label: str = "Speed"):
    """Floating horizontal bars for each rower's confidence interval (adjusted
    speed). Equivalent to the old ``generate_confidence_bars_with_gradient``."""
    rows = []
    for idx, row in side_df.iterrows():
        mean = row.get("Speed_Adjusted", row.get("Speed"))
        upper = row.get("Upper_Adjusted", row.get("Upper"))
        lower = row.get("Lower_Adjusted", row.get("Lower"))
        std = (upper - lower) / 3.92
        if std < 1e-10:
            lo = hi = mean
        else:
            lo = scipy.stats.norm.ppf(0.5 - (confidence / 2.0) / 100.0, mean, std)
            hi = scipy.stats.norm.ppf(0.5 + (confidence / 2.0) / 100.0, mean, std)
        rows.append({"rower": str(idx), "speed": mean, "lo": round(lo, 2), "hi": round(hi, 2)})

    rows.sort(key=lambda r: r["speed"], reverse=True)  # fastest (lowest speed) at top
    names = [r["rower"] for r in rows]
    base = [round(r["lo"], 3) for r in rows]
    span = [round(r["hi"] - r["lo"], 3) for r in rows]

    height = max(140, len(names) * 30 + 60)
    option = {
        "grid": {"left": 8, "right": 24, "top": 16, "bottom": 36, "containLabel": True},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {"type": "value", "name": value_label, "nameLocation": "middle", "nameGap": 24},
        "yAxis": {"type": "category", "data": names, "axisLabel": {"interval": 0}},
        "series": [
            {"type": "bar", "stack": "ci", "itemStyle": {"color": "transparent"},
             "data": base, "silent": True, "tooltip": {"show": False}},
            {"name": value_label, "type": "bar", "stack": "ci",
             "itemStyle": {"color": BLUE, "borderRadius": 3}, "data": span},
        ],
    }
    return ui.echart(option).style(f"height: {height}px").classes("w-full")


def pdf_bars_svg(values, y_max: float, width: int = 260, height: int = 34,
                color: str = BLUE) -> str:
    """Tiny inline bar-chart (sparkline) of a probability density. All sparklines
    in a side share ``y_max`` and the same x-range, so bar height encodes density
    and a confident rower shows a tall narrow spike vs. a flat broad one."""
    n = len(values)
    if n == 0:
        return ""
    if y_max <= 0:
        y_max = 1.0
    bw = width / n
    bars = []
    for i, v in enumerate(values):
        h = max(0.0, min(1.0, v / y_max)) * (height - 2)
        x = i * bw
        y = height - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bw - 0.4, 0.6):.1f}" '
                    f'height="{h:.1f}" fill="{color}"/>')
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'preserveAspectRatio="none" style="display:block">{"".join(bars)}</svg>')


def build_side_confidence_table(side_df: pd.DataFrame, color: str = BLUE,
                                width: int = 260, n_points: int = 40) -> Optional[dict]:
    """Per-rower confidence table: Rower | Behind | Coeff | distribution sparkline.

    Faithful port of the old ``generate_side_chart`` BarChartColumn: each rower's
    row shows the probability density of their true performance across a shared
    adjusted-seconds axis (fastest rower's lower CI to slowest rower's upper CI).
    Returns a dict with the table HTML and the x-axis range, or None if empty.
    """
    if side_df is None or side_df.empty:
        return None
    df = side_df.copy()

    try:
        x_min = float(df.loc[df["Speed_Adjusted"].idxmin(), "Lower_Adjusted"])
        x_max = float(df.loc[df["Speed_Adjusted"].idxmax(), "Upper_Adjusted"])
    except Exception:  # noqa: BLE001
        x_min, x_max = float(df["Lower_Adjusted"].min()), float(df["Upper_Adjusted"].max())
    if not (np.isfinite(x_min) and np.isfinite(x_max)) or x_max <= x_min:
        x_min = float(df["Lower_Adjusted"].min())
        x_max = float(df["Upper_Adjusted"].max())
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = (x_min if np.isfinite(x_min) else 0.0), \
                           (x_min + 1.0 if np.isfinite(x_min) else 1.0)

    x_vals = np.linspace(x_min, x_max, n_points)
    pdfs, y_max = {}, 1e-9
    for idx, row in df.iterrows():
        y = likelihood_curve(row["Coefficient_Adjusted"], row["Lower_Adjusted"],
                             row["Upper_Adjusted"], x_vals)
        y = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in y]
        pdfs[idx] = y
        y_max = max(y_max, max(y) if y else 0.0)
    y_max *= 1.05

    line = "border-bottom:1px solid #e5e7eb"
    rows = []
    for i, (idx, row) in enumerate(df.sort_values("Speed_Adjusted").iterrows()):
        spark = pdf_bars_svg(pdfs[idx], y_max, width=width, color=color)
        title = (f"{idx}: coefficient {row['Coefficient']:.1f} "
                 f"(95% CI {row['Lower']:.1f} to {row['Upper']:.1f})")
        bg = "background:#f8fafc" if i % 2 else ""  # zebra striping for row tracking
        rows.append(
            f'<tr style="{bg}">'
            f'<td style="padding:3px 8px;white-space:nowrap;{line};border-right:1px solid #eef2f7">'
            f'{html.escape(str(idx))}</td>'
            f'<td style="padding:3px 8px;text-align:right;white-space:nowrap;{line};'
            f'border-right:1px solid #eef2f7">{html.escape(str(row["Behind_Adjusted"]))}</td>'
            f'<td style="padding:3px 8px;text-align:right;white-space:nowrap;{line};'
            f'border-right:1px solid #eef2f7">{row["Coefficient"]:.1f}</td>'
            f'<td style="padding:1px 8px;width:{width}px;{line}" '
            f'title="{html.escape(title)}">{spark}</td>'
            "</tr>")

    hcell = "text-align:left;padding:4px 8px;border-bottom:2px solid #cbd5e1;font-weight:600"
    hcell_r = hcell.replace("text-align:left", "text-align:right")
    thead = (
        '<thead><tr>'
        f'<th style="{hcell}">Rower</th>'
        f'<th style="{hcell_r}">Behind</th>'
        f'<th style="{hcell_r}">Coeff</th>'
        f'<th style="{hcell}">Confidence</th>'
        '</tr></thead>')
    table = (f'<table style="border-collapse:collapse;width:100%;font-size:13px">'
             f'{thead}<tbody>{"".join(rows)}</tbody></table>')
    return {"html": table, "x_min": x_min, "x_max": x_max}


def probability_matrix_numeric(side_df: pd.DataFrame) -> pd.DataFrame:
    """Numeric (0-100) version of the one-on-one probability matrix."""
    df = side_df.copy()
    rowers = df.index.tolist()
    df["StdDev"] = (df["Upper"] - df["Lower"]) / 3.92
    mat = pd.DataFrame(index=rowers, columns=rowers, dtype=float)
    for a in rowers:
        for b in rowers:
            if a == b:
                mat.loc[a, b] = np.nan
                continue
            var = df.loc[a, "StdDev"] ** 2 + df.loc[b, "StdDev"] ** 2
            ca, cb = df.loc[a, "Coefficient"], df.loc[b, "Coefficient"]
            if var < 1e-10:
                mat.loc[a, b] = 100.0 if ca < cb else (0.0 if ca > cb else 50.0)
            else:
                z = (cb - ca) / np.sqrt(var)
                mat.loc[a, b] = float(norm.cdf(z) * 100)
    return mat


def echart_probability_heatmap(side_df: pd.DataFrame):
    """Heatmap of one-on-one probabilities: cell (row a, col b) = P(a faster than b).
    Replaces the cramped, name-truncated matrix grid – names sit on the axes
    (rotated) and the value is shown in each cell, coloured red->green."""
    mat = probability_matrix_numeric(side_df)
    names = [str(n) for n in mat.index.tolist()]
    data = []
    for yi, a in enumerate(mat.index):
        for xi, b in enumerate(mat.columns):
            v = mat.loc[a, b]
            if pd.isna(v):
                continue
            data.append([xi, yi, round(float(v))])

    n = len(names)
    show_label = n <= 16
    px = max(300, n * 30 + 140)
    option = {
        "grid": {"left": 8, "right": 16, "top": 8, "bottom": 8, "containLabel": True},
        "tooltip": {"position": "top"},
        "xAxis": {"type": "category", "data": names, "splitArea": {"show": True},
                  "axisLabel": {"rotate": 55, "interval": 0, "fontSize": 10}},
        "yAxis": {"type": "category", "data": names, "splitArea": {"show": True},
                  "inverse": True, "axisLabel": {"interval": 0, "fontSize": 10}},
        "visualMap": {"min": 0, "max": 100, "show": False, "calculable": True,
                      "inRange": {"color": ["#E4572E", "#f7f7f7", "#2E8B57"]}},
        "series": [{"type": "heatmap", "data": data,
                    "label": {"show": show_label, "fontSize": 9, "formatter": "{@[2]}"},
                    "emphasis": {"itemStyle": {"shadowBlur": 6, "shadowColor": "rgba(0,0,0,0.3)"}}}],
    }
    return ui.echart(option).style(f"height: {px}px").classes("w-full")


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def section_title(text: str, subtitle: str = ""):
    ui.label(text).classes("text-lg font-semibold mt-2")
    if subtitle:
        ui.label(subtitle).classes("text-xs text-gray-500 italic -mt-1 mb-1")
