"""Individual tab: leave-one-piece-out influence analysis per athlete.

The compute (``_compute_leave_one_out``) is a Streamlit-free port of the original
``run_leave_one_out_analysis``: for each piece (or day/week), the model is refit
with that piece removed, and the change in each athlete's coefficient/speed/rank is
recorded. It runs in a worker thread so the UI stays responsive.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from nicegui import ui, run

from ..loo_worker import run_job

LEVELS = {"per_race": "Per Race", "per_day": "Per Day", "per_week": "Per Week"}
SPEED_COLORS = {"good": "#4CAF50", "bad": "#F44336",
                "good_indirect": "#A5D6A7", "bad_indirect": "#FFCCCB", "neutral": "#CCCCCC"}


# --------------------------------------------------------------------------- #
# Compute (ported, Streamlit-free)
# --------------------------------------------------------------------------- #
def _piece_groups(analysis, level):
    comparison_df = analysis.final_results["comparison"]
    orig_df = analysis.df.copy()
    if level == "per_race":
        all_pieces = comparison_df["Piece"].unique().tolist()
        return ({p: [p] for p in all_pieces}, {p: p for p in all_pieces}, all_pieces, orig_df)

    orig_df["Date"] = pd.to_datetime(orig_df["Race Session (date)"])
    if level == "per_day":
        orig_df["Key"] = orig_df["Date"].dt.strftime("%Y-%m-%d")
        prefix = "Day"
    else:
        orig_df["Key"] = orig_df["Date"].dt.to_period("W").astype(str)
        prefix = "Week"

    grouped: dict[str, list] = {}
    for _, row in orig_df.iterrows():
        grouped.setdefault(row["Key"], [])
        if row["Piece"] not in grouped[row["Key"]]:
            grouped[row["Key"]].append(row["Piece"])

    piece_groups, group_labels = {}, {}
    for key, pieces in grouped.items():
        for piece in pieces:
            piece_groups[piece] = pieces
            group_labels[piece] = f"{prefix}: {key}"
    return piece_groups, group_labels, list(set(piece_groups.keys())), orig_df


def _run_jobs(jobs):
    """Run the per-piece refits, in parallel across processes when there are
    enough of them (each refit is independent). Falls back to sequential if a
    process pool can't be created."""
    if len(jobs) <= 3:
        return [run_job(j) for j in jobs]
    try:
        workers = min(8, len(jobs), max(1, (os.cpu_count() or 2)))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run_job, jobs))
    except Exception:  # noqa: BLE001 - any pool/pickling issue -> sequential
        return [run_job(j) for j in jobs]


def compute_leave_one_out(analysis, level):
    comparison_df = analysis.final_results["comparison"]
    piece_groups, group_labels, all_pieces, orig_df = _piece_groups(analysis, level)

    all_athletes = set()
    if "athletes" in analysis.final_results:
        all_athletes.update(analysis.final_results["athletes"].index)
    if analysis.final_results.get("dropped_athletes") is not None:
        all_athletes.update(analysis.final_results["dropped_athletes"].index)

    athlete_pieces = {a: [r["Piece"] for _, r in comparison_df.iterrows() if a in r["Crew"]]
                      for a in all_athletes}

    # Crew / pace info per (piece, athlete), taken from the main comparison frame.
    crew_info = {}
    for piece in all_pieces:
        info = {}
        for p in piece_groups[piece]:
            for _, row in comparison_df[comparison_df["Piece"] == p].iterrows():
                for ath in row["Crew"].split("/"):
                    info.setdefault(ath, {"crew": row["Crew"], "actual_pace": row["Actual Pace"],
                                          "model_pace": row["Model Pace"], "delta": row["Delta"]})
        crew_info[piece] = info

    # Build one refit job per piece (each removes that piece's group).
    kwargs = dict(halflife=analysis.halflife, weight_close=analysis.weight_close,
                  weight_stern=analysis.weight_stern, include_coxswains=analysis.include_coxswains,
                  seat_breakdown=analysis.seat_breakdown, lookback=analysis.lookback,
                  erg_scores=analysis.erg_scores, shell_class=analysis.shell_class)
    model_class = analysis.__class__
    jobs = []
    for piece in all_pieces:
        filtered = orig_df[~orig_df["Piece"].isin(piece_groups[piece])].copy()
        if level == "per_race":
            sub = orig_df[orig_df["Piece"] == piece]
            race_date = sub["Race Session (date)"].iloc[0] if not sub.empty else None
            piece_number = sub["PieceNumber"].iloc[0] if not sub.empty else None
        else:
            race_date, piece_number = group_labels[piece], None
        jobs.append({"model_class": model_class, "kwargs": kwargs, "df": filtered,
                     "piece": piece, "group_label": group_labels[piece],
                     "race_date": race_date, "piece_number": piece_number})

    by_piece = {r["piece"]: r for r in _run_jobs(jobs)}

    athlete_data = {}
    active_athletes = analysis.final_results.get("athletes", pd.DataFrame())
    for athlete in all_athletes:
        in_main = athlete in active_athletes.index
        athlete_data[athlete] = {"status": "active" if in_main else "dropped"}
        if not in_main:
            continue
        current_coef = float(active_athletes.loc[athlete]["Coefficient"])
        athlete_data[athlete]["position_info"] = analysis.get_athlete_position_info(athlete)

        influences = []
        for piece in all_pieces:
            r = by_piece.get(piece, {})
            participated = any(p in athlete_pieces.get(athlete, []) for p in piece_groups[piece])
            act = r.get("active", {}).get(athlete)
            if act is not None:
                new_coef, speed, rank, dropped = act["new_coef"], act["speed"], act["rank"], False
            elif athlete in r.get("dropped", []):
                new_coef, speed, rank, dropped = None, None, None, True
            else:
                new_coef, speed, rank, dropped = None, None, None, None
            ci = crew_info.get(piece, {}).get(athlete, {})
            influences.append({
                "Piece": r.get("group_label", piece), "Race Date": r.get("race_date"),
                "Piece Number": r.get("piece_number"),
                "Crew": ci.get("crew", "Athlete not in race"),
                "New Coefficient": new_coef,
                "Coefficient Change": (current_coef - new_coef) if new_coef is not None else None,
                "Dropped in Analysis": dropped, "Athlete Participated": participated,
                "Position Speed": speed, "Position Rank": rank})
        athlete_data[athlete]["piece_influences"] = influences

    return {"athlete_data": athlete_data, "all_pieces": all_pieces}


def _sort_key(row):
    date_str = "0000-00-00"
    rd = row["Race Date"]
    if pd.notnull(rd):
        if isinstance(rd, str) and (rd.startswith("Day:") or rd.startswith("Week:")):
            return rd
        try:
            date_str = (rd if isinstance(rd, pd.Timestamp) else pd.to_datetime(rd)).strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            date_str = str(rd)
    piece_part = "00000"
    if pd.notnull(row["Piece Number"]):
        try:
            piece_part = str(int(row["Piece Number"])).zfill(5)
        except Exception:  # noqa: BLE001
            piece_part = str(row["Piece Number"])
    return f"{date_str}_{piece_part}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _speed_color(participated, change):
    if change is None or change == 0:
        return SPEED_COLORS["neutral"]
    if participated:
        return SPEED_COLORS["good"] if change > 0 else SPEED_COLORS["bad"]
    return SPEED_COLORS["good_indirect"] if change > 0 else SPEED_COLORS["bad_indirect"]


def build(dash):
    s = dash.state
    analysis = s.analysis
    all_athletes = sorted(set(analysis.final_results.get("athletes", pd.DataFrame()).index))

    async def _run(level):
        spinner = ui.spinner(size="lg")
        ui.notify(f"Running {LEVELS[level]} analysis...", type="info")
        try:
            bundle = await run.io_bound(compute_leave_one_out, analysis, level)
            s.loo_results[level] = bundle
            s.loo_level = level
            if s.loo_athlete is None and all_athletes:
                s.loo_athlete = all_athletes[0]
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Analysis failed: {exc}", type="negative")
        finally:
            spinner.delete()
        dash.rebuild("Individual")

    with ui.row().classes("items-center gap-2 flex-wrap"):
        ui.button("Run per-race", on_click=lambda: _run("per_race")).props("outline size=sm")
        ui.button("Run per-day", on_click=lambda: _run("per_day")).props("outline size=sm")
        ui.button("Run per-week", on_click=lambda: _run("per_week")).props("outline size=sm")

        completed = [lvl for lvl in LEVELS if lvl in s.loo_results]
        if completed:
            if s.loo_level not in completed:
                s.loo_level = completed[0]
            ui.radio({lvl: LEVELS[lvl] for lvl in completed}, value=s.loo_level,
                     on_change=lambda e: (setattr(s, "loo_level", e.value),
                                          dash.rebuild("Individual"))).props("inline dense")

    if not s.loo_results:
        ui.label("Run an analysis to see how each race influences an athlete's rating.") \
            .classes("text-gray-500 mt-2")
        return

    ui.select(all_athletes, value=s.loo_athlete, label="Select athlete",
              on_change=lambda e: (setattr(s, "loo_athlete", e.value), dash.rebuild("Individual"))) \
        .props("dense outlined").classes("w-full max-w-sm")

    bundle = s.loo_results.get(s.loo_level)
    if not bundle or not s.loo_athlete:
        return
    adata = bundle["athlete_data"].get(s.loo_athlete, {})
    if adata.get("status") != "active":
        ui.label("This athlete was dropped from the model (high colinearity).") \
            .classes("text-gray-500 mt-2")
        return

    info = adata["position_info"]
    df = pd.DataFrame(adata["piece_influences"])
    df["Speed Change"] = df["Position Speed"].fillna(0) - info["speed"]
    df["Rank Change"] = df["Position Rank"].fillna(0) - info["rank"]
    df["sort_key"] = df.apply(_sort_key, axis=1)
    df = df.sort_values("sort_key")

    with ui.row().classes("gap-3 mt-2"):
        for label, value in (("Coefficient", f"{info['coefficient']:.2f}s"),
                             ("Speed", f"+{info['speed']:.2f}s"),
                             ("Rank", f"{info['rank']}/{info['total_in_position']}")):
            with ui.card().classes("p-3 min-w-[140px]"):
                ui.label(label).classes("text-xs text-gray-500")
                ui.label(value).classes("text-xl font-semibold")

    pieces = df["Piece"].tolist()
    bar_data = [{"value": round(float(c), 3),
                 "itemStyle": {"color": _speed_color(p, c)}}
                for c, p in zip(df["Speed Change"], df["Athlete Participated"])]
    rank_data = [round(float(r), 3) for r in df["Rank Change"]]
    option = {
        "title": {"text": f"{LEVELS[s.loo_level]} impact (higher is better)", "left": "center",
                  "textStyle": {"fontSize": 13}},
        "grid": {"left": 8, "right": 24, "top": 40, "bottom": 80, "containLabel": True},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Speed Change", "Rank Change"], "top": 20},
        "xAxis": {"type": "category", "data": pieces,
                  "axisLabel": {"rotate": 45, "interval": 0, "fontSize": 9}},
        "yAxis": [{"type": "value", "name": "Speed Change (s)"},
                  {"type": "value", "name": "Rank Change", "position": "right"}],
        "series": [
            {"name": "Speed Change", "type": "bar", "yAxisIndex": 0, "data": bar_data},
            {"name": "Rank Change", "type": "scatter", "yAxisIndex": 1, "symbolSize": 10,
             "itemStyle": {"color": "#D4A017"}, "data": rank_data},
        ],
    }
    ui.echart(option).style("height: 420px").classes("w-full")

    with ui.expansion("Show data table").classes("w-full mt-2"):
        show = df[["Piece", "Crew", "New Coefficient", "Position Speed", "Speed Change",
                   "Position Rank", "Rank Change", "Athlete Participated"]].copy()
        from ..ui_common import df_to_grid
        df_to_grid(show.reset_index(drop=True), height=360, auto_height=False)
