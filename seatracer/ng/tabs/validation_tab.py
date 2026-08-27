"""Validation tab: model-fit metrics, actual-vs-model table, and data sanity checks."""
from __future__ import annotations

from nicegui import ui

from ..ui_common import df_to_grid


def _metric(label, value):
    with ui.card().classes("p-3 min-w-[180px]"):
        ui.label(label).classes("text-xs text-gray-500")
        ui.label(value).classes("text-xl font-semibold")


def build(dash):
    analysis = dash.state.analysis
    comparison = analysis.final_results["comparison"]
    delta = comparison["Delta"].abs()

    ui.label("Model fit").classes("text-base font-semibold")
    with ui.row().classes("gap-3"):
        _metric("Average model error", f'+/-{delta.mean():.2f}" / 500m')
        _metric("Greatest model error", f'+/-{delta.max():.2f}" / 500m')
        _metric("Squared model error", f'{(comparison["Delta"] ** 2).sum():.2f}"')

    ui.label("Actual vs. model").classes("text-base font-semibold mt-4")

    all_athletes = sorted({a.strip() for crew in comparison["Crew"]
                           for a in str(crew).split("/") if a.strip()})

    table_box = ui.column().classes("w-full")

    def _render_table(selected):
        table_box.clear()
        df = comparison
        if selected:
            mask = df["Crew"].apply(lambda crew: all(a in str(crew) for a in selected))
            df = df[mask]
        with table_box:
            df_to_grid(df.reset_index(drop=True), height=420, auto_height=False)

    ui.select(all_athletes, multiple=True, label="Filter lineups containing athletes",
              on_change=lambda e: _render_table(list(e.value or []))) \
        .props("use-chips dense outlined").classes("w-full max-w-2xl")
    _render_table([])

    # Possible errors: an athlete appearing in more than one boat in the same piece.
    ui.label("Possible errors").classes("text-base font-semibold mt-4")
    df = analysis.df.copy()
    duplicates = []
    for (session, piece), group in df.groupby(["Race Session (date)", "Piece"]):
        if len(group) <= 1:
            continue
        seen = {}
        for _, row in group.iterrows():
            for athlete in str(row["Personnel"]).split("/"):
                athlete = athlete.strip()
                if not athlete or athlete == "Coxᶜ":
                    continue
                seen.setdefault(athlete, []).append(row["Personnel"])
        for athlete, boats in seen.items():
            if len(boats) > 1:
                duplicates.append((f"{session} - Piece {piece}", athlete, boats))

    if duplicates:
        with ui.column().classes("gap-1"):
            ui.label(f"Found {len(duplicates)} athlete(s) in multiple boats").classes("text-amber-700")
            for race, athlete, boats in duplicates:
                ui.label(f"- {athlete} appears in {len(boats)} boats in {race}").classes("text-sm")
    else:
        ui.label("No athletes found in multiple boats.").classes("text-green-700")
