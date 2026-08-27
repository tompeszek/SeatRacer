"""Athletes tab: manage 2k erg scores used to seed weighted/gradient models."""
from __future__ import annotations

import pandas as pd
from nicegui import ui

from seatracer.utils.data_handler import DataHandler
from ..ui_common import df_to_grid


def _roster_from_data(df: pd.DataFrame):
    """Unique raw athlete names (no rigging suffix) from the Personnel column."""
    if df is None or df.empty or "Personnel" not in df.columns:
        return []
    names = df["Personnel"].astype(str).str.split("/", expand=True).stack().unique()
    return sorted(n for n in names if n)


def _ensure_erg_df(s):
    """Create/extend the erg DataFrame so it covers the current roster."""
    roster = _roster_from_data(s.current_data)
    if not roster:
        return
    if s.athlete_ergs_df is None or s.athlete_ergs_df.empty:
        s.athlete_ergs_df = (
            pd.DataFrame({"Athlete": roster, "2k Erg": ["7:00.0"] * len(roster)})
            .sort_values("Athlete").set_index("Athlete")
        )
    else:
        existing = s.athlete_ergs_df.index.tolist()
        new = [a for a in roster if a not in existing]
        if new:
            add = (pd.DataFrame({"Athlete": new, "2k Erg": ["7:00.0"] * len(new)})
                   .sort_values("Athlete").set_index("Athlete"))
            s.athlete_ergs_df = pd.concat([s.athlete_ergs_df, add])


def build(dash):
    s = dash.state
    handler = DataHandler("erg_data")

    async def _set_ergs(df: pd.DataFrame):
        s.athlete_ergs_df = df.set_index("Athlete") if "Athlete" in df.columns else df
        await dash.recompute()
        dash.rebuild("Athletes")

    async def _load_example(name):
        await _set_ergs(handler.load_dataset(name))
        ui.notify(f"Loaded erg data: {name}", type="positive")

    async def _on_upload(e):
        try:
            df = pd.read_csv(e.content)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Could not read CSV: {exc}", type="negative")
            return
        await _set_ergs(df)
        ui.notify(f"Loaded erg data from {e.name}", type="positive")

    with ui.row().classes("w-full items-start gap-6"):
        with ui.column().classes("gap-1"):
            ui.label("Load example erg data").classes("text-base font-semibold")
            datasets = handler.get_available_datasets()
            if datasets:
                with ui.row().classes("gap-2 flex-wrap"):
                    for name in datasets:
                        ui.button(name.replace(".csv", ""),
                                  on_click=lambda n=name: _load_example(n)).props("outline size=sm")
            else:
                ui.label("No example erg files available.").classes("text-gray-500")
        with ui.column().classes("gap-1"):
            ui.label("Upload erg data (CSV)").classes("text-base font-semibold")
            ui.upload(on_upload=_on_upload, auto_upload=True).props("accept=.csv").classes("max-w-sm")

    ui.separator().classes("my-3")

    if not s.has_data:
        ui.label("Load racing data first - the roster is taken from your race data.") \
            .classes("text-gray-500")
        return

    _ensure_erg_df(s)
    roster = set(_roster_from_data(s.current_data))
    display = s.athlete_ergs_df.loc[s.athlete_ergs_df.index.isin(roster)].copy()

    ui.label("2k erg times (format m:ss.s)").classes("text-base font-semibold")
    grid = df_to_grid(display, index_label="Athlete", editable_columns=["2k Erg"],
                      height=min(520, (len(display) + 1) * 32 + 40), auto_height=False)

    async def _save():
        rows = await grid.get_client_data()
        for r in rows:
            athlete = r.get("Athlete")
            if athlete in s.athlete_ergs_df.index:
                s.athlete_ergs_df.at[athlete, "2k Erg"] = r.get("2k Erg")
        ui.notify("Erg scores saved", type="positive")
        await dash.recompute()

    with ui.row().classes("gap-2 mt-2"):
        ui.button("Save changes", icon="save", on_click=_save).props("color=primary size=sm")
        ui.button("Discard", icon="undo", on_click=lambda: dash.rebuild("Athletes")).props("outline size=sm")
