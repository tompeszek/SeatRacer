"""Data tab: load example datasets, upload a CSV, and edit the racing data."""
from __future__ import annotations

import pandas as pd
from nicegui import ui

from seatracer.utils.data_handler import DataHandler
from ..ui_common import df_to_grid

EXPECTED_COLUMNS = ["Race Session (date)", "Piece", "KM", "Rigging", "Personnel", "Result"]


def build(dash):
    s = dash.state
    handler = DataHandler("data")

    async def _load_example(name):
        s.current_data = handler.load_dataset(name)
        s.athlete_ergs_df = None      # rebuild erg table for the new roster
        s.loo_results, s.loo_level = {}, None
        ui.notify(f"Loaded {name} ({len(s.current_data)} rows)", type="positive")
        await dash.recompute()

    async def _on_upload(e):
        try:
            df = pd.read_csv(e.content)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Could not read CSV: {exc}", type="negative")
            return
        s.current_data = df
        s.athlete_ergs_df = None
        s.loo_results, s.loo_level = {}, None
        ui.notify(f"Loaded {len(df)} rows from {e.name}", type="positive")
        await dash.recompute()

    with ui.row().classes("w-full items-start gap-6"):
        with ui.column().classes("gap-1"):
            ui.label("Load example dataset").classes("text-base font-semibold")
            datasets = handler.get_available_datasets()
            if datasets:
                with ui.row().classes("gap-2 flex-wrap"):
                    for name in datasets:
                        ui.button(name.replace(".csv", ""),
                                  on_click=lambda n=name: _load_example(n)) \
                            .props("outline size=sm")
            else:
                ui.label("No example datasets available.").classes("text-gray-500")

        with ui.column().classes("gap-1"):
            ui.label("Upload racing data (CSV)").classes("text-base font-semibold")
            ui.upload(on_upload=_on_upload, auto_upload=True).props("accept=.csv").classes("max-w-sm")

    ui.separator().classes("my-3")

    ui.label("Edit racing data").classes("text-base font-semibold")
    if not s.has_data:
        ui.label("No data loaded yet. Pick an example or upload a CSV above.") \
            .classes("text-gray-500")
        return

    grid = df_to_grid(
        s.current_data,
        editable_columns=list(s.current_data.columns),
        height=420, auto_height=False,
    )
    grid.props("rowSelection=multiple")

    async def _save():
        rows = await grid.get_client_data()
        s.current_data = pd.DataFrame(rows)
        ui.notify("Changes saved", type="positive")
        await dash.recompute()

    def _add_row():
        blank = {c: "" for c in s.current_data.columns}
        grid.options["rowData"].append(blank)
        grid.update()

    async def _delete_selected():
        selected = await grid.get_selected_rows()
        if not selected:
            ui.notify("Select one or more rows first", type="warning")
            return
        rows = await grid.get_client_data()
        keep = [r for r in rows if r not in selected]
        grid.options["rowData"] = keep
        grid.update()
        ui.notify(f"Removed {len(selected)} row(s) - remember to Save", type="info")

    async def _clear():
        s.current_data = pd.DataFrame()
        s.analysis = None
        ui.notify("Data cleared", type="info")
        await dash.recompute()

    with ui.row().classes("gap-2 mt-2"):
        ui.button("Save changes", icon="save", on_click=_save).props("color=primary size=sm")
        ui.button("Add row", icon="add", on_click=_add_row).props("outline size=sm")
        ui.button("Delete selected", icon="delete", on_click=_delete_selected).props("outline size=sm")
        ui.button("Clear data", icon="clear", on_click=_clear).props("outline color=negative size=sm")
