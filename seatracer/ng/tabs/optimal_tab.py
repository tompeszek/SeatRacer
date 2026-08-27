"""Optimal Lineups tab: placeholder (the search was never implemented upstream)."""
from nicegui import ui


def build(dash):
    with ui.column().classes("items-center w-full mt-12 gap-2"):
        ui.icon("construction", size="40px").classes("text-gray-400")
        ui.label("Optimal lineup search is not implemented yet.").classes("text-gray-500")
        ui.label("Use the Lineup Testing tab to compare hand-picked lineups.") \
            .classes("text-xs text-gray-400")
