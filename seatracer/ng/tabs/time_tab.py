"""Over Time tab: rolling/cumulative performance trends per side."""
from __future__ import annotations

from nicegui import ui, run

from ..temporal_plot import build_position_figure

POSITION_SUFFIX = {"Starboard": "ˢ", "Port": "ᵖ", "Sculling": "ˣ", "Coxswain": "ᶜ"}


def build(dash):
    s = dash.state
    analysis = s.analysis

    ui.label("Performance over time").classes("text-base font-semibold")

    with ui.row().classes("items-center gap-6"):
        run_btn = ui.button("Run historical analysis", icon="timeline").props("color=primary")
        with ui.column().classes("gap-0"):
            ui.switch("X-axis by piece", value=s.ot_by_piece,
                      on_change=lambda e: setattr(s, "ot_by_piece", bool(e.value))).props("dense")
            ui.label().bind_text_from(
                s, "ot_by_piece",
                lambda v: "X-axis will be by piece." if v else "X-axis will be by date.") \
                .classes("text-xs text-gray-500")
        with ui.column().classes("gap-0 w-64"):
            ui.label().bind_text_from(s, "ot_lookback", lambda v: f"Lookback days: {v}") \
                .classes("text-xs text-gray-600")
            ui.slider(min=1, max=100, step=1, value=s.ot_lookback,
                      on_change=lambda e: setattr(s, "ot_lookback", int(e.value))) \
                .props("label").classes("w-full")

    result_box = ui.column().classes("w-full mt-2")

    async def _run():
        run_btn.disable()
        spinner = ui.spinner(size="lg")
        try:
            await run.io_bound(analysis.run_history, custom_lookback=s.ot_lookback,
                               by_piece=s.ot_by_piece)
            s.ot_ran = True
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Historical analysis failed: {exc}", type="negative")
        finally:
            spinner.delete()
            run_btn.enable()
        dash.rebuild("Over Time")

    run_btn.on_click(_run)

    has_history = getattr(analysis, "temporal_data", {}).get("time_series_df") is not None
    if not has_history:
        with result_box:
            ui.label("Click 'Run historical analysis' to see performance trends over time.") \
                .classes("text-gray-500 mt-2")
        return

    stats_df = analysis.temporal_data["stats_df"]
    available = [pos for pos, suf in POSITION_SUFFIX.items()
                 if any(str(r).endswith(suf) for r in stats_df["Rower"])]
    if not available:
        with result_box:
            ui.label("No position data available for visualization.").classes("text-gray-500")
        return

    if s.ot_position not in available:
        s.ot_position = available[0]

    with result_box:
        with ui.row().classes("items-center gap-2"):
            ui.label("Position").classes("text-sm")
            ui.radio(available, value=s.ot_position,
                     on_change=lambda e: (setattr(s, "ot_position", e.value), _draw())) \
                .props("inline dense")
        chart_box = ui.column().classes("w-full")

    def _draw():
        chart_box.clear()
        fig = build_position_figure(analysis, s.ot_position)
        with chart_box:
            if fig is None:
                ui.label(f"No data for {s.ot_position}.").classes("text-gray-500")
            else:
                ui.plotly(fig).classes("w-full").style("height: 560px")

    _draw()
