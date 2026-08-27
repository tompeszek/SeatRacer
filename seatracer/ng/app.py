"""SeatRacer – NiceGUI application shell.

Builds a per-client :class:`Dashboard` containing the sidebar (analysis controls)
and the tabbed result views. Unlike Streamlit, changing a control runs a single
``recompute()`` (re-fit the model once) and rebuilds only the visible tab.
"""
from __future__ import annotations

import pandas as pd
from nicegui import ui, run

from seatracer.optimize.lineup_optimizer import LineupOptimizer

from .state import AppState
from .models import (
    AVAILABLE_MODELS, DEFAULT_MODEL, RECENCY_OPTIONS,
    CLOSE_RACES_OPTIONS, STERN_BIAS_OPTIONS,
)
from .tabs import (
    data_tab, athletes_tab, performance_tab, individual_tab, fairness_tab,
    correlations_tab, validation_tab, synergy_tab, new_lineup_tab,
    lineup_tab, optimal_tab, time_tab, debug_tab,
)

# Tab name -> builder function. Order defines the tab strip.
TAB_BUILDERS = {
    "Data": data_tab.build,
    "Athletes": athletes_tab.build,
    "Performance": performance_tab.build,
    "Individual": individual_tab.build,
    "Fairness": fairness_tab.build,
    "Correlations": correlations_tab.build,
    "Validation": validation_tab.build,
    "Synergies": synergy_tab.build,
    "New Lineup": new_lineup_tab.build,
    "Lineup Testing": lineup_tab.build,
    "Optimal Lineups": optimal_tab.build,
    "Over Time": time_tab.build,
    "Debug": debug_tab.build,
}
# Tabs that are usable before/without a fitted analysis.
DATA_TABS = {"Data", "Athletes"}


def _fit_model(model_class, df, max_corr, halflife, weight_close, weight_stern,
               include_cox, ergs, shell_class):
    """Fit the model (runs in a worker thread via run.io_bound, so no UI here).

    Returns (analysis, sides_count, correlation_auto_adjusted, error_str)."""
    try:
        def _run(mc):
            model = model_class(
                df=df.copy(), max_correlation=mc, halflife=halflife,
                weight_close=weight_close, weight_stern=weight_stern,
                include_coxswains=include_cox, erg_scores=ergs, shell_class=shell_class,
            )
            return model.run_analysis()

        analysis = _run(max_corr)
        auto_adj = False
        athletes_df = analysis.final_results.get("athletes", pd.DataFrame())
        if athletes_df.empty and max_corr < 1.0:
            analysis = _run(1.0)
            auto_adj = True
        return analysis, (getattr(analysis, "sides_count", {}) or {}), auto_adj, None
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        return None, {}, False, f"{type(exc).__name__}: {exc}"


class Dashboard:
    def __init__(self):
        self.state = AppState()
        self.tab_containers: dict[str, ui.element] = {}
        self.current_tab = "Data"
        # sidebar elements updated dynamically
        self._model_caption: ui.label | None = None
        self._weights_box: ui.column | None = None
        self.loading: ui.element | None = None

    def set_loading(self, on: bool):
        if self.loading is not None:
            self.loading.set_visibility(on)

    # ------------------------------------------------------------------ #
    # Analysis pipeline
    # ------------------------------------------------------------------ #
    async def recompute(self):
        """Re-fit the selected model once (off the event loop) and refresh the
        visible tab, showing a loading bar while the fit runs."""
        s = self.state
        s.last_error = None
        s.correlation_auto_adjusted = False
        if not s.has_data:
            s.analysis = None
            s.optimizer = None
            self.rebuild_current()
            return

        info = AVAILABLE_MODELS[s.model_name]
        if info["uses_custom_weighting"]:
            weight_close = CLOSE_RACES_OPTIONS[s.weight_close]["value"]
            weight_stern = (STERN_BIAS_OPTIONS[s.weight_stern]["value"]
                            if info["can_have_stern_bias"] else None)
            halflife = RECENCY_OPTIONS[s.recency]
        else:
            weight_close = weight_stern = halflife = None

        self.set_loading(True)
        try:
            analysis, sides, auto_adj, err = await run.io_bound(
                _fit_model, info["class"], s.current_data.copy(), s.max_correlation,
                halflife, weight_close, weight_stern, s.include_coxswains,
                s.athlete_ergs_df, list(s.shell_class),
            )
        finally:
            self.set_loading(False)

        if err:
            s.analysis = None
            s.optimizer = None
            s.last_error = err
            ui.notify(f"Analysis failed: {err}", type="negative", timeout=8000)
        else:
            s.analysis = analysis
            s.sides_count = sides
            s.correlation_auto_adjusted = auto_adj
            s.optimizer = LineupOptimizer(analysis)

        self.rebuild_current()

    # ------------------------------------------------------------------ #
    # Tab rendering
    # ------------------------------------------------------------------ #
    def rebuild(self, name: str):
        container = self.tab_containers.get(name)
        if container is None:
            return
        container.clear()
        with container:
            if name not in DATA_TABS and not self.state.has_analysis:
                self._placeholder()
            else:
                try:
                    TAB_BUILDERS[name](self)
                except Exception as exc:  # noqa: BLE001
                    ui.label(f"Error rendering this tab: {type(exc).__name__}: {exc}") \
                        .classes("text-red-600")

    def rebuild_current(self):
        self.rebuild(self.current_tab)

    def on_tab_change(self, name: str):
        self.current_tab = name
        self.rebuild(name)

    def _placeholder(self):
        with ui.column().classes("items-center w-full mt-12 gap-2"):
            ui.icon("table_view", size="48px").classes("text-gray-400")
            ui.label("Load racing data on the Data tab to see this analysis.") \
                .classes("text-gray-500")

    # ------------------------------------------------------------------ #
    # Sidebar (built by the module-level _render_sidebar helper)
    # ------------------------------------------------------------------ #
    def _build_weights(self):
        """(Re)build the model-weight controls; visibility depends on the model."""
        s = self.state
        info = AVAILABLE_MODELS[s.model_name]
        box = self._weights_box
        box.clear()
        if not info["uses_custom_weighting"]:
            return
        with box:
            ui.separator().classes("my-2")
            ui.label("Model Weights").classes("text-sm font-bold text-gray-700")

            self._weight_radio(
                "Close Races", "weight_close", CLOSE_RACES_OPTIONS, s.weight_close,
                lambda key: CLOSE_RACES_OPTIONS[key]["caption"],
            )
            if info["can_have_stern_bias"]:
                self._weight_radio(
                    "Stern Bias", "weight_stern", STERN_BIAS_OPTIONS, s.weight_stern,
                    lambda key: STERN_BIAS_OPTIONS[key]["caption"],
                )

            def _recency_caption(key):
                hl = RECENCY_OPTIONS[key]
                return (f"At {hl:.0f} days, a result's weight is halved" if hl is not None
                        else "Older races are weighted the same as recent ones")
            self._weight_radio("Recency Weighting", "recency", RECENCY_OPTIONS,
                               s.recency, _recency_caption)

    def _weight_radio(self, title, attr, options, value, caption_fn):
        ui.label(title).classes("text-xs font-medium text-gray-600 mt-2")
        caption = ui.label(caption_fn(value)).classes("text-xs text-gray-500 italic")

        async def _on(e):
            setattr(self.state, attr, e.value)
            caption.text = caption_fn(e.value)
            await self.recompute()
        ui.radio(list(options.keys()), value=value, on_change=_on) \
            .props("inline dense").classes("-my-1")

    # ------------------------------------------------------------------ #
    async def _on_model_change(self, e):
        self.state.model_name = e.value
        if self._model_caption:
            self._model_caption.text = AVAILABLE_MODELS[e.value]["description"]
        self._build_weights()
        await self.recompute()

    async def _set(self, attr, value):
        setattr(self.state, attr, value)
        await self.recompute()


def _global_head():
    ui.add_head_html("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=B612&display=swap');
      body, .q-page, .nicegui-content { font-family: 'B612', sans-serif; }
      .ag-theme-balham, .ag-theme-balham-dark { font-family: 'B612', sans-serif; }
    </style>
    """)


@ui.page("/")
def main_page():
    _global_head()
    dash = Dashboard()

    with ui.header().classes("items-center bg-blue-900 text-white"):
        ui.button(icon="menu", on_click=lambda: drawer.toggle()).props("flat color=white")
        ui.label("SeatRacer").classes("text-xl font-bold")
        ui.label("Rowing lineup & seat-racing analysis").classes("text-sm opacity-70 ml-2")
        ui.space()
        dash.loading = ui.row().classes("items-center gap-2")
        with dash.loading:
            ui.spinner(size="sm", color="white")
            ui.label("Analyzing...").classes("text-sm")
        dash.loading.set_visibility(False)

    # Build sidebar; capture the drawer so the header menu button can toggle it.
    # A low breakpoint keeps it pinned beside content on normal screens (instead of
    # overlaying) while still collapsing to an overlay on narrow/mobile viewports.
    with ui.left_drawer(value=True, bordered=True).classes("bg-gray-50") \
            .props("width=320 breakpoint=500") as drawer:
        with ui.scroll_area().classes("w-full h-full"):
            _render_sidebar(dash)

    with ui.tabs().classes("w-full").props("dense") as tabs:
        for name in TAB_BUILDERS:
            ui.tab(name)
    with ui.tab_panels(tabs, value="Data").classes("w-full"):
        for name in TAB_BUILDERS:
            with ui.tab_panel(name):
                dash.tab_containers[name] = ui.column().classes("w-full")
    tabs.on_value_change(lambda e: dash.on_tab_change(e.value))

    dash.rebuild("Data")


def _render_sidebar(dash: "Dashboard"):
    """Sidebar content (kept as a function so the drawer can wrap it)."""
    s = dash.state
    ui.label("Data Filters").classes("text-sm font-bold text-gray-700 mt-1")
    ui.select(
        ["1x", "2-", "4-", "4+", "8+"], value=list(s.shell_class), multiple=True,
        label="Include shell classes",
        on_change=lambda e: dash._set("shell_class", list(e.value or [])),
    ).props("use-chips dense outlined").classes("w-full")

    ui.separator().classes("my-2")
    ui.label("Model").classes("text-sm font-bold text-gray-700")
    ui.radio(list(AVAILABLE_MODELS.keys()), value=s.model_name,
             on_change=dash._on_model_change).props("dense").classes("-my-1")
    dash._model_caption = ui.label(AVAILABLE_MODELS[s.model_name]["description"]) \
        .classes("text-xs text-gray-500 italic")

    dash._weights_box = ui.column().classes("w-full gap-0")
    dash._build_weights()

    ui.separator().classes("my-2")
    ui.label("Parameters").classes("text-sm font-bold text-gray-700")
    ui.label("Max allowed correlation").classes("text-xs text-gray-600 mt-1")
    corr_caption = ui.label("").classes("text-xs text-gray-500 italic")

    def _corr_caption():
        corr_caption.text = (f"Only show athletes with no correlation greater than "
                             f"{s.max_correlation:.2f} to any other athlete")
    _corr_caption()
    slider = ui.slider(min=0.5, max=1.0, step=0.01, value=s.max_correlation) \
        .props("label-always").classes("w-full")

    def _corr_live(e):
        s.max_correlation = round(float(e.value), 2)
        _corr_caption()
    slider.on_value_change(_corr_live)
    slider.on("change", lambda _: dash.recompute())

    cox_caption = ui.label("").classes("text-xs text-gray-500 italic")

    async def _on_cox(e):
        s.include_coxswains = bool(e.value)
        cox_caption.text = ("Include coxswains in analysis" if e.value
                            else "Ignore coxswains - assume minimal impact on crew speed")
        await dash.recompute()
    ui.switch("Evaluate coxswain performance", value=s.include_coxswains,
              on_change=_on_cox).props("dense").classes("mt-2")
    cox_caption.text = ("Include coxswains in analysis" if s.include_coxswains
                        else "Ignore coxswains - assume minimal impact on crew speed")
