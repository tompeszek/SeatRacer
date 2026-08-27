# SeatRacer (NiceGUI)

Rowing lineup & seat-racing analysis. This app was migrated from Streamlit to
[NiceGUI](https://nicegui.io) for a snappier, event-driven UI: changing a control
re-fits the model **once** and refreshes only the visible tab, instead of
re-running the whole script on every interaction.

## Run locally

```bash
python -m venv .venv && .venv\Scripts\activate      # (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python main.py
```

Then open http://localhost:8088. Load an example dataset on the **Data** tab (or
upload your own CSV) to populate the analysis tabs.

## Deploy on Railway

The repo already contains everything Railway/nixpacks needs:

- `requirements.txt` – Python dependencies (no Streamlit)
- `Procfile` / `railway.toml` – start command `python main.py`
- `runtime.txt` – Python version

`main.py` binds `0.0.0.0` and reads `$PORT`, so a fresh Railway project that
deploys this repo works with no extra configuration. Optionally set a
`STORAGE_SECRET` env var for signed per-client storage in production.

## Architecture

```
main.py                      entrypoint (reads $PORT, calls ui.run)
seatracer/
  analysis/  optimize/  utils/   <- statistical engine (framework-agnostic, reused as-is)
  ng/                            <- NiceGUI frontend (this migration)
    app.py            Dashboard: sidebar + tabs + recompute() pipeline
    state.py          AppState  (per-client session, replaces st.session_state)
    models.py         visible models + weighting option maps
    ui_common.py      AG Grid + ECharts builders, prob-matrix / confidence maths
    temporal_plot.py  Streamlit-free Plotly builder for the Over Time tab
    tabs/             one module per tab (Data, Performance, Individual, ...)
```

The old Streamlit UI (`seatracer/app.py`, `seatracer/ui/`, `seatracer/visualization/`)
has been removed; it lives on in git history if ever needed.

The engine (`analysis/`, `optimize/`, `utils/`) is unchanged except for three
small edits that removed its hard dependency on `streamlit` so it can run under
any frontend:

- `analysis/registry.py` – model registry uses a module-level dict instead of
  `st.session_state`.
- `analysis/analysis_base.py` – side counts are stored on the analysis object
  (`analysis.sides_count`) instead of being written into `st.session_state`.
- `utils/helpers.py` – a single `st.warning(...)` became `warnings.warn(...)`.

### Tabs

Data, Athletes (erg scores), Performance, Individual (leave-one-out influence),
Fairness, Correlations, Validation, Synergies, New Lineup, Lineup Testing,
Optimal Lineups (placeholder, as upstream), Over Time, Debug.

### UI building blocks

- **Tables** → `ui.aggrid` (sortable / filterable; editable on Data & Athletes).
- **Charts** → `ui.echart` (confidence-interval bars, bias bars, per-piece
  impact), inline SVG sparklines (per-rower confidence distributions on the
  Performance tab), and `ui.plotly` (Over Time trends).
- **Reactivity** → each tab is rebuilt on demand from `AppState`; sidebar control
  changes call `Dashboard.recompute()` which re-fits the model and refreshes the
  current tab only. The max-correlation slider recomputes on release (not on every
  drag step) to stay responsive.

## Notes

- Athlete names carry superscript rigging marks (ᵖ ˢ ᶜ ˣ). The browser renders
  these as UTF-8; `main.py` also reconfigures stdout/stderr to UTF-8 so Windows
  console logging doesn't choke on them.
- The Over Time view computes its position athlete list directly from the temporal
  stats frame. The engine's `get_position_athletes` name is shadowed by a later,
  suffix-based override, so the temporal builder doesn't rely on it.
