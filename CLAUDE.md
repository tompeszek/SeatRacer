# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

SeatRacer is a rowing lineup analysis and optimization tool. It analyzes seat
racing data to evaluate athlete performance, predict lineups, and provide
statistical insights for rowing teams.

The UI is built with **NiceGUI** (`seatracer/ng/`, entrypoint `main.py`). It was
migrated from an earlier Streamlit UI, which has been removed (recoverable from
git history if ever needed). See `README.md` for the architecture overview and
`DEPLOYMENT.md` for Railway notes.

## Running the Application

### Primary Command
```bash
python main.py
```

The app is available at http://localhost:8088 (binds `0.0.0.0`, reads `$PORT`).

### Installation
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components

**Analysis Engine (`seatracer/analysis/`)**
- `analysis_base.py` - Abstract base class for all analysis models
- `registry.py` - ModelRegistry for managing different analysis model types
- `temporal_analysis.py` - Time-series analysis functionality

**Analysis Models (`seatracer/analysis/models/`)**
- `statsmodels/` - Statistical models (OLS, GLM, RLM, WLS)
- `machine_learning/` - ML models (Random Forest, XGBoost, Elastic Net)
- `gradient_descent/` - Custom gradient descent implementation
- `trueskill/` - TrueSkill-based rating system

**Optimization (`seatracer/optimize/`)**
- `lineup_optimizer.py` - Primary lineup optimization using analysis results
- `lineup_optimizer2.py` - Alternative optimization implementation

**User Interface (`seatracer/ng/`)**
- `app.py` - Dashboard: sidebar controls + tabs + `recompute()` pipeline
- `state.py` - `AppState` per-client session state
- `ui_common.py` - AG Grid / ECharts builders, probability-matrix and confidence maths
- `temporal_plot.py` - Plotly builder for the Over Time tab
- `loo_worker.py` - ProcessPoolExecutor worker for leave-one-out refits
- `tabs/` - one module per tab (Data, Performance, Individual, ...)

**Utilities (`seatracer/utils/`)**
- `data_handler.py` - Data loading and processing
- `grouping.py` - Athlete correlation and grouping logic  
- `helpers.py` - Utility functions for time conversion, shell classification

### Model Registry System

The application uses a decorator-based model registry (`ModelRegistry`) to dynamically register analysis models. Models are registered with metadata like:
- Custom weighting support
- Stern bias capability  
- Athlete display settings
- UI ordering

Example registration:
```python
@ModelRegistry.register(
    key="ols",
    name="Linear Regression", 
    uses_custom_weighting=True,
    show_athletes=True
)
class OLSAnalysis(StatsModelBase):
    # Implementation
```

### Data Flow

1. **Data Loading**: CSV files with seat racing results loaded via the Data tab (`seatracer/ng/tabs/data_tab.py`)
2. **Analysis**: Selected model processes data with user-configured parameters
3. **Optimization**: `LineupOptimizer` uses analysis results to find optimal lineups
4. **Visualization**: Results displayed across multiple UI tabs

### Key Data Structures

- Race data includes: athletes, shell classes, race sessions, margins, positions
- Analysis outputs: athlete coefficients, performance metrics, correlations
- Optimization results: lineup predictions, performance estimates

## Configuration

### Model Parameters
Models support various weighting options:
- **Recency weighting** - Recent races weighted more heavily
- **Close race weighting** - Closer margins get higher weight  
- **Stern bias** - Position-based performance adjustments
- **Correlation filtering** - Remove highly correlated athletes

## Development Notes

- Per-client UI state lives in `seatracer/ng/state.py` (`AppState`); the engine is framework-agnostic and must not import any UI library
- Hot-reloading friendly model registry design
- Modular tab design allows independent development of each view
- All analysis models inherit from `Analysis` base class
- Data preprocessing includes shell class filtering and rigging assignment