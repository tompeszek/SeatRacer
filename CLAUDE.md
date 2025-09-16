# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

SeatRacer is a Streamlit-based rowing lineup analysis and optimization tool. The application analyzes seat racing data to evaluate athlete performance, predict optimal lineups, and provide statistical insights for rowing teams.

## Running the Application

### Primary Command
```bash
streamlit run seatracer/app.py
```

The app will be available at http://localhost:8501 by default.

### Development with Debugging
The app includes debugpy configuration for VS Code debugging on port 5678. The debugger will wait for attachment on startup.

### Installation
```bash
pip install -e .
# or
pip install -r seatracer/requirements.txt
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

**User Interface (`seatracer/ui/sections/`)**
- Modular UI sections for different analysis views (athletes, performance, correlations, etc.)
- Each section is a separate module for maintainability

**Utilities (`seatracer/utils/`)**
- `data_handler.py` - Data loading and processing
- `grouping.py` - Athlete correlation and grouping logic  
- `helpers.py` - Utility functions for time conversion, shell classification

**Visualization (`seatracer/visualization/`)**
- `charts.py` - Chart generation for analysis results
- `temporal_visualization.py` - Time-series visualization

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

1. **Data Loading**: CSV files with seat racing results loaded via `data_section.py`
2. **Analysis**: Selected model processes data with user-configured parameters
3. **Optimization**: `LineupOptimizer` uses analysis results to find optimal lineups
4. **Visualization**: Results displayed across multiple UI tabs

### Key Data Structures

- Race data includes: athletes, shell classes, race sessions, margins, positions
- Analysis outputs: athlete coefficients, performance metrics, correlations
- Optimization results: lineup predictions, performance estimates

## Configuration

### Streamlit Config
- `seatracer/_config.toml` - Sets B612 font for UI consistency

### Model Parameters
Models support various weighting options:
- **Recency weighting** - Recent races weighted more heavily
- **Close race weighting** - Closer margins get higher weight  
- **Stern bias** - Position-based performance adjustments
- **Correlation filtering** - Remove highly correlated athletes

## Development Notes

- The app uses Streamlit's session state extensively for persistence
- Hot-reloading friendly model registry design
- Debugpy integration for development debugging
- Modular UI design allows independent section development
- All analysis models inherit from `Analysis` base class
- Data preprocessing includes shell class filtering and rigging assignment