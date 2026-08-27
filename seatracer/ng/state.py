"""Per-client application state.

One :class:`AppState` is created per browser connection (inside the ``@ui.page``
handler), giving every user an isolated session – the NiceGUI equivalent of
Streamlit's ``st.session_state``, but without the full-script reruns.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

import pandas as pd

from .models import DEFAULT_MODEL


@dataclass
class AppState:
    # ---- data ----
    current_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    athlete_ergs_df: Optional[pd.DataFrame] = None

    # ---- computed analysis ----
    analysis: Any = None
    optimizer: Any = None
    sides_count: Dict[str, Dict[str, int]] = field(default_factory=dict)
    correlation_auto_adjusted: bool = False
    last_error: Optional[str] = None

    # ---- sidebar parameters ----
    shell_class: List[str] = field(default_factory=lambda: ["1x", "2-", "4-", "4+", "8+"])
    model_name: str = DEFAULT_MODEL
    weight_close: str = "Off"
    weight_stern: str = "Off"
    recency: str = "Off"
    max_correlation: float = 0.8
    include_coxswains: bool = True

    # ---- Performance tab ----
    perf_boat_class: str = "4x/-"
    perf_distance: int = 2000
    conf: Dict[str, int] = field(default_factory=lambda: {
        "starboard": 50, "port": 50, "scull": 50, "coxswain": 50,
    })

    # ---- Individual (leave-one-out) tab ----
    loo_results: Dict[str, dict] = field(default_factory=dict)   # level -> results bundle
    loo_level: Optional[str] = None                              # currently displayed level
    loo_athlete: Optional[str] = None

    # ---- Over Time tab ----
    ot_by_piece: bool = True
    ot_lookback: int = 50
    ot_ran: bool = False
    ot_position: Optional[str] = None

    # ---- Lineup Testing tab ----
    lt_boat_count: int = 2
    lt_lineups: List[dict] = field(default_factory=list)  # [{'boat_class': str, 'seats': [name,...]}]

    # ------------------------------------------------------------------
    @property
    def has_data(self) -> bool:
        return self.current_data is not None and not self.current_data.empty

    @property
    def has_analysis(self) -> bool:
        return self.analysis is not None and getattr(self.analysis, "final_results", None) is not None
