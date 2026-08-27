"""Tab builders for the SeatRacer NiceGUI app.

Each module exposes ``build(dash)`` which renders that tab's content into the
currently-open NiceGUI container (managed by ``Dashboard.rebuild``).
"""
from . import (  # noqa: F401
    data_tab, athletes_tab, performance_tab, individual_tab, fairness_tab,
    correlations_tab, validation_tab, synergy_tab, new_lineup_tab,
    lineup_tab, optimal_tab, time_tab, debug_tab,
)
