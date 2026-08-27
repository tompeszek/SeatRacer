"""Top-level worker for the leave-one-out analysis.

Kept as a module-level function (no closures, no UI imports) so it can be pickled
and run in a ``ProcessPoolExecutor`` under Windows' spawn start method. Each call
refits the model with one piece/day/week removed, in ``light`` mode, and returns
only the per-athlete coefficient + position metrics the caller needs.
"""
from __future__ import annotations


def run_job(job: dict) -> dict:
    model_class = job["model_class"]
    df = job["df"]
    kwargs = job["kwargs"]

    def _fit(max_corr):
        temp = model_class(df=df, max_correlation=max_corr, light=True, **kwargs)
        temp.run_analysis()
        return temp

    try:
        temp = _fit(1.0)
    except Exception:  # noqa: BLE001 - retry with a touch of regularization
        temp = _fit(0.99)

    fr = temp.final_results
    active = {}
    adf = fr.get("athletes")
    if adf is not None:
        for ath in adf.index:
            coef = float(adf.loc[ath, "Coefficient"])
            m = temp.calculate_position_metrics_for_coefficient(coef, ath[-1])
            active[ath] = {"new_coef": coef, "speed": m["speed"], "rank": m["rank"]}
    dropped = (list(fr["dropped_athletes"].index)
               if fr.get("dropped_athletes") is not None else [])
    return {
        "piece": job["piece"], "group_label": job["group_label"],
        "race_date": job["race_date"], "piece_number": job["piece_number"],
        "active": active, "dropped": dropped,
    }
