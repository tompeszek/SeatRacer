"""Golden-fixture generator for the TypeScript rewrite.

Runs the existing Python engine (pandas + statsmodels) over every bundled
dataset under a matrix of settings and writes JSON fixtures that the
TypeScript engine's tests must reproduce (see REWRITE_PLAN.md section 6).

Run from the repo root:  python tools/make_fixtures.py
Output:                  fixtures/*.json

This script is deleted along with the Python engine at the end of the
migration; the fixtures stay committed.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from seatracer.utils.data_handler import DataHandler  # noqa: E402
from seatracer.utils.helpers import (  # noqa: E402
    add_athlete_counts,
    determine_shell_class,
    time_to_seconds,
)
from seatracer.analysis.models.statsmodels.ols_analysis import OLSAnalysis  # noqa: E402
from seatracer.analysis.models.statsmodels.rlm_analysis import RLMAnalysis  # noqa: E402
from seatracer.analysis.models.statsmodels.glm_analysis import GLMAnalysis  # noqa: E402

OUT = REPO / "fixtures"
OUT.mkdir(exist_ok=True)

DATASETS = [
    "Olympic Selection - 2012.csv",
    "Olympic Selection - 2021.csv",
    "SDRC HOCR 2025 rate adjusted.csv",
    "SDRC HOCR 2025 raw.csv",
    "SDRC Masters Men HOCR Selection - 2024.csv",
    "fall_2025.csv",
]

# Settings matrix. Values mirror the UI options in seatracer/ng/models.py.
CONFIGS = {
    "base": dict(halflife=None, weight_close=None, weight_stern=None, include_coxswains=True),
    "recency": dict(halflife=56.0, weight_close=None, weight_stern=None, include_coxswains=True),
    "close": dict(halflife=None, weight_close=8.0, weight_stern=None, include_coxswains=True),
    "stern": dict(halflife=None, weight_close=None, weight_stern=0.5, include_coxswains=True),
    "combo": dict(halflife=21.0, weight_close=5.0, weight_stern=1.0, include_coxswains=False),
}

MODELS = {"ols": OLSAnalysis, "rlm": RLMAnalysis, "glm": GLMAnalysis}


def slug(name):
    return (
        name.replace(".csv", "")
        .lower()
        .replace(" - ", "-")
        .replace(" ", "-")
    )


def all_shell_classes(df):
    df = df.copy()
    add_athlete_counts(df)
    return sorted(df.apply(determine_shell_class, axis=1).unique().tolist())


def series_to_obj(s):
    return {str(k): float(v) for k, v in s.items()}


def run_model(model_class, df, shell_classes, cfg):
    analysis = model_class(
        df=df.copy(),
        max_correlation=1.1,  # keep every athlete; correlation dropping is display-only
        erg_scores=None,
        shell_class=shell_classes,
        light=True,
        **cfg,
    )
    analysis.run_analysis()
    results = analysis.final_results["results"]
    ci = results.conf_int()
    out = {
        "columns": [str(c) for c in results.params.index],
        "params": series_to_obj(results.params),
        "bse": series_to_obj(results.bse),
        "ci_lower": series_to_obj(ci[0]),
        "ci_upper": series_to_obj(ci[1]),
    }
    for attr in ("df_resid",):
        if hasattr(results, attr):
            try:
                out[attr] = float(getattr(results, attr))
            except Exception:
                pass
    return analysis, out


def prep_snapshot(analysis):
    """Capture the prepped rows and the exact design matrix inputs for one run.

    Reconstructs the same prep the engine does in run_analysis (weights and
    design columns), so the TS engine can be validated stage by stage.
    """
    from seatracer.utils.helpers import calculate_closest_margin

    df = analysis.df.copy()
    df["Race Session (date)"] = pd.to_datetime(df["Race Session (date)"])
    df["time_seconds"] = df["Result"].apply(time_to_seconds)
    df["time_per_500m"] = df["time_seconds"] / (df["KM"] * 2.0)
    df = calculate_closest_margin(df)
    df = analysis._apply_weights(df, analysis.weight_close, analysis.halflife)

    athletes = df["Personnel"].str.split("/", expand=True).stack().unique()
    athletes = [a for a in athletes if analysis.include_coxswains or not a.endswith("ᶜ")]
    athlete_weights = analysis._compute_athlete_weights(df, analysis.weight_stern)

    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "date": str(row["Race Session (date)"].date()),
                "piece": str(row["Piece"]),
                "km": float(row["KM"]),
                "personnel": str(row["Personnel"]),
                "rigging": str(row["Rigging"]),
                "shell_class": str(row["shell_class"]),
                "time_seconds": float(row["time_seconds"]),
                "time_per_500m": float(row["time_per_500m"]),
                "closest_margin": None
                if np.isinf(row["closest_margin"])
                else float(row["closest_margin"]),
                "closeness_factor": float(row["scaled_closeness_factor"]),
                "recency_factor": float(row["scaled_recency_factor"]),
                "total_weight": float(row["total_weight"]),
                "athlete_fractions": {
                    a: float(athlete_weights[a].at[idx])
                    for a in athletes
                    if a in athlete_weights and athlete_weights[a].at[idx] != 0.0
                },
            }
        )
    return {"athletes": sorted(map(str, athletes)), "rows": rows}


def helpers_fixture():
    """Small pure-function cases: time parsing and shell class detection."""
    times = ["13:05", "10:38", "03:28.5", "17:50.3", "00:59.95", "07:00"]
    shell_cases = []
    for rigging in ["c/p/s/p/s", "p/s", "x", "x/x", "x/x/c", "s/p/s/p", "c/s/p/s/p/s/p/s/p", "x/x/x/x"]:
        n = len(rigging.split("/"))
        rowers = len([r for r in rigging.split("/") if r != "c"])
        row = pd.Series({"athlete_count": n, "rower_count": rowers, "Rigging": rigging})
        shell_cases.append({"rigging": rigging, "shell_class": determine_shell_class(row)})
    return {
        "time_to_seconds": {t: float(time_to_seconds(t)) for t in times},
        "shell_class": shell_cases,
    }


def main():
    handler = DataHandler("data")
    (OUT / "helpers.json").write_text(
        json.dumps(helpers_fixture(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("wrote helpers.json")

    for dataset in DATASETS:
        raw = handler.load_dataset(dataset)
        shells = all_shell_classes(raw)
        ds = slug(dataset)
        for cfg_name, cfg in CONFIGS.items():
            fixture = {
                "dataset": dataset,
                "config": {**cfg, "shell_class": shells},
                "models": {},
            }
            snapshot_done = False
            for model_key, model_class in MODELS.items():
                analysis, out = run_model(model_class, raw, shells, cfg)
                fixture["models"][model_key] = out
                if not snapshot_done:
                    fixture["prep"] = prep_snapshot(analysis)
                    snapshot_done = True
            path = OUT / f"{ds}.{cfg_name}.json"
            path.write_text(
                json.dumps(fixture, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"wrote {path.name}  ({len(fixture['prep']['rows'])} rows)")


if __name__ == "__main__":
    main()
