"""Model metadata for the UI.

Mirrors the ``ALL_MODELS`` dict from the old ``seatracer/app.py`` so the visible
model set and weighting capabilities are identical to the Streamlit version.
"""
from seatracer.analysis.models.statsmodels.ols_analysis import OLSAnalysis
from seatracer.analysis.models.statsmodels.glm_analysis import GLMAnalysis
from seatracer.analysis.models.statsmodels.rlm_analysis import RLMAnalysis
from seatracer.analysis.models.statsmodels.wls_analysis import WLSAnalysis
from seatracer.analysis.models.gradient_descent.gradient_descent import GradientDescentAnalysis
from seatracer.analysis.models.machine_learning.en_analysis import ElasticNetAnalysis

ALL_MODELS = {
    "Ordinary Least Squares": {
        "class": OLSAnalysis,
        "description": "Finds rower contributions by minimizing squared differences between predicted and actual race times",
        "uses_custom_weighting": False,
        "can_have_stern_bias": False,
        "show_athletes": True,
    },
    "Generalized Linear Model": {
        "class": GLMAnalysis,
        "description": "Flexible generalized linear model, similar to Ordinary Least Squares (OLS) regression, but with the ability to apply weights.",
        "uses_custom_weighting": True,
        "can_have_stern_bias": True,
        "show_athletes": True,
    },
    "Robust Linear Model": {
        "class": RLMAnalysis,
        "description": "Robust linear regression resistant to outliers",
        "uses_custom_weighting": True,
        "can_have_stern_bias": True,
        "show_athletes": True,
    },
    "Weighted Least Squares": {
        "class": WLSAnalysis,
        "description": "Weighted least squares regression",
        "uses_custom_weighting": True,
        "can_have_stern_bias": True,
        "show_athletes": True,
    },
    "Gradient Descent": {
        "class": GradientDescentAnalysis,
        "description": "Iteratively adjusts each rower's estimated performance using absolute errors rather than squared errors, and can start with erg scores as initial values.",
        "uses_custom_weighting": True,
        "can_have_stern_bias": True,
        "show_athletes": True,
    },
    "Elastic Net": {
        "class": ElasticNetAnalysis,
        "description": "Elastic net regularized regression",
        "uses_custom_weighting": False,
        "can_have_stern_bias": False,
        "show_athletes": True,
    },
}

# Only these three appear in the UI (matches the old app's VISIBLE_MODELS).
VISIBLE_MODELS = ["Gradient Descent", "Generalized Linear Model", "Ordinary Least Squares"]
AVAILABLE_MODELS = {name: ALL_MODELS[name] for name in VISIBLE_MODELS}
DEFAULT_MODEL = "Generalized Linear Model"

# Weighting option maps (values mirror the old sidebar definitions).
RECENCY_OPTIONS = {"Off": None, "Low": 210.0, "Medium": 56.0, "High": 21.0}

CLOSE_RACES_OPTIONS = {
    "Off": {"value": None, "caption": "Margins do not affect race result weighting"},
    "Low": {"value": 12.0, "caption": 'Races determined by 1" are weighted twice as much as those with a 12" margin'},
    "Medium": {"value": 8.0, "caption": 'Races determined by 1" are weighted twice as much as those with an 8" margin'},
    "High": {"value": 5.0, "caption": 'Races determined by 1" are weighted twice as much as those with a 5" margin'},
}

STERN_BIAS_OPTIONS = {
    "Off": {"value": 0.0, "caption": "Rowers in all positions get the same credit or blame for every result"},
    "Low": {"value": 0.1, "caption": "Stroke seat gets 10% more credit or blame than bow seat"},
    "Medium": {"value": 0.5, "caption": "Stroke seat gets 50% more credit or blame than bow seat"},
    "High": {"value": 1.0, "caption": "Stroke seat gets 100% more credit or blame than bow seat"},
}
