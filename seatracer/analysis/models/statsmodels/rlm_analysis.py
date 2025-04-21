from seatracer.analysis.models.statsmodels.statsmodel_base import StatsModelAnalysis
from seatracer.analysis.registry import ModelRegistry

@ModelRegistry.register(
    key="rlm", 
    name="Robust Linear Model",
    description="Regression robust to outliers using M-estimation",
    recommended=False
)
class RLMAnalysis(StatsModelAnalysis):
    """
    Robust Linear Model regression analysis.
    
    RLM uses M-estimation to reduce the influence of outliers on the regression.
    This model is appropriate when:
    - The data contains outliers that should not be removed
    - You want the model to be less sensitive to extreme values
    - There are races with unusual conditions or performances
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.selected_model = 'rlm'