from seatracer.analysis.models.statsmodels.statsmodel_base import StatsModelAnalysis
from seatracer.analysis.registry import ModelRegistry

@ModelRegistry.register(
    key="wls", 
    name="Weighted Least Squares",
    description="Regression with weighted observations for varying reliability",
    recommended=False
)
class WLSAnalysis(StatsModelAnalysis):
    """
    Weighted Least Squares regression model analysis.
    
    WLS uses weights for each observation, giving more influence to some
    observations and less to others based on their reliability or importance.
    This model is appropriate when:
    - Residuals have heteroscedasticity (non-constant variance)
    - Some observations are more reliable than others
    - You want to give more weight to recent data or close races
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.selected_model = 'wls'