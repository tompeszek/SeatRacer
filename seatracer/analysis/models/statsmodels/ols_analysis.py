from seatracer.analysis.models.statsmodels.statsmodel_base import StatsModelAnalysis
from seatracer.analysis.registry import ModelRegistry

@ModelRegistry.register(
    key="ols", 
    name="Ordinary Least Squares",
    description="Finds rower contributions by minimizing squared differences between predicted and actual race times",
    uses_custom_weighting=True,
    can_have_stern_bias=True,
    show_athletes=True,
    recommended=True,
    order=3
)
class OLSAnalysis(StatsModelAnalysis):
    """
    Ordinary Least Squares regression model analysis.
    
    OLS minimizes the sum of squared residuals to find the best-fitting line.
    This model is appropriate when:
    - Residuals are homoscedastic (constant variance)
    - All observations should be weighted equally
    - There are few outliers
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.selected_model = 'ols'