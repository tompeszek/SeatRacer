from seatracer.analysis.models.statsmodels.statsmodel_base import StatsModelAnalysis
from seatracer.analysis.registry import ModelRegistry

@ModelRegistry.register(
    key="glm", 
    name="Generalized Linear Model",
    description="Flexible generalized linear model, similiar to Ordinary Least Squares (OLS) regression, but with the ability to apply weights.",
    recommended=False,
    order=2,
)
class GLMAnalysis(StatsModelAnalysis):
    """
    Generalized Linear Model regression analysis.
    
    GLM extends linear models to allow for non-normal error distributions.
    This implementation uses a Gaussian family (equivalent to OLS) but with
    the ability to apply frequency weights.
    
    This model is appropriate when:
    - You want the flexibility of the GLM framework
    - You plan to extend to other error distributions in the future
    - You need to apply weights to observations
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.selected_model = 'glm'