from seatracer.analysis.analysis_base import Analysis
import pandas as pd
import statsmodels.api as sm
from seatracer.utils.helpers import *

class StatsModelAnalysis(Analysis):
    """Base class for statsmodels-based analyses"""
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.selected_model = getattr(self, 'model_key', 'ols')  # Default to 'ols' if not specified
    
    def _run_regression(self, df, weights, athletes, shell_classes):
        """Implement the run_regression method for statsmodels"""
        # Prepare design matrix and response variable
        X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)])
        y = df['time_per_500m']

        # Ensure all columns are float64
        for col in X.columns:
            if X[col].dtype != 'float64':
                X[col] = X[col].astype("float64")

        # Run the selected regression model
        results = self._run_statsmodels_regression(self.selected_model, X, y, weights)

        # Generate comparison dataframe
        comparison_df = self._create_comparison_df(df, y, results, X)
        
        # Generate athlete statistics
        athletes_df, dropped_athletes_df = self._create_athlete_stats(results, athletes, X, self.max_correlation)
        
        # Generate shell class statistics
        shell_classes_df = self._create_shell_class_stats(results, shell_classes)
        
        # Generate other factors statistics
        other_factors_df = self._create_other_factors_stats(results, X, athletes)
        
        # Check race balance
        race_balance_info = self._check_race_balance(comparison_df)

        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        if not athletes_df.empty:
            self._add_correlations(athletes_df, athletes, X, corr_matrix)
        
        return {
            'results': results,
            'comparison': comparison_df,
            'athletes': athletes_df,
            'factors': other_factors_df,
            'shell_classes': shell_classes_df,
            'fitted': generate_fitted_values_vs_actual(df, results, athletes, shell_classes),
            'raw': df,
            'corr': X.corr(),
            'weights': weights,
            'dropped_athletes': dropped_athletes_df,
            'race_balance': race_balance_info,
            'pairs': self._create_athlete_pairs_df(df, y, results, X, athletes),
        }
    
    def _run_statsmodels_regression(self, model_type, X, y, weights):
        """Run regression using statsmodels"""
        match model_type:
            case 'rlm':
                model = sm.RLM(y, X)         
            case 'wls':
                model = sm.WLS(y, X, weights=weights)       
            case 'ols':
                # Use WLS when weights are provided, otherwise standard OLS
                if weights is not None and not np.allclose(weights, 1.0):
                    model = sm.WLS(y, X, weights=weights)
                else:
                    model = sm.OLS(y, X)
            case 'glm':            
                model = sm.GLM(y, X, family=sm.families.Gaussian(), freq_weights=weights)
            case _:
                model = sm.RLM(y, X)

        return model.fit()