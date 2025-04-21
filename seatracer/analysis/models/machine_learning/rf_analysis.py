from seatracer.analysis.analysis_base import Analysis
from seatracer.analysis.registry import ModelRegistry
from seatracer.utils.helpers import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RandomForestResults:
    """Custom results class to mimic statsmodels results interface"""
    def __init__(self, model, X, y, feature_importances=None):
        self.model = model
        self._X = X
        self._y = y
        
        # Create a params series to mimic statsmodels
        self.params = pd.Series(feature_importances, index=X.columns)
        
        # Calculate confidence intervals from feature importances
        # This is a simplification as RF doesn't provide direct CIs
        ci_lower = feature_importances - np.abs(feature_importances) * 0.2
        ci_upper = feature_importances + np.abs(feature_importances) * 0.2
        
        # Handle edge cases
        epsilon = 1e-6
        identical_indices = np.isclose(ci_lower, ci_upper, atol=epsilon)
        if np.any(identical_indices):
            ci_lower[identical_indices] -= epsilon
            ci_upper[identical_indices] += epsilon
        
        self._ci_lower = ci_lower
        self._ci_upper = ci_upper
    
    def predict(self, X):
        """Predict using the random forest model"""
        if isinstance(X, pd.DataFrame):
            # Ensure all columns needed by the model are present
            missing_cols = set(self._X.columns) - set(X.columns)
            if missing_cols:
                for col in missing_cols:
                    X[col] = 0  # Add missing columns with zeros
            
            # Reorder columns to match training data
            X = X[self._X.columns]
        
        return self.model.predict(X)
    
    def conf_int(self, alpha=0.05):
        """Return confidence intervals based on feature importance variation"""
        # Replace any potential infinity or NaN values
        lower = pd.Series(np.nan_to_num(self._ci_lower, nan=0.0, posinf=1e5, neginf=-1e5), index=self._X.columns)
        upper = pd.Series(np.nan_to_num(self._ci_upper, nan=0.0, posinf=1e5, neginf=-1e5), index=self._X.columns)
        
        # Ensure bounds are finite and well-behaved
        for col in lower.index:
            if not np.isfinite(lower[col]) or not np.isfinite(upper[col]):
                lower[col] = self.params[col] - abs(self.params[col]) * 0.2
                upper[col] = self.params[col] + abs(self.params[col]) * 0.2
            
            # Ensure lower is actually lower than upper
            if lower[col] > upper[col]:
                lower[col], upper[col] = upper[col], lower[col]
            
            # If they're still identical, add a small separation
            if np.isclose(lower[col], upper[col]):
                epsilon = max(1e-6, abs(lower[col] * 0.01))
                lower[col] -= epsilon
                upper[col] += epsilon
        
        return pd.DataFrame({0: lower, 1: upper})


class RandomForestWithOLS:
    """Wrapper class that combines Random Forest predictions with OLS confidence intervals"""
    def __init__(self, rf_results, ols_results=None):
        # Get feature importances from Random Forest
        self.params = rf_results.params
        
        # Store original results
        self._rf_results = rf_results
        self._ols_results = ols_results
        
        # If OLS results are provided, use them for statistical tests
        if ols_results is not None:
            # Calculate the width of OLS confidence intervals
            ols_ci = ols_results.conf_int()
            self._ci_width = ols_ci[1] - ols_ci[0]
    
    def predict(self, X):
        """Predict using random forest model"""
        return self._rf_results.predict(X)
    
    def conf_int(self, alpha=0.05):
        """Return confidence intervals"""
        if self._ols_results is not None:
            # Use OLS-based confidence intervals if available
            if alpha != 0.05:
                ols_ci = self._ols_results.conf_int(alpha=alpha)
                ci_width = ols_ci[1] - ols_ci[0]
            else:
                ci_width = self._ci_width
            
            # Center the confidence intervals on the RF parameters
            lower = self.params - ci_width / 2
            upper = self.params + ci_width / 2
            
            return pd.DataFrame({0: lower, 1: upper})
        else:
            # Fall back to RF-based intervals if OLS is not available
            return self._rf_results.conf_int(alpha=alpha)
    
    def summary(self):
        """Return summary from OLS or a simplified summary if not available"""
        if self._ols_results is not None:
            return self._ols_results.summary()
        else:
            # Create a simplified summary
            return f"Random Forest Model\n" \
                   f"Number of trees: {self._rf_results.model.n_estimators}\n" \
                   f"R² score: {self._rf_results.model.score(self._rf_results._X, self._rf_results._y):.4f}"
    
    # Add any other methods from the statsmodels results object that you need
    def __getattr__(self, name):
        """Delegate any missing attributes/methods to the OLS results if available"""
        if self._ols_results is not None and hasattr(self._ols_results, name):
            return getattr(self._ols_results, name)
        elif hasattr(self._rf_results, name):
            return getattr(self._rf_results, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


@ModelRegistry.register(
    key="random_forest", 
    name="Random Forest Regressor",
    description="Machine learning approach using Random Forest regression. This 'works' but the results are junk. Not imported.",
    uses_custom_weighting=True,
    can_have_stern_bias=True,
    show_athletes=True,
    order=1,
    recommended=False,
)
class RandomForestAnalysis(Analysis):
    """
    Random Forest-based analysis.
    
    Uses a random forest regressor to model rowing performance.
    This model is appropriate when:
    - You want a non-linear approach that can capture complex interactions
    - You want feature importance measurements to understand rower contributions
    - You have sufficient data to train a machine learning model
    - You want robust predictions less susceptible to outliers
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # Extract RF-specific parameters with defaults
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', None)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.bootstrap = kwargs.get('bootstrap', True)
        self.erg_scores = kwargs.get('erg_scores', None)
        self.use_ols_ci = kwargs.get('use_ols_ci', True)  # Whether to use OLS for confidence intervals
        self.test_size = kwargs.get('test_size', 0.2)  # For train/test split
        self.random_state = kwargs.get('random_state', 42)
    
    def _run_regression(self, df, weights, athletes, shell_classes):
        """Implement the run_regression method for random forest"""
        # Prepare design matrix and response variable
        X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)])
        y = df['time_per_500m']

        # Ensure all columns are float64
        for col in X.columns:
            if X[col].dtype != 'float64':
                X[col] = X[col].astype("float64")
        
        # Run random forest to get parameter estimates and feature importances
        rf_results = self._run_random_forest(X, y, weights)
        
        # Optionally run OLS in parallel to get confidence intervals
        ols_results = None
        if self.use_ols_ci:
            ols_model = sm.OLS(y, X)
            ols_results = ols_model.fit()
        
        # Create a combined results object
        results = RandomForestWithOLS(rf_results, ols_results)

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
        
        # Generate partial dependence plots for important features
        pdp_data = self._generate_partial_dependence(rf_results.model, X, athletes) if hasattr(self, '_generate_partial_dependence') else None
        
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
            'pdp': pdp_data  # Optional partial dependence plot data
        }
    
    # Only the _run_random_forest method and related fixes

    def _run_random_forest(self, X, y, weights):
        """Run random forest regressor to fit the model"""
        # Initialize the model with customizable parameters
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            random_state=self.random_state
        )
        
        # Apply sample weights if provided
        weights_array = weights.values if weights is not None else None
        
        # Fit the model
        rf.fit(X, y, sample_weight=weights_array)
        
        # Get feature importances and scale them to match the time scale
        # This conversion makes the feature importances interpretable as time effects
        raw_importances = rf.feature_importances_
        
        # Calculate median feature importance for non-zero features
        positive_importances = raw_importances[raw_importances > 0]
        median_importance = np.median(positive_importances) if len(positive_importances) > 0 else 1.0
        
        # Calculate average prediction to scale importances to time units
        avg_prediction = np.mean(rf.predict(X))
        
        # Estimate mean absolute difference between predictions for samples where a feature is present vs absent
        # This gives us time-scaled importance for each feature
        scaled_importances = np.zeros_like(raw_importances)
        
        for i, col in enumerate(X.columns):
            # Create two versions of X: one with feature i=0 and one with feature i=1
            X_without = X.copy()
            X_with = X.copy()
            
            # Set the feature to 0 or 1
            X_without[col] = 0
            X_with[col] = 1
            
            # Predict with both datasets
            pred_without = rf.predict(X_without)
            pred_with = rf.predict(X_with)
            
            # Calculate the average effect
            effect = np.mean(pred_with - pred_without)
            
            # Store the effect as the scaled importance
            scaled_importances[i] = effect
        
        # Adjust the sign of importances for athlete columns to match conventional interpretation
        # (negative = faster, positive = slower)
        for i, col in enumerate(X.columns):
            if any(athlete in col for athlete in X.columns if 'Piece' not in col):
                # For athlete columns, determine if presence of athlete tends to improve times
                # We'll use correlation between feature values and predictions to determine this
                
                # Get all rows where the athlete has any presence
                athlete_present = X[col] > 0
                
                if athlete_present.any():
                    # Create predictions for all rows
                    all_preds = rf.predict(X)
                    
                    # Calculate correlation between athlete's presence level and predictions
                    # Positive correlation means higher athlete value -> higher time (worse)
                    # Negative correlation means higher athlete value -> lower time (better)
                    athlete_values = X.loc[athlete_present, col]
                    pred_values = all_preds[athlete_present]
                    
                    if len(athlete_values) > 1:  # Need at least 2 points for correlation
                        correlation = np.corrcoef(athlete_values, pred_values)[0, 1]
                        
                        # If correlation is negative, athlete improves performance (make importance negative)
                        if correlation < 0:
                            scaled_importances[i] = -abs(scaled_importances[i])
                        else:
                            scaled_importances[i] = abs(scaled_importances[i])
        
        # Create results object
        return RandomForestResults(rf, X, y, scaled_importances)

    # Also need to fix the _create_comparison_df method to handle NumPy arrays
    def _create_comparison_df(self, df, y, results, X):
        """Create dataframe comparing actual vs. predicted values"""
        fitted_values = results.predict(X)
        
        # Convert numpy array to pandas Series if needed
        if isinstance(fitted_values, np.ndarray):
            fitted_values = pd.Series(fitted_values, index=y.index)
        
        comparison_df = pd.DataFrame({
            'Actual Pace': y.apply(lambda x: seconds_to_time(x)),
            'Actual Pace Seconds': y,
            'Model Pace': fitted_values.apply(lambda x: seconds_to_time(x)),
            'Model Pace Seconds': fitted_values
        })
        comparison_df['Piece'] = df['Piece']
        comparison_df['Crew'] = df['Personnel']
        comparison_df['shell_class'] = df['shell_class']
        comparison_df['athlete_count'] = df['athlete_count']
        comparison_df['Delta'] = (y - fitted_values).round(2)
        comparison_df = comparison_df[['Piece', 'Crew', 'Actual Pace', 'Model Pace', 'Delta', 'athlete_count', 'shell_class']]
        
        return comparison_df
    
    def _generate_partial_dependence(self, model, X, athletes):
        """Generate partial dependence data for visualizing feature effects"""
        # This method would generate partial dependence plots data
        # for the most important features (particularly athletes)
        # Implementation depends on your visualization framework
        pdp_data = {}
        
        # Placeholder for actual implementation
        # Would use sklearn.inspection.partial_dependence
        
        return pdp_data
    
    # Optional: Add methods for hyperparameter tuning, cross-validation, etc.
    def tune_hyperparameters(self, df, weights, athletes, shell_classes):
        """Tune random forest hyperparameters using cross-validation"""
        # Implementation would use GridSearchCV or RandomizedSearchCV
        pass