from seatracer.analysis.analysis_base import Analysis
from seatracer.analysis.registry import ModelRegistry
from seatracer.utils.helpers import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class XGBoostResults:
    """Custom results class to mimic statsmodels results interface"""
    def __init__(self, model, X, y, feature_importances=None):
        self.model = model
        self._X = X
        self._y = y
        
        # Create a params series to mimic statsmodels
        self.params = pd.Series(feature_importances, index=X.columns)
        
        # Calculate confidence intervals from feature importances
        # This is a simplification as XGBoost doesn't provide direct CIs
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
        """Predict using the XGBoost model"""
        if isinstance(X, pd.DataFrame):
            # Ensure all columns needed by the model are present
            missing_cols = set(self._X.columns) - set(X.columns)
            if missing_cols:
                for col in missing_cols:
                    X[col] = 0  # Add missing columns with zeros
            
            # Reorder columns to match training data
            X = X[self._X.columns]
        
        # Convert DataFrame to DMatrix for prediction
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)
    
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


class XGBoostWithOLS:
    """Wrapper class that combines XGBoost predictions with OLS confidence intervals"""
    def __init__(self, xgb_results, ols_results=None):
        # Get feature importances from XGBoost
        self.params = xgb_results.params
        
        # Store original results
        self._xgb_results = xgb_results
        self._ols_results = ols_results
        
        # If OLS results are provided, use them for statistical tests
        if ols_results is not None:
            # Calculate the width of OLS confidence intervals
            ols_ci = ols_results.conf_int()
            self._ci_width = ols_ci[1] - ols_ci[0]
    
    def predict(self, X):
        """Predict using XGBoost model"""
        return self._xgb_results.predict(X)
    
    def conf_int(self, alpha=0.05):
        """Return confidence intervals"""
        if self._ols_results is not None:
            # Use OLS-based confidence intervals if available
            if alpha != 0.05:
                ols_ci = self._ols_results.conf_int(alpha=alpha)
                ci_width = ols_ci[1] - ols_ci[0]
            else:
                ci_width = self._ci_width
            
            # Center the confidence intervals on the XGB parameters
            lower = self.params - ci_width / 2
            upper = self.params + ci_width / 2
            
            return pd.DataFrame({0: lower, 1: upper})
        else:
            # Fall back to XGB-based intervals if OLS is not available
            return self._xgb_results.conf_int(alpha=alpha)
    
    def summary(self):
        """Return summary from OLS or a simplified summary if not available"""
        if self._ols_results is not None:
            return self._ols_results.summary()
        else:
            # Create a simplified summary
            return f"XGBoost Model\n" \
                   f"Number of trees: {self._xgb_results.model.num_boosted_rounds()}\n" \
                   f"Training RMSE: {np.sqrt(self._xgb_results.model.best_score):.4f}"
    
    # Add any other methods from the statsmodels results object that you need
    def __getattr__(self, name):
        """Delegate any missing attributes/methods to the OLS results if available"""
        if self._ols_results is not None and hasattr(self._ols_results, name):
            return getattr(self._ols_results, name)
        elif hasattr(self._xgb_results, name):
            return getattr(self._xgb_results, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


@ModelRegistry.register(
    key="xgboost", 
    name="XGBoost Regressor",
    description="Advanced machine learning approach using gradient boosting. Runs but everyone comes out nearly the same.",
    uses_custom_weighting=True,
    can_have_stern_bias=True,
    show_athletes=True,
    order=1,
    recommended=False,
)
class XGBoostAnalysis(Analysis):
    """
    XGBoost-based analysis.
    
    Uses a gradient boosting regressor to model rowing performance.
    This model is appropriate when:
    - You want a powerful non-linear approach that outperforms Random Forests
    - You want stable feature importance measurements to understand rower contributions
    - You have sufficient data to train a machine learning model
    - You want robust predictions less susceptible to outliers
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # Extract XGBoost-specific parameters with defaults
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', 3)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.subsample = kwargs.get('subsample', 0.8)
        self.colsample_bytree = kwargs.get('colsample_bytree', 0.8)
        self.min_child_weight = kwargs.get('min_child_weight', 1)
        self.gamma = kwargs.get('gamma', 0)
        self.alpha = kwargs.get('alpha', 0)  # L1 regularization
        self.lambda_param = kwargs.get('lambda', 1)  # L2 regularization
        self.objective = kwargs.get('objective', 'reg:squarederror')
        self.erg_scores = kwargs.get('erg_scores', None)
        self.use_ols_ci = kwargs.get('use_ols_ci', True)  # Whether to use OLS for confidence intervals
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        self.test_size = kwargs.get('test_size', 0.2)  # For train/test split
        self.random_state = kwargs.get('random_state', 42)
    
    def _run_regression(self, df, weights, athletes, shell_classes):
        """Implement the run_regression method for XGBoost"""
        # Prepare design matrix and response variable
        X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)])
        y = df['time_per_500m']

        # Ensure all columns are float64
        for col in X.columns:
            if X[col].dtype != 'float64':
                X[col] = X[col].astype("float64")
        
        # Run XGBoost to get parameter estimates and feature importances
        xgb_results = self._run_xgboost(X, y, weights)
        
        # Optionally run OLS in parallel to get confidence intervals
        ols_results = None
        if self.use_ols_ci:
            ols_model = sm.OLS(y, X)
            ols_results = ols_model.fit()
        
        # Create a combined results object
        results = XGBoostWithOLS(xgb_results, ols_results)

        # Generate comparison dataframe - convert numpy arrays to pandas Series as needed
        comparison_df = self._create_comparison_df_safe(df, y, results, X)
        
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
            'fitted': self._generate_fitted_values_vs_actual_safe(df, results, athletes, shell_classes),
            'raw': df,
            'corr': X.corr(),
            'weights': weights,
            'dropped_athletes': dropped_athletes_df,
            'race_balance': race_balance_info,
            'pairs': self._create_athlete_pairs_df(df, y, results, X, athletes),
        }
    
    def _run_xgboost(self, X, y, weights):
        """Run XGBoost regressor to fit the model"""
        # Split data for early stopping
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Apply sample weights if provided
        weights_train = None
        if weights is not None:
            weights_train = weights.iloc[X_train.index].values
        
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        # Set up parameters
        params = {
            'objective': self.objective,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'lambda': self.lambda_param,
            'seed': self.random_state,
            'tree_method': 'hist',  # For faster training
            'verbosity': 0  # Silence output
        }
        
        # Evaluation watchlist
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            self.n_estimators,
            evals=watchlist,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        # Get feature importances (using weight by default, could be gain or cover)
        importances = model.get_score(importance_type='weight')
        
        # Convert to array to match the column order
        imp_array = np.zeros(len(X.columns))
        for i, col in enumerate(X.columns):
            if col in importances:
                imp_array[i] = importances[col]
        
        # Scale and determine sign of importances for interpretability
        scaled_importances = self._scale_and_sign_importances(model, X, imp_array)
        
        # Create results object
        return XGBoostResults(model, X, y, scaled_importances)
    
    def _scale_and_sign_importances(self, model, X, raw_importances):
        """
        Scale feature importances to be comparable with coefficients from linear models
        and determine their signs for athlete features.
        """
        # Create a full dataset DMatrix for prediction
        dmatrix = xgb.DMatrix(X)
        baseline_preds = model.predict(dmatrix)
        
        # Initialize scaled importances array
        scaled_importances = np.zeros_like(raw_importances)
        
        # Process each feature
        for i, col in enumerate(X.columns):
            if raw_importances[i] > 0:  # Only process features with non-zero importance
                # Create perturbed dataset with feature increased by 10%
                X_plus = X.copy()
                # For binary features, we'll use a smaller value to avoid going too far out of range
                if X[col].nunique() <= 2:
                    X_plus[col] = X_plus[col] + 0.1 * (X_plus[col] > 0).astype(float)
                else:
                    X_plus[col] = X_plus[col] * 1.1
                
                # Create perturbed dataset with feature decreased by 10%
                X_minus = X.copy()
                if X[col].nunique() <= 2:
                    X_minus[col] = X_minus[col] - 0.1 * (X_minus[col] > 0).astype(float)
                    X_minus[col] = X_minus[col].clip(0, 1)  # Ensure values stay in valid range
                else:
                    X_minus[col] = X_minus[col] * 0.9
                
                # Get predictions for perturbed datasets
                dmatrix_plus = xgb.DMatrix(X_plus)
                dmatrix_minus = xgb.DMatrix(X_minus)
                preds_plus = model.predict(dmatrix_plus)
                preds_minus = model.predict(dmatrix_minus)
                
                # Calculate average effect (how much the predictions change when feature is perturbed)
                mean_effect = np.mean(preds_plus - preds_minus)
                
                # Scale the importance by the effect direction
                scaled_importances[i] = raw_importances[i] * np.sign(mean_effect)
            
        # Determine sign specifically for athlete features
        for i, col in enumerate(X.columns):
            if any(athlete in col for athlete in X.columns if 'Piece' not in col) and raw_importances[i] > 0:
                # For athlete columns, look at correlation with predictions
                athlete_mask = X[col] > 0
                
                if np.sum(athlete_mask) > 1:  # Need at least 2 data points for correlation
                    # Calculate correlation between athlete value and predictions
                    corr = np.corrcoef(X.loc[athlete_mask, col], baseline_preds[athlete_mask])[0, 1]
                    
                    # If correlation is negative, athlete improves performance (makes times lower)
                    if corr < 0:
                        scaled_importances[i] = -abs(scaled_importances[i])
                    else:
                        scaled_importances[i] = abs(scaled_importances[i])
        
        # The raw XGBoost feature importances are always positive, so we need to scale them
        # to match the magnitude of the effects in the data
        
        # First, calculate the standard deviation of the target variable
        y_std = np.std(baseline_preds)
        
        # Find the most important feature
        max_imp_idx = np.argmax(np.abs(scaled_importances))
        max_imp = np.abs(scaled_importances[max_imp_idx])
        
        # Scale factor: We want the most important feature to have a meaningful effect
        # For rowing data, a meaningful effect might be around 5% of the target standard deviation
        if max_imp > 0:
            scale_factor = (0.05 * y_std) / max_imp
            scaled_importances = scaled_importances * scale_factor
        
        return scaled_importances

    def _create_comparison_df_safe(self, df, y, results, X):
        """Create dataframe comparing actual vs. predicted values with safety checks"""
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

    def _generate_fitted_values_vs_actual_safe(self, df, results, athletes, shell_classes):
        """Safe version of generate_fitted_values_vs_actual that handles numpy arrays"""
        # Get the coefficients from the regression model
        coef = results.params

        # Prepare a new DataFrame that includes all original data
        df_fitted = df.copy()

        # Compute fitted values
        X_pred = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)], drop_first=False)
        fitted_values = results.predict(X_pred)
        
        # Convert numpy array to pandas Series if needed
        if isinstance(fitted_values, np.ndarray):
            fitted_values = pd.Series(fitted_values, index=df.index)
            
        df_fitted['Fitted'] = fitted_values

        # Generate the Breakdown column
        def breakdown(row):
            components = []
            
            # Intercept
            if 'const' in coef:
                components.append(f"Intercept: {coef['const']:.4f}")
            
            # Piece contributions
            piece_col = f"Piece_{row['Piece']}"
            if piece_col in coef:
                components.append(f"{piece_col}: {coef[piece_col]:.4f}")
            
            # Athlete contributions
            for athlete in athletes:
                if athlete in coef and row[athlete] > 0:
                    weight = row[athlete]
                    contribution = coef[athlete] * weight
                    components.append(f"{athlete} ({weight:.2f}): {contribution:.4f}")
            
            # Shell class contributions
            for shell_class in shell_classes:
                if shell_class in coef and row[shell_class] == 1:
                    components.append(f"{shell_class}: {coef[shell_class]:.4f}")

            return " + ".join(components)

        df_fitted['Breakdown'] = df_fitted.apply(breakdown, axis=1)

        return df_fitted