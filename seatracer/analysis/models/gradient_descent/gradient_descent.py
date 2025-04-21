from seatracer.analysis.analysis_base import Analysis
from seatracer.analysis.registry import ModelRegistry
from seatracer.utils.helpers import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

class GradientDescentResults:
    """Custom results class to mimic statsmodels results interface"""
    def __init__(self, params, X, y, param_history):
        self.params = pd.Series(params, index=X.columns)
        self._X = X
        self._y = y
        
        # Calculate confidence intervals from parameter history
        if param_history:
            param_history_array = np.array(param_history)
            # Handle any NaN values that might cause warnings
            param_history_array = np.nan_to_num(param_history_array, nan=0.0)
            self._param_min = np.min(param_history_array, axis=0)
            self._param_max = np.max(param_history_array, axis=0)
        else:
            # Fallback if we don't have history
            self._param_min = params - abs(params) * 0.2
            self._param_max = params + abs(params) * 0.2
        
        # Ensure no identical min/max that could cause division by zero elsewhere
        epsilon = 1e-6
        identical_indices = np.isclose(self._param_min, self._param_max, atol=epsilon)
        if np.any(identical_indices):
            self._param_min[identical_indices] -= epsilon
            self._param_max[identical_indices] += epsilon
    
    def predict(self, X):
        return X.dot(self.params)
    
    def conf_int(self, alpha=0.05):
        # Replace any potential infinity or NaN values
        lower = pd.Series(np.nan_to_num(self._param_min, nan=0.0, posinf=1e5, neginf=-1e5), index=self._X.columns)
        upper = pd.Series(np.nan_to_num(self._param_max, nan=0.0, posinf=1e5, neginf=-1e5), index=self._X.columns)
        
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


class GradientDescentWithOLS:
    """Wrapper class that uses gradient descent parameters with adjusted OLS confidence intervals"""
    def __init__(self, gd_results, ols_results):
        # Get parameter estimates from gradient descent
        self.params = gd_results.params
        
        # Store reference to the OLS results for confidence intervals
        self._ols_results = ols_results
        
        # Store original results for reference if needed
        self._gd_results = gd_results
        
        # Calculate the width of OLS confidence intervals
        ols_ci = ols_results.conf_int()
        self._ci_width = ols_ci[1] - ols_ci[0]
    
    def predict(self, X):
        """Predict using gradient descent parameters"""
        return X.dot(self.params)
    
    def conf_int(self, alpha=0.05):
        """Return confidence intervals centered on gradient descent parameters
        with the same width as the OLS confidence intervals"""
        # Get the OLS confidence interval width (for the specified alpha if needed)
        if alpha != 0.05:
            ols_ci = self._ols_results.conf_int(alpha=alpha)
            ci_width = ols_ci[1] - ols_ci[0]
        else:
            ci_width = self._ci_width
        
        # Center the confidence intervals on the gradient descent parameters
        lower = self.params - ci_width / 2
        upper = self.params + ci_width / 2
        
        return pd.DataFrame({0: lower, 1: upper})
    
    def summary(self):
        """Return summary from OLS but with gradient descent parameters"""
        # This is a simplified approach - a more comprehensive one would
        # modify the OLS summary to use gradient descent parameters
        return self._ols_results.summary()
    
    # Add any other methods from the statsmodels results object that you need
    def __getattr__(self, name):
        """Delegate any missing attributes/methods to the OLS results"""
        if hasattr(self._ols_results, name):
            return getattr(self._ols_results, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


@ModelRegistry.register(
    key="gradient_descent", 
    name="Gradient Descent",
    description="Iteratively adjusts each rower's estimated performance using absolute errors rather than squared errors, and can start with erg scores as initial values.",
    uses_custom_weighting=True,
    can_have_stern_bias=True,
    show_athletes=True,
    order=0,
    recommended=True,
)
class GradientDescentAnalysis(Analysis):
    """
    Gradient Descent-based analysis.
    
    Uses custom gradient descent optimization with an absolute error criterion.
    This model is appropriate when:
    - You want custom optimization beyond what statsmodels provides
    - You prefer to minimize absolute errors rather than squared errors
    - You want to initialize with external data like erg scores
    - You want to apply custom regularization or constraints
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # Extract GD-specific parameters with defaults
        self.gd_learning_rate = kwargs.get('gd_learning_rate', 0.01)
        self.gd_iterations = kwargs.get('gd_iterations', 1000000)
        self.gd_convergence_threshold = kwargs.get('gd_convergence_threshold', 1e-7)
        self.erg_scores = kwargs.get('erg_scores', None)
    
    def _run_regression(self, df, weights, athletes, shell_classes):
        """Implement the run_regression method for gradient descent"""
        # Prepare design matrix and response variable
        X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)])
        y = df['time_per_500m']

        # Ensure all columns are float64
        for col in X.columns:
            if X[col].dtype != 'float64':
                X[col] = X[col].astype("float64")

        # Run gradient descent to get parameter estimates
        gd_results = self._run_gradient_descent(X, y, weights, self.gd_learning_rate, self.gd_iterations, 
                                        self.gd_convergence_threshold, self.erg_scores, athletes)
        
        # Run OLS in parallel to get confidence intervals
        ols_model = sm.OLS(y, X)
        ols_results = ols_model.fit()
        
        # Create a GradientDescentWithOLS results object that combines both
        results = GradientDescentWithOLS(gd_results, ols_results)

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
    
    def _run_gradient_descent(self, X, y, weights, learning_rate, max_iterations, convergence_threshold, erg_scores, athletes):
        """Run gradient descent algorithm to fit the model using Adam optimizer"""
        # Prepare data for gradient descent
        X_array = X.values
        y_array = y.values
        weights_array = weights.values if weights is not None else np.ones(len(y))
        
        # Initialize parameters with arbitrary value of 10
        params = np.ones(X.shape[1]) * 10
        
        # Use erg scores for initialization if provided
        if erg_scores is not None and not erg_scores.empty:
            params = self._initialize_with_erg_scores(params, X, erg_scores, athletes)
        
        # Setup for gradient descent
        param_history = []
        convergence_window = max(50, int(max_iterations * 0.05))  # Last 5% of iterations or at least 50
        prev_loss = float('inf')
        no_improvement_count = 0
        patience = 50  # Number of iterations to allow without significant improvement
        best_params = params.copy()
        best_loss = float('inf')
        
        # Adam optimizer hyperparameters
        beta1 = 0.9  # Exponential decay rate for first moment
        beta2 = 0.999  # Exponential decay rate for second moment
        epsilon = 1e-8  # Small constant to prevent division by zero
        
        # Initialize Adam momentum variables
        m = np.zeros_like(params)  # First moment vector
        v = np.zeros_like(params)  # Second moment vector
        
        # Run gradient descent iterations
        for iteration in range(max_iterations):
            # Predict values
            predictions = X_array.dot(params)
            
            # Calculate weighted errors
            errors = predictions - y_array
            weighted_errors = errors * weights_array
            
            # Compute gradient using absolute error
            gradient = X_array.T.dot(np.sign(weighted_errors) * np.sqrt(np.abs(weighted_errors))) / len(y)
            
            # Adam optimizer update
            t = iteration + 1
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * gradient**2
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1**t)
            
            # Compute bias-corrected second moment estimate
            v_hat = v / (1 - beta2**t)
            
            # Update parameters
            params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Calculate loss
            loss = np.mean(np.abs(weighted_errors))
            
            # Track best parameters
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Check for convergence
            if iteration > 0 and abs(prev_loss - loss) < convergence_threshold:
                if iteration >= max_iterations - convergence_window:
                    param_history.append(params.copy())
                # Early stopping if we've reached convergence
                if no_improvement_count >= patience:
                    print(f"Early stopping at iteration {iteration}, no improvement for {patience} iterations")
                    break
            
            prev_loss = loss
            
            # Always track parameters in the final convergence window
            if iteration >= max_iterations - convergence_window:
                param_history.append(params.copy())
        
        # Use the best parameters found
        params = best_params
        
        # Create a results object
        return GradientDescentResults(params, X, y, param_history)

    def _initialize_with_erg_scores(self, params, X, erg_scores, athletes):
        """Initialize parameters using erg scores"""
        # Standardize erg scores - find fastest time (minimum)
        erg_times = {}
        for athlete, row in erg_scores.iterrows():
            # athlete = row.Name
            erg_time_str = row['2k Erg']
            if pd.notna(erg_time_str):
                # Convert m:ss.s format to seconds
                parts = erg_time_str.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    total_seconds = minutes * 60 + seconds
                    erg_times[athlete] = total_seconds
        
        # Find fastest time and calculate standardized scores
        if erg_times:
            fastest_time = min(erg_times.values())
            standardized_erg_scores = {athlete: (time - fastest_time) / 4 for athlete, time in erg_times.items()}
            median_erg_score = np.median(list(standardized_erg_scores.values())) if standardized_erg_scores else 0
            
            # Set initial parameters based on standardized erg scores
            for i, col in enumerate(X.columns):
                base_name = ''.join(c for c in col if c not in ['ᵖ', 'ˢ', 'ᶜ', 'ˣ'])  # Remove rigging suffixes
                
                # Check if it's an athlete column and if we have a matching erg score
                if col in athletes or base_name in athletes:
                    match_found = False
                    
                    # Try exact match first
                    if col in standardized_erg_scores:
                        params[i] = standardized_erg_scores[col]
                        match_found = True
                    
                    # Try base name match if needed
                    elif base_name in standardized_erg_scores:
                        params[i] = standardized_erg_scores[base_name]
                        match_found = True
                    
                    # If no match found, use the median erg score
                    elif not match_found:
                        params[i] = median_erg_score
        
        return params