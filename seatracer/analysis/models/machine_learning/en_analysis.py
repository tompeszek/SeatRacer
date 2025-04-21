from seatracer.analysis.analysis_base import Analysis
from seatracer.analysis.registry import ModelRegistry
from seatracer.utils.helpers import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ElasticNetResults:
    """Custom results class to mimic statsmodels results interface"""
    def __init__(self, model, X, y, cv_score=None, scaler=None):
        self.model = model
        self._X = X
        self._y = y
        self.cv_score = cv_score
        self.scaler = scaler
        
        # Create a params series from model coefficients
        self.params = pd.Series(np.concatenate(([0], model.coef_)), index=['const'] + list(X.columns))
        
        # Use bootstrapping to estimate confidence intervals
        self._generate_confidence_intervals(X, y)
    
    def _generate_confidence_intervals(self, X, y, n_bootstraps=100, ci_level=0.95):
        """Generate confidence intervals using bootstrapping"""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        alpha = (1 - ci_level) / 2
        
        # Initialize storage for bootstrap coefficients
        boot_coefs = np.zeros((n_bootstraps, n_features))
        
        # Run bootstrapping
        np.random.seed(42)  # For reproducibility
        for i in range(n_bootstraps):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X.iloc[indices], y.iloc[indices]
            
            # Fit model on bootstrap sample
            model_boot = ElasticNet(
                alpha=self.model.alpha,
                l1_ratio=self.model.l1_ratio,
                max_iter=self.model.max_iter,
                tol=self.model.tol,
                random_state=i
            )
            
            # Apply standardization if needed
            if self.scaler is not None:
                X_boot_scaled = self.scaler.transform(X_boot)
                model_boot.fit(X_boot_scaled, y_boot)
            else:
                model_boot.fit(X_boot, y_boot)
            
            # Store coefficients
            boot_coefs[i] = model_boot.coef_
        
        # Calculate confidence intervals
        lower = np.percentile(boot_coefs, alpha * 100, axis=0)
        upper = np.percentile(boot_coefs, (1 - alpha) * 100, axis=0)
        
        # Add intercept CI using a simpler approach (since ElasticNet doesn't store intercept)
        self._ci_lower = np.concatenate(([self.model.intercept_ - 0.2], lower))
        self._ci_upper = np.concatenate(([self.model.intercept_ + 0.2], upper))
    
    def predict(self, X):
        """Predict using the elastic net model"""
        if isinstance(X, pd.DataFrame):
            # Check for missing columns
            missing_cols = set(self._X.columns) - set(X.columns)
            if missing_cols:
                for col in missing_cols:
                    X[col] = 0  # Add missing columns with zeros
            
            # Reorder columns to match training data
            X = X[self._X.columns]
        
        # Apply standardization if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X)
    
    def conf_int(self, alpha=0.05):
        """Return confidence intervals"""
        # Create DataFrame with lower and upper bounds
        ci_df = pd.DataFrame({
            0: pd.Series(self._ci_lower, index=['const'] + list(self._X.columns)),
            1: pd.Series(self._ci_upper, index=['const'] + list(self._X.columns))
        })
        return ci_df


@ModelRegistry.register(
    key="elastic_net", 
    name="Elastic Net Regression",
    description="Regularized linear regression with L1 and L2 penalties to handle correlated features",
    uses_custom_weighting=True,
    can_have_stern_bias=True,
    show_athletes=True,
    order=1,
    recommended=False
)
class ElasticNetAnalysis(Analysis):
    """
    Elastic Net regression analysis.
    
    Uses Elastic Net (combination of L1 and L2 regularization) to model rowing performance.
    This model is appropriate when:
    - You want to differentiate between rower contributions
    - You have many correlated features (rowers who frequently row together)
    - You want to reduce overfitting with regularization
    - You want a model that performs feature selection (can set coefficients to 0)
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # Extract elastic net parameters with defaults
        self.alpha = kwargs.get('alpha', 0.0001)  # Reduced by 10x for less regularization 0.001 default?
        self.l1_ratio = kwargs.get('l1_ratio', 0.85)  # Increased to favor Lasso behavior (better differentiation)
        self.max_iter = kwargs.get('max_iter', 100)  # Originally like 10k
        self.tol = kwargs.get('tol', 1e-5)  # Tightened tolerance for more precise results
        self.cv = kwargs.get('cv', 5)  # Number of cross-validation folds
        self.random_state = kwargs.get('random_state', 42)
        self.erg_scores = kwargs.get('erg_scores', None)
        self.test_size = kwargs.get('test_size', 0.2)  # For train/test split
        self.standardize = kwargs.get('standardize', False)  # Whether to standardize features
    
    def _run_regression(self, df, weights, athletes, shell_classes):
        """Implement the run_regression method for elastic net"""
        # Prepare design matrix and response variable
        X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)])
        y = df['time_per_500m']

        # Ensure all columns are float64
        for col in X.columns:
            if X[col].dtype != 'float64':
                X[col] = X[col].astype("float64")
        
        # Run elastic net regression to get parameter estimates
        results = self._run_elastic_net(X, y, weights)
        
        # Generate comparison dataframe with safety checks for numpy arrays
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
    
    def _run_elastic_net(self, X, y, weights):
        """Run elastic net regression to fit the model"""
        # Initialize model with parameters
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            fit_intercept=True
        )
        
        # Initialize scaler if standardization is needed
        scaler = None
        if self.standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Apply sample weights if provided
        if weights is not None:
            # Elastic Net doesn't directly support sample weights, 
            # so we need to implement weighted fitting ourselves
            
            # Create sqrt of weights for weighted fitting
            sqrt_weights = np.sqrt(weights.values)
            
            # Apply weights to X and y
            if self.standardize:
                X_weighted = X_scaled.copy()
            else:
                X_weighted = X.copy()
            y_weighted = y.copy()
            
            # Multiply each sample by sqrt of weight
            for i in range(len(X)):
                if self.standardize:
                    X_weighted[i] = X_scaled[i] * sqrt_weights[i]
                else:
                    X_weighted.iloc[i] = X.iloc[i] * sqrt_weights[i]
                y_weighted.iloc[i] = y.iloc[i] * sqrt_weights[i]
            
            # Fit model on weighted data
            model.fit(X_weighted, y_weighted)
        else:
            # Regular unweighted fit
            model.fit(X_scaled, y)
        
        # Run cross-validation to get model performance
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        def custom_cv_score(model, X, y, cv):
            scores = []
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Apply standardization if needed
                if self.standardize:
                    scaler_cv = StandardScaler()
                    X_train_scaled = scaler_cv.fit_transform(X_train)
                    X_test_scaled = scaler_cv.transform(X_test)
                    
                    # Fit model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test_scaled)
                else:
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                
                # Calculate score
                mse = mean_squared_error(y_test, y_pred)
                scores.append(-mse)  # Negative MSE to match sklearn's scoring convention
            
            return np.array(scores)
        
        cv_scores = custom_cv_score(model, X, y, cv)
        rmse_cv = np.sqrt(-cv_scores.mean())
        
        # Create results object
        results = ElasticNetResults(model, X, y, cv_score=rmse_cv, scaler=scaler if self.standardize else None)
        
        return results
    
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