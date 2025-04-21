from seatracer.analysis.analysis_base import Analysis
from seatracer.analysis.registry import ModelRegistry
from seatracer.utils.helpers import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import math

class TrueSkill2Results:
    """Custom results class to mimic statsmodels results interface"""
    def __init__(self, params, X, y, uncertainty_estimates):
        self.params = pd.Series(params, index=X.columns)
        self._X = X
        self._y = y
        self._uncertainty = uncertainty_estimates
        
    def predict(self, X):
        return X.dot(self.params)
    
    def conf_int(self, alpha=0.05):
        # Calculate Z value for the given confidence level
        z = stats.norm.ppf(1 - alpha/2)
        
        # Calculate lower and upper bounds
        lower = self.params - z * self._uncertainty
        upper = self.params + z * self._uncertainty
        
        return pd.DataFrame({0: lower, 1: upper})


class TrueSkill2WithOLS:
    """Wrapper class that uses TrueSkill2 parameters with adjusted OLS confidence intervals"""
    def __init__(self, ts_results, ols_results):
        # Get parameter estimates from TrueSkill2
        self.params = ts_results.params
        
        # Store reference to the OLS results for confidence intervals
        self._ols_results = ols_results
        
        # Store original results for reference if needed
        self._ts_results = ts_results
        
        # Calculate the width of OLS confidence intervals
        ols_ci = ols_results.conf_int()
        self._ci_width = ols_ci[1] - ols_ci[0]
    
    def predict(self, X):
        """Predict using TrueSkill2 parameters"""
        return X.dot(self.params)
    
    def conf_int(self, alpha=0.05):
        """Return confidence intervals centered on TrueSkill2 parameters
        with the same width as the OLS confidence intervals"""
        # Get the OLS confidence interval width (for the specified alpha if needed)
        if alpha != 0.05:
            ols_ci = self._ols_results.conf_int(alpha=alpha)
            ci_width = ols_ci[1] - ols_ci[0]
        else:
            ci_width = self._ci_width
        
        # Center the confidence intervals on the TrueSkill2 parameters
        lower = self.params - ci_width / 2
        upper = self.params + ci_width / 2
        
        return pd.DataFrame({0: lower, 1: upper})
    
    def summary(self):
        """Return summary from OLS but with TrueSkill2 parameters"""
        return self._ols_results.summary()
    
    # Add any other methods from the statsmodels results object that you need
    def __getattr__(self, name):
        """Delegate any missing attributes/methods to the OLS results"""
        if hasattr(self._ols_results, name):
            return getattr(self._ols_results, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


@ModelRegistry.register(
    key="trueskill2", 
    name="TrueSkill2",
    description="Bayesian skill rating system. Ideally could be used to evaluate without accurate margins. Not functional yet.",
    uses_custom_weighting=True,
    can_have_stern_bias=True,
    show_athletes=True,
    order=0,
    recommended=False
)
class TrueSkill2Analysis(Analysis):
    """
    TrueSkill 2-based analysis.
    
    Uses a Bayesian skill rating system to estimate rower ability.
    This model is appropriate when:
    - You want to account for uncertainty in skill estimates
    - You want a probabilistic model that converges quickly
    - You want to model both the mean skill and its variance
    - Your data has significant noise or few data points per athlete
    - Allows for skill changes over time
    """
    
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # Extract TrueSkill2-specific parameters with defaults
        self.ts_beta = kwargs.get('ts_beta', 4.0)  # Dynamics factor (how quickly skills can change)
        self.ts_iterations = kwargs.get('ts_iterations', 100)  # Number of iterations for convergence
        self.ts_draw_probability = kwargs.get('ts_draw_probability', 0.0)  # Rowing doesn't usually have draws
        self.ts_tau = kwargs.get('ts_tau', 0.3)  # Small additive dynamics factor
        self.erg_scores = kwargs.get('erg_scores', None)  # Optional erg scores for prior initialization
    
    def _run_regression(self, df, weights, athletes, shell_classes):
        """Implement the run_regression method for TrueSkill2"""
        # Prepare design matrix and response variable
        X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)])
        y = df['time_per_500m']

        # Ensure all columns are float64
        for col in X.columns:
            if X[col].dtype != 'float64':
                X[col] = X[col].astype("float64")

        # Run TrueSkill2 to get parameter estimates
        ts_results = self._run_trueskill2(df, X, y, weights, athletes, shell_classes)
        
        # Run OLS in parallel to get confidence intervals
        ols_model = sm.OLS(y, X)
        ols_results = ols_model.fit()
        
        # Create a TrueSkill2WithOLS results object that combines both
        results = TrueSkill2WithOLS(ts_results, ols_results)

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
    
    def _run_trueskill2(self, df, X, y, weights, athletes, shell_classes):
        """Run TrueSkill2 algorithm to estimate parameters"""
        # Initialize parameters
        num_features = X.shape[1]
        skill_means = np.zeros(num_features)
        skill_variances = np.ones(num_features) * 4.0  # Start with high uncertainty
        
        # Initialize with erg scores if available
        if self.erg_scores is not None and not self.erg_scores.empty:
            skill_means = self._initialize_with_erg_scores(skill_means, X, self.erg_scores, athletes)
        
        # Get all unique pieces
        pieces = df['Piece'].unique()
        
        # Create an acceleration structure for athlete lookup
        athlete_to_idx = {col: i for i, col in enumerate(X.columns) if col in athletes}
        shell_to_idx = {col: i for i, col in enumerate(X.columns) if col in shell_classes}
        piece_to_idx = {col: i for i, col in enumerate(X.columns) if col.startswith('Piece')}
        
        # For each iteration
        for iteration in range(self.ts_iterations):
            # Shuffle pieces to avoid systematic bias
            np.random.shuffle(pieces)
            
            # For each piece, update the skills of participants
            for piece in pieces:
                piece_df = df[df['Piece'] == piece]
                
                if len(piece_df) <= 1:
                    continue  # Skip pieces with only one lineup (no comparison possible)
                
                # Create piece-specific copies of skill estimates
                piece_means = skill_means.copy()
                piece_variances = skill_variances.copy()
                
                # Get all lineups in this piece
                lineups = []
                times = []
                lineup_weights = []
                
                for _, row in piece_df.iterrows():
                    # Build the lineup features
                    lineup_idx = []
                    
                    # Add athletes in the lineup
                    for athlete in athletes:
                        if row[athlete] == 1:
                            if athlete in athlete_to_idx:
                                lineup_idx.append(athlete_to_idx[athlete])
                    
                    # Add shell class
                    for shell in shell_classes:
                        if row[shell] == 1:
                            if shell in shell_to_idx:
                                lineup_idx.append(shell_to_idx[shell])
                    
                    # Add piece
                    piece_col = f"Piece_{piece}"
                    if piece_col in piece_to_idx:
                        lineup_idx.append(piece_to_idx[piece_col])
                    
                    # Store lineup
                    lineups.append(lineup_idx)
                    times.append(row['time_per_500m'])
                    
                    # Get weight if available
                    if weights is not None:
                        lineup_weights.append(weights.loc[row.name])
                    else:
                        lineup_weights.append(1.0)
                
                # Skip if we don't have at least 2 valid lineups
                if len(lineups) < 2:
                    continue
                
                # Sort lineups by time (faster times first)
                sorted_indices = np.argsort(times)
                sorted_lineups = [lineups[i] for i in sorted_indices]
                sorted_weights = [lineup_weights[i] for i in sorted_indices]
                
                # Apply TrueSkill2 update
                self._update_skills(sorted_lineups, sorted_weights, skill_means, skill_variances)
            
            # Apply small dynamics factor to increase uncertainty slightly
            skill_variances += self.ts_tau**2
        
        # Return results object with final skill estimates
        return TrueSkill2Results(skill_means, X, y, np.sqrt(skill_variances))
    
    def _update_skills(self, lineups, weights, skill_means, skill_variances):
        """Update skills using TrueSkill2 algorithm"""
        # Initialize team skills
        team_means = []
        team_vars = []
        
        # Calculate team performance distributions
        for lineup in lineups:
            if not lineup:  # Skip empty lineups
                continue
                
            # Get mean and variance for the team
            team_mean = sum(skill_means[idx] for idx in lineup)
            team_var = sum(skill_variances[idx] for idx in lineup) + self.ts_beta**2
            
            team_means.append(team_mean)
            team_vars.append(team_var)
        
        if len(team_means) < 2:
            return  # Not enough teams to compare
        
        # Process in pairs (each team vs the next faster team)
        for i in range(len(team_means) - 1):
            # Get faster team (i) and slower team (i+1)
            faster_mean = team_means[i]
            faster_var = team_vars[i]
            slower_mean = team_means[i+1]
            slower_var = team_vars[i+1]
            
            # Calculate performance difference distribution
            diff_mean = faster_mean - slower_mean
            diff_var = faster_var + slower_var
            
            # Calculate probability that faster team is actually faster
            c = math.sqrt(2 * diff_var)
            if c == 0:  # Avoid division by zero
                continue
                
            p_faster_is_faster = 1.0 - self.ts_draw_probability
            v = diff_mean / c
            
            # Calculate update factors
            v_faster = stats.norm.pdf(v) / stats.norm.cdf(v)
            v_slower = stats.norm.pdf(v) / (1 - stats.norm.cdf(v))
            
            w_faster = v_faster * v_faster + v * v_faster
            w_slower = v_slower * v_slower - v * v_slower
            
            # Apply weight to the update
            weight = (weights[i] + weights[i+1]) / 2
            
            # Update skills for athletes in faster team
            for idx in lineups[i]:
                # Calculate scaling factor based on team size
                scale = self.ts_beta**2 / (faster_var * len(lineups[i]))
                
                # Update mean
                skill_means[idx] += scale * weight * v_faster * diff_var / c
                
                # Update variance
                skill_variances[idx] *= max(1.0 - scale * weight * w_faster, 0.001)
            
            # Update skills for athletes in slower team
            for idx in lineups[i+1]:
                # Calculate scaling factor based on team size
                scale = self.ts_beta**2 / (slower_var * len(lineups[i+1]))
                
                # Update mean
                skill_means[idx] -= scale * weight * v_slower * diff_var / c
                
                # Update variance
                skill_variances[idx] *= max(1.0 - scale * weight * w_slower, 0.001)
    
    def _initialize_with_erg_scores(self, params, X, erg_scores, athletes):
        """Initialize parameters using erg scores"""
        # Standardize erg scores - find fastest time (minimum)
        erg_times = {}
        for _, row in erg_scores.iterrows():
            athlete = row.name
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