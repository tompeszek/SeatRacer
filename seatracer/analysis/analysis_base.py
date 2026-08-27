from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Any
from seatracer.utils.helpers import add_athlete_counts, append_rigging_to_names, determine_shell_class, get_rower_sides_count, time_to_seconds, seconds_to_time, calculate_closest_margin
from seatracer.utils.grouping import group_highly_correlated_parameters

@dataclass
class Analysis(ABC):
    """Base Analysis class with common functionality for all model types"""
    df: object
    max_correlation: float = float("inf")
    halflife: float = None
    weight_close: float = None
    weight_stern: float = None
    include_coxswains: bool = False
    seat_breakdown: bool = True
    lookback: int = 10000
    erg_scores: object = None
    shell_class: object = None
    # When True, _run_regression returns only athlete/shell coefficients and skips
    # the expensive comparison / pairs / fitted / correlation outputs. Used by the
    # leave-one-out analysis, which refits the model many times but only needs each
    # athlete's coefficient. Default False keeps full behaviour everywhere else.
    light: bool = False
    
    def __post_init__(self):
        self.df = self.df.copy().sort_values(by=['Race Session (date)', 'Piece'])
        self.time_series_df = None
        self.stats_df = None
        self.final_results = None
        
        # Initialize temporal analysis data
        self.temporal_data = {
            'time_series_df': None,
            'stats_df': None,
            'all_athletes': set(),
            'results_by_date': {}
        }

        # Side counts (Starboard/Port/Scull/Coxswain per athlete). Populated below
        # once the dataframe has been prepared; kept on the instance so the engine
        # has no dependency on any web-framework session state.
        self.sides_count = {}

        add_athlete_counts(self.df)

        self.df['shell_class'] = self.df.apply(determine_shell_class, axis=1)

        # Apply shell class filter
        self.df = self.df[self.df['shell_class'].isin(self.shell_class)]
        # Add sides to names  (also adds coxswain to personnel if needed)
        self.df = append_rigging_to_names(self.df)

        # Only go forward if there is data:
        if not self.df.empty:
            
            # Add piece names
            self.df['PieceNumber'] = self.df['Piece']
            self.df['Piece'] = self.df['Race Session (date)'].astype(str) + " #" + self.df['Piece'].astype(str)            

            # Sides count
            self.sides_count = get_rower_sides_count(self.df)

    def run_analysis(self, get_history=False, by_piece=False):
        """
        Main analysis method that handles the overall analysis pipeline.
        
        Parameters:
        -----------
        get_history : bool
            Whether to calculate results over time
        by_piece : bool
            If True, analyze progression by individual pieces rather than by date
        """
        df = self.df.copy()

        # Ensure date column is datetime
        df['Race Session (date)'] = pd.to_datetime(df['Race Session (date)'])
        
        # Store results 
        coefficients_by_point = {}
        results_by_point = {}
        all_athletes = set()

        # Convert Result times to seconds and then calculate time per 500m
        df['time_seconds'] = df['Result'].apply(time_to_seconds)
        df['time_per_500m'] = df['time_seconds'] / (df['KM'] * 2.0)  # Adjust time per 500m

        # Apply weights
        df = calculate_closest_margin(df)
        df = self._apply_weights(df, self.weight_close, self.halflife)
        
        # Extract athletes and shell classes
        athletes = df['Personnel'].str.split('/', expand=True).stack().unique()
        athletes = [athlete for athlete in athletes if self.include_coxswains or not athlete.endswith('ᶜ')]
        shell_classes = df['shell_class'].unique()

        # Apply position-based weighting to athletes in each boat
        athlete_weights = self._compute_athlete_weights(df, self.weight_stern)
        
        # Add athlete columns to dataframe
        for athlete in athletes:
            df[athlete] = athlete_weights[athlete]

        # Add shell class columns
        for shell_class in shell_classes:
            df[shell_class] = df['shell_class'].apply(lambda x: 1 if x == shell_class else 0)

        if get_history:
            if by_piece:
                # Sort data by date and piece to ensure chronological order
                df = df.sort_values(['Race Session (date)', 'Piece'])
                
                # Get all unique pieces in order
                all_pieces = df['Piece'].unique()
                
                # Process each piece incrementally
                cumulative_df = pd.DataFrame()
                
                for piece_idx, piece in enumerate(all_pieces):
                    # Add this piece to the cumulative data
                    piece_df = df[df['Piece'] == piece]
                    
                    if piece_idx == 0:
                        cumulative_df = piece_df.copy()
                    else:
                        cumulative_df = pd.concat([cumulative_df, piece_df])
                    
                    # # Skip if not enough data
                    # if len(cumulative_df) < 5:
                    #     continue
                    
                    # Adjust recency weights if needed
                    if self.halflife is not None:
                        current_date = piece_df['Race Session (date)'].iloc[0]
                        cumulative_df = self._recalculate_recency_weights(cumulative_df, current_date, self.halflife)
                    
                    # Get weights specific to this cumulative dataset
                    window_weights = cumulative_df['total_weight']
                    
                    # Get athletes present in this data
                    window_personnel = cumulative_df['Personnel'].str.split('/', expand=True).stack().unique()
                    window_athletes = [athlete for athlete in athletes if athlete in window_personnel]
                    
                    # Run regression on cumulative data through this piece
                    piece_results = self._run_regression(cumulative_df, window_weights, window_athletes, shell_classes)
                    
                    # Store results with this piece
                    results_by_point[piece] = piece_results
                    
                    # Extract coefficients
                    athletes_df = piece_results['athletes']
                    
                    # Store coefficients for this piece - only for athletes present
                    piece_coeffs = {'point': piece, 'date': piece_df['Race Session (date)'].iloc[0]}
                    for idx, row in athletes_df.iterrows():
                        athlete = idx
                        all_athletes.add(athlete)
                        piece_coeffs[athlete] = row['Coefficient']
                    
                    coefficients_by_point[piece] = piece_coeffs
                    
                    # Update final results to most recent
                    self.final_results = piece_results
                
            else:  # Process by date (original method)
                # Get unique dates in chronological order
                unique_dates = sorted(df['Race Session (date)'].unique())
                
                # Process each date
                for idx, current_date in enumerate(unique_dates):
                    # Define the lookback window
                    lookback_start = current_date - pd.Timedelta(days=self.lookback)
                    
                    # Filter data within the lookback window
                    window_df = df[(df['Race Session (date)'] >= lookback_start) & 
                                    (df['Race Session (date)'] <= current_date)].copy()
                    
                    # # Skip if not enough data in this window
                    # if len(window_df) < 5:  # Minimum number of rows needed
                    #     continue
                    
                    # Adjust recency weights to be relative to current date if using recency
                    if self.halflife is not None:
                        window_df = self._recalculate_recency_weights(window_df, current_date, self.halflife)
                    
                    # Get the weights specific to this window
                    window_weights = window_df['total_weight']
                    
                    # Get the athletes present in this window
                    window_personnel = window_df['Personnel'].str.split('/', expand=True).stack().unique()
                    window_athletes = [athlete for athlete in athletes if athlete in window_personnel]
                    
                    # Run regression on this window
                    window_results = self._run_regression(window_df, window_weights, window_athletes, shell_classes)
                    
                    # Store results with this date
                    results_by_point[current_date] = window_results
                    
                    # Extract coefficients
                    athletes_df = window_results['athletes']
                    
                    # Store coefficients for this date - only for athletes present in this window
                    date_coeffs = {'point': current_date, 'date': current_date}
                    for idx, row in athletes_df.iterrows():
                        athlete = idx
                        all_athletes.add(athlete)
                        date_coeffs[athlete] = row['Coefficient']
                    
                    coefficients_by_point[current_date] = date_coeffs
                    
                    # Update final results to the most recent date
                    self.final_results = window_results
            
            # Create time series dataframe with NaN for missing athletes at each point
            if coefficients_by_point:
                self.time_series_df = pd.DataFrame(list(coefficients_by_point.values()))
                
                # Sort correctly based on whether by_piece is True
                if by_piece:
                    # First ensure we have a proper date column in datetime format
                    self.time_series_df['date'] = pd.to_datetime(self.time_series_df['date'])
                    
                    # Extract date part and possibly piece number for sorting
                    self.time_series_df['sort_date'] = self.time_series_df['date'].dt.date
                    
                    # Try to extract piece number if available
                    # Assuming piece format might be like "2023-09-14 #2" or similar
                    try:
                        self.time_series_df['sort_piece'] = self.time_series_df['point'].astype(str).str.extract(r'#(\d+)').astype(float)
                    except:
                        # If extraction fails, just use a default ordering
                        self.time_series_df['sort_piece'] = range(len(self.time_series_df))
                    
                    # Sort by date then by piece number
                    self.time_series_df = self.time_series_df.sort_values(['sort_date', 'sort_piece'])
                    
                    # Remove temporary sorting columns
                    self.time_series_df = self.time_series_df.drop(columns=['sort_date', 'sort_piece'])
                else:
                    # Sort by date for the date-based analysis
                    self.time_series_df = self.time_series_df.sort_values('date')
                
                # Calculate statistics for each rower
                self.stats_df = self._calculate_athlete_statistics(all_athletes)
                
                # Store data for temporal analysis
                self.temporal_data = {
                    'time_series_df': self.time_series_df,
                    'stats_df': self.stats_df,
                    'all_athletes': all_athletes, 
                    'results_by_point': results_by_point,
                    'by_piece': by_piece
                }
        
        else:
            # Get the weights for regular analysis
            weights = df['total_weight']
            
            # Run regression on the entire dataset
            self.final_results = self._run_regression(df, weights, athletes, shell_classes)
            
        return self
    
    def run_history(self, custom_lookback=None, by_piece=False):
        """
        Run temporal analysis to track athlete performance over time.
        This is an explicit method to process historical data which may be time-consuming.
        
        Parameters:
        -----------
        custom_lookback : int, optional
            Override the default lookback period for this analysis
        by_piece : bool, optional
            If True, analyze progression by individual pieces rather than by date
            
        Returns:
        --------
        self : Analysis
            Returns self for method chaining
        """
        original_lookback = self.lookback
        
        # Override lookback if provided
        if custom_lookback is not None:
            self.lookback = custom_lookback
            
        try:
            # Run analysis with history enabled and the by_piece parameter
            self.run_analysis(get_history=True, by_piece=by_piece)
        finally:
            # Restore original lookback value
            self.lookback = original_lookback
            
        return self
        
    @abstractmethod
    def _run_regression(self, df, weights, athletes, shell_classes):
        """Abstract method to be implemented by specific model subclasses"""
        pass
    
    def _apply_weights(self, df, weight_close, halflife):
        """Apply closeness and recency weights to the dataframe"""
        # Apply closeness weights if specified
        if weight_close is not None:
            max_margin = 12
            df['scaled_closeness_factor'] = 1 * np.exp(-np.log(2) * np.clip(df['closest_margin'], None, max_margin) / weight_close)
            df['scaled_closeness_factor'] = df['scaled_closeness_factor'].clip(lower=0.1)    
        else:
            df['scaled_closeness_factor'] = 1

        # Apply recency weights if specified
        if halflife is not None and isinstance(halflife, float) and halflife > 0:
            df['Race Session (date)'] = pd.to_datetime(df['Race Session (date)'])
            df = df.sort_values('Race Session (date)')
            df['days_since_latest'] = (df['Race Session (date)'].max() - df['Race Session (date)']).dt.days
            df['recency_factor'] = np.exp(-df['days_since_latest'] / halflife)
            df['recency_factor'] = df['recency_factor'].clip(lower=0.1)
            scale_factor = 1 / df['recency_factor'].min()
            df['scaled_recency_factor'] = df['recency_factor'] * scale_factor
            df['scaled_recency_factor'] = df['scaled_recency_factor'].clip(upper=10)
        else:
            df['scaled_recency_factor'] = 1

        # Combine weights
        df['total_weight'] = df['scaled_recency_factor'] * df['scaled_closeness_factor']
        
        return df

    def _recalculate_recency_weights(self, df, current_date, halflife):
        """Recalculate recency weights relative to the current analysis date"""
        df['days_since_current'] = (current_date - df['Race Session (date)']).dt.days
        df['recency_factor'] = np.exp(-df['days_since_current'] / halflife)
        df['recency_factor'] = df['recency_factor'].clip(lower=0.1)
        scale_factor = 1 / df['recency_factor'].min()
        df['scaled_recency_factor'] = df['recency_factor'] * scale_factor
        df['scaled_recency_factor'] = df['scaled_recency_factor'].clip(upper=10)
        
        # If total_weight already exists, update it, otherwise create it
        if 'total_weight' in df.columns:
            # Assume closeness factor is in the data
            if 'scaled_closeness_factor' in df.columns:
                df['total_weight'] = df['scaled_recency_factor'] * df['scaled_closeness_factor']
            else:
                df['total_weight'] = df['scaled_recency_factor']
        else:
            df['total_weight'] = df['scaled_recency_factor']
            
        return df
    
    def _calculate_athlete_statistics(self, all_athletes):
        """Calculate statistics for each athlete's time series"""
        stats = []
        
        for athlete in all_athletes:
            if athlete in self.time_series_df.columns:
                athlete_data = self.time_series_df[athlete].dropna()
                
                if len(athlete_data) > 0:
                    stats.append({
                        'Rower': athlete,
                        'Mean': athlete_data.mean(),
                        'Std': athlete_data.std(),
                        'Min': athlete_data.min(),
                        'Max': athlete_data.max(), 
                        'Days_Present': len(athlete_data),
                        'Days_Total': len(self.time_series_df),
                        'Coverage': len(athlete_data) / len(self.time_series_df)
                    })
        
        stats_df = pd.DataFrame(stats)
        if not stats_df.empty:
            stats_df = stats_df.sort_values('Mean')
            
        return stats_df
    
    def get_final_results(self):
        """Get the results from the most recent analysis window"""
        return self.final_results
    
    def get_athlete_trend(self, athlete):
        """Get time series for a specific athlete"""
        if self.temporal_data['time_series_df'] is None:
            raise ValueError("No temporal analysis results available. Run run_history() first.")
        
        if athlete not in self.temporal_data['time_series_df'].columns:
            raise ValueError(f"Athlete '{athlete}' not found in results")
        
        return self.temporal_data['time_series_df'][['date', athlete]].dropna()
    
    def get_position_athletes(self, position):
        """Get list of athletes for a specific position"""
        if self.temporal_data['stats_df'] is None:
            raise ValueError("No temporal analysis results available. Run run_history() first.")
        
        position_suffix_map = {
            'Starboard': 'ˢ',
            'Port': 'ᵖ',
            'Sculling': 'ˣ',
            'Coxswain': 'ᶜ'
        }
        
        if position not in position_suffix_map:
            raise ValueError(f"Position must be one of {list(position_suffix_map.keys())}")
        
        suffix = position_suffix_map[position]
        
        return [a for a in self.temporal_data['stats_df']['Rower'] if a.endswith(suffix)]
    
    def get_temporal_data(self):
        """
        Get all temporal analysis data.
        
        Returns:
        --------
        temporal_data : dict
            Dictionary containing time_series_df, stats_df, and other temporal analysis data
        """
        return self.temporal_data
    
    def _create_comparison_df(self, df, y, results, X):
        """Create dataframe comparing actual vs. predicted values with contribution breakdown"""
        fitted_values = results.predict(X)
        
        # Get model parameters
        params = results.params
        
        # Create dataframe for basic comparison
        comparison_df = pd.DataFrame({
            'Actual Pace': y.apply(lambda x: seconds_to_time(x)),
            'Actual Pace Seconds': y,
            'Model Pace': fitted_values.apply(lambda x: seconds_to_time(x)),
            'Model Pace Seconds': fitted_values
        })
        
        # Add metadata columns
        comparison_df['Piece'] = df['Piece']
        comparison_df['Crew'] = df['Personnel']
        comparison_df['shell_class'] = df['shell_class']
        comparison_df['athlete_count'] = df['athlete_count']
        comparison_df['Delta'] = (y - fitted_values).round(2)
        comparison_df['KM'] = df['KM']
        
        # Create contribution breakdowns
        contribution_breakdowns = []
        
        for i, row in X.iterrows():
            # Calculate contribution of each parameter
            contributions = {}
            
            # Only include non-zero contributions to keep the output manageable
            for col in X.columns:
                if row[col] != 0 and col in params:
                    contribution = row[col] * params[col]
                    if abs(contribution) > 0.01:  # Filter out very small contributions
                        contributions[col] = round(contribution, 2)
            
            # Categorize contributions
            athlete_contributions = {}
            coxswain_contributions = {}
            shell_contributions = {}
            piece_contributions = {}
            other_contributions = {}
            
            for param, value in contributions.items():
                if param.startswith('athlete_'):
                    athlete_contributions[param] = value
                elif param.startswith('coxswain_'):
                    coxswain_contributions[param] = value
                elif param.startswith('shell_class_'):
                    shell_contributions[param] = value
                elif param.startswith('piece_'):
                    piece_contributions[param] = value
                else:
                    other_contributions[param] = value
            
            # Sort athletes alphabetically
            sorted_athlete_contributions = sorted(athlete_contributions.items())
            sorted_coxswain_contributions = sorted(coxswain_contributions.items())
            sorted_shell_contributions = sorted(shell_contributions.items())
            sorted_piece_contributions = sorted(piece_contributions.items())
            sorted_other_contributions = sorted(other_contributions.items())
            
            # Format as string with linebreaks between categories
            formatted_parts = []
            
            # Add athletes
            if sorted_athlete_contributions:
                athlete_part = "\n".join([f"{param}: {value}" for param, value in sorted_athlete_contributions])
                formatted_parts.append(athlete_part)
            
            # Add coxswains
            if sorted_coxswain_contributions:
                coxswain_part = "\n".join([f"{param}: {value}" for param, value in sorted_coxswain_contributions])
                formatted_parts.append(coxswain_part)
            
            # Add shell class
            if sorted_shell_contributions:
                shell_part = "\n".join([f"{param}: {value}" for param, value in sorted_shell_contributions])
                formatted_parts.append(shell_part)
            
            # Add piece
            if sorted_piece_contributions:
                piece_part = "\n".join([f"{param}: {value}" for param, value in sorted_piece_contributions])
                formatted_parts.append(piece_part)
            
            # Add other contributions
            if sorted_other_contributions:
                other_part = "\n".join([f"{param}: {value}" for param, value in sorted_other_contributions])
                formatted_parts.append(other_part)
            
            # Join all parts with an extra linebreak between categories
            breakdown = "\n\n".join(formatted_parts)
            contribution_breakdowns.append(breakdown)
        
        # Add contribution breakdown column
        comparison_df['Contribution Breakdown'] = contribution_breakdowns
        
        # Select and order columns
        comparison_df = comparison_df[['Piece', 'KM', 'Crew', 'Actual Pace', 'Model Pace', 'Actual Pace Seconds', 'Model Pace Seconds', 'Delta', 'Contribution Breakdown', 'athlete_count', 'shell_class']]
        
        return comparison_df
    
    def _create_athlete_pairs_df(self, df, y, results, X, athletes):
        """
        Create dataframe analyzing how athlete pairs perform together relative to the model.
        """
        # Get predicted values
        fitted_values = results.predict(X)
        
        # Create residuals (actual - predicted)
        residuals = y - fitted_values
        
        # Add residuals to original dataframe for easier analysis
        analysis_df = df.copy()
        analysis_df['residual'] = residuals
        
        # Initialize storage for pair results
        pairs_data = []
        
        # Iterate through all possible athlete pairs
        for i, athlete1 in enumerate(athletes):
            for j, athlete2 in enumerate(athletes[i+1:], i+1):  # Start from i+1 to avoid duplicates
                # Find races where both athletes participated
                # We'll assume an athlete participated if their name is in the Personnel column
                joint_races = analysis_df[analysis_df['Personnel'].str.contains(athlete1) & 
                                        analysis_df['Personnel'].str.contains(athlete2)]
                
                # If they raced together at least once
                if len(joint_races) > 0:
                    # Calculate metrics
                    avg_residual = joint_races['residual'].mean()
                    std_residual = joint_races['residual'].std()
                    count = len(joint_races)
                    
                    # Calculate a simple t-statistic as a measure of significance
                    # t = mean / (std / sqrt(n))
                    if std_residual > 0 and count > 1:
                        t_stat = avg_residual / (std_residual / np.sqrt(count))
                        # Calculate p-value (two-tailed test)
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=count-1))
                    else:
                        t_stat = np.nan
                        p_value = np.nan
                    
                    # Store the pair's results
                    pairs_data.append({
                        'Athlete1': athlete1,
                        'Athlete2': athlete2,
                        'AvgDelta': avg_residual,
                        'StdDelta': std_residual,
                        'Races': count,
                        't_stat': t_stat,
                        'p_value': p_value
                    })
        
        # Convert to DataFrame
        pairs_df = pd.DataFrame(pairs_data)
        
        # Sort by absolute t-statistic to find the most significant pairs
        if not pairs_df.empty:
            pairs_df['abs_t_stat'] = abs(pairs_df['t_stat'])
            pairs_df = pairs_df.sort_values('abs_t_stat', ascending=False)
            pairs_df = pairs_df.drop('abs_t_stat', axis=1)
        
        return pairs_df

    def _create_athlete_stats(self, results, athletes, X, max_correlation):
        """Create athlete statistics dataframe"""
        # Create basic athlete stats
        athletes_df = pd.DataFrame({
            'Rower': athletes,
            'Coefficient': results.params[athletes].round(1),
            'Lower': results.conf_int()[0][athletes].round(1),
            'Upper': results.conf_int()[1][athletes].round(1)
        })
        athletes_df.set_index('Rower', inplace=True)
        
        # Handle correlated athletes
        X_filtered = X.loc[:, X.columns.intersection(athletes)]
        correlations = group_highly_correlated_parameters(X_filtered.corr(), threshold=max_correlation)
        
        athletes_to_remove = set()
        athlete_groups = {}
        
        for group in correlations:
            for athlete in group:
                athletes_to_remove.add(athlete)
                athlete_groups[athlete] = group
        
        # Create dataframe for dropped athletes
        dropped_athletes_df = athletes_df.loc[athletes_df.index.intersection(athletes_to_remove)].copy()
        
        # Add group information
        dropped_athletes_df["Group Members"] = dropped_athletes_df.index.map(
            lambda x: ", ".join(sorted(set(athlete_groups[x])))
        )
        
        # Compute group-wide stats
        dropped_athletes_df["Group Coefficient Sum"] = dropped_athletes_df["Group Members"].map(
            lambda members: athletes_df.loc[members.split(", "), "Coefficient"].sum()
        )
        dropped_athletes_df["Group Upper Sum"] = dropped_athletes_df["Group Members"].map(
            lambda members: athletes_df.loc[members.split(", "), "Upper"].sum()
        )
        dropped_athletes_df["Group Lower Sum"] = dropped_athletes_df["Group Members"].map(
            lambda members: athletes_df.loc[members.split(", "), "Lower"].sum()
        )
        
        # Remove dropped athletes from main dataframe
        athletes_df = athletes_df.drop(index=athletes_to_remove, errors='ignore')
        athletes_df = self._add_side_aware_speed(athletes_df)
        
        return athletes_df, dropped_athletes_df

    def _create_shell_class_stats(self, results, shell_classes):
        """Create shell class statistics dataframe"""
        shell_classes_df = pd.DataFrame({
            'Shell Class': shell_classes,
            'Coefficient': results.params[shell_classes].round(1),
            'Lower': results.conf_int()[0][shell_classes].round(1),
            'Upper': results.conf_int()[1][shell_classes].round(1)
        })
        shell_classes_df.set_index('Shell Class', inplace=True)
        
        return shell_classes_df

    def _create_other_factors_stats(self, results, X, athletes):
        """Create statistics for other factors"""
        other_factors = [col for col in X.columns if col not in athletes]
        
        other_factors_df = pd.DataFrame({
            'Factor': other_factors,
            'Coefficient': results.params[other_factors].round(1),
            'Lower': results.conf_int()[0][other_factors].round(1),
            'Upper': results.conf_int()[1][other_factors].round(1)
        })
        other_factors_df.set_index('Factor', inplace=True)
        
        return other_factors_df

    def _check_race_balance(self, comparison_df):
        """Check if predictions are balanced within each race"""
        race_balance_check = comparison_df.groupby('Piece')['Delta'].sum().abs()
        max_imbalance = race_balance_check.max()
        avg_imbalance = race_balance_check.mean()
        imbalanced_races = race_balance_check[race_balance_check > 1.0].to_dict()
        
        return {
            'max_imbalance': max_imbalance,
            'avg_imbalance': avg_imbalance,
            'imbalanced_races': imbalanced_races
        }
    
    def _add_side_aware_speed(self, df):
        """Add side-aware speed calculations to the dataframe"""
        df = df.copy()

        # Extract suffix (ᵖ, ˢ, ᶜ, ˣ) from athlete names
        df["Suffix"] = df.index.to_series().str.extract(r'([ᵖˢᶜˣ])$')[0]

        # Determine the fastest athlete per suffix group
        fastest_by_suffix = df.groupby("Suffix")["Coefficient"].transform("min")

        # Compute speed relative to the fastest in each suffix group
        df["Speed"] = df["Coefficient"] - fastest_by_suffix
        df["Behind"] = df["Speed"].apply(lambda x: f"+{round(x, 1)}" if x > 0 else "-")
        df["Max/Min"] = df.apply(
            lambda row: f"{round(row['Lower'] - row['Coefficient'], 1)} to {round(row['Upper'] - row['Coefficient'], 1)}",
            axis=1
        )

        return df
    
    def _add_correlations(self, athletes_df, athletes, X, corr_matrix):
        """Add correlation information to the athletes dataframe"""
        # Add max and min correlation columns to athletes_df
        if not athletes_df.empty:
            # Get athlete columns from X
            athlete_columns = [col for col in X.columns if any(athlete in col for athlete in athletes)]
            
            # For each athlete in athletes_df
            max_correlations = []
            max_correlated_params = []
            min_correlations = []
            min_correlated_params = []
            
            for athlete in athletes_df.index:
                # Find columns related to this athlete
                athlete_cols = [col for col in athlete_columns if athlete in col]
                
                if athlete_cols:
                    # Find other athlete columns (excluding this athlete)
                    other_athlete_cols = [col for col in athlete_columns if athlete not in col]
                    
                    # Track overall max and min correlations for this athlete
                    overall_max_corr = -float('inf')
                    overall_min_corr = float('inf')
                    overall_max_corr_params = []
                    overall_min_corr_params = []
                    
                    for col in athlete_cols:
                        # Get correlations only with other athletes
                        if other_athlete_cols:
                            correlations = corr_matrix[col][other_athlete_cols]
                            if not correlations.empty:
                                # Find maximum correlation (without using abs)
                                max_corr = correlations.max()
                                
                                # Handle ties for maximum
                                if max_corr > overall_max_corr:
                                    overall_max_corr = max_corr
                                    overall_max_corr_params = [correlations.index[correlations == max_corr].tolist()]
                                elif max_corr == overall_max_corr:
                                    overall_max_corr_params.append(correlations.index[correlations == max_corr].tolist())
                                
                                # Find minimum correlation (without using abs)
                                min_corr = correlations.min()
                                
                                # Handle ties for minimum
                                if min_corr < overall_min_corr:
                                    overall_min_corr = min_corr
                                    overall_min_corr_params = [correlations.index[correlations == min_corr].tolist()]
                                elif min_corr == overall_min_corr:
                                    overall_min_corr_params.append(correlations.index[correlations == min_corr].tolist())
                    
                    # Flatten the lists of parameters and remove duplicates
                    if overall_max_corr_params:
                        flat_max_params = []
                        for param_list in overall_max_corr_params:
                            flat_max_params.extend(param_list)
                        # Remove duplicates while preserving order
                        max_params_unique = []
                        for item in flat_max_params:
                            if item not in max_params_unique:
                                max_params_unique.append(item)
                        max_correlations.append(overall_max_corr)
                        max_correlated_params.append(", ".join(max_params_unique))
                    else:
                        max_correlations.append(0)
                        max_correlated_params.append("")
                    
                    if overall_min_corr_params:
                        flat_min_params = []
                        for param_list in overall_min_corr_params:
                            flat_min_params.extend(param_list)
                        # Remove duplicates while preserving order
                        min_params_unique = []
                        for item in flat_min_params:
                            if item not in min_params_unique:
                                min_params_unique.append(item)
                        min_correlations.append(overall_min_corr)
                        min_correlated_params.append(", ".join(min_params_unique))
                    else:
                        min_correlations.append(0)
                        min_correlated_params.append("")
                else:
                    max_correlations.append(0)
                    max_correlated_params.append("")
                    min_correlations.append(0)
                    min_correlated_params.append("")

            # Add columns to athletes_df
            athletes_df['max_correlation'] = max_correlations
            athletes_df['max_correlated'] = max_correlated_params
            athletes_df['min_correlation'] = min_correlations
            athletes_df['min_correlated'] = min_correlated_params
            
    def _compute_athlete_weights(self, df, weight_stern):
        """Compute weights for athletes based on their position in the boat. Returns a dictionary of series, with weights for that athlete per piece."""
        athlete_weights = {}
        df["athlete_count"] = df["Personnel"].apply(lambda x: len(x.split("/")))
        
        for idx, row in df.iterrows():
            athletes_in_boat = row["Personnel"].split("/")
            n_athletes = len(athletes_in_boat)
            
            if weight_stern is not None and n_athletes > 1:
                # Create a linear weight distribution that still sums to 1
                # Bow (last position) gets lowest weight, stroke (first position) gets highest
                total_weight = 1.0
                base_weight = total_weight / n_athletes  # Equal distribution as starting point
                
                # Calculate adjustment factor based on weight_stern parameter
                # weight_stern represents how much extra weight to give to stroke vs bow
                adjustment_range = base_weight * weight_stern
                
                # Create linearly distributed weights
                position_weights = []
                for pos in range(n_athletes):
                    # pos=0 is stroke (stern), pos=n_athletes-1 is bow
                    # Higher positions (closer to stern) get higher weights
                    relative_position = (n_athletes - 1 - pos) / max(1, (n_athletes - 1))
                    position_weight = base_weight + (relative_position * adjustment_range) - (adjustment_range / 2)
                    position_weights.append(position_weight)
                
                # Ensure weights sum to exactly 1 to avoid introducing bias
                position_weights = np.array(position_weights) / sum(position_weights)
                
                # Assign weights to each athlete in the boat
                for athlete_idx, athlete in enumerate(athletes_in_boat):
                    if athlete not in athlete_weights:
                        athlete_weights[athlete] = pd.Series(0.0, index=df.index, dtype=float)
                    
                    athlete_weights[athlete].at[idx] = position_weights[athlete_idx]
            else:
                # Original even distribution of weight
                weight = 1.0 / n_athletes if n_athletes > 0 else 0.0
                for athlete in athletes_in_boat:
                    if athlete not in athlete_weights:
                        athlete_weights[athlete] = pd.Series(0.0, index=df.index, dtype=float)
                    
                    athlete_weights[athlete].at[idx] = weight
                    
        return athlete_weights

    def compare_lineups(self, lineups):
        """
        Compare multiple lineups and provide detailed breakdown of each.
        """
        if self.final_results is None:
            raise ValueError("No analysis results available. Run analysis first.")
        
        # Get all model parameters
        params = self.final_results['results'].params
        
        # Initialize results list
        results = []
        
        # Process each lineup
        for lineup in lineups:
            name = lineup['name']
            personnel = lineup['personnel']
            shell_class = lineup['shell_class']
            
            # Get the predicted time
            predicted_time = self.predict_lineup(personnel, shell_class)
            formatted_time = seconds_to_time(predicted_time)
            
            # Get shell class contribution
            shell_contribution = params.get(shell_class, 0)
            
            # Calculate athlete contributions
            temp_df = pd.DataFrame({
                'Personnel': ['/'.join(personnel)],
                'shell_class': [shell_class],
                'athlete_count': [len(personnel)]
            })
            
            athlete_weights = self._compute_athlete_weights(temp_df, self.weight_stern)
            athlete_contributions = []
            
            # Get individual contributions
            for i, athlete in enumerate(personnel):
                position = len(personnel) - i  # Stern = highest number, Bow = 1
                weight = athlete_weights[athlete].iloc[0] if athlete in athlete_weights else 0
                coefficient = params.get(athlete, 0)
                contribution = coefficient * weight
                
                athlete_contributions.append({
                    'athlete': athlete,
                    'position': position,
                    'weight': weight,
                    'coefficient': coefficient,
                    'contribution': contribution
                })
            
            # Calculate total athlete contribution
            total_athlete_contribution = sum(ac['contribution'] for ac in athlete_contributions)
            
            # Add to results
            results.append({
                'name': name,
                'shell_class': shell_class,
                'personnel': personnel,
                'predicted_time': predicted_time,
                'formatted_time': formatted_time,
                'shell_contribution': shell_contribution,
                'athlete_contribution': total_athlete_contribution,
                'athlete_details': athlete_contributions
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([{
            'Lineup': r['name'],
            'Shell Class': r['shell_class'],
            'Personnel': ', '.join(r['personnel']),
            'Predicted Time': r['formatted_time'],
            'Time (seconds)': round(r['predicted_time'], 1),
            'Shell Contribution': round(r['shell_contribution'], 1),
            'Athlete Contribution': round(r['athlete_contribution'], 1)
        } for r in results])
        
        # Sort by predicted time
        comparison_df = comparison_df.sort_values('Time (seconds)')
        
        return comparison_df, results

    def predict_lineup(self, personnel, shell_class, return_formatted=False):
        """
        Predict the speed (time per 500m) for an arbitrary lineup with a given shell class.
        """
        if self.final_results is None:
            raise ValueError("No analysis results available. Run analysis first.")
            
        # Create a temporary dataframe with one row for prediction
        temp_df = pd.DataFrame({
            'Personnel': ['/'.join(personnel)],
            'shell_class': [shell_class],
            'athlete_count': [len(personnel)]
        })
        
        # Apply position-based weighting to athletes
        athlete_weights = self._compute_athlete_weights(temp_df, self.weight_stern)
        
        # Create a design matrix for prediction
        # First, get all the parameters from the model
        all_params = self.final_results['results'].params.index.tolist()
        
        # Initialize a dataframe with zeros for all parameters
        X_pred = pd.DataFrame(0, index=[0], columns=all_params)
        
        # Set the shell class to 1
        if shell_class in X_pred.columns:
            X_pred[shell_class] = 1
        
        # Set athlete contributions based on weights
        for athlete in personnel:
            if athlete in X_pred.columns:
                # Use the computed weight for this athlete
                X_pred[athlete] = athlete_weights[athlete].iloc[0]
        
        # Make the prediction
        predicted_time = self.final_results['results'].predict(X_pred).iloc[0]
        
        # Return formatted time if requested
        if return_formatted:
            return seconds_to_time(predicted_time)
        
        return predicted_time

    def _add_side_aware_speed(self, df):
        """Add side-aware speed calculations to the dataframe"""
        df = df.copy()

        # Extract suffix (ᵖ, ˢ, ᶜ, ˣ) from athlete names
        df["Suffix"] = df.index.to_series().str.extract(r'([ᵖˢᶜˣ])$')[0]

        # Determine the fastest athlete per suffix group
        fastest_by_suffix = df.groupby("Suffix")["Coefficient"].transform("min")

        # Compute speed relative to the fastest in each suffix group
        df["Speed"] = df["Coefficient"] - fastest_by_suffix
        df["Behind"] = df["Speed"].apply(lambda x: f"+{round(x, 1)}" if x > 0 else f"{round(x, 1)}")
        df["Max/Min"] = df.apply(
            lambda row: f"{round(row['Lower'] - row['Coefficient'], 1)} to {round(row['Upper'] - row['Coefficient'], 1)}",
            axis=1
        )
        
        # Add position rankings within each group
        df["Rank"] = df.groupby("Suffix")["Coefficient"].rank(method="min")
        df["Total In Position"] = df.groupby("Suffix")["Coefficient"].transform("count")

        return df

    def get_athlete_position_info(self, athlete_name):
        """
        Get position information for a specific athlete.
        
        Parameters:
        -----------
        athlete_name : str
            Name of the athlete to get position info for
            
        Returns:
        --------
        dict
            Dictionary containing position, speed, rank, and other data
        """
        if self.final_results is None or 'athletes' not in self.final_results:
            return None
        
        # Check if athlete exists in the results
        athletes_df = self.final_results['athletes']
        if athlete_name not in athletes_df.index:
            # Check if they're in dropped athletes
            if ('dropped_athletes' in self.final_results and 
                self.final_results['dropped_athletes'] is not None and 
                athlete_name in self.final_results['dropped_athletes'].index):
                # Return minimal info for dropped athletes
                return {
                    'name': athlete_name,
                    'status': 'dropped',
                    'position_suffix': athlete_name[-1] if len(athlete_name) > 0 else '',
                    'position': self._suffix_to_position(athlete_name[-1] if len(athlete_name) > 0 else '')
                }
            return None
        
        # Get athlete's row
        athlete_row = athletes_df.loc[athlete_name]
        
        # Determine position from suffix
        position_suffix = athlete_name[-1] if len(athlete_name) > 0 else ''
        position = self._suffix_to_position(position_suffix)
        
        # Get enhanced dataframe with speed calculations
        enhanced_df = self._add_side_aware_speed(athletes_df)
        
        # Extract relevant data
        if athlete_name in enhanced_df.index:
            enhanced_row = enhanced_df.loc[athlete_name]
            result = {
                'name': athlete_name,
                'status': 'active',
                'coefficient': float(athlete_row['Coefficient']),
                'lower': float(athlete_row['Lower']),
                'upper': float(athlete_row['Upper']),
                'position_suffix': position_suffix,
                'position': position,
                'speed': float(enhanced_row['Speed']),
                'rank': int(enhanced_row['Rank']),
                'total_in_position': int(enhanced_row['Total In Position'])
            }
            return result
        
        return None

    def _suffix_to_position(self, suffix):
        """Convert a suffix character to a position name"""
        position_map = {
            'ˢ': 'Starboard',
            'ᵖ': 'Port',
            'ˣ': 'Scull',
            'ᶜ': 'Coxswain'
        }
        return position_map.get(suffix, 'Unknown')

    def get_position_athletes(self, position_suffix):
        """
        Get all athletes from a specific position.
        
        Parameters:
        -----------
        position_suffix : str
            Position suffix (ˢ, ᵖ, ˣ, ᶜ)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with position-specific metrics for all athletes in that position
        """
        if self.final_results is None or 'athletes' not in self.final_results:
            return pd.DataFrame()
        
        # Get athletes dataframe
        athletes_df = self.final_results['athletes']
        
        # Filter for athletes in the specified position
        position_athletes = athletes_df.index.str.endswith(position_suffix)
        position_df = athletes_df[position_athletes].copy()
        
        # Add position-based metrics
        if not position_df.empty:
            position_df = self._add_side_aware_speed(position_df)
        
        return position_df

    def calculate_position_metrics_for_coefficient(self, coefficient, position_suffix):
        """
        Calculate speed and rank for a hypothetical coefficient in a given position.
        
        Parameters:
        -----------
        coefficient : float
            Coefficient to calculate metrics for
        position_suffix : str
            Position suffix (ˢ, ᵖ, ˣ, ᶜ)
            
        Returns:
        --------
        dict
            Dictionary with speed and rank information
        """
        if self.final_results is None or 'athletes' not in self.final_results:
            return {'speed': None, 'rank': None, 'total_in_position': 0}
        
        # Get athletes dataframe
        athletes_df = self.final_results['athletes']
        
        # Filter for athletes in the specified position
        position_athletes = athletes_df.index.str.endswith(position_suffix)
        position_df = athletes_df[position_athletes].copy()
        
        if position_df.empty:
            return {'speed': None, 'rank': None, 'total_in_position': 0}
        
        # Determine speed (difference from best in position)
        best_coefficient = position_df['Coefficient'].min()
        speed = coefficient - best_coefficient
        
        # Determine rank
        position_df = position_df.sort_values('Coefficient')
        coefficients = list(position_df['Coefficient'])
        
        # Insert the hypothetical coefficient to determine rank
        rank = 1
        for coeff in coefficients:
            if coefficient <= coeff:
                break
            rank += 1
        
        return {
            'speed': speed,
            'rank': rank,
            'total_in_position': len(position_df) + 1  # +1 to include the hypothetical athlete
        }