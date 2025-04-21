import pandas as pd
import numpy as np
from utils.helpers import *

class TemporalAnalysis:
    """
    Class for analyzing rower performance over time.
    """
    def __init__(self, df, lookback=30):
        """
        Initialize the analysis with rowing data.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing rowing race data
        lookback : int
            Number of days to look back for each analysis point
        """
        self.df = df.copy()
        self.lookback = lookback
        self.time_series_df = None
        self.stats_df = None
        self.final_results = None
    
    def run_analysis(self, regression_func, **kwargs):
        """
        Run regression analysis over time and store results.
        
        Parameters:
        -----------
        regression_func : function
            Function to run regression analysis (e.g., run_regression)
        **kwargs : dict
            Additional parameters to pass to the regression function
            
        Returns:
        --------
        self : TemporalAnalysis
            Returns self for method chaining
        """
        # Ignore overly-correlated athletes - will likely happen in smaller windows
        kwargs['max_correlation'] = 2.0

        # Ensure date column is datetime
        self.df['Race Session (date)'] = pd.to_datetime(self.df['Race Session (date)'])
        
        # Get unique dates in chronological order
        unique_dates = sorted(self.df['Race Session (date)'].unique())
        
        # Store results for all dates
        coefficients_by_date = {}
        results_by_date = {}
        all_athletes = set()
        
        # Process each date
        for idx, current_date in enumerate(unique_dates):
            # Define the lookback window
            lookback_start = current_date - pd.Timedelta(days=self.lookback)
            
            # Filter data within the lookback window
            window_df = self.df[(self.df['Race Session (date)'] >= lookback_start) & 
                               (self.df['Race Session (date)'] <= current_date)].copy()
            
            # # Skip if not enough data in this window
            # if len(window_df) < 5:  # Minimum number of rows needed
            #     continue
            
            # Adjust recency weights to be relative to current date if using recency
            if 'halflife' in kwargs and kwargs['halflife'] is not None:
                window_df = self._recalculate_recency_weights(window_df, current_date, kwargs['halflife'])
            
            try:
                # Run regression on this window
                window_results = regression_func(window_df, **kwargs)
                
                # Store results with this date
                results_by_date[current_date] = window_results
                
                # Extract coefficients
                athletes_df = window_results['athletes']
                
                # Store coefficients for this date
                date_coeffs = {'date': current_date}
                for idx, row in athletes_df.iterrows():
                    athlete = idx
                    all_athletes.add(athlete)
                    date_coeffs[athlete] = row['Coefficient']
                
                coefficients_by_date[current_date] = date_coeffs
                
                # Update final results to the most recent date
                self.final_results = window_results
                
            except Exception as e:
                print(f"Error processing date {current_date}: {e}")
                continue
        
        # Create time series dataframe
        if coefficients_by_date:
            self.time_series_df = pd.DataFrame(list(coefficients_by_date.values()))
            self.time_series_df = self.time_series_df.sort_values('date')
            
            # Calculate statistics for each rower
            self.stats_df = self._calculate_athlete_statistics(all_athletes)
        
        return self
    
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
        if self.time_series_df is None:
            raise ValueError("No analysis results available")
        
        if athlete not in self.time_series_df.columns:
            raise ValueError(f"Athlete '{athlete}' not found in results")
        
        return self.time_series_df[['date', athlete]].dropna()
    
    def get_position_athletes(self, position):
        """Get list of athletes for a specific position"""
        if self.stats_df is None:
            raise ValueError("No analysis results available")
        
        position_suffix_map = {
            'Starboard': 'ˢ',
            'Port': 'ᵖ',
            'Sculling': 'ˣ',
            'Coxswain': 'ᶜ'
        }
        
        if position not in position_suffix_map:
            raise ValueError(f"Position must be one of {list(position_suffix_map.keys())}")
        
        suffix = position_suffix_map[position]
        
        return [a for a in self.stats_df['Rower'] if a.endswith(suffix)]