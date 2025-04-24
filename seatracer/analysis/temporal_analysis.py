import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

class TemporalAnalysis:
    """
    Class for analyzing rower performance over time using an Analysis instance.
    Allows tracking changes in athlete performance across multiple time windows.
    """
    def __init__(self, analysis_instance, lookback=30):
        """
        Initialize the temporal analysis with an Analysis instance.
        
        Parameters:
        -----------
        analysis_instance : Analysis
            An instance of a class derived from the Analysis base class
        lookback : int
            Number of days to look back for each analysis point
        """
        self.analysis = analysis_instance
        self.lookback = lookback
        self.time_series_df = None
        self.stats_df = None
        self.final_results = None
    
    def run_temporal_analysis(self, get_history=True):
        """
        Run analysis over time windows and store results.
            
        Returns:
        --------
        self : TemporalAnalysis
            Returns self for method chaining
        """
        # Get a reference to the dataframe from the analysis object
        df = self.analysis.df.copy()
        
        # Ensure date column is datetime
        df['Race Session (date)'] = pd.to_datetime(df['Race Session (date)'])
        
        # Get unique dates in chronological order
        unique_dates = sorted(df['Race Session (date)'].unique())
        
        # Store results for all dates
        coefficients_by_date = {}
        results_by_date = {}
        all_athletes = set()
        
        # Process each date
        for idx, current_date in enumerate(unique_dates):
            # Define the lookback window
            lookback_start = current_date - pd.Timedelta(days=self.lookback)
            
            # Filter data within the lookback window
            window_df = df[(df['Race Session (date)'] >= lookback_start) & 
                          (df['Race Session (date)'] <= current_date)].copy()
            
            # Skip if not enough data in this window
            if len(window_df) < 5:  # Minimum number of rows needed
                continue
            
            # Create a temporary Analysis object with the window data
            window_analysis = self._create_window_analysis(window_df, current_date)
            
            try:
                # Run analysis on this window
                window_analysis.run_analysis(get_history=False)
                window_results = window_analysis.get_final_results()
                
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
    
    def _create_window_analysis(self, window_df, current_date):
        """
        Create an Analysis instance for a specific time window.
        Preserves all settings from the original analysis object.
        
        Parameters:
        -----------
        window_df : pandas DataFrame
            DataFrame containing rowing data for a specific time window
        current_date : datetime
            Current date being analyzed
            
        Returns:
        --------
        window_analysis : Analysis
            New Analysis instance for the window
        """
        # Create a clone of the analysis with the window data
        # We'll copy the analysis object and update its dataframe
        window_analysis = self._clone_analysis(window_df)
        
        # If the analysis uses recency weighting, recalculate based on current date
        if hasattr(window_analysis, 'halflife') and window_analysis.halflife is not None:
            # Recalculate recency weights relative to current date
            window_df = self._recalculate_recency_weights(window_df, current_date, window_analysis.halflife)
            window_analysis.df = window_df
            
        return window_analysis
    
    def _clone_analysis(self, window_df):
        """
        Create a clone of the original analysis object with a new dataframe.
        
        Parameters:
        -----------
        window_df : pandas DataFrame
            DataFrame to use in the new analysis object
            
        Returns:
        --------
        cloned_analysis : Analysis
            A new Analysis instance with the same parameters as the original
        """
        # Get the class of the original analysis
        analysis_class = self.analysis.__class__
        
        # Create a new instance with the same parameters but the window dataframe
        # We extract all the constructor parameters from the original analysis
        # except for the dataframe which we replace with window_df
        
        # Get all the attributes that are constructor parameters
        constructor_params = {}
        
        # Copy all dataclass fields (assuming Analysis is a dataclass)
        for field in self.analysis.__dataclass_fields__.keys():
            if field != 'df':  # Skip the dataframe
                constructor_params[field] = getattr(self.analysis, field)
        
        # Create a new instance with the window dataframe and copied parameters
        cloned_analysis = analysis_class(df=window_df, **constructor_params)
        
        return cloned_analysis
    
    def _recalculate_recency_weights(self, df, current_date, halflife):
        """
        Recalculate recency weights relative to the current analysis date.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing rowing data
        current_date : datetime
            Current date being analyzed
        halflife : float
            Halflife parameter for recency weighting
            
        Returns:
        --------
        df : pandas DataFrame
            DataFrame with updated recency weights
        """
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
        """
        Calculate statistics for each athlete's time series.
        
        Parameters:
        -----------
        all_athletes : set
            Set of all athletes found in the analysis
            
        Returns:
        --------
        stats_df : pandas DataFrame
            DataFrame containing statistics for each athlete
        """
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
        """
        Get the results from the most recent analysis window.
        
        Returns:
        --------
        final_results : dict
            Results from the most recent analysis window
        """
        return self.final_results
    
    def get_athlete_trend(self, athlete):
        """
        Get time series for a specific athlete.
        
        Parameters:
        -----------
        athlete : str
            Name of the athlete
            
        Returns:
        --------
        trend_df : pandas DataFrame
            DataFrame containing time series data for the athlete
        """
        if self.time_series_df is None:
            raise ValueError("No analysis results available")
        
        if athlete not in self.time_series_df.columns:
            raise ValueError(f"Athlete '{athlete}' not found in results")
        
        return self.time_series_df[['date', athlete]].dropna()
    
    def get_position_athletes(self, position):
        """
        Get list of athletes for a specific position.
        
        Parameters:
        -----------
        position : str
            Position name (Starboard, Port, Sculling, Coxswain)
            
        Returns:
        --------
        athletes : list
            List of athletes in the specified position
        """
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