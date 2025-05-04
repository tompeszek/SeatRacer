import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from seatracer.utils.helpers import add_side_aware_speed

class TemporalVisualizer:
    """
    Class for visualizing temporal analysis results in Streamlit.
    """
    def __init__(self, analysis):
        """
        Initialize the visualizer with analysis results.
        
        Parameters:
        -----------
        analysis : Analysis
            Instance of Analysis with temporal results
        """
        self.analysis = analysis
    
    def plot_position_trends(self, position, top_n=8, figsize=(12, 8), default_visible=None):
        """
        Create a Plotly figure for position-based performance trends with side-aware speed.
        
        Parameters:
        -----------
        position : str
            Position to plot ('starboard', 'port', 'sculling', 'coxswain')
        top_n : int
            Number of top athletes to include
        figsize : tuple
            Figure size (width, height)
        default_visible : list
            List of athlete names to display by default
            
        Returns:
        --------
        plotly.graph_objs.Figure
            Figure that can be displayed in Streamlit
        """
        try:
            # Get athletes for this position
            position_athletes = self.analysis.get_position_athletes(position)
            
            if not position_athletes:
                return None
            
            # Get temporal data  
            temporal_data = self.analysis.get_temporal_data()
            time_series_df = temporal_data['time_series_df']
            stats_df = temporal_data['stats_df']
            by_piece = temporal_data.get('by_piece', False)
            
            # Get stats for these athletes
            position_stats = stats_df[stats_df['Rower'].isin(position_athletes)]
            position_stats = position_stats.sort_values('Mean')
            
            # Determine which athletes to show by default
            if default_visible is None:
                default_visible = position_stats.head(top_n)['Rower'].tolist()
                
            # Create figure
            fig = go.Figure()
            
            # Initialize new_time_series with an empty dataframe
            new_time_series = pd.DataFrame()
            
            # Process each time point and compute speed relative to fastest athlete for each
            time_points = time_series_df['point'].unique()
            
            for time_point in time_points:
                # Get data for this time point
                point_data = time_series_df[time_series_df['point'] == time_point].copy()
                
                # Filter for just athletes of this position
                point_athletes = {athlete: point_data[athlete].iloc[0] for athlete in position_athletes 
                                if athlete in point_data.columns and not pd.isna(point_data[athlete].iloc[0])}
                
                if not point_athletes:  # Skip this point if no athletes from this position
                    continue
                    
                # Create a dataframe for side-aware speed calculation
                athlete_df = pd.DataFrame({
                    'Coefficient': point_athletes
                })
                athlete_df.index.name = 'Rower'
                
                # Extract suffix (ᵖ, ˢ, ᶜ, ˣ) from athlete names
                athlete_df["Suffix"] = athlete_df.index.to_series().str.extract(r'([ᵖˢᶜˣ])$')[0]

                # Determine the fastest athlete per suffix group
                fastest_by_suffix = athlete_df.groupby("Suffix")["Coefficient"].transform("min")

                # Compute speed relative to the fastest in each suffix group
                athlete_df["Speed"] = athlete_df["Coefficient"] - fastest_by_suffix
                
                # Store back in the time_series
                for athlete in athlete_df.index:
                    if athlete in point_data.columns:
                        point_data.loc[point_data.index[0], f"{athlete}_Speed"] = athlete_df.loc[athlete, 'Speed']
                
                # Append to new_time_series
                new_time_series = pd.concat([new_time_series, point_data])
            
            # Add trace for each athlete
            for athlete in position_stats['Rower']:
                # Create speed column name for this athlete
                speed_col = f"{athlete}_Speed"
                
                # Check if we have data for this athlete
                if speed_col not in new_time_series.columns:
                    continue
                    
                athlete_data = new_time_series[['point', 'date', athlete, speed_col]].dropna(subset=[speed_col])
                
                if athlete_data.empty:
                    continue
                    
                # Set visibility based on whether athlete is in default_visible list
                visible = True if athlete in default_visible else 'legendonly'
                
                # Get min/max from stats
                min_val = position_stats.loc[position_stats['Rower'] == athlete, 'Min'].iloc[0]
                max_val = position_stats.loc[position_stats['Rower'] == athlete, 'Max'].iloc[0]
                
                # Determine x-axis values based on by_piece
                if by_piece:
                    # Use piece numbers/names for x-axis
                    x_values = athlete_data['point']
                else:
                    # Use dates for x-axis
                    x_values = athlete_data['date']
                
                # Use the raw negative speed so higher is better
                y_values = -athlete_data[speed_col]

                # Create a list of hover templates, one for each point
                hover_texts = []
                for val in -athlete_data[speed_col]:
                    if val == 0:
                        hover_texts.append(f'{athlete}: Leader')
                    else:
                        hover_texts.append(f'{athlete}: +{-val:.1f}"/500m')

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=athlete,
                    visible=visible,
                    line=dict(width=2),
                    marker=dict(size=6),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    customdata=-athlete_data[speed_col]  # Still keep the customdata for other uses if needed
                ))
                
            # Set layout        
            # Set title and axis labels based on by_piece
            if by_piece:
                x_title = "Race Piece"
                title = f"Performance Trends by Race for {position.capitalize()} Position"
            else:
                x_title = "Date"
                title = f"Performance Trends by Date for {position.capitalize()} Position"
                
            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Speed (higher is better)",
                width=figsize[0]*100,
                height=figsize[1]*100,
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating position trend plot: {e}")
            return None
    
    def plot_athlete_comparison(self, athletes, figsize=(12, 8)):
        """
        Create a Plotly figure comparing multiple athletes using side-aware speed.
        
        Parameters:
        -----------
        athletes : list
            List of athlete names to compare
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plotly.graph_objs.Figure
            Figure that can be displayed in Streamlit
        """
        fig = go.Figure()
        
        # Get temporal data
        temporal_data = self.analysis.get_temporal_data()
        time_series_df = temporal_data['time_series_df'].copy()  # Make a copy to avoid modifying the original
        stats_df = temporal_data['stats_df']
        by_piece = temporal_data.get('by_piece', False)
        
        # Ensure proper sorting of the time series data
        if by_piece:
            # Convert date to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(time_series_df['date']):
                time_series_df['date'] = pd.to_datetime(time_series_df['date'])
            
            # Sort by date first, then by point if possible
            time_series_df = time_series_df.sort_values('date')
        else:
            # For date-based views, simply sort by date
            time_series_df = time_series_df.sort_values('date')
        
        # Initialize new_time_series with an empty dataframe
        new_time_series = pd.DataFrame()
        
        # Process each time point and compute speed relative to fastest athlete for each
        for _, point_data in time_series_df.groupby('point', sort=False):
            # Filter for just the selected athletes
            point_athletes = {athlete: point_data[athlete].iloc[0] for athlete in athletes 
                            if athlete in point_data.columns and not pd.isna(point_data[athlete].iloc[0])}
            
            if not point_athletes:  # Skip this point if no selected athletes
                continue
                
            # Create a dataframe for side-aware speed calculation
            athlete_df = pd.DataFrame({
                'Coefficient': point_athletes
            })
            athlete_df.index.name = 'Rower'
            
            # Extract suffix (ᵖ, ˢ, ᶜ, ˣ) from athlete names
            athlete_df["Suffix"] = athlete_df.index.to_series().str.extract(r'([ᵖˢᶜˣ])$')[0]

            # Determine the fastest athlete per suffix group
            fastest_by_suffix = athlete_df.groupby("Suffix")["Coefficient"].transform("min")

            # Compute speed relative to the fastest in each suffix group
            athlete_df["Speed"] = athlete_df["Coefficient"] - fastest_by_suffix
            
            # Store back in the time_series
            for athlete in athlete_df.index:
                if athlete in point_data.columns:
                    point_data.loc[point_data.index[0], f"{athlete}_Speed"] = athlete_df.loc[athlete, 'Speed']
            
            # Append to new_time_series
            new_time_series = pd.concat([new_time_series, point_data])
        
        # Ensure new_time_series maintains the same sorting as time_series_df
        new_time_series = new_time_series.merge(
            time_series_df[['point', 'date']], 
            on=['point', 'date'],
            how='left'
        )
        
        # Add trace for each athlete
        for athlete in athletes:
            try:
                # Create speed column name for this athlete
                speed_col = f"{athlete}_Speed"
                
                # Check if we have data for this athlete
                if speed_col not in new_time_series.columns:
                    continue
                    
                athlete_data = new_time_series[['point', 'date', athlete, speed_col]].dropna(subset=[speed_col])
                
                if athlete_data.empty:
                    continue
                    
                # Get min/max from stats for this athlete if available
                if athlete in stats_df['Rower'].values:
                    min_val = stats_df.loc[stats_df['Rower'] == athlete, 'Min'].iloc[0]
                    max_val = stats_df.loc[stats_df['Rower'] == athlete, 'Max'].iloc[0]
                    min_max_text = f"<br>Min/Max: {min_val:.1f}/{max_val:.1f}"
                else:
                    min_max_text = ""
                    
                # Determine x-axis values based on by_piece
                if by_piece:
                    # Use piece numbers/names for x-axis
                    x_values = athlete_data['point']
                else:
                    # Use dates for x-axis
                    x_values = athlete_data['date']
                    
                # Hover template with correct information
                hovertemplate = f'Athlete: {athlete}<br>Speed: +%{{y:.1f}}'#<br>Coefficient: %{{customdata:.1f}}{min_max_text}<extra></extra>'
                
                # Use the raw negative speed so higher is better
                y_values = -athlete_data[speed_col]
                
                # Add coefficient as custom data for hover
                customdata = athlete_data[athlete]
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=athlete,
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate=hovertemplate,
                    customdata=customdata
                ))
            except Exception as e:
                print(f"Error plotting athlete {athlete}: {e}")
                continue
                
        # Set layout
        # Set title and axis labels based on by_piece
        if by_piece:
            x_title = "Race Piece"
            title = "Athlete Performance Comparison by Race"
        else:
            x_title = "Date"
            title = "Athlete Performance Comparison by Date"
            
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title="Speed (higher is better)",
            width=figsize[0]*100,
            height=figsize[1]*100,
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        )
        
        # Ensure the x-axis is in the correct order
        if by_piece:
            # Create a category order based on the sorted dates
            unique_points = new_time_series.sort_values('date')['point'].unique()
            fig.update_xaxes(categoryorder='array', categoryarray=unique_points)
        
        return fig
    
    def create_streamlit_position_chart(self, position, top_n=8):
        """
        Create an interactive Streamlit chart for a position.
        
        Parameters:
        -----------
        position : str
            Position to plot ('starboard', 'port', 'sculling', 'coxswain')
        top_n : int
            Number of top athletes to include by default
        """
        try:
            # Get athletes for this position
            position_athletes = self.analysis.get_position_athletes(position)
            
            if not position_athletes:
                st.warning(f"No data available for {position} position.")
                return
                
            # Get temporal data
            temporal_data = self.analysis.get_temporal_data()
            stats_df = temporal_data['stats_df']
            
            # Get stats for these athletes
            position_stats = stats_df[stats_df['Rower'].isin(position_athletes)]
            position_stats = position_stats.sort_values('Mean')
            
            # Create chart with all position athletes visible
            if position_athletes:
                fig = self.plot_position_trends(
                    position=position,
                    default_visible=position_athletes
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("Please select at least one athlete to display the chart.")
                
        except ValueError as e:
            st.error(f"Error: {e}")
    
    def create_streamlit_interface(self):
        """
        Create a complete Streamlit interface for temporal analysis results.
        """
        # Check if temporal analysis has been run
        temporal_data = self.analysis.get_temporal_data()
        if temporal_data['time_series_df'] is None or temporal_data['stats_df'] is None:
            st.warning("No temporal analysis results available. Please run analysis.run_history() first.")
            return
                    
        # Get available positions
        position_suffix_map = {
            'Starboard': 'ˢ',
            'Port': 'ᵖ',
            'Sculling': 'ˣ',
            'Coxswain': 'ᶜ'
        }
        
        stats_df = temporal_data['stats_df']
        
        available_positions = []
        for position, suffix in position_suffix_map.items():
            if any(rower.endswith(suffix) for rower in stats_df['Rower']):
                available_positions.append(position)
                
        if not available_positions:
            st.warning("No position data available for visualization.")
            return
            
        # Create position selector
        selected_position = st.radio(
            "Select Position to Analyze",
            options=available_positions,
            index=0
        )
        
        # Create the chart for selected position
        self.create_streamlit_position_chart(selected_position)