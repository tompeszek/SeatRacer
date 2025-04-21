import streamlit as st
import plotly.graph_objects as go

class TemporalVisualizer:
    """
    Class for visualizing temporal analysis results in Streamlit.
    """
    def __init__(self, temporal_analysis):
        """
        Initialize the visualizer with temporal analysis results.
        
        Parameters:
        -----------
        temporal_analysis : TemporalAnalysis
            Instance of TemporalAnalysis with results
        """
        self.analysis = temporal_analysis
    
    def plot_position_trends(self, position, top_n=8, figsize=(12, 8), default_visible=None):
        """
        Create a Plotly figure for position-based performance trends.
        
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
                
            # Get stats for these athletes
            position_stats = self.analysis.stats_df[self.analysis.stats_df['Rower'].isin(position_athletes)]
            position_stats = position_stats.sort_values('Mean')
            
            # Determine which athletes to show by default
            if default_visible is None:
                default_visible = position_stats.head(top_n)['Rower'].tolist()
                
            # Create figure
            fig = go.Figure()
            
            # Add trace for each athlete
            for athlete in position_stats['Rower']:
                athlete_data = self.analysis.get_athlete_trend(athlete)
                
                if athlete_data.empty:
                    continue
                    
                # Set visibility based on whether athlete is in default_visible list
                visible = True if athlete in default_visible else 'legendonly'
                
                fig.add_trace(go.Scatter(
                    x=athlete_data['date'],
                    y=athlete_data[athlete],
                    mode='lines+markers',
                    name=athlete,
                    visible=visible,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
            # Set layout
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                title=f"Performance Trends for {position.capitalize()} Position",
                xaxis_title="Date",
                yaxis_title="Coefficient (lower is better)",
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
            
        except ValueError as e:
            print(f"Error creating position trend plot: {e}")
            return None
    
    def plot_athlete_comparison(self, athletes, figsize=(12, 8)):
        """
        Create a Plotly figure comparing multiple athletes.
        
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
        
        for athlete in athletes:
            try:
                athlete_data = self.analysis.get_athlete_trend(athlete)
                
                if not athlete_data.empty:
                    fig.add_trace(go.Scatter(
                        x=athlete_data['date'],
                        y=athlete_data[athlete],
                        mode='lines+markers',
                        name=athlete,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
            except ValueError:
                continue
                
        # Set layout
        fig.update_layout(
            title="Athlete Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Coefficient (lower is better)",
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
                
            # Get stats for these athletes
            position_stats = self.analysis.stats_df[self.analysis.stats_df['Rower'].isin(position_athletes)]
            position_stats = position_stats.sort_values('Mean')
            
            # Default to showing top N athletes
            default_athletes = position_stats.head(top_n)['Rower'].tolist()
            
            # Create athlete selector
            # st.subheader(f"Select {position.capitalize()} Athletes to Display")
            # selected_athletes = st.multiselect(
            #     "Athletes:",
            #     options=position_athletes,
            #     default=default_athletes
            # )
            
            # Create chart with selected athletes
            if position_athletes:
                fig = self.plot_position_trends(
                    position=position,
                    default_visible=position_athletes
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # # Show statistics for selected athletes
                # with st.expander("Show Statistics"):
                #     stats_df = self.analysis.stats_df[self.analysis.stats_df['Rower'].isin(selected_athletes)]
                #     stats_df = stats_df.sort_values('Mean')
                #     st.dataframe(stats_df)
            else:
                st.info("Please select at least one athlete to display the chart.")
                
        except ValueError as e:
            st.error(f"Error: {e}")
    
    def create_streamlit_interface(self):
        """
        Create a complete Streamlit interface for temporal analysis results.
        """
        if self.analysis.time_series_df is None or self.analysis.stats_df is None:
            st.warning("No temporal analysis results available.")
            return
            
        # Get available positions
        position_suffix_map = {
            'Starboard': 'ˢ',
            'Port': 'ᵖ',
            'Sculling': 'ˣ',
            'Coxswain': 'ᶜ'
        }
        
        available_positions = []
        for position, suffix in position_suffix_map.items():
            if any(rower.endswith(suffix) for rower in self.analysis.stats_df['Rower']):
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
        
        # # Create custom comparison section
        # with st.expander("Compare Specific Athletes"):
        #     st.subheader("Custom Athlete Comparison")
            
        #     # Get all athletes
        #     all_athletes = self.analysis.stats_df['Rower'].tolist()
            
        #     # Create athlete selector
        #     custom_athletes = st.multiselect(
        #         "Select athletes to compare:",
        #         options=all_athletes,
        #         default=[]
        #     )
            
        #     if custom_athletes:
        #         fig = self.plot_athlete_comparison(custom_athletes)
        #         st.plotly_chart(fig, use_container_width=True)