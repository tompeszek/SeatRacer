import streamlit as st

from visualization.temporal_visualization import TemporalVisualizer

def render():
    analysis = st.session_state.analysis
    if analysis is not None:
        st.subheader("Performance Over Time")
        
        # Controls row
        col1, col2, col3 = st.columns([1,1,3])

        with col1:
            run_analysis = st.button("Run historical analysis")
        
        with col2:
            show_per_piece = st.checkbox(
                "Show timeline per piece",
                key='show_timeline_per_piece'
            )
        
        with col3:
            lookback_days = st.slider(
                'Lookback Days', 
                1, 100, 50,
                key='lookback_days'
            )
        
        # Run analysis if button clicked
        if run_analysis:
            with st.spinner("Running temporal analysis..."):
                analysis.run_history(
                    custom_lookback=lookback_days, 
                    by_piece=show_per_piece
                )
                st.session_state.history_analysis_complete = True
        
        # Show visualization if analysis exists
        if hasattr(analysis, 'temporal_data') and analysis.temporal_data.get('time_series_df') is not None:
            visualizer = TemporalVisualizer(analysis)
            visualizer.create_streamlit_interface()
        elif not run_analysis:  # Only show this if not currently running
            st.info("Click 'Run historical analysis' to see performance trends over time.")
    else:
        st.warning("No data available. Please load and analyze some racing data first.")