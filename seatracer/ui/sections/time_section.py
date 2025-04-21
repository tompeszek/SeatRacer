import streamlit as st

from seatracer.analysis.temporal_analysis import TemporalAnalysis
from visualization.temporal_visualization import TemporalVisualizer

def render(filtered_data, run_history, lookback_days, run_regression, models, max_correlation, halflife, weight_close_factor, weight_stern_factor, include_coxswains):
    st.subheader("Performance Over Time")

    analysis = st.session_state.analysis
    if analysis is not None:
        if run_history:
            if st.session_state.rerun or st.session_state.temporal_analysis is None:
                # Run temporal analysis on the filtered data
                # Note that this could be moved earlier in your code to avoid redundant processing
                st.session_state.temporal_analysis = TemporalAnalysis(filtered_data, lookback=lookback_days)
                
                # Run the analysis using your existing run_regression function
                # Pass the same parameters you used for the main analysis
                st.session_state.temporal_analysis.run_analysis(
                    run_regression,
                    selected_model=models[select_model],
                    max_correlation=max_correlation,
                    halflife=halflife,
                    weight_close=weight_close_factor,
                    weight_stern=weight_stern_factor,
                    include_coxswains=include_coxswains,
                    erg_scores=st.session_state.athlete_ergs_df if 'athlete_ergs_df' in st.session_state else None,
                    gd_learning_rate=0.01,
                    gd_iterations=100000  # Reduced for faster processing
                )

            # Create visualizer and display results
            visualizer = TemporalVisualizer(st.session_state.temporal_analysis)
            visualizer.create_streamlit_interface()
            
        else:
            st.warning("To see performance over time, switch on the toggle for historical analysis in the sidebar.")        
        
    else:
        st.warning("No data available. Please load and analyze some racing data first.")