import streamlit as st

from seatracer.utils.grouping import group_highly_correlated_parameters

def render(max_correlation):
    st.dataframe(st.session_state.analysis.final_results['shell_classes'])
    st.dataframe(st.session_state.analysis.final_results['athletes'])

    if st.session_state.athlete_ergs_df is None:
        st.write("No athlete ergs data available.")
    else:
        st.subheader("Athlete Ergs")
        st.dataframe(st.session_state.athlete_ergs_df)

    analysis = st.session_state.analysis
    if analysis is not None:
        analysis.final_results['raw']

        st.subheader("Piece Weights")
        st.dataframe(analysis.final_results['weights'])

        # debug data
        st.subheader("Correlated Groups")   
        highly_correlated_groups = group_highly_correlated_parameters(analysis.final_results['corr'], threshold=max_correlation)

        for i, group in enumerate(highly_correlated_groups, 1):
            st.write(f"Group {i}: {group}")    
    else:
        st.write("No data available.")