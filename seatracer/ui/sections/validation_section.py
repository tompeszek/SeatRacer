import streamlit as st

def render():
    analysis = st.session_state.analysis
    if analysis is not None:
        st.subheader("Model Fit")
        abs_delta_sum = analysis.final_results['comparison']['Delta'].abs().sum()
        abs_delta_avg = analysis.final_results['comparison']['Delta'].abs().mean()        
        abs_delta_max = analysis.final_results['comparison']['Delta'].abs().max()
        squared_delta_sum = (analysis.final_results['comparison']['Delta'] ** 2).sum()

        col_avg, col_max, col_sqr  = st.columns([1, 1, 1])
        with col_avg:
            st.metric(label="Average Model Error", value=f'±{abs_delta_avg:.2f}" / 500m')
        with col_max:
            st.metric(label="Greatest Model Error", value=f'±{abs_delta_max:.2f}" / 500m')
        with col_sqr:
            st.metric(label="Squared Model Error", value=f'{squared_delta_sum:.2f}"')
        
        st.subheader("Actual vs. Model")
        st.dataframe(analysis.final_results['comparison'], hide_index=True) #height=300
    else:
        st.write("No data available.")