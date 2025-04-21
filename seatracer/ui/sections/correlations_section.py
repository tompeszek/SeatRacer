import streamlit as st

def render():
    st.subheader("Correlation Matrix")
    st.write("_Shows how often certain rowers are boated with others. If the correlation is too great (≥ 0.5 or ≤ -0.5), the model cannot effectively separate the performances of each rower_")
    
    analysis = st.session_state.analysis
    if analysis is not None:
        st.dataframe(analysis.final_results['corr'].round(2))
    else:
        st.write("No data available.")