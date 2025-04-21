import streamlit as st

from seatracer.optimize.lineup_optimizer2 import LineupOptimizer2

def render():
    analysis = st.session_state.analysis
    if analysis is not None:
        st.subheader("Lineup Testing")
        st.write("_Select rowers and boat class to test different lineups_")

        new_optimizer = LineupOptimizer2(analysis)