import debugpy

if not debugpy.is_client_connected():  # Check if a debugger is already attached
    debugpy.listen(("localhost", 5678))
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()  # Pause execution until the debugger is attached

import streamlit as st
import pandas as pd
import os

from seatracer.analysis.analysis_base import *
from analysis.registry import ModelRegistry
from optimize.lineup_optimizer import LineupOptimizer
from utils.grouping import *
from utils.helpers import (
    add_athlete_counts,
    determine_shell_class,
    get_rower_sides_count,
    append_rigging_to_names
)
from visualization.charts import *

from seatracer.ui.sections import (
    athletes_section,
    data_section,
    performance_section,
    correlations_section, 
    optimal_section,
    validation_section,
    synergy_section,
    lineup_section,
    new_lineup_section,
    debug_section,
    time_section
)

st.set_page_config(
    layout="wide",
    page_title="SeatRacer",
    menu_items={
        'Get Help': 'mailto:tompeszek@gmail.com',
        'Report a bug': "mailto:tompeszek@gmail.com",
        'About': "# SeatRacer"
    }
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=B612&display=swap');

html, body, [class*="css"] {
    font-family: 'B612', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

default_session_values = {
    'action_pills': None,
    'data_action_pills': None,
    'analysis': None,
    'optimizer': None, 
    'temporal_analysis': None,
    'sides_count': None,
    'athlete_ergs_df': pd.DataFrame(),
    'current_data': pd.DataFrame(),
    'rerun': False,
    'reset_athletes': True
}

for key, default_value in default_session_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

### Sidebar

## Data Filters
st.sidebar.subheader("Data Filters")
shell_class = st.sidebar.segmented_control(
    'Include Shell Classes', 
    options=['2-', '4-', '4+', '8+'],
    selection_mode='multi',
    default=['2-', '4-', '4+', '8+'],
    on_change=lambda: setattr(st.session_state, 'rerun', True)
)

### Models
model_display_names = [choice['label'] for choice in ModelRegistry.get_model_choices()]

st.sidebar.divider()
st.sidebar.subheader("Models")
select_model = st.sidebar.radio(
    "Model", model_display_names, index=0, label_visibility='collapsed', on_change=lambda: setattr(st.session_state, 'rerun', True)
)
# st.sidebar.markdown("_Models with * are not recommended_")
selected_model = ModelRegistry.get_model_class_by_name(select_model)
st.sidebar.caption(selected_model.model_description)

run_history = st.sidebar.toggle("Run historical analysis", value=False)

# Weighting
if selected_model.uses_custom_weighting:
    if not st.session_state.current_data.empty:
        days_diff = (pd.to_datetime(st.session_state.current_data['Race Session (date)']).max() - pd.to_datetime(st.session_state.current_data['Race Session (date)']).min()).days
    else:
        days_diff = 0
        
    recency_options = {
        "Off": None,
        "Low": 210.0,
        "Medium": 56.0,
        "High": 21.0,
    }

    st.sidebar.divider()
    st.sidebar.header("Model Weights")

    # Define the options with captions and values, using underscores for italics
    close_races_options = {
        "Off": {"value": None, "caption": "_Margins do not affect race result weighting_"},
        "Low": {"value": 12.0, "caption": '_Races determined by 1" are weighted twice as much as those with a 12" margin_'},
        "Medium": {"value": 8.0, "caption": '_Races determined by 1" are weighted twice as much as those with a 8" margin_'},
        "High": {"value": 5.0, "caption": '_Races determined by 1" are weighted twice as much as those with a 5" margin_'},
    }

    stern_bias_options = {
        "Off": {"value": 0.0, "caption": "_Rowers in all positions get the same credit or blame for every result_"},
        "Low": {"value": 0.1, "caption": "_Stroke seat gets 10% more credit or blame than bow seat_"}, # 1.1 is wildly bad intervals
        "Medium": {"value": 0.5, "caption": "_Stroke seat gets 50% more credit or blame than bow seat_"},
        "High": {"value": 1.0, "caption": "_Stroke seat gets 100% more credit or blame than bow seat_"}, # meh intervals at 2
    }

    # Close Races widget
    st.sidebar.markdown("### Close Races")
    weight_close = st.sidebar.radio("Close Races", list(close_races_options.keys()), horizontal=False, index=0, label_visibility='collapsed', on_change=lambda: setattr(st.session_state, 'rerun', True)) # maybe index=2
    weight_close_text = close_races_options[weight_close]["caption"]
    st.sidebar.caption(weight_close_text)

    # Stern Bias widget
    if selected_model.can_have_stern_bias:
        st.sidebar.markdown("### Stern Bias")
        weight_stern = st.sidebar.radio("Stern Bias", list(stern_bias_options.keys()), horizontal=False, index=0, label_visibility='collapsed', on_change=lambda: setattr(st.session_state, 'rerun', True))# maybe index=1
        weight_stern_text = stern_bias_options[weight_stern]["caption"]
        st.sidebar.caption(weight_stern_text)
    else:
        weight_stern = None

    # Recency Weighting
    st.sidebar.markdown("### Recency Weighting")
    recency_halflife = st.sidebar.radio("Recency Weighting", list(recency_options.keys()), horizontal=False, index=0, label_visibility='collapsed', on_change=lambda: setattr(st.session_state, 'rerun', True))# maybe index=2
    halflife = recency_options[recency_halflife]
    halflife_text = f"{halflife:.0f}" if halflife is not None else "Off"
    if recency_halflife != "Off":
        st.sidebar.caption(f"_At {halflife_text}{" days, a result's weight is reduced by half" if halflife_text != 'Off' else ''}_")
    else:
        st.sidebar.caption(f"_Older races are weighted the same as more recent races_")



## Parameters
st.sidebar.divider()
st.sidebar.subheader("Parameters")

max_correlation = st.sidebar.slider("Max Allowed Correlation", min_value = 0.5, max_value = 1.0, value = 0.8, step = 0.01, on_change=lambda: setattr(st.session_state, 'rerun', True))

# Checkbox options
include_coxswains = st.sidebar.checkbox('Coxswains')
if include_coxswains :
    st.sidebar.caption(f"_Include coxswains in analysis_")
else:
    st.sidebar.caption(f"_Ignore coxswains - assume every cox has minimal impact on crew performance_")

## Over Time
st.sidebar.divider()
st.sidebar.subheader("Evalation Over Time")

lookback_days = st.sidebar.slider('Lookback Days', 1, 100, 50, on_change=lambda: setattr(st.session_state, 'rerun', True))

### App Code
# Copy data and add fields (athlete counts, shell class)
if not st.session_state.current_data.empty:
    df = st.session_state.current_data.copy()
    
    add_athlete_counts(df)

    df['shell_class'] = df.apply(determine_shell_class, axis=1)

    # Apply shell class filter
    filtered_data = df[df['shell_class'].isin(shell_class)]
    # Add sides to names  (also adds coxswain to personnel if needed)
    filtered_data = append_rigging_to_names(filtered_data)

    # Only go forward if there is data:
    if not filtered_data.empty:
        
        # Add piece names
        filtered_data['Piece'] = filtered_data['Race Session (date)'].astype(str) + " #" + filtered_data['Piece'].astype(str)

        # Sides count
        st.session_state.sides_count = get_rower_sides_count(filtered_data)

        if selected_model.uses_custom_weighting:
            weight_close_factor = close_races_options[weight_close]["value"]
            weight_stern_factor = stern_bias_options[weight_stern]["value"]
        else:
            weight_close_factor = None
            weight_stern_factor = None
            halflife = None

        if st.session_state.rerun or 'analysis' not in st.session_state or st.session_state.analysis is None:
            analysis = selected_model(
                df=filtered_data,                
                max_correlation=max_correlation,
                halflife=halflife,
                weight_close=weight_close_factor,
                weight_stern=weight_stern_factor,
                include_coxswains=include_coxswains,
                erg_scores=st.session_state.athlete_ergs_df if 'athlete_ergs_df' in st.session_state else None
            )

            st.session_state.analysis = analysis.run_analysis()
            st.session_state.optimizer = LineupOptimizer(analysis)
        else:
            analysis = st.session_state.analysis         

### Main UI
show_athletes = (
    selected_model.show_athletes
    and 'current_data' in st.session_state
    and not st.session_state.current_data.empty
)

if show_athletes:
    (
        data_tab, athletes_tab, performance_tab, corr_tab, validation_tab,
        syn_tab, new_lineup_tab, lineup_tab, optimal_tab, time_tab, debug_tab
    ) = st.tabs([
        "Data", "Athletes", "Performance", "Correlations", "Validation",
        "Synergies", "New Lineup", "Lineup Testing", "Optimal Lineups", "Over Time", "Debug"
    ])
else:
    (
        data_tab, performance_tab, corr_tab, validation_tab, syn_tab,
        new_lineup_tab, lineup_tab, optimal_tab, time_tab, debug_tab
    ) = st.tabs([
        "Data", "Performance", "Correlations", "Validation", "Synergies",
        "New Lineup", "Lineup Testing", "Optimal Lineups", "Over Time", "Debug"
    ])

with data_tab:
    data_section.render()

if show_athletes:
    with athletes_tab:
        athletes_section.render()

with time_tab:
    # time_section.render(analysis, filtered_data, run_history, lookback_days, run_regression, models, max_correlation, halflife, weight_close_factor, weight_stern_factor, include_coxswains)
    st.write("Not implemented yet.")

with performance_tab:
    performance_section.render(st.session_state.sides_count)

with corr_tab:
    correlations_section.render()

with new_lineup_tab:
    new_lineup_section.render()
        
with lineup_tab:
    lineup_section.render()

with validation_tab:
    validation_section.render()    

with syn_tab:
    synergy_section.render()

with debug_tab:
    debug_section.render(max_correlation)

with optimal_tab:
    optimal_section.render()

# Finally, set rerun analysis to False by default
st.session_state.rerun = False