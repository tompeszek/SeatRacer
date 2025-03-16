from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
from helpers import *
from charts import *
from weighting import *
from grouping import *
import os

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

results = None

def clear_data():
    st.session_state.current_data = pd.DataFrame()

# Edit Race Data section (only visible if editor toggle is on)
def save_changes():
    st.session_state.current_data = edited_dataframe

# Sidebar dataset selection
data_folder = './data'
# def load_data_on_change(dataset_selected):    
#     file_path = os.path.join(data_folder, dataset_selected)
#     st.session_state.current_data = pd.read_csv(file_path)

if 'current_data' not in st.session_state:
    clear_data()



### Sidebar

## Data Filters
st.sidebar.subheader("Data Filters")
shell_class = st.sidebar.segmented_control(
    'Include Shell Classes', 
    options=['2-', '4-', '4+', '8+'],
    selection_mode='multi',
    default=['2-', '4-', '4+', '8+']
)

# Models
st.sidebar.divider()
st.sidebar.subheader("Models")

# Grouping the models
models = {
    # "Ridge Regression": "ridge",
    "Generalized Linear Model": "glm",
    "Weighted Least Squares": "wls",
    "Robust Linear Model*": "rlm", # does internal weighing
    "Ordinary Least Squares*": "ols"
}

select_model = st.sidebar.radio(
    "Model", models, index=0, label_visibility='collapsed'
)
st.sidebar.markdown("_Models with * are not recommended_")

# Weighting
if not st.session_state.current_data.empty:
    days_diff = (pd.to_datetime(st.session_state.current_data['Race Session (date)']).max() - pd.to_datetime(st.session_state.current_data['Race Session (date)']).min()).days
else:
    days_diff = 0
# options = {
#     "Off": None,         # No weighting
#     "Some": days_diff / 4.0,
#     "Medium": days_diff / 2.0,
#     "Max": days_diff * 2.0,
# }
recency_options = {
    "Off": None,         # No weighting
    "Some": 21.0,
    "Medium": 56.0,
    "Max": 210.0,
}
if models[select_model] in (['glm', 'wls']):
    st.sidebar.divider()
    st.sidebar.subheader("Model Weights")
    weight_close = st.sidebar.slider("Close Pieces (not working)", min_value=0, max_value=100, value=50, step=1)
    recency_halflife = st.sidebar.radio("Recency Weighting", list(recency_options.keys()), horizontal=False, index=2)
    halflife = recency_options[recency_halflife]
    halflife_text = f"{halflife:.0f}" if halflife is not None else "Off"
    st.sidebar.write(f"**Halflife = {halflife_text}{' days' if halflife_text != 'Off' else ''}**")


## Parameters
st.sidebar.divider()
st.sidebar.subheader("Parameters")

max_correlation = st.sidebar.slider("Max Allowed Correlation", min_value = 0.5, max_value = 1.0, value = 0.75, step = 0.01)
max_uncertainty = st.sidebar.slider("Max Allowed Uncertainty", min_value = 5, max_value = 100, value = 20, step = 1)

# Checkbox options
include_equipment = st.sidebar.checkbox('Equipment')
include_coxswains = st.sidebar.checkbox('Coxswains')

## Over Time
st.sidebar.divider()
st.sidebar.subheader("Evalation Over Time")

lookback_days = st.sidebar.slider('Lookback Days', 1, 100, 50)
lookback_weighting = st.sidebar.segmented_control('Lookback Weighting', ['Uniform', 'Linear', 'Log', 'Exp'])



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
        sides_count = get_rower_sides_count(filtered_data)

        # Run regression
        results = run_regression(filtered_data, models[select_model], max_correlation, halflife)
        athletes_df = results['athletes']
        shell_classes_df = results['shell_classes']        

### Main UI
data_tab, performance_tab, corr_tab, validation_tab, debug_tab = st.tabs(["Data", "Performance", "Correlations", "Validation", "Debug"])

with data_tab:

    ## Dataset section
    st.subheader("Load Example Datasets")

    dataset_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    for file in dataset_files:
        if st.button(file):
            file_path = os.path.join(data_folder, file)
            st.session_state.current_data = pd.read_csv(file_path)
            st.rerun()

    st.divider()

    # File upload section
    # if show_uploader:
    st.subheader("Upload Racing Data")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Read the file
        dataframe = pd.read_csv(uploaded_file)
        st.session_state.current_data = dataframe

        # I hate this, this can't be right
        st.rerun()

    # if show_editor:
    # Display editable dataframe
    st.divider()
    st.subheader("Edit Racing Data")
    edited_dataframe = st.data_editor(st.session_state.current_data, num_rows= "dynamic")

    buttons_1, buttons_2, _ = st.columns([1, 1, 3])
    buttons_1.button("Save changes", on_click=save_changes)
    buttons_2.button("Clear data", on_click=clear_data)

with performance_tab:
    if results is not None:
        starboard_rowers = [rower for rower, sides in sides_count.items() if sides['Starboard'] > 0]
        port_rowers = [rower for rower, sides in sides_count.items() if sides['Port'] > 0]
        coxswains = [rower for rower, sides in sides_count.items() if sides['Coxswain'] > 0]
        scullers = [rower for rower, sides in sides_count.items() if sides['Scull'] > 0]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Starboard Coefficients")
            starboard_df = athletes_df.loc[athletes_df.index.isin(starboard_rowers)].copy()
            generate_side_chart(st, starboard_df, "Starboard Coefficients")
            st.divider()
            starboard_confidence = st.slider("Confidence", key="starboard_confidence", min_value=0, max_value=100, value=50, step=1, format="%d%%")
            starboard_bar_chart = generate_confidence_bars_with_gradient(starboard_df, starboard_confidence)
            st.altair_chart(starboard_bar_chart, use_container_width=True)        

        with col2:
            st.subheader("Port Coefficients")
            port_df = athletes_df.loc[athletes_df.index.isin(port_rowers)].copy()
            generate_side_chart(st, port_df, "Port Coefficients")
            st.divider()
            port_confidence = st.slider("Confidence", key="port_confidence", min_value=0, max_value=100, value=50, step=1, format="%d%%")
            port_bar_chart = generate_confidence_bars_with_gradient(port_df, port_confidence)
            st.altair_chart(port_bar_chart, use_container_width=True)        





        st.divider()

        col3, col4 = st.columns([1, 1])
        prob_matrix = compute_probability_matrix(starboard_df)
        col3.subheader("Starboard 1v1")
        col3.dataframe(prob_matrix)

        prob_matrix = compute_probability_matrix(port_df)
        col4.subheader("Port 1v1")
        col4.dataframe(prob_matrix)

        st.divider()

        col5, col6 = st.columns([1, 1])
        # col5.subheader("Boat Classes")
        # generate_side_chart(col5, shell_classes_df, "Boat Classes")

        col5.subheader("Coxswains")
        coxswains_df = athletes_df.loc[athletes_df.index.isin(coxswains)].copy()
        # generate_side_chart(col5, coxswains_df, "Coxswains")
    else:
        st.write("No data available.")

with corr_tab:
    st.subheader("Correlation Matrix")
    if results is not None:
        st.dataframe(results['corr'].round(2))
    else:
        st.write("No data available.")

with validation_tab:
    st.subheader("Lineup Testing")

    if results is not None:
        athletes_list = sorted(athletes_df.index.tolist())
        classes_list = sorted(shell_classes_df.index.tolist(), reverse=True)

        col_v1, col_v2 = st.columns([1, 1])
        with col_v1:
            boat_class_1 = st.selectbox("Test Boat Class #1", classes_list, index=0)
            boat_class_1 = st.selectbox("Rigging", classes_list, index=0)
            left, mid, right = st.columns([2, 1, 2])
            with mid:
                st.write("Position")            
            with left:
                lineup_1 = st.multiselect("Left", athletes_list)
            with right:
                lineup_1_r = st.multiselect("Right", athletes_list)
        
        with col_v2:
            boat_class_2 = st.selectbox("Test Boat Class #2", classes_list, index=0)
            lineup_2 = st.multiselect("Test Lineup #2", athletes_list)

        if lineup_1 and lineup_2:
            lineup_1_shell_class = determine_shell_class_from_list(lineup_1)
            lineup_2_shell_class = determine_shell_class_from_list(lineup_2)
            
            
            if lineup_1_shell_class and lineup_2_shell_class and lineup_1_shell_class in classes_list and lineup_2_shell_class in classes_list:
                lineup_1_coefficient = sum(athletes_df.loc[lineup_1, "Coefficient"]) + shell_classes_df.loc[lineup_1_shell_class, "Coefficient"]
                lineup_2_coefficient = sum(athletes_df.loc[lineup_2, "Coefficient"]) + shell_classes_df.loc[lineup_2_shell_class, "Coefficient"]
                st.write(f"Lineup 1: {lineup_1_coefficient:.2f}")
                st.write(f"Lineup 2: {lineup_2_coefficient:.2f}")
                st.write(f"Lineup 1 vs. Lineup 2: {lineup_1_coefficient - lineup_2_coefficient:.2f}")
            elif lineup_1 and lineup_2:
                if lineup_1_shell_class not in classes_list and lineup_2_shell_class not in classes_list:
                    if lineup_1_shell_class == lineup_2_shell_class:
                        st.write(f"No data available for shell class {lineup_1_shell_class}.")
                    else:    
                        st.write(f"No data available for shell classes {lineup_1_shell_class} and {lineup_2_shell_class}.")
                elif lineup_1_shell_class not in classes_list:
                    st.write(f"No shell class data available for Lineup 1: {lineup_1_shell_class}")
                elif lineup_2_shell_class not in classes_list:
                    st.write(f"No shell class data available for Lineup 2: {lineup_2_shell_class}")


        st.subheader("Actual vs. Model")
        st.dataframe(results['comparison'], hide_index=True) #height=300
    else:
        st.write("No data available.")
    

with debug_tab:
    if results is not None:
        # debug data    
        highly_correlated_groups = group_highly_correlated_parameters(results['corr'], threshold=max_correlation)

        for i, group in enumerate(highly_correlated_groups, 1):
            print(f"Group {i}: {group}")    
    else:
        st.write("No data available.")