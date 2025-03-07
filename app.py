from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
from helpers import *
from charts import *

st.set_page_config(
    layout="wide",
    page_title="SeatRacer",
    menu_items={
        'Get Help': 'mailto:tompeszek@gmail.com',
        'Report a bug': "mailto:tompeszek@gmail.com",
        'About': "# SeatRacer"
    }
)

# Edit Race Data section (only visible if editor toggle is on)
def save_changes():
    st.session_state.current_data = edited_dataframe

if 'current_data' not in st.session_state:
    # Define the initial dataframe with empty values
    # st.session_state.current_data = pd.DataFrame({
    #     'Race Session (date)': [''] * 5,
    #     'Piece': [0.0] * 5,
    #     'KM': [0.0] * 5,
    #     'Rigging': [''] * 5,
    #     'Personnel': [''] * 5,
    #     'Result': ['00:00.00'] * 5
    # })
    file_path = r'./data/2024_hocr_racing.csv'  # Adjust the file path if necessary
    st.session_state.current_data = pd.read_csv(file_path)

if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

if 'show_editor' not in st.session_state:
    st.session_state.show_editor = True

# Sidebar dataset selection
st.sidebar.subheader("Dataset")
# dataset_selected = st.sidebar.radio("Dataset", ('HOCR 2024', 'Olympics 2021'), label_visibility='collapsed')


# Always show both toggles in the sidebar
show_uploader = st.sidebar.toggle('Show file uploader', key="uploader", value=st.session_state.show_uploader)
show_editor = st.sidebar.toggle('Edit Race Data', key="editor", value=st.session_state.show_editor)

# File upload section (only visible if uploader toggle is on)
if show_uploader:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Read the file
        dataframe = pd.read_csv(uploaded_file)
        st.session_state.current_data = dataframe

        # Update session state to hide uploader and show editor
        st.session_state.show_uploader = False
        st.session_state.show_editor = True

        # I hate this, this can't be right
        st.rerun()



if show_editor:
    # Display editable dataframe
    st.subheader("Edit Racing Data")
    edited_dataframe = st.data_editor(st.session_state.current_data, num_rows= "dynamic")
    st.button("Save changes", on_click=save_changes)
    st.divider()

st.sidebar.divider()
st.sidebar.subheader("Include Parameters")

# Checkbox options
include_equipment = st.sidebar.checkbox('Equipment')
include_coxswains = st.sidebar.checkbox('Coxswains')

st.sidebar.divider()
st.sidebar.subheader("Evalation Over Time")

# Slider for range selection

lookback_days = st.sidebar.slider('Lookback Days', 1, 100, 50)
lookback_weighting = st.sidebar.segmented_control('Lookback Weighting', ['Uniform', 'Linear', 'Log', 'Exp'])




# Run the OLS regression
results = run_ols_regression(st.session_state.current_data)
athletes_df = results['athletes']



col1, col2 = st.columns([1, 1])
starboard_rowers = [rower for rower, sides in results['sides'].items() if sides['Starboard'] > 0]
port_rowers = [rower for rower, sides in results['sides'].items() if sides['Port'] > 0]
coxswains = [rower for rower, sides in results['sides'].items() if sides['Coxswain'] > 0]
scullers = [rower for rower, sides in results['sides'].items() if sides['Scull'] > 0]


col1.subheader("Starboard")
starboard_df = athletes_df.loc[athletes_df.index.isin(starboard_rowers)]
generate_side_chart(col1, starboard_df, "Starboard")




col2.subheader("Port")
port_df = athletes_df.loc[athletes_df.index.isin(port_rowers)]
generate_side_chart(col2, port_df, "Port")


st.divider()

col3, col4 = st.columns([1, 1])
prob_matrix = compute_probability_matrix(starboard_df)
col3.subheader("Starboard Matrix")
col3.dataframe(prob_matrix)

prob_matrix = compute_probability_matrix(port_df)
col4.subheader("Port Matrix")
col4.dataframe(prob_matrix)

# col3.subheader("Shells")
# starboard_df = athletes_df.loc[athletes_df.index.isin(starboard_rowers)]
# generate_side_chart(col3, starboard_df, "Starboard")

# col4.subheader("Coxswains")
# starboard_df = athletes_df.loc[athletes_df.index.isin(starboard_rowers)]
# generate_side_chart(col4, starboard_df, "Starboard")

# st.divider()

# col5, col6 = st.columns([1, 1])

# col5.subheader("Factors")
# col5.dataframe(results['factors'])

st.subheader("Actual vs. Model")
st.dataframe(results['comparison'], hide_index=True) #height=300