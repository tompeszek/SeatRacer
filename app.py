from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
from helpers import *
from charts import *
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

# Edit Race Data section (only visible if editor toggle is on)
def save_changes():
    st.session_state.current_data = edited_dataframe

if 'current_data' not in st.session_state:
    file_path = r'./data/2012.csv'
    st.session_state.current_data = pd.read_csv(file_path)

if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

if 'show_editor' not in st.session_state:
    st.session_state.show_editor = True

# Sidebar dataset selection
data_folder = './data'
def load_data_on_change():    
    file_path = os.path.join(data_folder, dataset_selected)
    st.session_state.current_data = pd.read_csv(file_path)


### Sidebar
## Dataset section
st.sidebar.subheader("Dataset")
dataset_selected = st.sidebar.radio(
    "Dataset",
    [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))],
    label_visibility='collapsed',
    on_change=load_data_on_change
)

# Toggles
show_uploader = st.sidebar.toggle('Show file uploader', key="uploader", value=st.session_state.show_uploader)
show_editor = st.sidebar.toggle('Edit Race Data', key="editor", value=st.session_state.show_editor)

## Data Filters
st.sidebar.divider()
st.sidebar.subheader("Data Filters")
shell_class = st.sidebar.segmented_control(
    'Include Shell Classes', 
    options=['2-', '4-', '4+', '8+'],
    selection_mode='multi',
    default=['4-', '4+', '8+']
)

remove_mixed = st.sidebar.radio("Remove Mixed Class Results (not working)", ["Yes", "No"], index=1)

weight_closeness = st.sidebar.slider("Weight Closeness", min_value=0, max_value=100, value=50, step=1)

## Parameters
st.sidebar.divider()
st.sidebar.subheader("Include Parameters")

# Checkbox options
include_equipment = st.sidebar.checkbox('Equipment')
include_coxswains = st.sidebar.checkbox('Coxswains')

# Models
st.sidebar.divider()
st.sidebar.subheader("Models")

# Grouping the models
models = {
    "Generalized Linear Model": "glm",
    "Weighted Least Squares": "wls",
    "Robust Linear Model*": "rlm",
    "Ordinary Least Squares*": "ols"
}

select_model = st.sidebar.radio(
    "Model", models, index=0, label_visibility='collapsed'
)
st.sidebar.markdown("_Models with * are not recommended_")


## Over Time
st.sidebar.divider()
st.sidebar.subheader("Evalation Over Time")

lookback_days = st.sidebar.slider('Lookback Days', 1, 100, 50)
lookback_weighting = st.sidebar.segmented_control('Lookback Weighting', ['Uniform', 'Linear', 'Log', 'Exp'])


### Main UI
# File upload section (only visible if uploader toggle is on)
if show_uploader:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Read the file
        dataframe = pd.read_csv(uploaded_file)
        st.session_state.current_data = dataframe

        # Update session state to hide uploader and show editor
        st.session_state.show_uploader = False
        # st.session_state.show_editor = True

        # I hate this, this can't be right
        st.rerun()

if show_editor:
    # Display editable dataframe
    st.subheader("Edit Racing Data")
    edited_dataframe = st.data_editor(st.session_state.current_data, num_rows= "dynamic")
    st.button("Save changes", on_click=save_changes)
    st.divider()



### App Code
# Copy data and add fields (athlete counts, shell class)
df = st.session_state.current_data.copy()
add_athlete_counts(df)
df['shell_class'] = df.apply(determine_shell_class, axis=1)

# Apply shell class filter
filtered_data = df[df['shell_class'].isin(shell_class)]

# Add sides to names  (also adds coxswain to personnel if needed)
filtered_data = append_rigging_to_names(filtered_data)

# Add piece names
filtered_data['Piece'] = filtered_data['Race Session (date)'].astype(str) + " #" + filtered_data['Piece'].astype(str)

# Remove mixed if needed
if remove_mixed == 'Yes':
    # filtered_data = filtered_data[filtered_data['shell_class'] != 'Mixed']    
    print(df[['shell_class', 'Piece']].head(30))

# Add piece weights


# Sides count
sides_count = get_rower_sides_count(filtered_data)

# Run regression
results = run_regression(filtered_data, models[select_model])
athletes_df = results['athletes']
shell_classes_df = results['shell_classes']


# debug data
st.dataframe(results['fitted'])


# real presentation
col1, col2 = st.columns([1, 1])
starboard_rowers = [rower for rower, sides in sides_count.items() if sides['Starboard'] > 0]
port_rowers = [rower for rower, sides in sides_count.items() if sides['Port'] > 0]
coxswains = [rower for rower, sides in sides_count.items() if sides['Coxswain'] > 0]
scullers = [rower for rower, sides in sides_count.items() if sides['Scull'] > 0]


col1.subheader("Starboard")
starboard_df = athletes_df.loc[athletes_df.index.isin(starboard_rowers)].copy()
generate_side_chart(col1, starboard_df, "Starboard")

col2.subheader("Port")
port_df = athletes_df.loc[athletes_df.index.isin(port_rowers)].copy()
generate_side_chart(col2, port_df, "Port")


st.divider()

col3, col4 = st.columns([1, 1])
prob_matrix = compute_probability_matrix(starboard_df)
col3.subheader("Starboard Matrix")
col3.dataframe(prob_matrix)

prob_matrix = compute_probability_matrix(port_df)
col4.subheader("Port Matrix")
col4.dataframe(prob_matrix)

st.divider()

col5, col6 = st.columns([1, 1])
col5.subheader("Boat Classes")
generate_side_chart(col5, shell_classes_df, "Boat Classes")

col6.subheader("Coxswains")
coxswains_df = athletes_df.loc[athletes_df.index.isin(coxswains)]
generate_side_chart(col6, coxswains_df, "Coxswains")

# st.divider()

# col5, col6 = st.columns([1, 1])

# col5.subheader("Factors")
# col5.dataframe(results['factors'])

st.subheader("Actual vs. Model")
st.dataframe(results['comparison'], hide_index=True) #height=300