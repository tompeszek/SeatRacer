import streamlit as st
import pandas as pd
import os

from utils.data_handler import DataHandler

# TODO: remove use of os.path.join

def render():
    ## Dataset section
    results_data_handler = DataHandler('data')
    
    st.subheader("Load Example Datasets")
    dataset_files = results_data_handler.get_available_datasets()

    if dataset_files:
        for file in dataset_files:
            if st.button(file):
                st.session_state.current_data = results_data_handler.load_dataset(file)
                setattr(st.session_state, 'reset_athletes', True)
                setattr(st.session_state, 'rerun', True)
                st.rerun()
    else:
        st.info("No example datasets available. Please upload your own CSV file below.")

    st.divider()

    # File upload section
    st.subheader("Upload Racing Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your seat racing data as a CSV file"
    )
    if uploaded_file is not None:
        try:
            # Read the file
            dataframe = pd.read_csv(uploaded_file)
            st.session_state.current_data = dataframe
            setattr(st.session_state, 'reset_athletes', True)
            setattr(st.session_state, 'rerun', True)
            st.success(f"Successfully loaded {len(dataframe)} rows of data from {uploaded_file.name}")
            st.rerun()
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")

    st.divider()
    st.subheader("Edit Racing Data")
    edited_dataframe = st.data_editor(st.session_state.current_data, num_rows="dynamic")

    # Define callback function for pill selection
    def handle_data_action():
        action = st.session_state.data_action_pills
        if action == "Save Changes":
            st.session_state.current_data = edited_dataframe
        elif action == "Clear Data":
            st.session_state.current_data = pd.DataFrame()
        
        # Reset the selection
        st.session_state.data_action_pills = None

    # Use pills with key for session state
    st.pills(
        "Data Action Pills",
        options=["Save Changes", "Clear Data"],
        key="data_action_pills",
        on_change=handle_data_action,
        label_visibility="collapsed",
    )