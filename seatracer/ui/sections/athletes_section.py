import streamlit as st
import pandas as pd
import streamlit_antd_components as sac


from utils.data_handler import DataHandler

def render():
    erg_data_handler = DataHandler('erg_data')
    data_folder = './erg__data'

    # Dataset section
    st.subheader("Load Example Erg Data")

    # Display available datasets as buttons
    for file in erg_data_handler.get_available_datasets():
        if st.button(file):
            df = erg_data_handler.load_dataset(file)
            st.session_state.athlete_ergs_df = df.set_index('Athlete') if 'Athlete' in df.columns else df
            st.session_state.original_athlete_ergs_df = st.session_state.athlete_ergs_df.copy()
            st.session_state.rerun = True
            st.rerun()
    
    st.divider()
    
    # Upload section
    st.subheader("Upload Erg Data")
    uploaded_file = st.file_uploader("Choose a file with athlete data")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.athlete_ergs_df = df.set_index('Athlete') if 'Athlete' in df.columns else df
        st.session_state.original_athlete_ergs_df = st.session_state.athlete_ergs_df.copy()
        st.session_state.rerun = True
        st.rerun()
    
    # Get unique athletes from current data
    if getattr(st.session_state, 'reset_athletes', False):
        athletes_list = st.session_state.current_data['Personnel'].str.split('/', expand=True).stack().unique()
        st.session_state.current_athletes_list = athletes_list
        
        # Initialize or update athlete dataframe
        if 'athlete_ergs_df' not in st.session_state or getattr(st.session_state.get('athlete_ergs_df'), 'empty', True):
            # Create new dataframe if it doesn't exist
            st.session_state.athlete_ergs_df = pd.DataFrame({
                'Athlete': athletes_list,
                '2k Erg': ['7:00.0'] * len(athletes_list),
            }).sort_values('Athlete').set_index('Athlete')
        else:
            # Add any new athletes to existing dataframe
            current_athletes = st.session_state.athlete_ergs_df.index.tolist()
            new_athletes = [a for a in athletes_list if a not in current_athletes]
            
            if new_athletes:
                new_df = pd.DataFrame({
                    'Athlete': new_athletes,
                    '2k Erg': ['7:00.0'] * len(new_athletes),
                }).sort_values('Athlete').set_index('Athlete')
                st.session_state.athlete_ergs_df = pd.concat([st.session_state.athlete_ergs_df, new_df])
        
        st.session_state.original_athlete_ergs_df = st.session_state.athlete_ergs_df.copy()
        st.session_state.reset_athletes = False
    
    # Display editable dataframe with save/discard buttons
    if 'athlete_ergs_df' in st.session_state:
        # Store a copy of the original dataframe for the discard functionality
        if 'original_athlete_ergs_df' not in st.session_state:
            st.session_state.original_athlete_ergs_df = st.session_state.athlete_ergs_df.copy()
        
        # Filter to show only athletes in the current athletes_list
        display_df = st.session_state.athlete_ergs_df.copy()
        if hasattr(st.session_state, 'current_athletes_list'):
            display_df = display_df.loc[display_df.index.isin(st.session_state.current_athletes_list)]
        
        # Create the data editor
        num_rows = len(display_df)
        calculated_height = (num_rows + 1) * 35 + 3
        
        # Get the edited dataframe
        edited_df = st.data_editor(
            display_df,
            key="Erg_Editor",
            width=400,
            use_container_width=False,
            column_config={
                "Athlete": st.column_config.TextColumn("Athlete"),
                "2k Erg": st.column_config.TextColumn(
                    "2k Erg Time",
                    help="Format: m:ss.s",
                )
            },
            height=calculated_height,            
        )

        # Create buttons for save/discard in columns for better layout
        # col1, col2, col3 = st.columns([1,1,5])
        
    # with col1:
        if st.button("Save Changes", key="save_btn"):
            # Store the edited values back into the main dataframe
            for athlete in edited_df.index:
                for col in edited_df.columns:
                    st.session_state.athlete_ergs_df.at[athlete, col] = edited_df.at[athlete, col]
            
            # Update the original copy too
            st.session_state.original_athlete_ergs_df = st.session_state.athlete_ergs_df.copy()
            st.session_state.rerun = True
            st.rerun()
            
    # with col2:
        if st.button("Discard Changes", key="discard_btn"):
            st.rerun()