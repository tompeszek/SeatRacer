import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def render():
    """
    Individual athlete analysis module that shows how each race influences
    an athlete's coefficient using a leave-one-piece-out regression approach.
    """
    analysis = st.session_state.analysis
    if analysis is None:
        st.write("No data available.")
        return
    
    # Initialize session state variables if they don't exist
    if 'leave_one_out_results' not in st.session_state:
        st.session_state.leave_one_out_results = None
    if 'leave_one_out_complete' not in st.session_state:
        st.session_state.leave_one_out_complete = False
    if 'leave_one_out_per_day_complete' not in st.session_state:
        st.session_state.leave_one_out_per_day_complete = False
    if 'leave_one_out_per_day_results' not in st.session_state:
        st.session_state.leave_one_out_per_day_results = None
    if 'leave_one_out_per_week_complete' not in st.session_state:
        st.session_state.leave_one_out_per_week_complete = False
    if 'leave_one_out_per_week_results' not in st.session_state:
        st.session_state.leave_one_out_per_week_results = None
    if 'selected_analysis_level' not in st.session_state:
        st.session_state.selected_analysis_level = "per_race"
    
    # Two-step process: First run analysis, then select athlete to visualize
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        run_analysis = st.button("Run Per-Race Analysis", key="run_leave_one_out")
    
    with col2:
        run_day_analysis = st.button("Run Per-Day Analysis", key="run_leave_one_out_per_day")
    
    with col3:
        run_week_analysis = st.button("Run Per-Week Analysis", key="run_leave_one_out_per_week")
    
    # Get all unique athletes
    all_athletes = set()
    if 'athletes' in analysis.final_results:
        all_athletes.update(analysis.final_results['athletes'].index)
    if 'dropped_athletes' in analysis.final_results and analysis.final_results['dropped_athletes'] is not None:
        all_athletes.update(analysis.final_results['dropped_athletes'].index)
    
    # Radio buttons for selecting analysis level
    with col4:
        analysis_levels = []
        if st.session_state.leave_one_out_complete:
            analysis_levels.append("per_race")
        if st.session_state.leave_one_out_per_day_complete:
            analysis_levels.append("per_day")
        if st.session_state.leave_one_out_per_week_complete:
            analysis_levels.append("per_week")
        
        if analysis_levels:
            analysis_level_labels = {
                "per_race": "Per Race",
                "per_day": "Per Day",
                "per_week": "Per Week"
            }
            selected_level = st.radio(
                "Analysis Level:",
                analysis_levels,
                format_func=lambda x: analysis_level_labels.get(x, x),
                key="analysis_level_radio"
            )
            st.session_state.selected_analysis_level = selected_level
        else:
            st.radio(
                "Analysis Level (run an analysis first):",
                ["Run analysis first"],
                disabled=True,
                key="analysis_level_disabled"
            )
    
    # Athlete selector row
    athlete_col = st.columns([1])[0]
    with athlete_col:
        # Enable the athlete selector if any analysis has been run
        any_analysis_complete = (
            st.session_state.leave_one_out_complete or 
            st.session_state.leave_one_out_per_day_complete or 
            st.session_state.leave_one_out_per_week_complete
        )
        
        if any_analysis_complete:
            athlete_for_analysis = st.selectbox(
                "Select athlete:",
                sorted(list(all_athletes)),
                key="athlete_influence_analysis"
            )
        else:
            st.selectbox(
                "Select athlete (run analysis first):",
                ["Run analysis first"],
                disabled=True,
                key="athlete_influence_disabled"
            )
    
    # First step: Run the leave-one-out analysis for all races and athletes
    if run_analysis:
        # Run per-race analysis
        st.session_state.selected_analysis_level = "per_race"
        run_leave_one_out_analysis(analysis, all_athletes, "per_race")
    
    # Run per-day analysis
    if run_day_analysis:
        # Run per-day analysis
        st.session_state.selected_analysis_level = "per_day"
        run_leave_one_out_analysis(analysis, all_athletes, "per_day")
    
    # Run per-week analysis  
    if run_week_analysis:
        # Run per-week analysis  
        st.session_state.selected_analysis_level = "per_week"
        run_leave_one_out_analysis(analysis, all_athletes, "per_week")
    
    # Second step: Visualize the pre-computed data for the selected athlete
    if (st.session_state.leave_one_out_complete or 
        st.session_state.leave_one_out_per_day_complete or 
        st.session_state.leave_one_out_per_week_complete) and 'athlete_influence_analysis' in st.session_state:
        
        # Determine which results to use based on selected level
        selected_level = st.session_state.selected_analysis_level
        
        results_key = {
            "per_race": "leave_one_out_results",
            "per_day": "leave_one_out_per_day_results",
            "per_week": "leave_one_out_per_week_results"
        }.get(selected_level)
        
        if results_key not in st.session_state:
            st.warning(f"No results available for {selected_level} analysis.")
            return
        
        athlete_for_analysis = st.session_state.athlete_influence_analysis
        athlete_data = st.session_state[results_key]['athlete_data'].get(athlete_for_analysis, {})
        
        if not athlete_data or athlete_data['status'] != 'active':
            return
        
        # Get the core data
        position_info = athlete_data['position_info']
        influence_df = pd.DataFrame(athlete_data['piece_influences'])
        
        # Add a descriptive title based on the analysis level
        title_prefix = {
            "per_race": "Race",
            "per_day": "Daily",
            "per_week": "Weekly"
        }.get(selected_level, "Race")
        
        # Process the data for visualization
        influence_df['sort_key'] = influence_df.apply(create_sort_key, axis=1)
        influence_df = influence_df.sort_values('sort_key')
        
        # Create categorical ordering for pieces - ensure unique categories
        ordered_pieces = influence_df['Piece'].unique().tolist()
        influence_df['Piece_Ordered'] = pd.Categorical(influence_df['Piece'], categories=ordered_pieces, ordered=True)
        
        # Fill NaN values
        influence_df['Dropped in Analysis'] = influence_df['Dropped in Analysis'].fillna(False)
        influence_df['Coefficient Change'] = influence_df['Coefficient Change'].fillna(0)
        influence_df['Position Speed'] = influence_df['Position Speed'].fillna(0)
        influence_df['Position Rank'] = influence_df['Position Rank'].fillna(0)
        
        # Calculate Speed Change (compared to current speed)
        current_position_info = position_info
        current_speed = current_position_info['speed']
        current_rank = current_position_info['rank']
        current_coefficient = current_position_info['coefficient']
        
        # Calculate Speed Change - positive value means speed got worse when race was removed
        # (which means the race improved the athlete's speed in the full dataset)
        influence_df['Speed Change'] = influence_df['Position Speed'] - current_speed
        
        # Calculate Rank Change - positive value means rank got worse when race was removed
        # (which means removing the race makes the athlete's rank worse, so the race helped them)
        influence_df['Rank Change'] = influence_df['Position Rank'] - current_rank
        
        # Create compact header with athlete info
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Coefficient", f"{position_info['coefficient']:.2f}s")
        with metrics_col2:
            st.metric("Speed", f"+{position_info['speed']:.2f}s")
        with metrics_col3:
            st.metric("Rank", f"{position_info['rank']}/{position_info['total_in_position']}")
        
        # Set up the layered visualization
        # Create a base chart for the main axis
        base = alt.Chart(influence_df).encode(
            x=alt.X('Piece_Ordered:N', title=None, axis=alt.Axis(labelAngle=-45))
        )
        
        # Color encoding based on participation and performance
        influence_df['Speed Color'] = 'neutral'
        
        # CHANGE: We're NOT inverting the speed change anymore
        # Positive speed change = removing race makes athlete's gap to best worse
        # This means the race had a positive effect on the athlete (good performance)
        
        # Athlete participated and removing race worsens gap to best (positive change is GOOD)
        influence_df.loc[(influence_df['Athlete Participated'] == True) & 
                        (influence_df['Speed Change'] > 0), 'Speed Color'] = 'good'
        # Athlete participated and removing race improves gap to best (negative change is BAD)
        influence_df.loc[(influence_df['Athlete Participated'] == True) & 
                        (influence_df['Speed Change'] < 0), 'Speed Color'] = 'bad'
        # Athlete didn't participate but removing race worsens gap
        influence_df.loc[(influence_df['Athlete Participated'] == False) & 
                        (influence_df['Speed Change'] > 0), 'Speed Color'] = 'good_indirect'
        # Athlete didn't participate but removing race improves gap
        influence_df.loc[(influence_df['Athlete Participated'] == False) & 
                        (influence_df['Speed Change'] < 0), 'Speed Color'] = 'bad_indirect'
        
        # Create color scale for speed bars
        speed_color_scale = alt.Scale(
            domain=['good', 'bad', 'good_indirect', 'bad_indirect', 'neutral'],
            range=['#4CAF50', '#F44336', '#A5D6A7', '#FFCCCB', '#CCCCCC']
        )
        
        # Calculate max absolute values for each metric for symmetric scales
        max_speed_abs = max(abs(influence_df['Speed Change'].max()), abs(influence_df['Speed Change'].min()))
        max_rank_abs = max(abs(influence_df['Rank Change'].max()), abs(influence_df['Rank Change'].min()))
        
        # Determine the appropriate bar size based on number of items
        # Using Altair's width scaling to make responsive bars
        n_items = len(ordered_pieces)
        
        # 1. Speed Change bars - with responsive sizing
        speed_bars = base.mark_bar().encode(
            y=alt.Y('Speed Change:Q', 
                   title='Speed Change (s)', 
                   axis=alt.Axis(grid=True),
                   # Set symmetric domain around zero
                   scale=alt.Scale(domain=[-max_speed_abs * 1.1, max_speed_abs * 1.1])),
            color=alt.Color('Speed Color:N', scale=speed_color_scale, legend=None),
            tooltip=[
                alt.Tooltip('Piece:N', title='Piece'),
                alt.Tooltip('Position Speed:Q', title='Speed', format='+.2f'),
                alt.Tooltip('Speed Change:Q', title='Speed Change', format='+.2f'),
                alt.Tooltip('Position Rank:Q', title='Rank'),
                alt.Tooltip('Rank Change:Q', title='Rank Change', format='+d'),
                alt.Tooltip('New Coefficient:Q', title='Coefficient', format='.2f'),
                alt.Tooltip('Coefficient Change:Q', title='Coefficient Change', format='+.2f')
            ]
        )
        
        # 2. Rank Change - using points instead of bars
        rank_points = base.mark_point(size=80, filled=True, color='#FFD700').encode(
            y=alt.Y('Rank Change:Q', 
                  title='Rank Change',
                  axis=alt.Axis(
                      grid=False,
                      format='d'  # Format as integer (no decimal places)
                  ),
                  # Set symmetric domain around zero
                  scale=alt.Scale(domain=[-max_rank_abs * 1.1, max_rank_abs * 1.1])),
            tooltip=[
                alt.Tooltip('Piece:N', title='Piece'),
                alt.Tooltip('Position Rank:Q', title='Rank'),
                alt.Tooltip('Rank Change:Q', title='Rank Change', format='+d')  # Format as integer with sign
            ]
        )
        
        # Layer the charts - without zero reference lines
        chart = alt.layer(
            speed_bars,
            rank_points
        ).resolve_scale(
            y='independent'  # Independent y-scales, but visually aligned at zero
        ).properties(
            title=f'{title_prefix} Impact Analysis (higher is better for both measures)',
            height=400
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        
        # Show the chart
        st.altair_chart(chart, use_container_width=True)
        
        
        # Better legend with color samples and bullet points
        st.markdown("""
        **Legend:**
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #4CAF50; margin-right: 10px;"></div>
            <div>Green bars: Speed improved due to race (race performance was good)</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #F44336; margin-right: 10px;"></div>
            <div>Red bars: Speed worsened due to race (race performance was poor)</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #FFD700; margin-right: 10px;"></div>
            <div>Yellow points: Rank change (higher points indicate better rank performance)</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #A5D6A7; margin-right: 10px;"></div>
            <div>Light Green: Indirect positive effects (races without the athlete)</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: #FFCCCB; margin-right: 10px;"></div>
            <div>Light Red: Indirect negative effects (races without the athlete)</div>
        </div>
        <div style="margin-top: 10px;">
            <strong>Interpretation:</strong> Bars and points positioned higher on the chart show races 
            where the athlete performed well. Removing these races would worsen the athlete's overall assessment.
        </div>
        """, unsafe_allow_html=True)
        
        # Show data table
        with st.expander("Show Data Table"):
            # Show data table
            display_df = influence_df.copy()
            
            # Format numeric columns
            display_df['New Coefficient'] = display_df['New Coefficient'].apply(
                lambda x: f"{x:.2f}" if x is not None else "N/A"
            )
            display_df['Coefficient Change'] = display_df['Coefficient Change'].apply(
                lambda x: f"{x:.2f}" if x is not None else "N/A"
            )
            display_df['Position Speed'] = display_df['Position Speed'].apply(
                lambda x: f"+{x:.2f}" if x is not None and x > 0 else (f"{x:.2f}" if x is not None else "N/A")
            )
            display_df['Speed Change'] = display_df['Speed Change'].apply(
                lambda x: f"+{x:.2f}" if x is not None and x > 0 else (f"{x:.2f}" if x is not None else "N/A")
            )
            
            st.dataframe(
                display_df[[
                    'Piece', 'Crew', 'New Coefficient', 'Position Speed', 
                    'Speed Change', 'Position Rank', 'Rank Change', 'Athlete Participated'
                ]],
                hide_index=True
            )

def run_leave_one_out_analysis(analysis, all_athletes, level="per_race"):
    """
    Run a leave-one-out analysis at the specified level (per_race, per_day, or per_week).
    
    Parameters:
    -----------
    analysis : Analysis object
        The main analysis object
    all_athletes : set
        Set of all athletes to analyze
    level : str
        Level of analysis - "per_race", "per_day", or "per_week"
    """
    with st.spinner(f"Processing {level} analysis..."):
        # Get the comparison dataframe and original dataframe
        comparison_df = analysis.final_results['comparison']
        orig_df = analysis.df.copy()
        
        # Group pieces based on the specified level
        if level == "per_race":
            # Each piece is its own group
            all_pieces = comparison_df['Piece'].unique().tolist()
            piece_groups = {piece: [piece] for piece in all_pieces}
            group_labels = {piece: piece for piece in all_pieces}
        elif level == "per_day":
            # Group by race date
            orig_df['Date'] = pd.to_datetime(orig_df['Race Session (date)'])
            orig_df['Date_Str'] = orig_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Create groups by date
            grouped_pieces = {}
            for _, row in orig_df.iterrows():
                date_str = row['Date_Str']
                piece = row['Piece']
                if date_str not in grouped_pieces:
                    grouped_pieces[date_str] = []
                if piece not in grouped_pieces[date_str]:
                    grouped_pieces[date_str].append(piece)
            
            # Now create piece groups and labels
            piece_groups = {}
            group_labels = {}
            for date_str, pieces in grouped_pieces.items():
                group_key = f"Day: {date_str}"
                for piece in pieces:
                    piece_groups[piece] = pieces
                    group_labels[piece] = group_key
            
            all_pieces = list(set(piece_groups.keys()))
        elif level == "per_week":
            # Group by race week
            orig_df['Date'] = pd.to_datetime(orig_df['Race Session (date)'])
            orig_df['Week'] = orig_df['Date'].dt.to_period('W').astype(str)
            
            # Create groups by week
            grouped_pieces = {}
            for _, row in orig_df.iterrows():
                week = row['Week']
                piece = row['Piece']
                if week not in grouped_pieces:
                    grouped_pieces[week] = []
                if piece not in grouped_pieces[week]:
                    grouped_pieces[week].append(piece)
            
            # Now create piece groups and labels
            piece_groups = {}
            group_labels = {}
            for week, pieces in grouped_pieces.items():
                group_key = f"Week: {week}"
                for piece in pieces:
                    piece_groups[piece] = pieces
                    group_labels[piece] = group_key
            
            all_pieces = list(set(piece_groups.keys()))
        
        # Get all unique athletes
        all_athletes = set()
        if 'athletes' in analysis.final_results:
            all_athletes.update(analysis.final_results['athletes'].index)
        if 'dropped_athletes' in analysis.final_results and analysis.final_results['dropped_athletes'] is not None:
            all_athletes.update(analysis.final_results['dropped_athletes'].index)
        
        # Setup progress bar
        progress_bar = st.progress(0)
        total_pieces = len(all_pieces)
        
        # Create a mapping of athletes to the pieces they participated in
        athlete_pieces = {}
        for athlete in all_athletes:
            athlete_pieces[athlete] = []
            for idx, row in comparison_df.iterrows():
                if athlete in row['Crew']:
                    athlete_pieces[athlete].append(row['Piece'])
        
        # Initialize storage for results
        piece_results = {}
        
        # For each piece/group, run analysis without that piece/group
        for i, piece_to_omit in enumerate(all_pieces):
            # Update progress bar
            progress_bar.progress(i / total_pieces)
            
            # Get all pieces in this group
            pieces_to_omit = piece_groups[piece_to_omit]
            
            # Create filtered dataframe
            filtered_df = orig_df[~orig_df['Piece'].isin(pieces_to_omit)].copy()
            
            # Create temporary analysis with filtered data
            temp_analysis = analysis.__class__(
                df=filtered_df,
                max_correlation=1.0,
                halflife=analysis.halflife,
                weight_close=analysis.weight_close,
                weight_stern=analysis.weight_stern,
                include_coxswains=analysis.include_coxswains,
                seat_breakdown=analysis.seat_breakdown,
                lookback=analysis.lookback,
                erg_scores=analysis.erg_scores,
                shell_class=analysis.shell_class
            )
            
            # Run the analysis
            try:
                temp_analysis.run_analysis()
            except Exception as e:
                # If there's an error (like divide by zero), try with a small regularization
                # This means we need to adjust the model slightly to avoid numerical issues
                st.warning(f"Encountered an error with {piece_to_omit}. Adding regularization.")
                # Create a new temporary analysis with a slightly adjusted configuration
                temp_analysis = analysis.__class__(
                    df=filtered_df,
                    max_correlation=0.99,  # Slight adjustment to prevent perfect correlation
                    halflife=analysis.halflife,
                    weight_close=analysis.weight_close,
                    weight_stern=analysis.weight_stern,
                    include_coxswains=analysis.include_coxswains,
                    seat_breakdown=analysis.seat_breakdown,
                    lookback=analysis.lookback,
                    erg_scores=analysis.erg_scores,
                    shell_class=analysis.shell_class
                )
                temp_analysis.run_analysis()
            
            # Store metadata about the piece or group
            if level == "per_race":
                # For single piece, store direct metadata
                race_date = orig_df[orig_df['Piece'] == piece_to_omit]['Race Session (date)'].iloc[0] if not orig_df[orig_df['Piece'] == piece_to_omit].empty else None
                piece_number = orig_df[orig_df['Piece'] == piece_to_omit]['PieceNumber'].iloc[0] if not orig_df[orig_df['Piece'] == piece_to_omit].empty else None
            else:
                # For groups, use the group label
                race_date = group_labels[piece_to_omit]
                piece_number = None
            
            # Store the results keyed by piece
            piece_results[piece_to_omit] = {
                'piece': piece_to_omit,
                'group_label': group_labels[piece_to_omit],
                'race_date': race_date,
                'piece_number': piece_number,
                'results': temp_analysis.final_results,
                'analysis': temp_analysis
            }
            
            # Find crew information for athletes in this piece
            for piece in pieces_to_omit:
                for idx, row in comparison_df[comparison_df['Piece'] == piece].iterrows():
                    crew = row['Crew']
                    actual_pace = row['Actual Pace']
                    model_pace = row['Model Pace']
                    delta = row['Delta']
                    
                    # Split crew into individual athletes
                    crew_athletes = crew.split('/')
                    for athlete in crew_athletes:
                        if athlete not in piece_results[piece_to_omit]:
                            piece_results[piece_to_omit][athlete] = {
                                'athlete_in_piece': True,
                                'crew': crew,
                                'actual_pace': actual_pace,
                                'model_pace': model_pace,
                                'delta': delta
                            }
        
        # Complete progress bar
        progress_bar.progress(1.0)
        
        # Pre-process data for all athletes
        athlete_data = {}
        for athlete in all_athletes:
            athlete_data[athlete] = {}
            
            # Determine athlete status in the main analysis
            in_main_results = athlete in analysis.final_results['athletes'].index if 'athletes' in analysis.final_results else False
            in_dropped_results = (
                analysis.final_results['dropped_athletes'] is not None and 
                athlete in analysis.final_results['dropped_athletes'].index
            ) if 'dropped_athletes' in analysis.final_results else False
            
            athlete_data[athlete]['status'] = 'active' if in_main_results else ('dropped' if in_dropped_results else 'unknown')
            
            # Get current coefficient and position info (for active athletes)
            if in_main_results:
                current_coefficient = float(analysis.final_results['athletes'].loc[athlete]['Coefficient'])
                position_info = analysis.get_athlete_position_info(athlete)
                
                athlete_data[athlete]['current_coefficient'] = current_coefficient
                athlete_data[athlete]['position_info'] = position_info
                
                # Get influence data for each piece
                piece_influences = []
                for piece, piece_data in piece_results.items():
                    # Default values
                    influence = None
                    speed = None
                    position_rank = None
                    dropped_status = None
                    new_coefficient = None
                    athlete_in_piece = any(p in athlete_pieces.get(athlete, []) for p in piece_groups[piece])
                    
                    # Get values if in active results
                    if 'results' in piece_data and 'athletes' in piece_data['results']:
                        if athlete in piece_data['results']['athletes'].index:
                            # Athlete is in this result
                            new_coefficient = float(piece_data['results']['athletes'].loc[athlete]['Coefficient'])
                            dropped_status = False
                            
                            # Calculate influence
                            influence = current_coefficient - new_coefficient
                            
                            # Get position metrics
                            position_suffix = athlete[-1]
                            position_metrics = piece_data['analysis'].calculate_position_metrics_for_coefficient(
                                new_coefficient, position_suffix
                            )
                            speed = position_metrics['speed']
                            position_rank = position_metrics['rank']
                        elif (piece_data['results']['dropped_athletes'] is not None and 
                              athlete in piece_data['results']['dropped_athletes'].index):
                            # Athlete is dropped in this result
                            new_coefficient = None
                            dropped_status = True
                        else:
                            # Athlete not found in results
                            new_coefficient = None
                            dropped_status = None
                    
                    # Get crew info for this athlete
                    crew_info = piece_data.get(athlete, {})
                    
                    # Create piece influence record
                    piece_label = piece_data['group_label']
                    
                    influence_record = {
                        'Piece': piece_label,
                        'Race Date': piece_data['race_date'],
                        'Piece Number': piece_data['piece_number'],
                        'Crew': crew_info.get('crew', "Unknown" if athlete_in_piece else "Athlete not in race"),
                        'Actual Pace': crew_info.get('actual_pace'),
                        'Model Pace': crew_info.get('model_pace'),
                        'Delta': crew_info.get('delta'),
                        'New Coefficient': new_coefficient,
                        'Coefficient Change': influence,
                        'Dropped in Analysis': dropped_status,
                        'Athlete Participated': athlete_in_piece,
                        'Position Speed': speed,
                        'Position Rank': position_rank
                    }
                    piece_influences.append(influence_record)
                
                # Store piece influences
                athlete_data[athlete]['piece_influences'] = piece_influences
        
        # Store all results in session state
        results_key = {
            "per_race": "leave_one_out_results",
            "per_day": "leave_one_out_per_day_results",
            "per_week": "leave_one_out_per_week_results"
        }.get(level)
        
        st.session_state[results_key] = {
            'piece_results': piece_results,
            'athlete_data': athlete_data,
            'all_pieces': all_pieces
        }
        
        # Mark the appropriate analysis as complete
        complete_key = {
            "per_race": "leave_one_out_complete",
            "per_day": "leave_one_out_per_day_complete",
            "per_week": "leave_one_out_per_week_complete"
        }.get(level)
        
        st.session_state[complete_key] = True
        
        # Set the selected analysis level to the one we just completed
        st.session_state.selected_analysis_level = level

def create_sort_key(row):
    """Create a sort key for chronological ordering"""
    # Start with a default very early date if missing
    date_str = '0000-00-00'
    
    if pd.notnull(row['Race Date']):
        try:
            # Check if the Race Date is already a formatted string like "Day: YYYY-MM-DD" or "Week: YYYY-WNN"
            if isinstance(row['Race Date'], str) and (row['Race Date'].startswith('Day:') or row['Race Date'].startswith('Week:')):
                return row['Race Date']  # Use as-is for sorting
            
            # Convert to datetime if not already
            date_obj = row['Race Date']
            if not isinstance(date_obj, pd.Timestamp):
                date_obj = pd.to_datetime(date_obj)
            
            # Format as YYYY-MM-DD for reliable string sorting
            date_str = date_obj.strftime('%Y-%m-%d')
        except:
            # If any error, use string representation
            date_str = str(row['Race Date'])
    
    # Default piece number part
    piece_part = '00000'
    
    if pd.notnull(row['Piece Number']):
        try:
            # Zero-pad to 5 digits
            piece_part = str(int(row['Piece Number'])).zfill(5)
        except:
            piece_part = str(row['Piece Number'])
    
    # Combine with an underscore separator
    return f"{date_str}_{piece_part}"

def get_gap_to_next_best(coefficient, position_suffix, analysis):
    """
    For athletes who are already the best in their position (speed=0),
    calculate the gap to the next best athlete.
    
    Parameters:
    -----------
    coefficient : float
        The coefficient of the athlete
    position_suffix : str
        The position suffix character (ˢ, ᵖ, etc.)
    analysis : Analysis
        The analysis object containing results
        
    Returns:
    --------
    float
        The gap to the next best athlete (negative number)
    """
    # Get position-specific athletes
    position_df = analysis.get_position_athletes(position_suffix)
    
    if len(position_df) <= 1:
        return 0
    
    # Sort by coefficient
    position_df = position_df.sort_values('Coefficient')
    
    # If this athlete's coefficient is already the best
    if coefficient <= position_df['Coefficient'].min():
        # Calculate gap to second best
        second_best = position_df.iloc[1]['Coefficient'] if len(position_df) > 1 else coefficient
        return coefficient - second_best  # This will be negative (better)
    
    return