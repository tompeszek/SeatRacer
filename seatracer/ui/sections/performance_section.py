import streamlit as st

from seatracer.visualization.charts import compute_probability_matrix, generate_confidence_bars_with_gradient, generate_side_chart

def render(sides_count):

    analysis = st.session_state.analysis
    if analysis is None:
        st.write("No data available.")
        return

    athletes_df = analysis.final_results['athletes']
    dropped_athletes_df = analysis.final_results['dropped_athletes']

    starboard_rowers = [rower for rower, sides in sides_count.items() if sides['Starboard'] > 0]
    port_rowers = [rower for rower, sides in sides_count.items() if sides['Port'] > 0]
    coxswains = [rower for rower, sides in sides_count.items() if sides['Coxswain'] > 0]
    scullers = [rower for rower, sides in sides_count.items() if sides['Scull'] > 0]

    st.subheader("Speed Coefficients")
    coefficient_boat_classes = ["8+", "4x/-", "2x/-", "1x"]
    coefficient_race_distances = [500, 1000, 1500, 2000, 4000, 6000]
    selected_boat_class = st.radio("Boat Class:", coefficient_boat_classes, index=1, horizontal=True, key="boat_class")
    selected_race_distance = st.radio("Race Distance:", coefficient_race_distances, index = 3, horizontal=True, key="race_distance")
    st.caption(f"_Number of seconds, over {selected_race_distance}m in a {selected_boat_class}, slower than best rower on the same side_")

    # Process all dataframes in a consistent way
    athlete_groups = {
        'starboard': starboard_rowers,
        'port': port_rowers,
        'coxswain': coxswains,
        'sculler': scullers
    }

    # Create and process all dataframes
    dfs = {
        group: adjust_metrics(
            athletes_df.loc[athletes_df.index.isin(indices)].copy(),
            selected_boat_class, 
            selected_race_distance
        ) for group, indices in athlete_groups.items()
    }

    # Unpack the processed dataframes
    starboard_df = dfs['starboard']
    port_df = dfs['port']
    coxswain_df = dfs['coxswain']
    sculler_df = dfs['sculler']

    # Note: Auto-adjustment of max_correlation is now handled in app.py

    if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
        col1, col2, col3 = st.columns([1, 1, 1])
    else:
        col1, col2 = st.columns([1, 1])

    with col1:
        st.write("Starboard")
        # starboard_confidence = st.slider("Max Uncertainty", key="starboard_uncertainty", min_value = 5, max_value = 100, value = 10, step = 1)
        generate_side_chart(st, starboard_df)

    with col2:
        st.write("Port")
        # starboard_confidence = st.slider("Max Uncertainty", key="port_uncertainty", min_value = 5, max_value = 100, value = 10, step = 1)
        generate_side_chart(st, port_df)
        
    if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
        with col3:
            st.write("Coxswains")
            generate_side_chart(st, coxswain_df)
    
    if len(dropped_athletes_df) > 0:
        # need to fix speed
        starboard_dropped_df = dropped_athletes_df.loc[dropped_athletes_df.index.isin(starboard_rowers)].sort_index().copy()
        port_dropped_df = dropped_athletes_df.loc[dropped_athletes_df.index.isin(port_rowers)].sort_index().copy()
        coxswain_dropped_df = dropped_athletes_df.loc[dropped_athletes_df.index.isin(coxswains)].sort_index().copy()

        # Calculate the best coefficients for each category
        starboard_best = min(starboard_df['Coefficient']) if not starboard_df.empty else None
        port_best = min(port_df['Coefficient']) if not port_df.empty else None
        coxswain_best = min(coxswain_df['Coefficient']) if not coxswain_df.empty else None
        sculler_best = min(sculler_df['Coefficient']) if not sculler_df.empty else None

        def calculate_average_behind(df, starboard_best, port_best, coxswain_best, sculler_best):
            starboard_best = starboard_best if starboard_best is not None else 0
            port_best = port_best if port_best is not None else 0
            coxswain_best = coxswain_best if coxswain_best is not None else 0
            sculler_best = sculler_best if sculler_best is not None else 0

            # Count occurrences of 'ᵖ', 'ˢ', 'ˣ', and 'ᶜ' in the specified group column
            df['Port Count'] = df['Group Members'].str.count("ᵖ")
            df['Starboard Count'] = df['Group Members'].str.count("ˢ")
            df['Sculler Count'] = df['Group Members'].str.count("ˣ")
            df['Coxswain Count'] = df['Group Members'].str.count("ᶜ")
            
            # Calculate the 'Average Behind' value
            df['Average Behind'] = round(
                (df['Group Coefficient Sum'] - 
                (df['Starboard Count'] * starboard_best) - 
                (df['Port Count'] * port_best) - 
                (df['Sculler Count'] * sculler_best) - 
                (df['Coxswain Count'] * coxswain_best)) / 
                (df['Starboard Count'] + df['Port Count'] + df['Sculler Count'] + df['Coxswain Count']), 1
            ).apply(lambda x: f"+{round(x, 1)}" if x > 0 else f"{round(x, 1)}")
            
            return df

        
        starboard_dropped_df = calculate_average_behind(starboard_dropped_df, starboard_best, port_best, coxswain_best, sculler_best)
        port_dropped_df = calculate_average_behind(port_dropped_df, starboard_best, port_best, coxswain_best, sculler_best)
        coxswain_dropped_df = calculate_average_behind(coxswain_dropped_df, starboard_best, port_best, coxswain_best, sculler_best)

        st.subheader("Dropped Rowers")
        st.write("_Rowers with high uncertainty due to high colinearity_")

        if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
            col_star, col_port, col_cox = st.columns([1, 1, 1])
        else:
            col_star, col_port = st.columns([1, 1])

        with col_star:
            st.write("Starboard")                
            st.dataframe(starboard_dropped_df, column_order=["Group Members", "Average Behind"])
        
        with col_port:
            st.write("Port")                
            st.dataframe(port_dropped_df, column_order=["Group Members", "Average Behind"])

        if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
            with col_cox:
                st.write("Coxswains")                
                st.dataframe(coxswain_dropped_df, column_order=["Group Members", "Average Behind"])

    
    st.subheader("Confidence Intervals")
    if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
        bars_chart_starboard, bars_chart_port, bars_chart_cox = st.columns([1, 1, 1])
    else:
        bars_chart_starboard, bars_chart_port = st.columns([1, 1])
    with bars_chart_starboard:
        st.write("Starboard")
        starboard_confidence = st.slider("Confidence", key="starboard_confidence", min_value=0, max_value=99, value=50, step=1, format="%d%%")
        starboard_bar_chart = generate_confidence_bars_with_gradient(starboard_df, starboard_confidence)
        st.altair_chart(starboard_bar_chart, use_container_width=True)       
    
    with bars_chart_port:
        st.write("Port")            
        port_confidence = st.slider("Confidence", key="port_confidence", min_value=0, max_value=99, value=50, step=1, format="%d%%")
        port_bar_chart = generate_confidence_bars_with_gradient(port_df, port_confidence)
        st.altair_chart(port_bar_chart, use_container_width=True)

    if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
        with bars_chart_cox:
            st.write("Coxswains")            
            coxswain_confidence = st.slider("Confidence", key="coxswain_confidence", min_value=0, max_value=99, value=50, step=1, format="%d%%")
            port_bar_chart = generate_confidence_bars_with_gradient(coxswain_df, coxswain_confidence)
            st.altair_chart(port_bar_chart, use_container_width=True)

    st.subheader("One-on-One Probabilities")
    st.write("_Probability of the rower in the first column outperforming the rowers listed in the first row_")
    if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
        col3, col4, col5 = st.columns([1, 1, 1])
    else:
        col3, col4 = st.columns([1, 1])
    prob_matrix = compute_probability_matrix(starboard_df).sort_index()
    prob_matrix = prob_matrix[sorted(prob_matrix.columns)]
    col3.write("Starboard")
    col3.dataframe(prob_matrix)

    prob_matrix = compute_probability_matrix(port_df).sort_index()
    prob_matrix = prob_matrix[sorted(prob_matrix.columns)]
    col4.write("Port")
    col4.dataframe(prob_matrix)

    if st.session_state['include_coxswains'] and len(coxswain_df) > 1:
        prob_matrix = compute_probability_matrix(coxswain_df).sort_index()
        prob_matrix = prob_matrix[sorted(prob_matrix.columns)]
        col5.write("Coxswains")
        col5.dataframe(prob_matrix)

def standardize_speed(speed, boat_class, meters):
    number_of_rowers = int(boat_class[0])
    if boat_class[0] == '8' and st.session_state.get('include_coxswains', False):
        number_of_rowers = 9
    standardized_speed = (speed / 2000.0 * 4.0) * meters / number_of_rowers
    return standardized_speed

def adjust_metrics(df, boat_class, race_distance):
    """Adjust Speed, Lower, and Upper metrics. Use user-selected boat/distance"""
    # Create adjusted columns in one pass
    for col in ['Speed', 'Lower', 'Upper', 'Coefficient']:
        df[f'{col}_Adjusted'] = df[col].apply(
            lambda val: standardize_speed(val, boat_class, race_distance)
        )
    
    # Format the Behind_Adjusted column
    df['Behind_Adjusted'] = df['Speed_Adjusted'].apply(
        lambda x: f"+{round(x, 1)}" if x > 0 else f"{round(x, 1)}"
    )
    
    return df