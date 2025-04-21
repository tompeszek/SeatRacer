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

    starboard_df = athletes_df.loc[athletes_df.index.isin(starboard_rowers)].copy()
    port_df = athletes_df.loc[athletes_df.index.isin(port_rowers)].copy()
    coxswain_df = athletes_df.loc[athletes_df.index.isin(coxswains)].copy()
    sculler_df = athletes_df.loc[athletes_df.index.isin(scullers)].copy()

    st.subheader("Speed Coefficients")
    coefficient_boat_classes = ["8+", "4+/-", "2-"]
    st.write("_Number of seconds, over 2k in a 4-, slower than best rower on the same side_")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("Starboard")
        # starboard_confidence = st.slider("Max Uncertainty", key="starboard_uncertainty", min_value = 5, max_value = 100, value = 10, step = 1)
        generate_side_chart(st, starboard_df)

    with col2:
        st.write("Port")
        # starboard_confidence = st.slider("Max Uncertainty", key="port_uncertainty", min_value = 5, max_value = 100, value = 10, step = 1)
        generate_side_chart(st, port_df)
        
    if len(dropped_athletes_df) > 0:
        # need to fix speed
        starboard_dropped_df = dropped_athletes_df.loc[dropped_athletes_df.index.isin(starboard_rowers)].sort_index().copy()
        port_dropped_df = dropped_athletes_df.loc[dropped_athletes_df.index.isin(port_rowers)].sort_index().copy()

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

        st.subheader("Dropped Rowers")
        st.write("_Rowers with high uncertainty due to high colinearity_")
        col_star, col_port = st.columns([1, 1])

        with col_star:
            st.write("Starboard")                
            st.dataframe(starboard_dropped_df, column_order=["Group Members", "Average Behind"])
        
        with col_port:
            st.write("Port")                
            st.dataframe(port_dropped_df, column_order=["Group Members", "Average Behind"])

    
    st.subheader("Confidence Intervals")
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

    st.subheader("One-on-One Probabilities")
    st.write("_Probability of the rower in the first column outperforming the rowers listed in the first row_")
    col3, col4 = st.columns([1, 1])
    st.markdown(
        "<style> td:first-child { font-weight: bold; } </style>", 
        unsafe_allow_html=True
    )

    prob_matrix = compute_probability_matrix(starboard_df).sort_index()
    prob_matrix = prob_matrix[sorted(prob_matrix.columns)]
    col3.write("Starboard")
    col3.dataframe(prob_matrix)

    prob_matrix = compute_probability_matrix(port_df).sort_index()
    prob_matrix = prob_matrix[sorted(prob_matrix.columns)]
    col4.write("Port")
    col4.dataframe(prob_matrix)

    st.divider()

    col5, col6 = st.columns([1, 1])
    # col5.subheader("Boat Classes")
    # generate_side_chart(col5, shell_classes_df, "Boat Classes")

    # col5.subheader("Coxswains")
    # generate_side_chart(col5, coxswain_df)
    # coxswain_matrix = compute_probability_matrix(coxswain_df).sort_index()
    # coxswain_matrix = coxswain_matrix[sorted(coxswain_matrix.columns)]
    # col5.dataframe(coxswain_matrix)