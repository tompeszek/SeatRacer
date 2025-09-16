import streamlit as st

def render():
    analysis = st.session_state.analysis
    if analysis is not None:
        st.subheader("Model Fit")
        abs_delta_sum = analysis.final_results['comparison']['Delta'].abs().sum()
        abs_delta_avg = analysis.final_results['comparison']['Delta'].abs().mean()        
        abs_delta_max = analysis.final_results['comparison']['Delta'].abs().max()
        squared_delta_sum = (analysis.final_results['comparison']['Delta'] ** 2).sum()

        col_avg, col_max, col_sqr  = st.columns([1, 1, 1])
        with col_avg:
            st.metric(label="Average Model Error", value=f'±{abs_delta_avg:.2f}" / 500m')
        with col_max:
            st.metric(label="Greatest Model Error", value=f'±{abs_delta_max:.2f}" / 500m')
        with col_sqr:
            st.metric(label="Squared Model Error", value=f'{squared_delta_sum:.2f}"')
        
        st.subheader("Actual vs. Model")
        
        # Get all unique athletes from the comparison dataframe
        all_athletes = set()
        for crew in analysis.final_results['comparison']['Crew']:
            athletes = crew.split('/')
            for athlete in athletes:
                all_athletes.add(athlete.strip())
        
        # Create a multiselect filter for athletes
        selected_athletes = st.multiselect(
            "Filter lineups containing athletes:",
            sorted(list(all_athletes)),
            key="lineup_filter"
        )
        
        # Filter the comparison dataframe based on selected athletes
        if selected_athletes:
            filtered_comparison = analysis.final_results['comparison'].copy()
            mask = filtered_comparison['Crew'].apply(
                lambda crew: all(athlete in crew for athlete in selected_athletes)
            )
            filtered_comparison = filtered_comparison[mask]
            st.dataframe(filtered_comparison, hide_index=True)
        else:
            # Show all data if no athletes are selected
            st.dataframe(analysis.final_results['comparison'], hide_index=True)

        st.subheader("Possible Errors")
        # Check for athletes in multiple boats
        df = analysis.df.copy()
        duplicates = []
        
        # Group by race session and piece
        grouped = df.groupby(['Race Session (date)', 'Piece'])
        
        for (race_session, piece), group in grouped:
            if len(group) <= 1:  # Skip if only one boat
                continue
                
            # Get all athletes from all boats
            all_athletes = {}
            for idx, row in group.iterrows():
                personnel = row['Personnel'].split('/')
                for athlete in personnel:
                    athlete = athlete.strip()
                    if not athlete:
                        continue

                    if athlete == "Coxᶜ":
                        continue
                        
                    if athlete not in all_athletes:
                        all_athletes[athlete] = []
                    all_athletes[athlete].append(row['Personnel'])
            
            # Find duplicates
            for athlete, boats in all_athletes.items():
                if len(boats) > 1:
                    duplicates.append({
                        'Race': f"{race_session} - Piece {piece}",
                        'Athlete': athlete,
                        'Boats': boats
                    })
        
        # Display results
        if duplicates:
            st.warning(f"Found {len(duplicates)} athletes in multiple boats")
            for dup in duplicates:
                st.write(f"- {dup['Athlete']} appears in {len(dup['Boats'])} boats in {dup['Race']}")
                for boat in dup['Boats']:
                    st.write(f"  • {boat}")
        else:
            st.success("No athletes found in multiple boats")

    else:
        st.write("No data available.")