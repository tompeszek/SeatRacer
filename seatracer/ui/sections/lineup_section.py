import streamlit as st
import pandas as pd
from utils.helpers import rig_map_reverse

def render():    
    analysis = st.session_state.analysis
    if analysis is None:
        st.write("No data available.")
        return
    
    athletes_df = analysis.final_results['athletes']    
    shell_classes_df = analysis.final_results['shell_classes']   
        
    athletes_list = sorted(athletes_df.index.tolist())
    classes_list = sorted(shell_classes_df.index.tolist(), reverse=True)

    # Allow the user to select how many lineups to test
    boat_count = st.slider("Number of Lineups to Test", min_value=2, max_value=6, value=2, step=1, key="boat_count")

    # Create dynamic columns based on the number of lineups
    lineup_cols = st.columns(boat_count)

    # Dictionary to store all lineup data for comparison
    all_lineups = {}

    # Loop through each column to create lineup selections
    for i, col in enumerate(lineup_cols):
        with col:
            st.markdown(f"### Lineup #{i+1}")
            
            # Select boat class
            boat_class = st.selectbox(f"Boat Class #{i+1}", classes_list, index=0, key=f"boat_class_{i}")
            rower_count = int(boat_class[0])
            
            # Select rowers
            selectboxes = []
            for j in range(rower_count - 1, -1, -1):
                position_name = 'Stroke' if (j == rower_count-1) else 'Bow' if (j == 0) else f'Seat #{j+1}'
                rower_select = st.selectbox(
                    position_name, 
                    key=f"boat_{i}_{j}", 
                    options=athletes_list, 
                    index=0
                )
                selectboxes.append(rower_select)
            
            # Check validity
            valid_lineup = True
            
            if len(set(selectboxes)) != len(selectboxes):
                st.warning("Duplicate rowers selected.")
                # valid_lineup = False

            # Count number of port/starboard rowers. Warn as invalid if p<>s
            rig_count = {'p': 0, 's': 0, 'c': 0, 'x': 0}
            for rower in selectboxes:
                rig = rig_map_reverse[rower[-1]]
                rig_count[rig] += 1

            if rig_count['p'] != rig_count['s'] and 'x' not in boat_class:
                st.warning(f"Invalid lineup. Port and starboard rowers must be equal.")
                # valid_lineup = False

            # Make prediction if lineup is valid
            if valid_lineup:
                # try:
                # Get prediction from the analysis                        
                predicted_time = analysis.predict_lineup(selectboxes, boat_class, return_formatted=True)
                predicted_seconds = analysis.predict_lineup(selectboxes, boat_class)
                
                # Display the prediction
                st.markdown(f"**Predicted: {predicted_time} / 500m**")
                
                # Store lineup data for comparison
                all_lineups[f"Lineup #{i+1}"] = {
                    'name': f"Lineup #{i+1}",
                    'personnel': selectboxes,
                    'shell_class': boat_class,
                    'predicted_time': predicted_time,
                    'predicted_seconds': predicted_seconds
                }
                    
                # except Exception as e:
                #     st.error(f"Error: {str(e)}")
                #     st.markdown("Some selected athletes or boat class might not be in the model.")

    # Compare all lineups if we have valid data
    if len(all_lineups) > 1:
        st.markdown("---")
        st.markdown("## Lineup Comparison")
        
        # Prepare data for compare_lineups
        lineups_for_comparison = [
            {'name': data['name'], 'personnel': data['personnel'], 'shell_class': data['shell_class']}
            for data in all_lineups.values()
        ]
        
        try:
            # Get comparison data
            comparison_df, details = analysis.compare_lineups(lineups_for_comparison)
            
            # Display comparison table
            st.dataframe(comparison_df)
                            
            # Detailed breakdown in expander
            with st.expander("Show detailed breakdown of all lineups"):
                for detail in details:
                    st.markdown(f"### {detail['name']} - {detail['formatted_time']}")
                    st.markdown(f"**Shell Class ({detail['shell_class']})**: {round(detail['shell_contribution'], 1)} seconds")
                    st.markdown(f"**Athletes Total**: {round(detail['athlete_contribution'], 1)} seconds")
                    
                    # Create a table for individual contributions
                    athlete_data = []
                    for athlete_detail in detail['athlete_details']:
                        position_idx = athlete_detail['position']
                        rower_count = len(detail['personnel'])
                        position_name = 'Stroke' if position_idx == rower_count else 'Bow' if position_idx == 1 else f'Seat #{position_idx}'
                        
                        athlete_data.append({
                            'Position': position_name,
                            'Athlete': athlete_detail['athlete'],
                            'Weight': f"{athlete_detail['weight']:.2f}",
                            'Coefficient': f"{athlete_detail['coefficient']:.1f}",
                            'Contribution': f"{athlete_detail['contribution']:.1f}"
                        })
                    
                    athlete_df = pd.DataFrame(athlete_data)
                    st.table(athlete_df)
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Error comparing lineups: {str(e)}")