import streamlit as st
import pandas as pd

def render():
    analysis = st.session_state.analysis
    if analysis is None:
        st.write("No data available.")
        return

    top_n = 10
    min_races = 5

    pairs_df = analysis.final_results['pairs']
    if pairs_df.empty:
        st.write("No athlete pairs found with sufficient data.")
        return
    
    
    # Only include pairs with at least 2 races for statistical relevance
    filtered_df = pairs_df[pairs_df['Races'] >= min_races].copy()

    if filtered_df.empty:
        st.write("No athlete pairs found with at least 2 races together.")
        return
        
    syn_col_1, syn_col_2 = st.columns([1, 3])

    with syn_col_1:
        # Radio select for p-value
        p_value_options = {
            "All": {"value": 1, "text": "Showing all pairs, regardless of statistical significance"},
            "0.05": {"value": 0.05, "text": "Showing pairs with p-value <= 0.05, which is statistically significant"},
            "0.01": {"value": 0.01, "text": "Showing pairs with p-value <= 0.01, which is highly statistically significant"},
            "0.001": {"value": 0.001, "text": "Showing pairs with p-value <= 0.001, which is extremely statistically significant"},
        }

        select_p_value = st.radio("Select p-value threshold for significance", list(p_value_options.keys()), index=1)
        st.caption(p_value_options[select_p_value]['text'])

    filtered_df = filtered_df[filtered_df['p_value'] <= p_value_options[select_p_value]['value']]

    # Add a "Synergy Score" column - combination of average delta and significance
    # Negative is good (faster than predicted)
    # filtered_df['Synergy'] = -filtered_df['AvgDelta']
    
    # Format p-value
    filtered_df['Significance'] = filtered_df['p_value'].apply(
        lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
    )
    
    # Format delta as time difference
    filtered_df['Performance'] = filtered_df['AvgDelta'].apply(
        lambda x: f"{x:.2f}s" + (" (faster)" if x < 0 else " (slower)")
    )
    
    # Create pair name column
    filtered_df['Pair'] = filtered_df.apply(
        lambda row: f"{row['Athlete1']} + {row['Athlete2']}", axis=1
    )
    
    # Configure columns for display
    column_config = {
        "Pair": st.column_config.TextColumn("Athlete Pair"),
        "Performance": st.column_config.TextColumn("Actual Seconds Per 500m Compared to Model"),
        "Races": st.column_config.NumberColumn("# of Races"),
        "Significance": st.column_config.TextColumn("Significance (p-value)"),
    }
    
    bottom_pairs = filtered_df.sort_values("AvgDelta", ascending=True)#.head(top_n)# original uses Synergy which is really just delta
                
    # Display synergistic pairs
    with syn_col_2:
        st.subheader(f"Pairs Showing Synergy or Discord")
        st.dataframe(
            bottom_pairs,
            column_config=column_config,
            column_order=["Pair", "Performance", "Races", "Significance"],
            hide_index=True
        )