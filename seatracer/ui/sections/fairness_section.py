import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy import stats

def analyze_prediction_bias(analysis):
    """
    Analyzes prediction bias for each athlete by comparing predicted vs actual results.
    Returns a dataframe with bias metrics for each athlete.
    """
    # Check if we have the final results available
    if analysis.final_results is None or 'comparison' not in analysis.final_results:
        st.warning("No comparison data available. Run analysis first.")
        return pd.DataFrame()
    
    # Get the comparison dataframe that contains actual vs predicted times
    comparison_df = analysis.final_results['comparison']
    
    # Create a dictionary to store each athlete's prediction errors
    athlete_errors = {}
    
    # Process each row in the comparison dataframe
    for idx, row in comparison_df.iterrows():
        # Extract actual and predicted times
        actual_time = row['Actual Pace Seconds']
        predicted_time = row['Model Pace Seconds']
        # The error is already calculated in the Delta column
        error = row['Delta']  # positive means model predicted too slow
        
        # Get the crew/lineup
        crew = row['Crew'].split('/')
        
        # Distribute the error among athletes in this lineup
        if crew:
            # Add this error to each athlete's record
            for athlete in crew:
                if athlete not in athlete_errors:
                    athlete_errors[athlete] = []
                athlete_errors[athlete].append(error)
    
    # Create a dataframe with metrics for each athlete
    bias_data = []
    for athlete, errors in athlete_errors.items():
        # Only include athletes with at least 2 races for meaningful statistics
        if len(errors) >= 2:
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            races_count = len(errors)
            
            # Determine bias direction
            # Note: Positive error means model predicted time was slower than actual
            # So positive = overestimated time (predicted slower than reality)
            # Negative = underestimated time (predicted faster than reality)
            bias_direction = "Underperformed" if avg_error < 0 else "Overperformed"
            
            # Calculate consistency (lower std dev = more consistent bias)
            consistency = 1 / (1 + std_error)
            
            # Calculate statistical significance (t-test)
            t_stat = avg_error / (std_error / np.sqrt(races_count)) if std_error > 0 else float('inf')
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=races_count-1)) if std_error > 0 else 0
            
            bias_data.append({
                'Athlete': athlete,
                'Average Error': round(avg_error, 2),
                'Standard Deviation': round(std_error, 2),
                'Races': races_count,
                'Bias Direction': bias_direction,
                'Consistency': round(consistency, 3),
                'Normalized Error': round(avg_error / (std_error if std_error > 0 else 1), 2),
                'T-Statistic': round(t_stat, 2),
                'P-Value': round(p_value, 3),
                'Significant': p_value < 0.05
            })
    
    return pd.DataFrame(bias_data)

def render(sides_count):
    """
    Renders the performance bias analysis section in the Streamlit app.
    """
    analysis = st.session_state.analysis
    if analysis is None:
        st.write("No data available.")
        return
    
    st.header("Performance Bias Analysis")
    st.write("This section analyzes how consistently the model evaluates each athlete's performance across multiple events.")
    
    # Generate the bias analysis data
    bias_df = analyze_prediction_bias(analysis)
    
    if bias_df.empty:
        st.warning("Insufficient data for bias analysis. Athletes need to appear in at least 2 races with comparison data available.")
        return
    
    # Sort the data based on statistical significance and average error
    bias_df = bias_df.sort_values(['Significant', 'Average Error'], ascending=[False, True])
    
    # Add a term for the bias that sounds better than "unfair"
    bias_df['Performance vs Model'] = bias_df.apply(
        lambda x: "Consistently Underperforms Model" if x['Average Error'] < -0.5 and x['Significant'] else 
                  "Consistently Outperforms Model" if x['Average Error'] > 0.5 and x['Significant'] else 
                  "No Significant Bias",
        axis=1
    )
    
    # Extract athlete suffixes for position information
    bias_df['Position'] = bias_df['Athlete'].str.extract(r'([ᵖˢᶜˣ])$')[0].map({
        'ᵖ': 'Port',
        'ˢ': 'Starboard',
        'ᶜ': 'Coxswain',
        'ˣ': 'Scull'
    })
    
    # Create a dictionary to group athletes by their position
    position_groups = {
        'Starboard': [rower for rower, sides in sides_count.items() if sides['Starboard'] > 0],
        'Port': [rower for rower, sides in sides_count.items() if sides['Port'] > 0],
        'Coxswain': [rower for rower, sides in sides_count.items() if sides['Coxswain'] > 0],
        'Scull': [rower for rower, sides in sides_count.items() if sides['Scull'] > 0]
    }
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Performance Bias Overview", "Detailed Analysis by Position"])
    
    with tab1:
        # Add description to help understand the analysis
        st.markdown("""
        ## Understanding Performance Bias
        
        This analysis examines how consistently the model evaluates each athlete's contributions by comparing predicted vs. actual results:
        
        - **Positive values** indicate the athlete **outperforms the model's expectations** (boats with this athlete are faster than the model predicts)
        - **Negative values** indicate the athlete **underperforms the model's expectations** (boats with this athlete are slower than the model predicts)
        - **Values near zero** indicate the model accurately evaluates the athlete's contribution
        - **Statistical significance** (p < 0.05) indicates the pattern is unlikely to be random chance
        """)
        
        # Create a chart showing average error for each athlete, highlighting statistically significant results
        significant_df = bias_df[bias_df['Significant']]
        
        if not significant_df.empty:
            chart = alt.Chart(significant_df).mark_bar().encode(
                x=alt.X('Athlete:N', sort=None),
                y=alt.Y('Average Error:Q', title='Average Prediction Error (seconds)'),
                color=alt.Color('Performance vs Model:N', 
                              scale=alt.Scale(domain=['Consistently Underperforms Model', 'No Significant Bias', 'Consistently Outperforms Model'],
                                             range=['#FF9999', '#AAAAAA', '#99CCFF'])),
                tooltip=['Athlete', 'Average Error', 'Races', 'P-Value', 'Performance vs Model']
            ).properties(
                title='Statistically Significant Performance Bias by Athlete',
                width=600,
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No statistically significant performance bias detected for any athlete.")
            
            # Show a chart of all athletes anyway
            chart = alt.Chart(bias_df).mark_bar().encode(
                x=alt.X('Athlete:N', sort=None),
                y=alt.Y('Average Error:Q', title='Average Prediction Error (seconds)'),
                color=alt.Color('Bias Direction:N', scale=alt.Scale(domain=['Underperformed', 'Overperformed'], range=['#FF9999', '#99CCFF'])),
                tooltip=['Athlete', 'Average Error', 'Races', 'P-Value']
            ).properties(
                title='Performance Bias by Athlete (Not Statistically Significant)',
                width=600,
                height=400
            )
            
            st.altair_chart(chart, use_container_width=True)
        
        # Show the full data table
        st.subheader("Complete Performance Bias Metrics")
        st.dataframe(
            bias_df[[
                'Athlete', 'Position', 'Average Error', 'Standard Deviation', 
                'Races', 'P-Value', 'Significant', 'Performance vs Model'
            ]].sort_values(['Significant', 'Average Error'], ascending=[False, True]),
            use_container_width=True
        )
    
    with tab2:
        # Find which positions have data
        available_positions = [pos for pos, athletes in position_groups.items() 
                              if any(athlete in bias_df['Athlete'].values for athlete in athletes)]
        
        if not available_positions:
            st.warning("No position data available for analysis.")
            return
            
        # Create columns for different positions
        cols = st.columns(len(available_positions))
        
        for i, position in enumerate(available_positions):
            with cols[i]:
                st.subheader(position)
                position_athletes = position_groups[position]
                position_df = bias_df[bias_df['Athlete'].isin(position_athletes)]
                
                if not position_df.empty:
                    # Create a position-specific chart
                    significant_position_df = position_df[position_df['Significant']]
                    
                    if not significant_position_df.empty:
                        pos_chart = alt.Chart(significant_position_df).mark_bar().encode(
                            x=alt.X('Athlete:N', sort=None),
                            y=alt.Y('Average Error:Q', title='Avg Error (seconds)'),
                            color=alt.Color('Performance vs Model:N',
                                          scale=alt.Scale(domain=['Consistently Underperforms Model', 'No Significant Bias', 'Consistently Outperforms Model'],
                                                         range=['#FF9999', '#AAAAAA', '#99CCFF'])),
                            tooltip=['Athlete', 'Average Error', 'Races', 'P-Value']
                        ).properties(
                            title=f'Significant {position} Bias',
                            height=250
                        )
                        
                        st.altair_chart(pos_chart, use_container_width=True)
                    else:
                        # Create chart for all athletes in this position
                        pos_chart = alt.Chart(position_df).mark_bar().encode(
                            x=alt.X('Athlete:N', sort=None),
                            y=alt.Y('Average Error:Q', title='Avg Error (seconds)'),
                            color=alt.Color('Bias Direction:N', scale=alt.Scale(domain=['Underperformed', 'Overperformed'], range=['#FF9999', '#99CCFF'])),
                            tooltip=['Athlete', 'Average Error', 'Races', 'P-Value']
                        ).properties(
                            title=f'{position} (Not Significant)',
                            height=250
                        )
                        
                        st.altair_chart(pos_chart, use_container_width=True)
                    
                    # Display metrics table for this position
                    st.dataframe(
                        position_df[['Athlete', 'Average Error', 'Races', 'P-Value', 'Significant']].sort_values(
                            ['Significant', 'Average Error'], ascending=[False, True]
                        ),
                        use_container_width=True
                    )
                else:
                    st.write("No data available for this position.")
    
    # # Add an advanced analysis section
    # st.subheader("Advanced Analysis")
    
    # # Create a scatter plot showing races vs error magnitude, with statistical significance
    # scatter = alt.Chart(bias_df).mark_circle(size=100).encode(
    #     x=alt.X('Races:Q', title='Number of Races'),
    #     y=alt.Y('Average Error:Q', title='Average Performance Deviation (seconds)'),
    #     color=alt.Color('Bias Direction:N', scale=alt.Scale(domain=['Underperformed', 'Overperformed'], range=['#FF9999', '#99CCFF'])),
    #     size=alt.Size('abs(T-Statistic):Q', title='Statistical Strength', scale=alt.Scale(domain=[0, 5], range=[30, 300])),
    #     opacity=alt.condition(
    #         alt.datum.Significant,
    #         alt.value(1),
    #         alt.value(0.3)
    #     ),
    #     tooltip=['Athlete', 'Average Error', 'Races', 'P-Value', 'Significant', 'T-Statistic']
    # ).properties(
    #     title='Performance Bias vs. Sample Size',
    #     width=700,
    #     height=400
    # )
    
    # # Add a rule at y=0 for reference
    # zero_rule = alt.Chart().mark_rule(color='gray').encode(y=alt.datum(0))
    
    # st.altair_chart(scatter + zero_rule, use_container_width=True)
    
    # st.markdown("""
    # **Interpreting the Advanced Analysis:**
    # - **Larger circles** indicate stronger statistical evidence for performance bias
    # - **Opaque circles** represent statistically significant results (p < 0.05)
    # - **Blue points** represent athletes who consistently perform better than the model predicts
    # - **Red points** represent athletes who consistently perform worse than the model predicts
    # - **More races** generally provides more reliable evidence of consistent bias
    
    # This analysis can help identify whether certain athletes are consistently under or overvalued by the model, which could inform lineup decisions and model refinement.
    # """)
    
    # # Add a summary of the most significant performance biases
    # st.subheader("Summary of Notable Performance Biases")
    
    # # Identify the most significantly biased athletes
    # sig_underperform = bias_df[(bias_df['Average Error'] < -0.5) & (bias_df['Significant'])].sort_values('Average Error')
    # sig_overperform = bias_df[(bias_df['Average Error'] > 0.5) & (bias_df['Significant'])].sort_values('Average Error', ascending=False)
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.markdown("#### Consistently Underperform Model Expectations")
    #     if not sig_underperform.empty:
    #         st.dataframe(
    #             sig_underperform[['Athlete', 'Position', 'Average Error', 'Races', 'P-Value']],
    #             use_container_width=True
    #         )
            
    #         st.markdown("""
    #         **Implications:** 
    #         - These athletes may be overrated by the model
    #         - Consider checking for specific conditions these athletes perform in
    #         - Model may need adjustment for these athletes
    #         """)
    #     else:
    #         st.write("No athletes significantly underperform model expectations.")
    
    # with col2:
    #     st.markdown("#### Consistently Outperform Model Expectations")
    #     if not sig_overperform.empty:
    #         st.dataframe(
    #             sig_overperform[['Athlete', 'Position', 'Average Error', 'Races', 'P-Value']],
    #             use_container_width=True
    #         )
            
    #         st.markdown("""
    #         **Implications:**
    #         - These athletes may be underrated by the model
    #         - They might provide intangible benefits not captured by data
    #         - Consider these athletes for important races even if model ranks them lower
    #         """)
    #     else:
    #         st.write("No athletes significantly outperform model expectations.")
            
    # # Add recommendations based on findings
    # st.subheader("Recommendations")
    
    # if not sig_underperform.empty or not sig_overperform.empty:
    #     st.markdown("""
    #     Based on the performance bias analysis:
        
    #     1. **Model Refinement:** Consider adjusting coefficients for athletes with significant bias
        
    #     2. **Lineup Decisions:** 
    #        - Athletes who consistently outperform the model may be valuable additions to critical lineups
    #        - Athletes who consistently underperform may need additional evaluation or specific boat placements
        
    #     3. **Further Investigation:**
    #        - Look for patterns in boat types or racing conditions for biased athletes
    #        - Check if certain athlete combinations amplify or reduce these biases
    #     """)
    # else:
    #     st.markdown("""
    #     The model appears well-calibrated overall with no statistically significant performance biases detected. This suggests:
        
    #     1. **Reliable Predictions:** The model evaluates athlete contributions consistently
        
    #     2. **Effective Parameter Estimation:** Athlete coefficients accurately reflect their performance impact
        
    #     3. **Continued Monitoring:** While no significant biases were found, continue to monitor as more race data becomes available
    #     """)