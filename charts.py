import altair as alt
import scipy.stats
from helpers import *
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_probability_matrix_monte_carlo(df, num_samples=10000):
    df = df.copy()
    
    rowers = df.index.tolist()

    # Estimate standard deviation from confidence interval (assuming normal dist)
    df.loc[:, "StdDev"] = (df["Upper"] - df["Lower"]) / 3.92
    
    # Generate samples for each athlete
    samples = {
        rower: np.random.normal(df.loc[rower, "Coefficient"], df.loc[rower, "StdDev"], num_samples)
        for rower in rowers
    }

    # Create probability matrix
    prob_matrix = pd.DataFrame(index=rowers, columns=rowers, dtype=object)

    for row1 in rowers:
        for row2 in rowers:
            if row1 == row2:
                prob_matrix.loc[row1, row2] = '-'  # Set diagonal values as '-'
            else:
                # Compute probability as percentage and store it directly
                prob_matrix.loc[row1, row2] = f"{np.mean(samples[row1] < samples[row2]) * 100:.0f}%"

    return prob_matrix

def compute_probability_matrix(df):
    df = df.copy()
    rowers = df.index.tolist()
    
    # Estimate standard deviation from confidence interval (assuming normal dist)
    df.loc[:, "StdDev"] = (df["Upper"] - df["Lower"]) / 3.92

    # Create probability matrix
    prob_matrix = pd.DataFrame(index=rowers, columns=rowers, dtype=object)

    for row1 in rowers:
        for row2 in rowers:
            if row1 == row2:
                prob_matrix.loc[row1, row2] = '-'  # Set diagonal values as '-'
            else:
                mu_A, mu_B = df.loc[row1, "Coefficient"], df.loc[row2, "Coefficient"]
                sigma_A, sigma_B = df.loc[row1, "StdDev"], df.loc[row2, "StdDev"]
                
                # Compute probability using normal CDF
                z_score = (mu_B - mu_A) / np.sqrt(sigma_A**2 + sigma_B**2)
                prob_matrix.loc[row1, row2] = f"{norm.cdf(z_score) * 100:.0f}%"

    return prob_matrix

def generate_likelihood(row, x_vals):
    """Generate likelihood values (y-axis) based on a normal distribution."""
    mean = row["Coefficient"]
    std_dev = (row["Upper"] - row["Lower"]) / 3.92  # Approximate 95% CI to std dev
    return scipy.stats.norm.pdf(x_vals, mean, std_dev).tolist()

def generate_side_chart(col1, side_df):
    # add_speed(side_df)

    # Establish a common x-axis range
    global_x_min = side_df["Lower"].min()
    global_x_max = side_df["Upper"].max()
    num_points = 1000  # Resolution of the curve
    x_vals = np.linspace(global_x_min, global_x_max, num_points)  # Shared x-axis

    # Generate likelihoods
    side_df = side_df.copy()  # Avoid modifying a slice of athletes_df
    side_df.loc[:, "Likelihood Chart"] = side_df.apply(generate_likelihood, axis=1, x_vals=x_vals)

    # Define column configuration
    column_config = {
        "Likelihood Chart": st.column_config.AreaChartColumn(
            label="Confidence",
            y_min=0,
            y_max=0.25  # Auto-scale max -> None
        )
    }

    col1.dataframe(
        side_df.sort_values(by="Speed"),
        column_config=column_config,
        column_order=["Rower", "Behind", "Plus/Minus", "Likelihood Chart"]
    )

def generate_confidence_bars_with_gradient_working(side_df, confidence=50):
    """Generate the confidence bars with gradient effect and dynamically adjust for readability."""
    
    chart_data = []
    
    # Iterate through the dataframe rows and collect data for each rower
    for index, row in side_df.iterrows():
        # Coefficient (mean) and standard deviation
        mean = row["Coefficient"]
        std_dev = (row["Upper"] - row["Lower"]) / 3.92  # Approximate 95% CI to std dev
        
        # Find 25th and 75th percentiles for the confidence interval
        # lower_percentile = scipy.stats.norm.ppf(0.25, mean, std_dev)
        # upper_percentile = scipy.stats.norm.ppf(0.75, mean, std_dev) # 25/75% for 95% CI

        lower_percentile = scipy.stats.norm.ppf(0.5 - ((confidence/2.0) / 100.0), mean, std_dev)
        upper_percentile = scipy.stats.norm.ppf(0.5 + ((confidence/2.0) / 100.0), mean, std_dev) # 25/75% for 95% CI

        
        # Add rower and their confidence interval to the chart data
        chart_data.append({
            "Rower": index,
            "Coefficient": mean,
            "Lower Percentile": lower_percentile,
            "Upper Percentile": upper_percentile,
            "Bar Position": lower_percentile,  # Start of the bar (25th percentile)
            "Bar End": upper_percentile,  # End of the bar (75th percentile)
            "Color Intensity": 0.5  # Placeholder for gradient effect
        })
    
    # Create a DataFrame for Altair plotting
    df_chart = pd.DataFrame(chart_data)

    # Sort the chart data by Coefficient in ascending order
    df_chart = df_chart.sort_values('Coefficient')
    
    # Adjust the height per rower (reduce it for smaller bars)
    chart_height = side_df.shape[0] * 30  # Reduced space per rower (default was 50px)
    chart_height = min(chart_height, 1000)  # Cap the height at 1000px

    # Get the Streamlit theme colors
    # pc = st.get_option('theme.primaryColor')
    # bc = st.get_option('theme.backgroundColor')
    # sbc = st.get_option('theme.secondaryBackgroundColor')
    # tc = st.get_option('theme.textColor')
    # print(f"Primary Color: {pc}, Background Color: {bc}, Secondary Background Color: {sbc}, Text Color: {tc}")
    
    # Create the Altair chart with horizontal bars
    chart = alt.Chart(df_chart).mark_bar().encode(
        y=alt.Y('Rower:N', title='Rower', sort=None),  # Rowers as categorical on Y axis
        x=alt.X('Bar Position:Q', title='Coefficient'),  # Starting point of the confidence interval
        x2='Bar End:Q',  # Ending point of the confidence interval (x2)
        # color=alt.Color(
        #     'Rower:N', 
        #     scale=alt.Scale(domain=df_chart['Rower'].tolist(), range=[pc, sbc]),
        #     legend=None
        # ), 
        tooltip=['Rower', 'Lower Percentile', 'Upper Percentile', 'Coefficient']  # Tooltip for extra info
    ).configure_axis(
        labelLimit=1000,  # Increase label limit to prevent Altair from skipping or truncating labels
        labelAngle=0  # Optional: Adjust angle to 0 if you don't want to rotate the labels
    )
    return chart



def generate_confidence_bars_with_gradient(side_df, confidence=50):
    """Generate the confidence bars with gradient effect and dynamically adjust for readability."""
    
    chart_data = []
    
    # Iterate through the dataframe rows and collect data for each rower
    for index, row in side_df.iterrows():
        # Speed (mean) and standard deviation
        mean = row["Speed"]
        std_dev = (row["Upper"] - row["Lower"]) / 3.92  # Approximate 95% CI to std dev
        
        # Find 25th and 75th percentiles for the confidence interval
        # lower_percentile = scipy.stats.norm.ppf(0.25, mean, std_dev)
        # upper_percentile = scipy.stats.norm.ppf(0.75, mean, std_dev) # 25/75% for 95% CI

        lower_percentile = scipy.stats.norm.ppf(0.5 - ((confidence/2.0) / 100.0), mean, std_dev)
        upper_percentile = scipy.stats.norm.ppf(0.5 + ((confidence/2.0) / 100.0), mean, std_dev) # 25/75% for 95% CI

        lower_percentile_label = "Upper Limit"
        upper_percentile_label = "Lower Limit"
        
        # Add rower and their confidence interval to the chart data
        chart_data.append({
            "Rower": index,
            "Speed": mean,
            lower_percentile_label: round(lower_percentile,1),
            upper_percentile_label: round(upper_percentile,1),
            "Bar Position": lower_percentile,  # Start of the bar (25th percentile)
            "Bar End": upper_percentile,  # End of the bar (75th percentile)
            "Color Intensity": 0.5  # Placeholder for gradient effect
        })
    
    # Create a DataFrame for Altair plotting
    df_chart = pd.DataFrame(chart_data)

    # Sort the chart data by Speed in ascending order
    df_chart = df_chart.sort_values('Speed')
    
    # Adjust the height per rower (reduce it for smaller bars)
    chart_height = side_df.shape[0] * 30  # Reduced space per rower (default was 50px)
    chart_height = min(chart_height, 1000)  # Cap the height at 1000px

    # Get the Streamlit theme colors
    # pc = st.get_option('theme.primaryColor')
    # bc = st.get_option('theme.backgroundColor')
    # sbc = st.get_option('theme.secondaryBackgroundColor')
    # tc = st.get_option('theme.textColor')
    # print(f"Primary Color: {pc}, Background Color: {bc}, Secondary Background Color: {sbc}, Text Color: {tc}")
    
    # Create the Altair chart with horizontal bars
    chart = alt.Chart(df_chart).mark_bar().encode(
        y=alt.Y('Rower:N', title='Rower', sort=None),  # Rowers as categorical on Y axis
        x=alt.X('Bar Position:Q', title='Speed'),  # Starting point of the confidence interval
        x2='Bar End:Q',  # Ending point of the confidence interval (x2)
        # color=alt.Color(
        #     'Rower:N', 
        #     scale=alt.Scale(domain=df_chart['Rower'].tolist(), range=[pc, sbc]),
        #     legend=None
        # ), 
        tooltip=['Rower', 'Speed', lower_percentile_label, upper_percentile_label, ]  # Tooltip for extra info
    ).configure_axis(
        labelLimit=1000,  # Increase label limit to prevent Altair from skipping or truncating labels
        labelAngle=0  # Optional: Adjust angle to 0 if you don't want to rotate the labels
    )
    return chart