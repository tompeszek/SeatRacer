from helpers import *
import streamlit as st
import numpy as np
import pandas as pd

def compute_probability_matrix(df, num_samples=10000):
    rowers = df.index.tolist()

    # Estimate standard deviation from confidence interval (assuming normal dist)
    df["StdDev"] = (df["Upper"] - df["Lower"]) / 3.92

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


def generate_side_chart(col1, side_df, side_label):
    add_speed(side_df)

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
            y_max=None  # Auto-scale max
        )
    }

    col1.dataframe(
        side_df.sort_values(by="Speed"),
        column_config=column_config,
        column_order=["Rower", "Behind", "Plus/Minus", "Likelihood Chart"]
    )