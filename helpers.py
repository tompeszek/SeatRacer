# helpers.py
import statsmodels.api as sm
import numpy as np
import scipy.stats
import pandas as pd
from datetime import datetime

def generate_common_x_range(df, num_points=50):
    """Generate a shared x-axis range for all rows based on min/max of CI."""
    global_x_min = df["Lower"].min()
    global_x_max = df["Upper"].max()
    return np.linspace(global_x_min, global_x_max, num_points)  # Shared x-axis

def generate_likelihood(row, x_vals):
    """Generate likelihood values (y-axis) based on a normal distribution."""
    mean = row["Coefficient"]
    std_dev = (row["Upper"] - row["Lower"]) / 3.92  # Approximate 95% CI to std dev
    return scipy.stats.norm.pdf(x_vals, mean, std_dev).tolist()

def time_to_seconds(time_str):
    """Convert time in MM:SS.x format to seconds."""
    time_obj = datetime.strptime(time_str, '%M:%S.%f')
    return time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6

def seconds_to_time(seconds):
    """Convert seconds to MM:SS.x format."""
    minutes = int(seconds // 60)
    seconds_remaining = seconds % 60
    return f"{minutes:02}:{seconds_remaining:05.1f}"

def determine_shell_class(row):
    athletes = row['athlete_count']
    is_sculling = 'x' in row['Rigging']
    has_cox = athletes % 2 == 1 and athletes > 1
    rower_count = athletes - (1 if has_cox else 0)

    boat_class = str(rower_count)
    if is_sculling:
        boat_class = boat_class + "x"
    if has_cox:
        boat_class = boat_class + "+"
    if not is_sculling and not has_cox:
        boat_class = boat_class + "-"

    return boat_class

def add_speed(df):
    fastest = df['Coefficient'].min()
    df['Speed'] = df['Coefficient'] - fastest
    df['Behind'] = df['Speed'].apply(lambda x: f"+{round(x, 1)}" if x > 0 else "-")
    df['Max/Min'] = df.apply(
        lambda row: f"{round(row['Lower'] - row['Coefficient'], 1)} to {round(row['Upper'] - row['Coefficient'], 1)}",
        axis=1
    )

    # Create a list of two points: lower and upper CI
    # df['CI Chart'] = df.apply(lambda row: [row['Lower'], row['Upper']], axis=1)

    return df

def generate_likelihood(row, x_vals):
    """Generate likelihood values (y-axis) based on a normal distribution."""
    mean = row["Coefficient"]
    std_dev = (row["Upper"] - row["Lower"]) / 3.92  # Approximate 95% CI to std dev
    y_vals = scipy.stats.norm.pdf(x_vals, mean, std_dev)  # Compute normal distribution
    return y_vals.tolist()

def run_ols_regression(data):
    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    # All athlete names
    athletes = df['Personnel'].str.split('/', expand=True).stack().unique()

    # Check sides
    rower_sides_count = {p: {'Starboard': 0, 'Port': 0, 'Scull': 0, 'Coxswain': 0} for p in athletes}

    for _, row in df.iterrows():
        # Split the rigging and personnel to get them as lists
        rigging_list = row['Rigging'].split('/')
        personnel_list = row['Personnel'].split('/')
        
        # Iterate over the zip of rigging and personnel
        for r, p in zip(rigging_list, personnel_list):
            if r == 's':
                rower_sides_count[p]['Starboard'] += 1
            elif r == 'p':
                rower_sides_count[p]['Port'] += 1
            elif r == 'x':
                rower_sides_count[p]['Scull'] += 1
            elif r == 'c':
                rower_sides_count[p]['Coxswain'] += 1

    print(rower_sides_count)

    # Get shell class
    df['athlete_count'] = df['Rigging'].apply(lambda x: len(x.split('/')))
    df['shell_class'] = df.apply(determine_shell_class, axis=1)
    shell_classes = df['shell_class'].unique()

    # Convert Result times to seconds and then calculate time per 500m
    df['time_seconds'] = df['Result'].apply(time_to_seconds)
    df['time_per_500m'] = df['time_seconds'] / (df['KM'] * 2.0)  # Adjust time per 500m

    # Create the Race Session + Piece interaction term (categorical variable)
    df['Piece'] = df['Race Session (date)'].astype(str) + " #" + df['Piece'].astype(str)

    
    # One-hot encode the rowers' names by splitting and applying dummy encoding    
    for athlete in athletes:
        df[athlete] = df['Personnel'].apply(lambda x: 1 if athlete in x else 0)

    # One-hot encode the 'shell_class' column for each unique shell class
    for shell_class in shell_classes:
        df[shell_class] = df['shell_class'].apply(lambda x: 1 if x == shell_class else 0)

    # One-hot encode the Piece column (creates dummy columns for each unique value)
    X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)], drop_first=True)

    # Define the dependent variable (time per 500m)
    y = df['time_per_500m']

    X = X.astype(int)

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Fit the OLS model
    print(X)
    model = sm.OLS(y, X)
    results = model.fit()

    fitted_values = results.predict(X)

    comparison_df = pd.DataFrame({
        'Actual Pace': y.apply(lambda x: seconds_to_time(x)),
        'Model Pace': fitted_values.apply(lambda x: seconds_to_time(x))
    })
    comparison_df['Piece'] = df['Piece']
    comparison_df['Crew'] = df['Personnel']
    # comparison_df['Index'] = df['Piece'] + " " + df['Personnel']
    comparison_df['Delta'] = (y - fitted_values).round(2)
    comparison_df = comparison_df[['Piece', 'Crew', 'Actual Pace', 'Model Pace', 'Delta']]

    athletes_df = pd.DataFrame({
        'Rower': athletes,
        'Coefficient': results.params[athletes].round(1),
        'Lower': results.conf_int()[0][athletes].round(1),  # Lower bound of the confidence interval (2.5%)
        'Upper': results.conf_int()[1][athletes].round(1)   # Upper bound of the confidence interval (97.5%)
    })
    athletes_df.set_index('Rower', inplace=True)

    
    other_factors_df = pd.DataFrame({
        'Factor': [col for col in X.columns if col not in athletes],  # Get all factors that are not rowers
        'Coefficient': results.params[[col for col in X.columns if col not in athletes]].round(1),  # Get coefficients for non-rowers
        'Lower': results.conf_int()[0][[col for col in X.columns if col not in athletes]].round(1),  # Lower bound (2.5%)
        'Upper': results.conf_int()[1][[col for col in X.columns if col not in athletes]].round(1)   # Upper bound (97.5%)
    })
    other_factors_df.set_index('Factor', inplace=True)


    return {
        'results': results,
        'comparison': comparison_df,
        'athletes': athletes_df,
        'factors': other_factors_df,
        'sides': rower_sides_count
        }