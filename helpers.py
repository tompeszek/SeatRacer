# helpers.py
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import numpy as np
import scipy.stats
import pandas as pd
from datetime import datetime

from grouping import group_highly_correlated_parameters

def generate_common_x_range(df, num_points=50):
    """Generate a shared x-axis range for all rows based on min/max of CI."""
    global_x_min = df["Lower"].min()
    global_x_max = df["Upper"].max()
    return np.linspace(global_x_min, global_x_max, num_points)  # Shared x-axis


def time_to_seconds(time_str):
    """Convert time in MM:SS.x or MM:SS format to seconds."""
    if '.' in time_str:  # Handle MM:SS.x format
        time_obj = datetime.strptime(time_str, '%M:%S.%f')
        return time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    else:  # Handle MM:SS format
        time_obj = datetime.strptime(time_str, '%M:%S')
        return time_obj.minute * 60 + time_obj.second

def seconds_to_time(seconds):
    """Convert seconds to MM:SS.x format."""
    minutes = int(seconds // 60)
    seconds_remaining = seconds % 60
    return f"{minutes:02}:{seconds_remaining:04.1f}"

def add_athlete_counts(df):
    df['athlete_count'] = df['Rigging'].apply(lambda x: len(x.split('/')))    

def get_rigging_options(boat_class):
    match boat_class:
        case '2-':
            return ['p/s', 's/p']
        case '2x':
            return ['x/x']
        case '2x+':
            return ['x/x/c']
        case '4-':
            return ['p/s/p/s', 's/p/s/p']
        case '4x':
            return ['x/x/x/x']
        case '4x+':
            return ['x/x/x/x/c']
        case '8+':
            return ['c/p/s/p/s/p/s/p/s', 'c/s/p/s/p/s/p/s/p']
        case '8x':
            return ['x/x/x/x/x/x/x/x']
        case '8x+':
            return ['x/x/x/x/x/x/x/x/c']
        case _:
            return []    

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

def determine_shell_class_from_list(rowers):
    athletes = len(rowers)
    is_sculling = any('ˣ' in r for r in rowers)  # Assuming 'ˣ' marks sculling rowers
    has_cox = athletes % 2 == 1 and athletes > 1
    rower_count = athletes - (1 if has_cox else 0)

    boat_class = str(rower_count)
    if is_sculling:
        boat_class += "x"
    if has_cox:
        boat_class += "+"
    if not is_sculling and not has_cox:
        boat_class += "-"

    return boat_class


def add_speed(df):
    fastest = df['Coefficient'].min()
    df['Speed'] = df['Coefficient'] - fastest
    df['Behind'] = df['Speed'].apply(lambda x: f"+{round(x, 1)}" if x > 0 else "-")
    df['Max/Min'] = df.apply(
        lambda row: f"{round(row['Lower'] - row['Coefficient'], 1)} to {round(row['Upper'] - row['Coefficient'], 1)}",
        axis=1
    )
    return df



def pascal_case(name):
    if name.lower().startswith("mc") and len(name) > 2:
        return "Mc" + name[2:].capitalize()
    return name.title()

def get_rower_sides_count(df):
    # Check sides
    athletes = df['Personnel'].str.split('/', expand=True).stack().unique()
    rower_sides_count = {p: {'Starboard': 0, 'Port': 0, 'Scull': 0, 'Coxswain': 0} for p in athletes}

    for index, row in df.iterrows():
        # Split the rigging and personnel to get them as lists
        rigging_list = row['Rigging'].split('/')
        personnel_list = [pascal_case(name) for name in row['Personnel'].split('/')]

        if len(rigging_list) != len(personnel_list):
            raise ValueError(f"Rigging and Personnel lists are not the same length: {row} {rigging_list} {personnel_list}") 
        
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

    return rower_sides_count

def run_regression(data, selected_model, max_correlation=1.0, halflife=None): 
    # If recency_weight is not None, calculate the recency weight for each observation
    if halflife is not None and type(halflife) == float and halflife > 0:
        # Convert 'Race Session (date)' to datetime format
        data['Race Session (date)'] = pd.to_datetime(data['Race Session (date)'])
        
        # Sort data by date (most recent to earliest)
        data = data.sort_values('Race Session (date)')
        
        # Calculate the days since the most recent race
        data['days_since_latest'] = (data['Race Session (date)'].max() - data['Race Session (date)']).dt.days
        
        # Exponential decay: Weight should decay half for each halflife days
        data['recency_factor'] = np.exp(-data['days_since_latest'] / halflife)
        
        # Ensure a minimum weight of 0.1 to prevent vanishing weights for very old races
        data['recency_factor'] = data['recency_factor'].clip(lower=0.1)
        
        # Apply scale factor to adjust the weight range (e.g., scale the min weight to 1)
        scale_factor = 1 / data['recency_factor'].min()  # Make minimum weight 1
        data['scaled_recency_factor'] = data['recency_factor'] * scale_factor
        
        # Ensure that scaled recency factors are within a reasonable range (optional)
        data['scaled_recency_factor'] = data['scaled_recency_factor'].clip(upper=10)  # Set max weight cap
        
        # Use scaled_recency_factor as the weight for the GLM model
        weights = data['scaled_recency_factor']
    else:
        # If no halflife is provided, use equal weights (default 1 for all)
        weights = None

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)
    print(df)

    # All athlete names
    athletes = df['Personnel']\
        .str.split('/', expand=True)\
        .stack()\
        .apply(pascal_case)\
        .unique()
    
    # Get shell classes    
    shell_classes = df['shell_class'].unique()   

    # Convert Result times to seconds and then calculate time per 500m
    df['time_seconds'] = df['Result'].apply(time_to_seconds)
    df['time_per_500m'] = df['time_seconds'] / (df['KM'] * 2.0)  # Adjust time per 500m

    # Proportional encode (was one-hot) the rowers' names by splitting and applying dummy encoding    
    def apply_weight(row):
        if athlete in row['Personnel']:
            weight = (1.0 / row['athlete_count'])
            # print(f"Assigned weight for {athlete}: {weight}")  # Debugging line
            return weight
        else:
            return 0
    
    for athlete in athletes:
        # df[athlete] = df['Personnel'].apply(lambda x: 100 if athlete in x else 0)
        # df[athlete] = df.apply(lambda row: 1 / row['athlete_count'] if athlete in row['Personnel'] else 0, axis=1)
        df[athlete] = df.apply(apply_weight, axis=1)




    # One-hot encode the 'shell_class' column for each unique shell class
    for shell_class in shell_classes:
        df[shell_class] = df['shell_class'].apply(lambda x: 1 if x == shell_class else 0)

    # One-hot encode the Piece column (creates dummy columns for each unique value)
    X = pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)], drop_first=True)

    # Define the dependent variable (time per 500m)
    y = df['time_per_500m']

    for col in X.columns:
        if X[col].dtype != 'float64':  # Ensure we don't touch float64 columns
            X[col] = X[col].astype(int)

    # Add a constant (intercept) to the model
    # X = sm.add_constant(X)

    # Fit the OLS model
    match selected_model:
        case 'ridge':
            model = Ridge(alpha=1.0)  # alpha controls regularization strength
            model.fit(X, y)  # Pass X and y directly to the fit method
            results = model
            coefficients = model.coef_  # Coefficients for each feature
            coef_df = pd.DataFrame(coefficients, index=X.columns, columns=['Coefficient'])
            print(coef_df)
            print(coef_df.to_string())

        case 'rlm':
            model = sm.RLM(y, X)         
        case 'wls':
            model = sm.WLS(y, X, freq_weights=weights)       
        case 'ols':
            model = sm.OLS(y, X)
        case 'glm':            
            model = sm.GLM(y, X, family=sm.families.Gaussian(), freq_weights=weights)
        case _:  # Default to RLM
            model = sm.RLM(y, X)

    if selected_model != 'ridge':
        results = model.fit()

    print(results.summary())
    print(selected_model)
    print(f"recency_halflife: {halflife}")
    print(f"weights: {weights}")

    fitted_values = results.predict(X)
    # print(fitted_values)

    comparison_df = pd.DataFrame({
        'Actual Pace': y.apply(lambda x: seconds_to_time(x)),
        'Actual Pace Seconds': y,
        'Model Pace': fitted_values.apply(lambda x: seconds_to_time(x)),
        'Model Pace Seconds': fitted_values
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

    shell_classes_df = pd.DataFrame({
        'Shell Class': shell_classes,
        'Coefficient': results.params[shell_classes].round(1),
        'Lower': results.conf_int()[0][shell_classes].round(1),  # Lower bound of the confidence interval (2.5%)
        'Upper': results.conf_int()[1][shell_classes].round(1)   # Upper bound of the confidence interval (97.5%)
    })
    shell_classes_df.set_index('Shell Class', inplace=True)
        
    other_factors_df = pd.DataFrame({
        'Factor': [col for col in X.columns if col not in athletes],  # Get all factors that are not rowers
        'Coefficient': results.params[[col for col in X.columns if col not in athletes]].round(1),  # Get coefficients for non-rowers
        'Lower': results.conf_int()[0][[col for col in X.columns if col not in athletes]].round(1),  # Lower bound (2.5%)
        'Upper': results.conf_int()[1][[col for col in X.columns if col not in athletes]].round(1)   # Upper bound (97.5%)
    })
    other_factors_df.set_index('Factor', inplace=True)

    # Deal with correlations
    correlations = group_highly_correlated_parameters(X.corr(), threshold=max_correlation)

    athletes_to_remove = set()
    for group in correlations:
        athletes_to_remove.update(group)

    # Remove those athletes from athletes_df
    athletes_df = athletes_df.drop(index=athletes_to_remove, errors='ignore')

    return {
        'results': results,
        'comparison': comparison_df,
        'athletes': athletes_df,
        'factors': other_factors_df,
        'shell_classes': shell_classes_df,
        'fitted': generate_fitted_values_vs_actual(df, results, athletes, shell_classes),
        'raw': df,
        'corr': X.corr(),
        # 'removed': highly_correlated_groups = group_highly_correlated_parameters(results['corr'], threshold=0.85)
        }

def append_rigging_to_names(df):
    """Appends superscript rigging information to each rower's name in the Personnel column."""
    rig_map = {'p': 'ᵖ', 's': 'ˢ', 'c': 'ᶜ', 'x': 'ˣ'}  # Superscript mappings
    df = df.copy()  # Avoid modifying original DataFrame

    def process_row(row):
        rigging_list = row['Rigging'].split('/')
        personnel_list = row['Personnel'].split('/')

        # Handle Coxswain if rigging has one extra entry
        if len(rigging_list) - 1 == len(personnel_list):
            personnel_list.insert(0, 'Cox')
            df.at[row.name, 'Personnel'] = 'Cox/' + row['Personnel']  # Update DataFrame

        elif len(rigging_list) != len(personnel_list):
            raise ValueError(f"Rigging and Personnel lists are not the same length: {row}")

        return '/'.join(f"{name}{rig_map.get(rig, '')}" for name, rig in zip(personnel_list, rigging_list))

    df['Personnel'] = df.apply(process_row, axis=1)
    return df


def strip_rigging(name):
    """Removes appended superscript rigging information from a given name."""
    return name.rstrip('ᵖˢᶜˣ')  # Strip superscripts for port, starboard, cox, and unknown x


def generate_fitted_values_vs_actual(df, results, athletes, shell_classes):
    # Get the coefficients from the regression model
    coef = results.params

    # Prepare a new DataFrame that includes all original data
    df_fitted = df.copy()

    # Compute fitted values
    # df_fitted['Fitted'] = results.predict(sm.add_constant(pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)], drop_first=True)))
    df_fitted['Fitted'] = results.predict(pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)], drop_first=True))

    # Generate the Breakdown column
    def breakdown(row):
        components = []
        
        # Intercept
        if 'const' in coef:
            components.append(f"Intercept: {coef['const']:.4f}")
        
        # Piece contributions
        piece_col = f"Piece_{row['Piece']}"
        if piece_col in coef:
            components.append(f"{piece_col}: {coef[piece_col]:.4f}")
        
        # Athlete contributions
        for athlete in athletes:
            if athlete in coef and row[athlete] > 0:
                weight = row[athlete]
                contribution = coef[athlete] * weight
                components.append(f"{athlete} ({weight:.2f}): {contribution:.4f}")
        
        # Shell class contributions
        for shell_class in shell_classes:
            if shell_class in coef and row[shell_class] == 1:
                components.append(f"{shell_class}: {coef[shell_class]:.4f}")

        return " + ".join(components)

    df_fitted['Breakdown'] = df_fitted.apply(breakdown, axis=1)

    return df_fitted
