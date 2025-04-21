# helpers.py
import numpy as np
import pandas as pd
from datetime import datetime

def generate_common_x_range(df, num_points=50):
    """Generate a shared x-axis range for all rows based on min/max of CI."""
    global_x_min = df["Lower"].min()
    global_x_max = df["Upper"].max()
    return np.linspace(global_x_min, global_x_max, num_points)  # Shared x-axis


def time_to_seconds(time_str):
    """Convert time in MM:SS.x or MM:SS format to seconds."""
    if '.' in time_str:  # Handle MM:SS.x format
        time_obj = datetime.strptime(time_str, '%M:%S.%f')
        return time_obj.minute * 60.0 + time_obj.second + time_obj.microsecond / 1e6
    else:  # Handle MM:SS format
        time_obj = datetime.strptime(time_str, '%M:%S')
        return time_obj.minute * 60.0 + time_obj.second

def seconds_to_time(seconds):
    """Convert seconds to MM:SS.x format."""
    minutes = int(seconds // 60)
    seconds_remaining = seconds % 60
    return f"{minutes:02}:{seconds_remaining:04.1f}"

def add_athlete_counts(df):
    df['athlete_count'] = df['Rigging'].apply(lambda x: len(x.split('/')))
    df['rower_count'] = df['Rigging'].apply(lambda x: len([i for i in x.split('/') if 'c' not in i]))

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
    # print(row)
    athletes = row['athlete_count']
    rowers = row['rower_count']

    is_sculling = 'x' in row['Rigging']
    has_cox = athletes != rowers

    boat_class = str(rowers)
    if is_sculling:
        boat_class = boat_class + "x"
    if has_cox:
        boat_class = boat_class + "+"
    if not is_sculling and not has_cox:
        boat_class = boat_class + "-"

    return boat_class

def determine_shell_class_from_list(rowers):
    athletes = len(rowers)
    is_sculling = any('ˣ' in r for r in rowers)
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

def add_side_aware_speed(df):
    df = df.copy()

    # Extract suffix (ᵖ, ˢ, ᶜ, ˣ) from athlete names
    df["Suffix"] = df.index.to_series().str.extract(r'([ᵖˢᶜˣ])$')[0]

    # Determine the fastest athlete per suffix group
    fastest_by_suffix = df.groupby("Suffix")["Coefficient"].transform("min")

    # Compute speed relative to the fastest in each suffix group
    df["Speed"] = df["Coefficient"] - fastest_by_suffix
    df["Behind"] = df["Speed"].apply(lambda x: f"+{round(x, 1)}" if x > 0 else "-")
    df["Max/Min"] = df.apply(
        lambda row: f"{round(row['Lower'] - row['Coefficient'], 1)} to {round(row['Upper'] - row['Coefficient'], 1)}",
        axis=1
    )

    return df


def pascal_case(name):
    # if name.lower().startswith("mc") and len(name) > 2:
    #     return "Mc" + name[2:].capitalize()
    # return name.title()
    return name

def get_rower_sides_count(df):
    # Check sides
    athletes = [pascal_case(name) for name in df['Personnel'].str.split('/', expand=True).stack().unique()]
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

def calculate_closest_margin(df):
    """
    Calculates the closest margin for each row in the dataframe based on 'time_seconds' 
    within the same 'piece'. The closest margin is the absolute difference to the nearest 
    result within the same piece.
    """
    
    df = df.copy()  # Avoid modifying the original DataFrame
    df['closest_margin'] = np.inf  # Initialize column

    # Ensure 'time_seconds' is a numeric type, forcing conversion to float
    df['time_seconds'] = pd.to_numeric(df['time_seconds'], errors='coerce')

    for piece in df['Piece'].unique():
        piece_mask = df['Piece'] == piece
        times = df.loc[piece_mask, 'time_seconds'].values  # Ensure it's a numpy array of numbers

        if len(times) < 2:
            df.loc[piece_mask, 'closest_margin'] = np.inf  # No comparison possible
            continue

        # Compute pairwise absolute differences using broadcasting
        time_diffs = np.abs(times[:, None] - times)  # Matrix of differences
        np.fill_diagonal(time_diffs, np.inf)  # Ignore self-comparison

        # Get the minimum difference for each row
        closest_margins = np.min(time_diffs, axis=1)

        # Assign closest margin values back to the DataFrame
        df.loc[piece_mask, 'closest_margin'] = closest_margins

    return df


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
    df_fitted['Fitted'] = results.predict(pd.get_dummies(df[['Piece'] + list(athletes) + list(shell_classes)], drop_first=False))

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

rig_map = {'p': 'ᵖ', 's': 'ˢ', 'c': 'ᶜ', 'x': 'ˣ'}
rig_map_reverse = {'ᵖ': 'p', 'ˢ': 's', 'ᶜ': 'c', 'ˣ': 'x'}