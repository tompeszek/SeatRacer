import numpy as np
import pandas as pd
import networkx as nx

def group_highly_correlated_parameters(correlation_matrix, threshold=0.8):
    """
    Groups parameters that are highly correlated based on a given threshold.

    Parameters:
        correlation_matrix (pd.DataFrame): The correlation matrix of features.
        threshold (float): The correlation threshold above which features are considered highly correlated.

    Returns:
        list: A list of lists, where each inner list contains highly correlated parameters.
    """
    # Create a graph where nodes are features and edges exist if correlation > threshold
    G = nx.Graph()

    # Add nodes
    for col in correlation_matrix.columns:
        G.add_node(col)

    # Add edges for highly correlated pairs
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j])

    # Find connected components (highly correlated groups)
    correlated_groups = [list(component) for component in nx.connected_components(G) if len(component) > 1]

    return correlated_groups
