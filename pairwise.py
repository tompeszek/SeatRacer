import pandas as pd
from itertools import combinations

def expand_pairwise(df):
    df['Bow'] = df['Personnel'].str.split('/').str[-1]
    
    pairwise_rows = []
    
    for (i, row1), (j, row2) in combinations(df.iterrows(), 2):
        if row1['Race Session (date)'] == row2['Race Session (date)'] and row1['Piece'] == row2['Piece']:
            margin = row2['time_per_500m'] - row1['time_per_500m']
            pairwise_rows.append({
                **row1.to_dict(),
                'Margin': margin
            })
            pairwise_rows.append({
                **row2.to_dict(),
                'Margin': -margin
            })
    
    return pd.DataFrame(pairwise_rows)

# # Example usage
# df = pd.DataFrame({
#     'Race Session (date)': ['1/5/2012', '1/5/2012', '1/5/2012'],
#     'Piece': [1, 1, 1],
#     'KM': [4, 4, 4],
#     'Rigging': ['c/p/s/p/s', 'c/p/s/p/s', 'p/s'],
#     'Personnel': [
#         'Gennaro/Bertoldo/Banks/Monaghan',
#         'Struzyna/Otto/Shald/Koven',
#         'McEachern/Guregian'
#     ],
#     'Result': ['13:05', '13:09', '14:52']
# })

# expanded_df = expand_pairwise(df)
# print(expanded_df)
