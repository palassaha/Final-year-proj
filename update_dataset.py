import pandas as pd

input_file = 'dataset\parkinsons.data'
output_file = r'dataset\augmented_dataset.csv'

features = [
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'PPE', 'status'
]

df = pd.read_csv(input_file)
filtered_df = df[features].copy()

filtered_df['Jitter_Ratio_1'] = filtered_df['Jitter:DDP'] / filtered_df['MDVP:RAP']
filtered_df['Jitter_Ratio_2'] = filtered_df['MDVP:PPQ'] / filtered_df['MDVP:RAP']
filtered_df['Shimmer_Ratio'] = filtered_df['Shimmer:DDA'] / filtered_df['MDVP:Shimmer']
filtered_df['Jitter_Shimmer_Product'] = filtered_df['Jitter:DDP'] * filtered_df['Shimmer:DDA']

filtered_df = filtered_df.round(5)
filtered_df.to_csv(output_file, index=False)

print(f"Augmented dataset saved to {output_file}")
