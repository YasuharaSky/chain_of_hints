import pandas as pd
df = pd.read_parquet('/net/scratch/honglinbao/coh_2015.parquet')
id_df = pd.read_csv('id_only.csv')
nan_id_indices = df[df['id'].isna()].index
df.loc[nan_id_indices, 'id'] = id_df['id'].values
df['id'] = df['id'].astype(str)
df.to_parquet('/net/scratch/honglinbao/coh_2015.parquet', index=False)