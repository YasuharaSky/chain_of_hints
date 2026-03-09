import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tqdm import tqdm
df = pd.read_parquet('/net/scratch/honglinbao/coh_2015.parquet')
df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['embedding'] = df['embedding'].apply(np.array)
embedding_dim = len(df.iloc[0]['embedding'])
df = df[df['embedding'].apply(lambda x: len(x) == embedding_dim)]
df_non_term = df[~df['id'].astype(str).str.startswith('term')].copy()
df_non_term = df_non_term.reset_index(drop=True)
id_to_emb = dict(zip(df['id'], df['embedding']))

from itertools import combinations
highlight_terms = [f'terms{i}' for i in range(1, 201)]
highlight_df = df[df['id'].isin(highlight_terms)].copy()
highlight_df['embedding'] = highlight_df['embedding'].apply(np.array)  
highlight_df = highlight_df[highlight_df['embedding'].apply(lambda x: len(x) == len(highlight_df.iloc[0]['embedding']))].copy()
results = []
for (idx1, row1), (idx2, row2) in tqdm(combinations(highlight_df.iterrows(), 2), total=19900):  # C(200,2) = 19900
    if row1['FieldOfStudyName'] == row2['FieldOfStudyName']:
        continue  

    sim = cosine_similarity(row1['embedding'].reshape(1, -1), row2['embedding'].reshape(1, -1))[0][0]
    results.append({
        'id1': row1['id'],
        'id2': row2['id'],
        'FieldOfStudyName1': row1['FieldOfStudyName'],
        'FieldOfStudyName2': row2['FieldOfStudyName'],
        'similarity_ids': sim
    })

similarity_df = pd.DataFrame(results)
similarity_df['FieldPair'] = similarity_df.apply(
    lambda row: tuple(sorted([row['FieldOfStudyName1'], row['FieldOfStudyName2']])), axis=1
)
top_k = similarity_df.copy()
results = []

for _, row in tqdm(top_k.iterrows(), total=len(top_k)):
    id1, id2 = row['id1'], row['id2']
    emb1 = id_to_emb.get(id1)
    emb2 = id_to_emb.get(id2)
    
    if emb1 is None or emb2 is None:
        continue  # skip missing

    sims1 = cosine_similarity([emb1], list(df_non_term['embedding'].values))[0]
    idx1 = np.argmax(sims1)
    paper1_id = df_non_term.iloc[idx1]['id']
    paper1_emb = df_non_term.iloc[idx1]['embedding']
    sim_id1_paper1 = sims1[idx1]

    sims2 = cosine_similarity([emb2], list(df_non_term['embedding'].values))[0]
    idx2 = np.argmax(sims2)
    paper2_id = df_non_term.iloc[idx2]['id']
    paper2_emb = df_non_term.iloc[idx2]['embedding']
    sim_id2_paper2 = sims2[idx2]

    sim_paper1_paper2 = cosine_similarity([paper1_emb], [paper2_emb])[0][0]

    results.append({
        **row,
        'paper1': paper1_id,
        'paper2': paper2_id,
        'sim_id1_paper1': sim_id1_paper1,
        'sim_id2_paper2': sim_id2_paper2,
        'sim_paper1_paper2': sim_paper1_paper2,
    })
final_df = pd.DataFrame(results)
final_df.to_csv('pairwise.csv', index=False)