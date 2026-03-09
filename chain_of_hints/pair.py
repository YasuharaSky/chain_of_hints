import pandas as pd
from itertools import product, combinations
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
df = pd.read_csv('terms.csv')
df['term'] = df['term'].str.strip().str.lower()
field_groups = df.groupby('field')
field_to_terms = {field: group['term'].tolist() for field, group in field_groups}
field_pairs = list(combinations(field_to_terms.keys(), 2))
all_pairs = []
for f1, f2 in field_pairs:
    terms1 = field_to_terms[f1]
    terms2 = field_to_terms[f2]
    all_pairs.extend(product(terms1, terms2))
pair_df = pd.DataFrame(all_pairs, columns=['term1', 'term2'])
pair_df[['term1', 'term2']] = pair_df[['term1', 'term2']].apply(lambda row: sorted(row), axis=1, result_type='expand')
pair_df = pair_df.drop_duplicates().reset_index(drop=True)
embeddings1 = model.encode(pair_df['term1'].tolist(), convert_to_tensor=True)
embeddings2 = model.encode(pair_df['term2'].tolist(), convert_to_tensor=True)
similarities = util.cos_sim(embeddings1, embeddings2).diagonal().cpu().numpy()
pair_df['similarity'] = similarities
pair_df.to_csv('pairs.csv',index=False)