import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv('/home/honglinbao/chain_of_hints/conf-simu/nlp_cv.csv')
cols = ['raw', 'few_shot', 'cot', 'got', 'tot', 'one_shot']
df[cols] = (
    df[cols]
    .fillna("")
    .astype(str)
)
df[cols] = df[cols].mask(df[cols] == 'nan', "")

base_path = '/net/scratch/honglinbao/coh/'
df_paper_abs_all = pd.read_parquet(base_path + "2_1_all_pub_from_authors_title_abs.parquet")
df_post =df_paper_abs_all[df_paper_abs_all['year']>2014]
df_post = df_post[df_post['abstract'].astype(str) != 'None']
venues_target = [
    'Conference on Empirical Methods in Natural Language Processing',
    'Computer Vision and Pattern Recognition'
]
all_df = df_post[
    ~((df_post['year'] == 2015) & (df_post['venue'].isin(venues_target)))
]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')

recovered_embeddings = model.encode(all_df['abstract'].tolist(), convert_to_numpy=True, show_progress_bar=True)

raw_embeddings = model.encode(df['raw'].tolist(), convert_to_numpy=True, show_progress_bar=True)
few_shot_embeddings = model.encode(df['few_shot'].tolist(), convert_to_numpy=True, show_progress_bar=True)
cot_embeddings = model.encode(df['cot'].tolist(), convert_to_numpy=True, show_progress_bar=True)
got_embeddings = model.encode(df['got'].tolist(), convert_to_numpy=True, show_progress_bar=True)
tot_embeddings = model.encode(df['tot'].tolist(), convert_to_numpy=True, show_progress_bar=True)
one_shot_embeddings = model.encode(df['one_shot'].tolist(), convert_to_numpy=True, show_progress_bar=True)

def get_topk_matches(query_embs, ref_embs, k=100, method_name="raw"):
    query_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    ref_norm = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)
    cosine_scores = np.dot(query_norm, ref_norm.T)

    results = []
    for i in range(cosine_scores.shape[0]):
        topk_idx = np.argsort(-cosine_scores[i])[:k] 
        for rank, j in enumerate(topk_idx, start=1):
            results.append({
                'query_index': i,
                'method': method_name,
                'rank': rank,
                'corpusid': all_df['corpusid'].iloc[j],
                'similarity': float(cosine_scores[i, j])
            })
    return results

all_results = []
all_results.extend(get_topk_matches(raw_embeddings, recovered_embeddings, k=100, method_name="raw"))
all_results.extend(get_topk_matches(tot_embeddings, recovered_embeddings, k=100, method_name="tot"))
all_results.extend(get_topk_matches(got_embeddings, recovered_embeddings, k=100, method_name="got"))
all_results.extend(get_topk_matches(cot_embeddings, recovered_embeddings, k=100, method_name="cot"))
all_results.extend(get_topk_matches(one_shot_embeddings, recovered_embeddings, k=100, method_name="one_shot"))
all_results.extend(get_topk_matches(few_shot_embeddings, recovered_embeddings, k=100, method_name="few_shot"))

results_df = pd.DataFrame(all_results)
results_df.to_csv('/home/honglinbao/chain_of_hints/conf-simu/top100_matches_all_methods.csv', index=False)
print(f"✅ Done! Total {len(results_df)} matches saved.")
