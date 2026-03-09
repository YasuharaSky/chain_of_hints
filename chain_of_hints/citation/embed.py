import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv('/home/honglinbao/chain_of_hints/diff.csv')
all_df = pd.read_parquet('/net/scratch/honglinbao/coh_data-2016.parquet')

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')

cot_embeddings = model.encode(df['cot'].tolist(), convert_to_numpy=True, show_progress_bar=True)
got_embeddings = model.encode(df['got'].tolist(), convert_to_numpy=True, show_progress_bar=True)
tot_embeddings = model.encode(df['tot'].tolist(), convert_to_numpy=True, show_progress_bar=True)
one_shot_embeddings = model.encode(df['one_shot'].tolist(), convert_to_numpy=True, show_progress_bar=True)
raw_embeddings = np.load('/home/honglinbao/chain_of_hints/raw_embeddings.npy')
recovered_embeddings = np.load('/net/scratch/honglinbao/embeddings-2016.npy')

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
                'PaperID': all_df['PaperID'].iloc[j],
                'similarity': float(cosine_scores[i, j])
            })
    return results

all_results = []
all_results.extend(get_topk_matches(raw_embeddings, recovered_embeddings, k=100, method_name="raw"))
all_results.extend(get_topk_matches(tot_embeddings, recovered_embeddings, k=100, method_name="tot"))
all_results.extend(get_topk_matches(got_embeddings, recovered_embeddings, k=100, method_name="got"))
all_results.extend(get_topk_matches(cot_embeddings, recovered_embeddings, k=100, method_name="cot"))
all_results.extend(get_topk_matches(one_shot_embeddings, recovered_embeddings, k=100, method_name="one_shot"))

results_df = pd.DataFrame(all_results)
results_df.to_csv('/home/honglinbao/chain_of_hints/citation/top10_matches_all_methods.csv', index=False)
print(f"✅ Done! Total {len(results_df)} matches saved.")
