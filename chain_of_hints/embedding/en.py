import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
df = pd.read_parquet('/net/scratch/honglinbao/coh_2015.parquet')
terms = pd.read_csv('/home/honglinbao/chain_of_hints/terms.csv')
df_concat = pd.concat([df, terms], ignore_index=True)
for col in df_concat.select_dtypes(include='object').columns:
    df_concat[col] = df_concat[col].str.lower()

model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(df_concat['recovered_abstract'].astype(str).tolist(), show_progress_bar=True)
embeddings_rounded = np.round(embeddings, 2)

emb_df = pd.DataFrame({
    'id': df_concat['id'].values,
    'embedding': embeddings_rounded.tolist()  
})
emb_df.to_parquet('/net/scratch/honglinbao/coh_full_embedding.parquet', index=False)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_rounded)
embeddings_2d_rounded = np.round(embeddings_2d, 2)
df_concat['embedding'] = embeddings_2d_rounded.tolist()
df_concat.to_parquet('/net/scratch/honglinbao/coh_2015.parquet',index=False)