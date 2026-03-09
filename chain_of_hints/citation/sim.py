import pandas as pd
from sentence_transformers import SentenceTransformer, util
df = pd.read_csv('sample.csv')
# Load the all-mpnet-base-v2 model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Compute embeddings for raw and rewritten columns
embeddings_raw = model.encode(df['raw'].tolist(), convert_to_tensor=True)
embeddings_rewritten = model.encode(df['rewritten'].tolist(), convert_to_tensor=True)

# Calculate cosine similarity
similarities = util.cos_sim(embeddings_raw, embeddings_rewritten).diagonal()

# Add similarity column to DataFrame
df['similarity'] = similarities.cpu().numpy()
df.to_csv('sample.csv',index=False)