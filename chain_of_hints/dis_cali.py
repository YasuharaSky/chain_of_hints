import numpy as np
import pandas as pd
import re
from itertools import chain
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [x['text'] for x in dataset if x['text'].strip()]
tokenized = [re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()) for text in texts]
flat_words = list(set(chain.from_iterable(tokenized)))
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding corpus...")
corpus_embeddings = embedder.encode(flat_words, convert_to_numpy=True, show_progress_bar=True)
word2vec = {word: vec for word, vec in zip(flat_words, corpus_embeddings)}
df = pd.read_csv("same_diff.csv")
df = df.dropna()
df = df.applymap(lambda x: x.lower()) 
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def get_phrase_vector(phrase, word2vec):
    words = phrase.split()
    vectors = [word2vec.get(word) for word in words if word in word2vec]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

dist_diff = []
dist_same = []

for _, row in df.iterrows():
    v1 = get_phrase_vector(row['term1'], word2vec)
    v2 = get_phrase_vector(row['term2'], word2vec)
    v3 = get_phrase_vector(row['same_field_term1'], word2vec)

    dist_diff.append(euclidean_distance(v1, v2) if v1 is not None and v2 is not None else np.nan)
    dist_same.append(euclidean_distance(v1, v3) if v1 is not None and v3 is not None else np.nan)

df['dist_diff'] = dist_diff
df['dist_same'] = dist_same
df.to_csv("same_diff.csv", index=False)