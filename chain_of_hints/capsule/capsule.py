#same field distance 0.128
#diff field distance 0.343

import numpy as np
import pandas as pd
import ast

# --- Step 1: Load main embedding DataFrame ---
df = pd.read_parquet('/net/scratch/honglinbao/coh_2015.parquet')
if isinstance(df['embedding'].iloc[0], str):
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
df['embedding'] = df['embedding'].apply(np.array)

# --- Step 2: Load A-B pairs from CSV ---
pairs_df = pd.read_csv("../diff.csv")  

# --- Step 3: Define capsule function ---
from math import hypot, isclose
def is_in_capsule(P, A, B, k=0.235, tol=1e-12):
    """
    Return True if point P is inside/on the rectangle:
      - Long axis along AB
      - Height = |AB|
      - Fixed half-width = 0.235 (total width = 0.47), independent of |AB|
    """
    ax, ay = A
    bx, by = B
    px, py = P

    # AB vector and length (height)
    vx, vy = bx - ax, by - ay
    h = hypot(vx, vy)
    if isclose(h, 0.0, abs_tol=tol):
        raise ValueError("A and B must be distinct to define the field.")

    # Orthonormal basis: v along AB, u perpendicular to AB
    vx, vy = vx / h, vy / h      # unit along AB
    ux, uy = -vy, vx             # unit perpendicular

    # Center of the field
    cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0

    # Local coordinates of P in (u, v) frame
    rx, ry = px - cx, py - cy
    x_local = rx * ux + ry * uy   # perpendicular to AB (half-width check)
    y_local = rx * vx + ry * vy   # along AB (half-height check)

    half_height = h / 2.0
    half_width  = k  # fixed 0.235

    return (abs(x_local) <= half_width + tol) and (abs(y_local) <= half_height + tol)

# --- Step 4: Precompute all embeddings ---
id_to_vec = dict(zip(df['id'], df['embedding']))
points = np.vstack(df['embedding'].values)

# --- Step 5: Process each A-B pair ---
results = []
k = 0.235

for _, row in pairs_df.iterrows():
    ida = row['id1']
    idb = row['id2']
    
    if ida not in id_to_vec or idb not in id_to_vec:
        continue  # skip missing ids
    
    A = id_to_vec[ida]
    B = id_to_vec[idb]
    
    # Compute capsule membership
    in_capsule_mask = np.array([is_in_capsule(p, A, B, k) for p in points])
    capsule_df = df[in_capsule_mask].copy()
    capsule_df = capsule_df[~capsule_df['id'].isin([ida, idb])]
    
    # Sample
    sampled_df = capsule_df.sample(frac=0.05, random_state=42)
    sampled_ids = sampled_df['id'].tolist()
    total_in_capsule = len(capsule_df)

    # Append result
    results.append({
        'A': ida,
        'B': idb,
        'sampled_ids': sampled_ids,
        'total_count': total_in_capsule
    })

# --- Step 6: Save to new CSV ---
output_df = pd.DataFrame(results)
output_df.to_csv("capsule.csv", index=False)
#A, B, sampled_ids, total_count
#terms1, terms54, ['2626932174', '2340491260', '2203648219', '2950489932', '1195125264'], 417757
