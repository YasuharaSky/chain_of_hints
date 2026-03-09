#output_df format
#A, B, sampled_ids, total_count
#terms1, terms54, ['2626932174', '2340491260', '2203648219', '2950489932', '1195125264'], 417757

#index_df format
#raw_id,id1,id2,
#3,terms1,terms54

import pandas as pd
import random
random.seed(42)
import ast
output_df = pd.read_csv("capsule.csv")
index_df = pd.read_csv("../diff.csv")

# Only keep id1 and id2
index_df = index_df[['id1', 'id2']]

# Convert (id1, id2) pairs into a set of frozensets to handle unordered pairs
index_pairs_set = set(frozenset((row['id1'], row['id2'])) for _, row in index_df.iterrows())

output_df = output_df[output_df.apply(lambda row: frozenset((row['A'], row['B'])) in index_pairs_set, axis=1)]

results = []

for _, row in output_df.iterrows():
    A = row['A']
    B = row['B']

    # Convert sampled_ids string to actual list
    sampled_ids = ast.literal_eval(row['sampled_ids'])

    for _ in range(2000):
        N = random.randint(1, 9)
        middle_nodes = random.sample(sampled_ids, N)
        path = [A] + [str(node) for node in middle_nodes] + [B]
        path_str = ", ".join(path)
        results.append({
            'A': A,
            'B': B,
            'Sequence': path_str,
            'Length': N + 1
        })

# Save to CSV
final_df = pd.DataFrame(results)
final_df.to_csv("overall_paths.csv", index=False)
