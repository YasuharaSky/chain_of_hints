#output_df format
#A, B, sampled_ids, total_count
#terms1, terms54, ['2626932174', '2340491260', '2203648219', '2950489932', '1195125264'], 417757

import pandas as pd
import random
random.seed(42)
import ast

# Read the input file
output_df = pd.read_csv("capsule.csv")
output_df = output_df.sample(20)

results = []

for _, row in output_df.iterrows():
    A = row['A']
    B = row['B']

    # Convert sampled_ids string to actual list
    sampled_ids = ast.literal_eval(row['sampled_ids'])

    for _ in range(12000):
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
final_df.to_csv("sampled_paths.csv", index=False)
