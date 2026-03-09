import pandas as pd
import numpy as np
import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

# --------------------------
# Load models
# --------------------------
lm_model_name = "meta-llama/Llama-3.1-8B-Instruct"
cache_dir = "/net/scratch/honglinbao/siyang"
hf_token = "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"

tokenizer = AutoTokenizer.from_pretrained(lm_model_name, trust_remote_code=True, cache_dir=cache_dir, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    lm_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    cache_dir=cache_dir,
    token=hf_token
)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------
# Load and preprocess data
# --------------------------
df = pd.read_parquet('../overall_paths.parquet')
chunk_size = len(df) // 6
dfs = [df.iloc[i*chunk_size : (i+1)*chunk_size] for i in range(5)]
dfs.append(df.iloc[5*chunk_size:])  
df = dfs[0]
df['A'] = df['A'].astype(str)
df['B'] = df['B'].astype(str)
df['Sequence'] = df['Sequence'].astype(str)
meta = pd.read_parquet("/net/scratch/honglinbao/siyang/coh_2015.parquet")
meta['id'] = meta['id'].astype(str)

# Parse embeddings
if isinstance(meta['embedding'].iloc[0], str):
    meta['embedding'] = meta['embedding'].apply(ast.literal_eval)
meta['embedding'] = meta['embedding'].apply(np.array)

meta['input_ids'] = meta['recovered_abstract'].apply(
    lambda x: tokenizer(x, return_tensors='pt', add_special_tokens=False).input_ids[0].squeeze(0)
)
meta.set_index('id', inplace=True)

# Dict cache for fast access
id_to_abs = meta['recovered_abstract'].to_dict()
id_to_emb = meta['embedding'].to_dict()
id_to_ids = meta['input_ids'].to_dict()

# --------------------------
# Helper functions
# --------------------------
def gini(array):
    array = np.array(array)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-8
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def contextual_metrics(prefix_ids, target_ids):
    if target_ids.numel() == 0:
        return None
    input_ids = torch.cat([prefix_ids, target_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, -len(target_ids):] = input_ids[0, -len(target_ids):]

    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    per_token_nll = loss.item()
    per_token_ppl = np.exp(per_token_nll)
    T = len(target_ids)
    total_nll = per_token_nll * T
    return per_token_nll, per_token_ppl, T, total_nll

def get_mid_perplexity_given_AB_context(A_ids, B_ids, mid_ids):
    if mid_ids.numel() == 0:
        return None, 0
    context_ids = torch.cat([A_ids, B_ids])
    input_ids = torch.cat([context_ids, mid_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[0, -len(mid_ids):] = input_ids[0, -len(mid_ids):]

    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()
    return perplexity, len(mid_ids)

# --------------------------
# Initialize result columns
# --------------------------
df['TransitPPL'] = np.nan
df['TransitNLL'] = np.nan
df['OutsidePPL'] = np.nan
df['OutsideNLL'] = np.nan
df['Full_Length'] = np.nan
df['Gini'] = np.nan

# --------------------------
# Main processing loop
# --------------------------
save_interval = 1500
for idx, row in df.iterrows():
    try:
        seq_ids = [str(s).strip() for s in str(row['Sequence']).split(',')]
        A_id, B_id = row['A'], row['B']
        middle_ids = seq_ids[1:-1]
        all_ids = [A_id] + middle_ids + [B_id]

        # --- TransitPPL/NLL ---
        if all(mid in meta.index for mid in all_ids):
            prefix_ids = id_to_ids[A_id]
            total_nll_sum = 0.0
            total_T_sum = 0

            for mid in middle_ids + [B_id]:
                target_ids = id_to_ids[mid]
                r = contextual_metrics(prefix_ids, target_ids)
                if r:
                    _, _, T, total_nll = r
                    total_nll_sum += total_nll
                    total_T_sum += T
                prefix_ids = torch.cat([prefix_ids, target_ids])

            if total_T_sum > 0:
                path_per_token_nll = total_nll_sum / total_T_sum
                path_per_token_ppl = float(np.exp(path_per_token_nll))
                df.loc[idx, 'TransitNLL'] = path_per_token_nll
                df.loc[idx, 'TransitPPL'] = path_per_token_ppl

        # --- OutsidePPL/NLL ---
        if middle_ids:
            middle_concat = ' '.join([id_to_abs[mid] for mid in middle_ids])
            A_ids = id_to_ids[A_id]
            B_ids = id_to_ids[B_id]
            mid_ids = tokenizer(middle_concat, return_tensors='pt', add_special_tokens=False).input_ids[0]
            mid_ppl, _ = get_mid_perplexity_given_AB_context(A_ids, B_ids, mid_ids)
            df.loc[idx, 'OutsidePPL'] = mid_ppl
            df.loc[idx, 'OutsideNLL'] = float(math.log(mid_ppl))

        # --- Length & Gini ---
        coords = [id_to_emb[pid] for pid in seq_ids if pid in id_to_emb]
        if len(coords) >= 2:
            steps = [np.linalg.norm(coords[i + 1] - coords[i]) for i in range(len(coords) - 1)]
            df.loc[idx, 'Full_Length'] = np.sum(steps)
            df.loc[idx, 'Gini'] = gini(steps)

        # --- Periodic Save ---
        if (idx + 1) % save_interval == 0 or idx == len(df) - 1:
            df.iloc[:idx + 1].to_csv("overall1.csv", index=False)
            print(f"Saved up to row {idx + 1}")

    except Exception as e:
        print(f"[Row {idx}] Error: {e}")
        continue

# Final save
df.to_csv("overall1.csv", index=False)
print("Final save completed.")