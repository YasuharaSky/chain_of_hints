import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

# --------------------------
# Model / Tokenizer
# --------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
cache_dir = "/net/scratch/honglinbao/siyang"
hf_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    cache_dir=cache_dir,
    token=hf_token
)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG2E = 1 / math.log(2)  # nats -> bits

# --------------------------
# Helpers for MI
# --------------------------
def _encode_ids(text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id]
    return torch.tensor(ids, dtype=torch.long, device=device)

def _mean_nll_from_ids(ids_1d: torch.Tensor):
    input_ids = ids_1d.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return float(out.loss.item()), input_ids.shape[1]  # mean_nll_nats, T

def _mean_nll_cond_from_ids(ctx_ids_1d: torch.Tensor, ans_ids_1d: torch.Tensor):
    newline_ids = tokenizer.encode(" ", add_special_tokens=False)
    newline_token_id = newline_ids[0] if len(newline_ids) > 0 else tokenizer.eos_token_id
    newline_tensor = torch.tensor([newline_token_id], dtype=torch.long, device=ctx_ids_1d.device)

    full_ids = torch.cat([ctx_ids_1d, newline_tensor, ans_ids_1d], dim=0).unsqueeze(0).to(device)
    labels = full_ids.clone()
    ctx_len = ctx_ids_1d.shape[0] + 1 
    labels[:, :ctx_len] = -100  # mask context
    with torch.no_grad():
        out = model(full_ids, labels=labels)
    return float(out.loss.item()), ans_ids_1d.shape[0]

def _split_sentences(text: str):
    text = (" ".join(str(text).split())).strip()
    if not text:
        return []
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".?":
            parts.append("".join(buf).strip())
            buf = []
    if buf:
        rest = "".join(buf).strip()
        if rest:
            parts.append(rest)
    return [s for s in parts if s]

def _mi_bits_per_token(ctx_text: str, ans_text: str) -> float:
    ctx_ids = _encode_ids(ctx_text)
    ans_ids = _encode_ids(ans_text)
    Hy_nats, Ty = _mean_nll_from_ids(ans_ids)
    Hyx_nats, Ty2 = _mean_nll_cond_from_ids(ctx_ids, ans_ids)
    if Ty != Ty2 or Ty == 0:
        return float("nan")
    return (Hy_nats - Hyx_nats) * LOG2E  # bits/token

def pairwise_mi_records(cell_text: str):
    """
    [{'i':0,'j':1,'sent_i':..., 'sent_j':..., 'mi_bits_per_token':...}, ...]
    """
    sents = _split_sentences(cell_text)
    if len(sents) < 2:
        return []
    recs = []
    for i in range(len(sents) - 1):
        mi = _mi_bits_per_token(sents[i], sents[i+1])
        recs.append({
            'i': i,
            'j': i+1,
            'sent_i': sents[i],
            'sent_j': sents[i+1],
            'mi_bits_per_token': mi
        })
    return recs

# --------------------------
# Load new data
# --------------------------
df = pd.read_csv('/home/honglinbao/chain_of_hints/capsule/run/max2.csv')

# Ensure relevant columns are string
for col in ['raw', 'cot', 'got', 'tot', 'one_shot','few_shot','gen_abs']:
    df[col] = df[col].astype(str).fillna(' ').replace(r'^\s*$', ' ', regex=True)

# --------------------------
# Compute pairwise MI and average; save both
# --------------------------
target_cols = ['raw', 'cot', 'got', 'tot', 'one_shot','few_shot','gen_abs']

all_pair_rows = []  

for row_idx, row in df.iterrows():
    for col in target_cols:
        recs = pairwise_mi_records(row[col])

        vals = [r['mi_bits_per_token'] for r in recs if not (r['mi_bits_per_token'] != r['mi_bits_per_token'])]
        avg = sum(vals) / len(vals) if len(vals) > 0 else float('nan')
        df.at[row_idx, f'{col}_mutual_info'] = avg

        for r in recs:
            all_pair_rows.append({
                'row_index': row_idx,   
                'column': col,          
                **r                     # i, j, sent_i, sent_j, mi_bits_per_token
            })

df.to_csv('mutual_info_sample.csv', index=False)
pairs_df = pd.DataFrame(all_pair_rows)
pairs_df.to_csv('mutual_info_sample.csv', index=False)