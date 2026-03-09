import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from zlib_ng import zlib_ng

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
# Utilities
# --------------------------
def _encode(text: str):
    return tokenizer(text, return_tensors="pt", truncation=True)

def _mean_nll_from_ids(ids_1d: torch.Tensor):
    input_ids = ids_1d.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return float(out.loss.item()), input_ids.shape[1]  # mean_nll_nats, T

def _zlib_total_bits(s: str) -> int:
    return len(zlib_ng.compress(s.encode('utf-8'))) * 8

def compute_ratio(text: str):
    text = str(text).strip()
    if not text:
        return None
    enc = _encode(text)
    if enc.input_ids.numel() == 0:
        return None
    mean_nll_nats, T = _mean_nll_from_ids(enc.input_ids[0])
    total_nll_bits = mean_nll_nats * T * LOG2E
    z_bits = _zlib_total_bits(text)
    if z_bits == 0:
        return None
    return total_nll_bits / z_bits

# --------------------------
# Load new data
# --------------------------
df = pd.read_csv('/home/honglinbao/chain_of_hints/diff.csv')

# Ensure relevant columns are string
for col in ['raw', 'cot', 'got', 'tot', 'one_shot']:
    df[col] = df[col].astype(str).fillna(' ').replace(r'^\s*$', ' ', regex=True)

# --------------------------
# Compute ratios for each target column
# --------------------------
target_cols = ['raw', 'cot', 'got', 'tot', 'one_shot']
for col in target_cols:
    ratio_col = f"{col}_confusion"
    ratios = []
    for val in df[col]:
        ratios.append(compute_ratio(val))
    df[ratio_col] = ratios
    print(f"Computed {ratio_col}")

# Save
df.to_csv('confusion.csv', index=False)
print("Saved to diff_with_ratios.csv")
