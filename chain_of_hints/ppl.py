import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

# Load dataset
df = pd.read_csv('/home/honglinbao/chain_of_hints/diff.csv')

# Ensure relevant columns are string
for col in ['abstract1', 'abstract2', 'raw', 'cot', 'got', 'tot', 'one_shot']:
    df[col] = df[col].astype(str).fillna(' ').replace(r'^\s*$', ' ', regex=True)

# Load model and tokenizer
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

def get_perplexity_and_token_count(context, answer):
    context_ids = tokenizer(context, return_tensors='pt', add_special_tokens=False).input_ids[0]
    answer_ids = tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids[0]
    if answer_ids.numel() == 0:
        return None, 0  

    input_ids = torch.cat([context_ids, answer_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Only compute loss on the answer tokens
    labels = torch.full_like(input_ids, -100)
    labels[0, -len(answer_ids):] = input_ids[0, -len(answer_ids):]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Mean loss over target tokens only

    perplexity = torch.exp(loss).item()
    answer_token_count = len(answer_ids)
    return perplexity, answer_token_count

# Perplexity computation
raw_ppl, raw_tokens = [], []
cot_ppl, cot_tokens = [], []
tot_ppl, tot_tokens = [], []
got_ppl, got_tokens = [], []
one_shot_ppl, one_shot_tokens = [], []

for i, row in tqdm(df.iterrows(), total=len(df)):
    context = row['abstract1'].strip() + " " + row['abstract2'].strip()

    ppl, tokens = get_perplexity_and_token_count(context, row['raw'])
    raw_ppl.append(ppl)

    ppl, tokens = get_perplexity_and_token_count(context, row['cot'])
    cot_ppl.append(ppl)

    ppl, tokens = get_perplexity_and_token_count(context, row['tot'])
    tot_ppl.append(ppl)

    ppl, tokens = get_perplexity_and_token_count(context, row['got'])
    got_ppl.append(ppl)

    ppl, tokens = get_perplexity_and_token_count(context, row['one_shot'])
    one_shot_ppl.append(ppl)

# Store results
df['raw_ppl'] = raw_ppl
df['cot_ppl'] = cot_ppl
df['got_ppl'] = got_ppl
df['tot_ppl'] = tot_ppl
df['one_shot_ppl'] = one_shot_ppl

# Save to CSV
df.to_csv('/home/honglinbao/chain_of_hints/ppl.csv', index=False)
