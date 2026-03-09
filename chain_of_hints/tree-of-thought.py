import os
import pandas as pd
from openai import OpenAI

# OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# --- ToT knobs ---
K_GENERATIONS = 3   # breadth (number of candidate thoughts)
N_VOTES = 3         # how many times to vote (aggregates to reduce randomness)
# ------------------

# Load CSV
csv_path = "/home/honglinbao/chain_of_hints/capsule/run/max2.csv"
text_column = "abstract2"
context_column = "abstract1"
output_column = "gen_abs"

df = pd.read_csv(csv_path)
df[text_column] = df[text_column].fillna("")
df[context_column] = df[context_column].fillna("")

# Merge the two paper abstracts
df["merged"] = "paper1:\n" + df[context_column] + "\n\n" + "paper2:\n" + df[text_column]

# ---- Step 1: generator (UNCHANGED PROMPT) ----
def generate_one(text):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a creative, experienced scientist."},
            {
                "role": "user",
                "content": (
                    f"You are reading two papers to initiate an interdisciplinary research project.\n\n"
                    f"{text}\n\n"
                    "Your task: Based on these two abstracts, write an original abstract for a publishable paper "
                    "that meaningfully integrates the core ideas from both sources.\n\n"
                    "Instructions:\n"
                    "1. The generated abstract should be **clear, feasible, novel, and well-reasoned**.\n"
                    "2. **Avoid vague** or purely directional statements.\n"
                    "3. Only return the plain abstract starting with \"Abstract:\". **Do not include any other commentary**.\n"
                    "4. **Be brief** and limit your response to approximately 200 words.\n\n"
                    "Example output:\n"
                    "Abstract: <your response here>\n\n"
                    "Now, give me your responses:\n"
                ),
            },
        ],
        model="meta-llama/llama-3.1-8b-instruct",
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def generate_candidates(text, k=K_GENERATIONS):
    return [generate_one(text) for _ in range(k)]

# ---- Step 2: voting selector (new, but keeps gen prompt untouched) ----
def vote_best(text, candidates, n_votes=N_VOTES):
    """
    Ask the model to select the best candidate given the original task constraints.
    We repeat the vote n_votes times and use majority / fallback to first.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Build the selector message once
    choices_block = "\n\n".join([f"Candidate {i+1}:\n{c}" for i, c in enumerate(candidates)])
    selector_user = (
        "Task: Select the **SINGLE** best candidate that most closely follows the instructions "
        "(clarity, feasibility, specificity, novelty, possibility, and how well it integrates the two provided paper abstracts (the context)).\n\n"
        f"Context (The Provided Paper Abstracts):\n{text}\n\n"
        f"Candidate Abstracts for Evaluation:\n{choices_block}\n\n"
        "Answer strictly with the candidate number only (e.g., '2')."
    )

    votes = []
    for _ in range(n_votes):
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a critical scientific evaluator."},
                {"role": "user", "content": selector_user},
            ],
            model="meta-llama/llama-3.1-70b-instruct",
            temperature=0.3,  # lower temp for more stable votes
        )
        raw = resp.choices[0].message.content.strip()
        # Extract the first integer we see; fallback to 1
        import re
        m = re.search(r"\b(\d+)\b", raw)
        pick = int(m.group(1)) if m else 1
        pick = min(max(1, pick), len(candidates))
        votes.append(pick)

    # Majority vote; tie-break by first occurrence
    from collections import Counter
    ctr = Counter(votes)
    winner_idx_1based = sorted(ctr.items(), key=lambda kv: (-kv[1], votes.index(kv[0])))[0][0]
    return candidates[winner_idx_1based - 1]

# Wrapper that runs ToT per row
def run_tot(text):
    if not text.strip():
        return ""
    try:
        cands = generate_candidates(text, K_GENERATIONS)
        best = vote_best(text, cands, N_VOTES)
        return best
    except Exception as e:
        print(f"Error: {e}")
        return f"ERROR: {str(e)}"

# Process rows and save periodically
results = []
for idx, row in df.iterrows():
    text = row["merged"]
    best_abs = run_tot(text)
    results.append(best_abs)

    # Save checkpoint every 100 rows
    if (idx + 1) % 10 == 0:
        temp_df = df.loc[:idx, ["raw_id"]].copy()
        temp_df[output_column] = results
        temp_df.to_csv("tot.csv", index=False)
        print(f"Checkpoint saved at row {idx+1}")

# Final save
df[output_column] = results
final_df = df[["raw_id", output_column]]
final_df.to_csv("tot.csv", index=False)
