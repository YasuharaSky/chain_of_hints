import re
import time
from typing import List, Tuple

NUM_CANDIDATES = 3      
GEN_TEMPERATURE = 0.7   
SCORE_TEMPERATURE = 0.3 
RETRY = 2        

import os
import pandas as pd
from openai import OpenAI

# OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

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

def _gen_once(text: str) -> str:
    for attempt in range(RETRY + 1):
        try:
            resp = client.chat.completions.create(
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
                temperature=GEN_TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < RETRY:
                #time.sleep(0.8 * (attempt + 1))
                continue
            return f"ERROR_GEN: {e}"

def _score_candidate(plan_text: str, cand: str) -> int:
    score_prompt = (
        "Task: Evaluate the following candidate abstract for whether it closely follows the instructions "
        "(clarity, feasibility, specificity, novelty, possibility, and how well it integrates the two provided paper abstracts (the context)).\n\n"
        "Give a single integer score from 1 (poor) to 10 (excellent). Then one short reason on a new line.\n\n"
        f"Context (The Provided Paper Abstracts):\n{plan_text}\n\n"
        f"Candidate Abstract for Evaluation:\n{cand}\n\n"
        "Answer strictly as:\nScore: <integer 1-10>\nReason: <one short sentence>"
    )
    for attempt in range(RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": "You are a critical scientific evaluator."},
                    {"role": "user", "content": score_prompt},
                ],
                temperature=SCORE_TEMPERATURE,
            )
            txt = resp.choices[0].message.content.strip()
            m = re.search(r"Score:\s*([1-9]|10)\b", txt)
            if m:
                return int(m.group(1))
            m2 = re.search(r"\b(10|[1-9])\b", txt)
            return int(m2.group(1)) if m2 else 0
        except Exception:
            if attempt < RETRY:
                #time.sleep(0.8 * (attempt + 1))
                continue
            return 0

def extract_hypotheses(text: str) -> str:
    if not text.strip():
        return ""

    candidates: List[str] = []
    for _ in range(NUM_CANDIDATES):
        cand = _gen_once(text)
        candidates.append(cand)

    if all(c.startswith("ERROR_GEN") for c in candidates):
        return candidates[0]

    scored: List[Tuple[int, str]] = []
    plan_text = f"Inputs (for Evaluation):\n\n{text}\n"
    for cand in candidates:
        score = _score_candidate(plan_text, cand)
        scored.append((score, cand))

    best_score, best_cand = max(scored, key=lambda x: x[0])
    return best_cand

# Process rows and save periodically
results = []
for idx, row in df.iterrows():
    text = row["merged"]
    result = extract_hypotheses(text)  
    results.append(result)

    # Save checkpoint every 100 rows
    if (idx + 1) % 10 == 0:
        temp_df = df.loc[:idx, ["raw_id"]].copy()
        temp_df[output_column] = results
        temp_df.to_csv("got.csv", index=False)
        print(f"Checkpoint saved at row {idx+1}")

# Final save
df[output_column] = results
final_df = df[["raw_id", output_column]]
final_df.to_csv("got.csv", index=False)
