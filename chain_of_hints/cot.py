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

# Define the function for generating hypotheses
def extract_hypotheses(text):
    if not text.strip():
        return ""

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative, experienced scientist.",
                },
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
                        "**Think step by step**, using intermediate concepts to ensure a smooth transition throughout your generated abstract.\n\n"
                        "Now, give me your responses:\n"
                    ),
                },
            ],
            model="meta-llama/llama-3.1-8b-instruct",
            temperature=0.7,
        )
        output_text = response.choices[0].message.content.strip()
        return output_text

    except Exception as e:
        print(f"Error at row: {e}")
        return f"ERROR: {str(e)}"

# Process rows and save periodically
results = []
for idx, row in df.iterrows():
    text = row["merged"]
    result = extract_hypotheses(text)
    results.append(result)

    # Save checkpoint every 100 rows
    if (idx + 1) % 100 == 0:
        temp_df = df.loc[:idx, ["raw_id"]].copy()
        temp_df[output_column] = results
        temp_df.to_csv("cot.csv", index=False)
        print(f"Checkpoint saved at row {idx+1}")

# Final save
df[output_column] = results
final_df = df[["raw_id", output_column]]
final_df.to_csv("cot.csv", index=False)