import os
import pandas as pd
from openai import OpenAI

# OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# Load CSV
csv_path = "max2.csv"
text_column = "B_abs"
context_column = "A_abs"
intermedia_column = "Sequence_abs"
output_column = "gen_abs"

df = pd.read_csv(csv_path)
df[text_column] = df[text_column].fillna("")
df[context_column] = df[context_column].fillna("")
df[intermedia_column] = df[intermedia_column].fillna("")

# Merge the two paper abstracts
df["merged"] = "paper1:\n" + df[context_column] + "\n\n" + "paper2:\n" + df[text_column]

# Define the function for generating hypotheses
def extract_hypotheses(text, intermedia_value):
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
                        f"You are reading two papers to initiate an interdisciplinary research project.\n"
                        f"{text}\n\n"
                        f"Your are also given the following **intermediate idea(s)** as a bridge between the two given papers:\n"
                        f"{intermedia_value}\n\n"
                        "These intermediate idea(s) represent possible linking concepts, methods, or shared applications "
                        "that could logically connect the research in paper 1 and paper 2.\n\n"
                        "Your task: write an original abstract for a publishable paper that meaningfully combines the core ideas from paper 1 and paper 2, "
                        "using the intermediate idea(s) as the thematic and methodological bridge.\n\n"
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

    except Exception as e:
        print(f"Error at row {e}")
        return f"ERROR: {str(e)}"

# Process rows and save periodically
results = []
for idx, row in df.iterrows():
    text = row["merged"]
    intermedia_value = row[intermedia_column]
    result = extract_hypotheses(text, intermedia_value)
    results.append(result)

    # Save checkpoint every 100 rows
    if (idx + 1) % 100 == 0:
        temp_df = df.loc[:idx, ["index"]].copy()
        temp_df[output_column] = results
        temp_df.to_csv("raw.csv", index=False)
        print(f"Checkpoint saved at row {idx+1}")

# Final save
df[output_column] = results
final_df = df[["index", output_column]]
final_df.to_csv("raw.csv", index=False)
