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
                        "Example input 1:\n"
                        "paper1:\nThis paper presents a novel graph neural network (GNN) architecture designed to model complex relational data across dynamic, multi-layered networks. By incorporating temporal dependencies and hierarchical connections, the approach enables improved representation learning for systems where interactions evolve over time, such as social and communication networks.\n\n"
                        "paper2:\nThis paper investigates how climate change drives cascading ecological effects by altering species distributions and interactions within boreal ecosystems. Using experiments, longitudinal field data, and empirically grounded simulation analysis, it reveals patterns of ecological reorganization, emphasizing shifts in trophic dynamics, habitat connectivity, and interspecific dependencies.\n\n"
                        "Example output 1:\n"
                        "Abstract: We apply graph neural networks to simulate climate-induced changes in species distributions across boreal ecosystems. By representing ecological relationships and environmental changes as dynamic, multi-layered graphs, our method captures cascading effects across trophic levels. Experiments demonstrate the improved predictive accuracy of ecosystem resilience and vulnerability under future climate scenarios.\n\n"
                        
                        "Example input 2:\n"
                        "paper1:\nThis paper proposes a reinforcement learning framework for optimizing urban traffic flow using decentralized control policies. By leveraging multi-agent coordination and real-time traffic signal adjustments, the method reduces congestion, travel time, and emissions in large-scale city networks.\n\n"
                        "paper2:\nThis paper examines the spread of infectious diseases in urban environments, focusing on human mobility patterns, contact networks, and environmental factors. Using epidemiological modeling and real-world mobility data, it predicts outbreak dynamics and evaluates targeted intervention strategies.\n\n"
                        "Example output 2:\n"
                        "Abstract: We integrate decentralized multi-agent reinforcement learning with epidemiological models to manage infectious disease spread in urban areas. By dynamically adjusting traffic flow and human mobility patterns through adaptive signal control, our approach mitigates contact rates during outbreaks while maintaining transport efficiency. Simulations demonstrate reduced transmission peaks and improved city resilience during epidemics.\n\n"

                        "Example input 3:\n"
                        "paper1:\nThis paper introduces a quantum-inspired algorithm for accelerating large-scale optimization problems in logistics planning. By leveraging quantum annealing principles on classical hardware, it efficiently solves complex routing and resource allocation tasks.\n\n"
                        "paper2:\nThis paper investigates the impact of supply chain disruptions caused by extreme weather events, focusing on delay propagation, cost escalation, and resilience strategies. Using historical disaster data and simulation models, it identifies vulnerabilities and adaptive planning measures.\n\n"
                        "Example output 3:\n"
                        "Abstract: We apply a quantum-inspired optimization framework to enhance supply chain resilience under extreme weather disruptions. By modeling logistics networks as combinatorial optimization problems, our method rapidly reconfigures routing and resource allocation to minimize delay propagation and cost increases. Experiments on historical disaster scenarios show improved recovery times and operational stability.\n\n"

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
        temp_df.to_csv("few_shot.csv", index=False)
        print(f"Checkpoint saved at row {idx+1}")

# Final save
df[output_column] = results
final_df = df[["raw_id", output_column]]
final_df.to_csv("few_shot.csv", index=False)