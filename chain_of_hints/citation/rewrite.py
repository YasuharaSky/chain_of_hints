import pandas as pd
from openai import AzureOpenAI

# Azure OpenAI configuration
endpoint = "https://hypoextraction.cognitiveservices.azure.com/"
model_name = "o3-mini"
deployment = "o3-mini"
subscription_key = "9E35cCPh8IS7Ulw0KYB3y2Ehl3vTv40TIVORaBKEMzJ9fa1N6BfgJQQJ99BFACYeBjFXJ3w3AAAAACOGtwQl"
api_version = "2024-12-01-preview"

# Initialize client
client = AzureOpenAI(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

# Function to rewrite text
def rewrite(text):
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful writer."},
                {"role": "user", "content": f"Rewrite the following text but keep the original meaning:\n\n{text}"}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Read CSV
df = pd.read_csv('sample.csv')
# Apply rewrite to each row in 'raw'
df["rewritten"] = df["raw"].apply(rewrite)
df.to_csv('sample.csv',index=False)