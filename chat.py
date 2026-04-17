import os
from datetime import datetime
from pathlib import Path

import azure.identity
import openai
from dotenv import load_dotenv

# Setup the OpenAI client to use either Azure, OpenAI.com, or Ollama API
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "openai")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = openai.OpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
        api_key=token_provider,
    )
    MODEL_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

elif API_HOST == "ollama":
    client = openai.OpenAI(base_url=os.environ["OLLAMA_ENDPOINT"], api_key="nokeyneeded")
    MODEL_NAME = os.environ["OLLAMA_MODEL"]

else:
    client = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
    MODEL_NAME = os.environ["OPENAI_MODEL"]


user_input = "What are the benefits of using Kubernetes NetworkPolicies?"

response = client.responses.create(
    model=MODEL_NAME,
    input=[
        {"role": "system", "content": "You are a knowledgeable Kubernetes administrator that provides accurate and detailed information."},
        {"role": "user", "content": user_input},
    ],
    store=False,
)

output_file = Path("response.txt")
with output_file.open("a", encoding="utf-8") as file_handle:
    file_handle.write(f"=== {datetime.now().isoformat(timespec='seconds')} ===\n")
    file_handle.write(f"User input: {user_input}\n\n")
    file_handle.write(response.output_text)
    file_handle.write("\n\n")

print(f"Response from {API_HOST}: \n")
print(response.output_text)
print(f"\nSaved response to {output_file.resolve()}")
