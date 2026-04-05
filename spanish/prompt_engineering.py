import os

import azure.identity
import openai
from dotenv import load_dotenv

# Configura el cliente de OpenAI para usar la API de Azure, OpenAI.com u Ollama
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "azure")

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


SYSTEM_MESSAGE = """
Quiero que actúes como Chespirito de El Chavo del 8.
Quiero que respondas y contestes como Chespirito utilizando el tono, manera y vocabulario que Chespirito usaría.
No escribas ninguna explicación. Solo responde como Chespirito.
Debes conocer todo el conocimiento de Chespirito, y nada más.
"""

USER_MESSAGE = """
¿Qué es un LLM?
"""

response = client.responses.create(
    model=MODEL_NAME,
    temperature=0.7,
    input=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_MESSAGE},
    ],
    store=False,
)

print(f"Response from {API_HOST}: \n")
print(response.output_text)
