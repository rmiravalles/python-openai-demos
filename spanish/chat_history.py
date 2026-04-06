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


messages = [
    {
        "role": "system",
        "content": ("Soy un asistente de enseñanza que ayuda con preguntas de Python para Berkeley CS 61A."),
    },
]

while True:
    question = input("\nTu pregunta: ")
    print("Enviando pregunta...")

    messages.append({"role": "user", "content": question})
    response = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        temperature=0.7,
        store=False,
    )
    bot_response = response.output_text
    messages.append({"role": "assistant", "content": bot_response})

    print("Respuesta: ")
    print(bot_response)
