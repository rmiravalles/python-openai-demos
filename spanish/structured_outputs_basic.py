import os

import azure.identity
import openai
import rich
from dotenv import load_dotenv
from pydantic import BaseModel

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


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


completion = client.responses.parse(
    model=MODEL_NAME,
    input=[
        {"role": "system", "content": "Extrae la info del evento."},
        {"role": "user", "content": "Alice y Bob van a ir a una feria de ciencias el viernes."},
    ],
    text_format=CalendarEvent,
    store=False,
)


if completion.output_parsed:
    event = completion.output_parsed
    rich.print(event)
else:
    rich.print(completion.output_text)
