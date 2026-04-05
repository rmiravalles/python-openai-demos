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


class GetDeliveryDate(BaseModel):
    order_id: str


response = client.responses.create(
    model=MODEL_NAME,
    input=[
        {
            "role": "system",
            "content": "Eres un bot de atención al cliente. Usá las herramientas para ayudar al usuario.",
        },
        {"role": "user", "content": "Hola, ¿me puedes decir cuándo llegará mi pedido #12345?"},
    ],
    tools=[
        {
            "type": "function",
            "name": "GetDeliveryDate",
            "description": "Obtiene la fecha de entrega del pedido de un cliente.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
    ],
    store=False,
)

tool_calls = [item for item in response.output if item.type == "function_call"]
if tool_calls:
    rich.print(f"name={tool_calls[0].name}")
    rich.print(f"arguments={tool_calls[0].arguments}")
