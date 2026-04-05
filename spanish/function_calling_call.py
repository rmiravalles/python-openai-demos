import json
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


def lookup_weather(city_name=None, zip_code=None):
    """Busca el clima para un nombre de ciudad o código postal dado."""
    print(f"Buscando el clima para {city_name or zip_code}...")
    return "¡Está soleado!"


tools = [
    {
        "type": "function",
        "name": "lookup_weather",
        "description": "Busca el clima para un nombre de ciudad o código postal dado.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "El nombre de la ciudad",
                },
                "zip_code": {
                    "type": "string",
                    "description": "El código postal",
                },
            },
            "required": ["city_name", "zip_code"],
            "additionalProperties": False,
        },
    }
]

response = client.responses.create(
    model=MODEL_NAME,
    input=[
        {"role": "system", "content": "Eres un chatbot del clima."},
        {"role": "user", "content": "¿está soleado en Berkeley, California?"},
    ],
    tools=tools,
    tool_choice="auto",
    store=False,
)

print(f"Respuesta de {MODEL_NAME} en {API_HOST}: \n")

tool_calls = [item for item in response.output if item.type == "function_call"]
if tool_calls:
    tool_call = tool_calls[0]
    function_name = tool_call.name
    arguments = json.loads(tool_call.arguments)
    if function_name == "lookup_weather":
        lookup_weather(**arguments)
else:
    print(response.output_text)
