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
    """Busca el clima según nombre de ciudad o código postal."""
    return {
        "city_name": city_name,
        "zip_code": zip_code,
        "weather": "soleado",
        "temperature": 75,
    }


tools = [
    {
        "type": "function",
        "name": "lookup_weather",
        "description": "Lookup the weather for a given city name or zip code.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": ["string", "null"],
                    "description": "The city name",
                },
                "zip_code": {
                    "type": ["string", "null"],
                    "description": "The zip code",
                },
            },
            "required": ["city_name", "zip_code"],
            "additionalProperties": False,
        },
    }
]

messages = [
    {"role": "system", "content": "Eres un chatbot de clima."},
    {"role": "user", "content": "Está soleado en Berkeley CA?"},
]
response = client.responses.create(
    model=MODEL_NAME,
    input=messages,
    tools=tools,
    tool_choice="auto",
    store=False,
)

print(f"Respuesta de {MODEL_NAME} en {API_HOST}: \n")

# Ahora llama a la función indicada

tool_calls = [item for item in response.output if item.type == "function_call"]
if tool_calls:
    tool_call = tool_calls[0]
    function_name = tool_call.name
    arguments = json.loads(tool_call.arguments)

    if function_name == "lookup_weather":
        result = lookup_weather(**arguments)
        messages.extend(response.output)
        messages.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": str(result)})
        response = client.responses.create(model=MODEL_NAME, input=messages, tools=tools, store=False)
        print(response.output_text)
