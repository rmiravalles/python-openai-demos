import json
import os
from collections.abc import Callable
from typing import Any

import azure.identity
import openai
from dotenv import load_dotenv

# Configura el cliente de OpenAI para usar Azure, OpenAI.com u Ollama
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


tools = [
    {
        "type": "function",
        "name": "lookup_weather",
        "description": "Lookup the weather for a given city name or zip code.",
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {"type": "string", "description": "The city name"},
                "zip_code": {"type": "string", "description": "The zip code"},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "lookup_movies",
        "description": "Lookup movies playing in a given city name or zip code.",
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {"type": "string", "description": "The city name"},
                "zip_code": {"type": "string", "description": "The zip code"},
            },
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Implementaciones de herramientas
# ---------------------------------------------------------------------------
def lookup_weather(city_name: str | None = None, zip_code: str | None = None) -> dict[str, Any]:
    """Devuelve un clima simulado para la ubicación proporcionada."""
    location = city_name or zip_code or "desconocido"
    return {
        "ubicacion": location,
        "condicion": "chubascos",
        "lluvia_mm_ult_24h": 7,
        "recomendacion": "Buen día para actividades bajo techo si no te gusta la llovizna.",
    }


def lookup_movies(city_name: str | None = None, zip_code: str | None = None) -> dict[str, Any]:
    """Devuelve una lista simulada de películas en cartelera."""
    location = city_name or zip_code or "desconocido"
    return {
        "ubicacion": location,
        "peliculas": [
            {"titulo": "El Arrecife Cuántico", "clasificacion": "PG-13"},
            {"titulo": "Tormenta Sobre Bahía Puerto", "clasificacion": "PG"},
            {"titulo": "Koala de Medianoche", "clasificacion": "R"},
        ],
    }


tool_mapping: dict[str, Callable[..., Any]] = {
    "lookup_weather": lookup_weather,
    "lookup_movies": lookup_movies,
}


# ---------------------------------------------------------------------------
# Bucle conversacional
# ---------------------------------------------------------------------------
messages: list[dict[str, Any]] = [
    {"role": "system", "content": "Eres un chatbot de turismo."},
    {"role": "user", "content": "¿Llueve lo suficiente en Sídney como para ir al cine y qué películas hay?"},
]

print(f"Modelo: {MODEL_NAME} en Host: {API_HOST}\n")

while True:
    print("Invocando el modelo...\n")
    response = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        tools=tools,
        tool_choice="auto",
        store=False,
    )

    tool_calls = [item for item in response.output if item.type == "function_call"]

    if not tool_calls:
        print("Asistente:")
        print(response.output_text)
        break

    # Agrega los items de function_call del response output
    messages.extend(response.output)

    for tool_call in tool_calls:
        fn_name = tool_call.name
        raw_args = tool_call.arguments or "{}"
        print(f"Solicitud de herramienta: {fn_name}({raw_args})")
        target_tool = tool_mapping.get(fn_name)
        parsed_args = json.loads(raw_args)
        tool_result = target_tool(**parsed_args)
        tool_result_str = json.dumps(tool_result)
        # Agrega la respuesta de la herramienta a la conversación
        messages.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": tool_result_str,
            }
        )
