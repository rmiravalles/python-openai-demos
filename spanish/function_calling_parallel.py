import json
import os
from concurrent.futures import ThreadPoolExecutor

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


tools = [
    {
        "type": "function",
        "name": "lookup_weather",
        "description": "Busca el clima según nombre de ciudad o código postal.",
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
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "lookup_movies",
        "description": "Buscar películas en cines según nombre de ciudad o código postal.",
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
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Tool (function) implementations
# ---------------------------------------------------------------------------
def lookup_weather(city_name: str | None = None, zip_code: str | None = None) -> str:
    """Looks up the weather for given city_name and zip_code."""
    location = city_name or zip_code or "unknown"
    # In a real implementation, call an external weather API here.
    return {
        "location": location,
        "condition": "rain showers",
        "rain_mm_last_24h": 7,
        "recommendation": "Good day for indoor activities if you dislike drizzle.",
    }


def lookup_movies(city_name: str | None = None, zip_code: str | None = None) -> str:
    """Returns a list of movies playing in the given location."""
    location = city_name or zip_code or "unknown"
    # A real implementation could query a cinema listings API.
    return {
        "location": location,
        "movies": [
            {"title": "The Quantum Reef", "rating": "PG-13"},
            {"title": "Storm Over Harbour Bay", "rating": "PG"},
            {"title": "Midnight Koala", "rating": "R"},
        ],
    }


messages = [
    {"role": "system", "content": "Eres un chatbot de turismo."},
    {
        "role": "user",
        "content": "¿Está lloviendo lo suficiente en Sídney como para ver películas y cuáles estan en los cines?",
    },
]
response = client.responses.create(
    model=MODEL_NAME,
    input=messages,
    tools=tools,
    tool_choice="auto",
    store=False,
)

print(f"Respuesta de {MODEL_NAME} en {API_HOST}: \n")

# Map function names to actual functions
available_functions = {
    "lookup_weather": lookup_weather,
    "lookup_movies": lookup_movies,
}

# Execute all tool calls in parallel using ThreadPoolExecutor
tool_calls = [item for item in response.output if item.type == "function_call"]
if tool_calls:
    print(f"El modelo solicitó {len(tool_calls)} llamada(s) de herramienta:\n")

    # Add the function call items from the response output
    messages.extend(response.output)

    with ThreadPoolExecutor() as executor:
        # Submit all tool calls to the thread pool
        futures = []
        for tool_call in tool_calls:
            function_name = tool_call.name
            arguments = json.loads(tool_call.arguments)
            print(f"Solicitud de herramienta: {function_name}({arguments})")

            if function_name in available_functions:
                future = executor.submit(available_functions[function_name], **arguments)
                futures.append((tool_call, function_name, future))

        # Add each tool result to the conversation
        for tool_call, function_name, future in futures:
            result = future.result()
            messages.append(
                {"type": "function_call_output", "call_id": tool_call.call_id, "output": json.dumps(result)}
            )

    # Get final response from the model with all tool results
    final_response = client.responses.create(model=MODEL_NAME, input=messages, tools=tools, store=False)
    print("Asistente:")
    print(final_response.output_text)
else:
    print(response.output_text)
