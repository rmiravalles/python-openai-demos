import json
import os

import azure.identity
import openai
from dotenv import load_dotenv

# Setup the OpenAI client to use either Azure, OpenAI.com, or Ollama API
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
                "city_name": {
                    "type": "string",
                    "description": "The city name",
                },
                "zip_code": {
                    "type": "string",
                    "description": "The zip code",
                },
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
                "city_name": {
                    "type": "string",
                    "description": "The city name",
                },
                "zip_code": {
                    "type": "string",
                    "description": "The zip code",
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


tool_mapping = {
    "lookup_weather": lookup_weather,
    "lookup_movies": lookup_movies,
}


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------
messages = [
    {"role": "system", "content": "You are a tourism chatbot."},
    {"role": "user", "content": "Is it rainy enough in Sydney to watch movies and which ones are on?"},
]

print(f"Model: {MODEL_NAME} on Host: {API_HOST}\n")

while True:
    print("Calling model...\n")
    response = client.responses.create(
        model=MODEL_NAME,
        input=messages,  # includes prior tool outputs
        tools=tools,
        tool_choice="auto",
        store=False,
    )

    tool_calls = [item for item in response.output if item.type == "function_call"]
    # If the assistant returned standard content with no tool calls, we're done.
    if not tool_calls:
        print("Assistant:")
        print(response.output_text)
        break

    # Append the function call items from response output
    messages.extend(response.output)

    # Execute each requested tool sequentially.
    for tool_call in tool_calls:
        fn_name = tool_call.name
        raw_args = tool_call.arguments or "{}"
        print(f"Tool request: {fn_name}({raw_args})")
        target_tool = tool_mapping.get(fn_name)
        parsed_args = json.loads(raw_args)
        tool_result = target_tool(**parsed_args)
        tool_result_str = json.dumps(tool_result)
        # Provide the tool output back to the model
        messages.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": tool_result_str,
            }
        )
