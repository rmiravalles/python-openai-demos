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
    }
]

response = client.responses.create(
    model=MODEL_NAME,
    input=[
        {"role": "system", "content": "You are a weather chatbot."},
        {"role": "user", "content": "Hi, whats the weather like in berkeley?"},
    ],
    tools=tools,
    store=False,
)

print(f"Response from {MODEL_NAME} on {API_HOST}: \n")

tool_calls = [item for item in response.output if item.type == "function_call"]
if tool_calls:
    tool_call = tool_calls[0]
    print(tool_call.name)
    print(tool_call.arguments)
else:
    print(response.output_text)
