import os

import azure.identity
import openai
import rich
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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


class CalendarEvent(BaseModel):
    name: str
    date: str = Field(..., description="A date in the format YYYY-MM-DD")
    participants: list[str]


completion = client.responses.parse(
    model=MODEL_NAME,
    input=[
        {
            "role": "system",
            "content": "Extract the event information. If no year is specified, assume the current year (2025).",
        },
        {"role": "user", "content": "Alice and Bob are going to a science fair on the 1st of april."},
    ],
    text_format=CalendarEvent,
    store=False,
)
CalendarEvent(name="Science Fair", date="2025-04-01", participants=["Alice", "Bob"])

if completion.output_parsed:
    event = completion.output_parsed
    rich.print(event)
else:
    rich.print(completion.output_text)
