import asyncio
import os

import azure.identity.aio
import openai
from dotenv import load_dotenv

# Setup the OpenAI client to use either Azure, OpenAI.com, or Ollama API
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "azure")

azure_credential = None  # Will hold the Azure credential so we can close it properly.
if API_HOST == "azure":
    azure_credential = azure.identity.aio.DefaultAzureCredential()
    token_provider = azure.identity.aio.get_bearer_token_provider(
        azure_credential, "https://cognitiveservices.azure.com/.default"
    )
    client = openai.AsyncOpenAI(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/v1/",
        api_key=token_provider,
    )
    MODEL_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
elif API_HOST == "ollama":
    client = openai.AsyncOpenAI(base_url=os.environ["OLLAMA_ENDPOINT"], api_key="nokeyneeded")
    MODEL_NAME = os.environ["OLLAMA_MODEL"]
else:
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_KEY"])
    MODEL_NAME = os.environ["OPENAI_MODEL"]


async def generate_response(location):
    print("Generating response for", location)
    response = await client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Name a single place I should visit on my trip to {location} and describe in one sentence",
            },
        ],
        temperature=0.7,
        store=False,
    )
    print("Got response for ", location)
    return response.output_text


async def single() -> None:
    """Run a single request example and handle cleanup."""
    print(await generate_response("Tokyo"))


async def multiple() -> None:
    """Run multiple requests concurrently and handle cleanup."""
    answers = await asyncio.gather(
        generate_response("Tokyo"),
        generate_response("Berkeley"),
        generate_response("Singapore"),
    )
    for answer in answers:
        print(answer, "\n")


async def close_clients() -> None:
    """Close the OpenAI async client and (if applicable) the Azure credential."""
    await client.close()
    if azure_credential is not None:
        await azure_credential.close()


async def main():
    try:
        await single()  # Change to await multiple() to run multiple requests concurrently
    finally:
        await close_clients()


if __name__ == "__main__":
    asyncio.run(main())
