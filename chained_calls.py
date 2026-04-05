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


response = client.responses.create(
    model=MODEL_NAME,
    temperature=0.7,
    input=[{"role": "user", "content": "Explain how LLMs work in a single paragraph."}],
    store=False,
)

explanation = response.output_text
print("Explanation: ", explanation)
response = client.responses.create(
    model=MODEL_NAME,
    temperature=0.7,
    input=[
        {
            "role": "user",
            "content": "You're an editor. Review the explanation and provide feedback (but don't edit yourself):\n\n"
            + explanation,
        }
    ],
    store=False,
)

feedback = response.output_text
print("\n\nFeedback: ", feedback)

response = client.responses.create(
    model=MODEL_NAME,
    temperature=0.7,
    input=[
        {
            "role": "user",
            "content": (
                "Revise the article using the following feedback, but keep it to a single paragraph."
                f"\nExplanation:\n{explanation}\n\nFeedback:\n{feedback}"
            ),
        }
    ],
    store=False,
)

final_article = response.output_text
print("\n\nFinal Article: ", final_article)
