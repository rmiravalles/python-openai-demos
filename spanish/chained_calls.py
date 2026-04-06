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


response = client.responses.create(
    model=MODEL_NAME,
    temperature=0.7,
    input=[{"role": "user", "content": "Explica cómo funcionan los LLM en un solo párrafo."}],
    store=False,
)

explanation = response.output_text
print("Explicación: ", explanation)
response = client.responses.create(
    model=MODEL_NAME,
    temperature=0.7,
    input=[
        {
            "role": "user",
            "content": (
                "Eres un editor. Revisa la explicación y proporciona comentarios detallados sobre claridad, coherencia "
                "y cautivación (pero no la edites tú mismo):\n\n"
            )
            + explanation,
        }
    ],
    store=False,
)

feedback = response.output_text
print("\n\nRetroalimentación: ", feedback)

response = client.responses.create(
    model=MODEL_NAME,
    temperature=0.7,
    input=[
        {
            "role": "user",
            "content": (
                "Revisa el artículo utilizando los siguientes comentarios, pero mantenlo a un solo párrafo."
                f"\nExplicación:\n{explanation}\n\nComentarios:\n{feedback}"
            ),
        }
    ],
    store=False,
)

final_article = response.output_text
print("\n\nFinal Article: ", final_article)
