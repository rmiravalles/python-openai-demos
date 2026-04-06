import csv
import os

import azure.identity
import openai
from dotenv import load_dotenv
from lunr import lunr

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

# Index the data from the CSV
with open("hybrid.csv") as file:
    reader = csv.reader(file)
    rows = list(reader)
documents = [{"id": (i + 1), "body": " ".join(row)} for i, row in enumerate(rows[1:])]
index = lunr(ref="id", fields=["body"], documents=documents)


def search(query):
    # Search the index for the user question
    results = index.search(query)
    matching_rows = [rows[int(result["ref"])] for result in results]

    # Format as a markdown table, since language models understand markdown
    matches_table = " | ".join(rows[0]) + "\n" + " | ".join(" --- " for _ in range(len(rows[0]))) + "\n"
    matches_table += "\n".join(" | ".join(row) for row in matching_rows)
    return matches_table


QUERY_REWRITE_SYSTEM_MESSAGE = """
You are a helpful assistant that rewrites user questions into good keyword queries
for an index of CSV rows with these columns: vehicle, year, msrp, acceleration, mpg, class.
Good keyword queries don't have any punctuation, and are all lowercase.
You will be given the user's new question and the conversation history.
Respond with ONLY the suggested keyword query, no other text.
"""

SYSTEM_MESSAGE = """
You are a helpful assistant that answers questions about cars based off a hybrid car data set.
You must use the data set to answer the questions, you should not provide any info that is not in the provided sources.
"""
messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

while True:
    question = input("\nYour question about electric cars: ")

    # Rewrite the query to fix typos and incorporate past context
    response = client.responses.create(
        model=MODEL_NAME,
        temperature=0.05,
        input=[
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_MESSAGE},
            {"role": "user", "content": f"New user question:{question}\n\nConversation history:{messages}"},
        ],
        store=False,
    )
    search_query = response.output_text
    print(f"Rewritten query: {search_query}")

    # Search the CSV for the question
    matches = search(search_query)
    print("Found matches:\n", matches)

    # Use the matches to generate a response
    messages.append({"role": "user", "content": f"{question}\nSources: {matches}"})
    response = client.responses.create(model=MODEL_NAME, temperature=0.3, input=messages, store=False)

    bot_response = response.output_text
    messages.append({"role": "assistant", "content": bot_response})

    print(f"\nResponse from {API_HOST} {MODEL_NAME}: \n")
    print(bot_response)
