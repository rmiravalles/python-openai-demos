# Python OpenAI demos

This repository contains a collection of Python scripts that demonstrate how to use the OpenAI Responses API.
[📺 Watch this video walkthrough of running these demos in GitHub Codespaces](https://www.youtube.com/watch?v=_daw48A-RZI)

- [Python OpenAI demos](#python-openai-demos)
  - [Examples](#examples)
    - [OpenAI Responses](#openai-responses)
    - [Function calling](#function-calling)
    - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Structured outputs](#structured-outputs)
  - [Setting up the Python environment](#setting-up-the-python-environment)
  - [Configuring the OpenAI environment variables](#configuring-the-openai-environment-variables)
  - [Using Azure OpenAI models](#using-azure-openai-models)
  - [Using OpenAI.com models](#using-openaicom-models)
  - [Using Ollama models](#using-ollama-models)
  - [Resources](#resources)

## Examples

### OpenAI Responses

These scripts use the openai Python package to demonstrate how to use the OpenAI Responses API.
In increasing order of complexity, the scripts are:

1. [`chat.py`](./chat.py): A simple script that demonstrates how to use the OpenAI Responses API to generate a response.
2. [`chat_stream.py`](./chat_stream.py): Adds `stream=True` to the API call to return a generator that streams the response text as it is being generated.
3. [`chat_history.py`](./chat_history.py): Adds a back-and-forth chat interface using `input()` which keeps track of past messages and sends them with each API call.
4. [`chat_history_stream.py`](./chat_history_stream.py): The same idea, but with `stream=True` enabled.

Plus these scripts to demonstrate additional features:

* [`chat_safety.py`](./chat_safety.py): The simple script with exception handling for Azure AI Content Safety filter errors.
* [`chat_async.py`](./chat_async.py): Uses the async clients to make asynchronous calls, including an example of sending off multiple requests at once using `asyncio.gather`.

### Function calling

These scripts demonstrate using the Responses API "tools" (a.k.a. function calling) feature, which lets the model decide when to call developer-defined functions and return structured arguments instead of (or before) a natural language answer.

In all of these examples, a list of functions is declared in the `tools` parameter. The model may respond with one or more tool calls as items in `response.output` (for example, items where `type == "function_call"`). Each tool call item includes the function `name` and a JSON string of `arguments` that match the declared schema. Your application is responsible for: (1) detecting tool calls, (2) executing the corresponding local / external logic, and (3) (optionally) sending the tool result back to the model for a final answer.

Scripts (in increasing order of capability):

1. [`function_calling_basic.py`](./function_calling_basic.py): Declares a single `lookup_weather` function and prompts the model. It prints the tool call (if any) or falls back to the model's normal content. No actual function execution occurs.
2. [`function_calling_call.py`](./function_calling_call.py): Executes the `lookup_weather` function if the model requests it by parsing the returned arguments JSON and calling the local Python function.
3. [`function_calling_extended.py`](./function_calling_extended.py): Shows a full round‑trip: after executing the function, it appends a `tool` role message containing the function result and asks the model again so it can incorporate real data into a final user-facing response.
4. [`function_calling_errors.py`](./function_calling_errors.py): Same as the extended example but adds robust error handling (malformed JSON arguments, missing tool implementations, execution exceptions, JSON serialization fallback).
5. [`function_calling_parallel.py`](./function_calling_parallel.py): Demonstrates the model returning multiple tool calls in a single response
6. [`function_calling_while_loop.py`](./function_calling_while_loop.py): An iterative conversation loop that keeps executing sequential tool calls (with error handling) until the model produces a final natural language answer.

You must use a model that supports function calling (such as the defaults `gpt-4o`, `gpt-4o-mini`, etc.). Some local or older models may not support the `tools` parameter.

### Retrieval-Augmented Generation (RAG)

These scripts demonstrate how to use the OpenAI API for Retrieval-Augmented Generation (RAG) tasks, where the model retrieves relevant information from a source and uses it to generate a response.

First install the RAG dependencies:

```bash
python -m pip install -r requirements-rag.txt
```

Then run the scripts (in order of increasing complexity):

* [`rag_csv.py`](./rag_csv.py): Retrieves matching results from a CSV file and uses them to answer user's question.
* [`rag_multiturn.py`](./rag_multiturn.py): The same idea, but with a back-and-forth chat interface using `input()` which keeps track of past messages and sends them with each API call.
* [`rag_queryrewrite.py`](./rag_queryrewrite.py): Adds a query rewriting step to the RAG process, where the user's question is rewritten to improve the retrieval results.
* [`rag_documents_ingestion.py`](./rag_documents_ingestion.py): Ingests PDFs by using pymupdf to convert to markdown, then using Langchain to split into chunks, then using OpenAI to embed the chunks, and finally storing in a local JSON file.
* [`rag_documents_flow.py`](./rag_documents_flow.py): A RAG flow that retrieves matching results from the local JSON file created by `rag_documents_ingestion.py`.
* [`rag_documents_hybrid.py`](./rag_documents_hybrid.py): A RAG flow that implements a hybrid retrieval with both vector and keyword search, merging with Reciprocal Rank Fusion (RRF), and semantic re-ranking with a cross-encoder model.

## Structured outputs

These scripts demonstrate how to use the OpenAI API to generate structured responses using Pydantic data models:

* [`structured_outputs_basic.py`](./structured_outputs_basic.py): Basic example extracting simple event information using a Pydantic model.
* [`structured_outputs_description.py`](./structured_outputs_description.py): Uses additional descriptions in Pydantic model fields to clarify to the model how to format the response.
* [`structured_outputs_enum.py`](./structured_outputs_enum.py): Uses enumerations (Enums) to restrict possible values in structured responses.
* [`structured_outputs_function_calling.py`](./structured_outputs_function_calling.py): Demonstrates how to use functions defined with Pydantic for automatic function calling based on user queries.
* [`structured_outputs_nested.py`](./structured_outputs_nested.py): Uses nested Pydantic models to handle more complex structured responses, such as events with participants having multiple attributes.

## Setting up the Python environment

If you open this up in a Dev Container or GitHub Codespaces, everything will be setup for you.
If not, follow these steps:

1. Set up a Python virtual environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Configuring the OpenAI environment variables

These scripts can be run with Azure OpenAI account, OpenAI.com, or local Ollama server,
depending on the environment variables you set. All the scripts reference the environment variables from a `.env` file, and an example `.env.sample` file is provided. Host-specific instructions are below.

## Using Azure OpenAI models

This project includes infrastructure as code (IaC) to provision Azure OpenAI deployments of "gpt-4o" and "text-embedding-3-large". The IaC is defined in the `infra` directory and uses the Azure Developer CLI to provision the resources.

1. Make sure the [Azure Developer CLI (azd)](https://aka.ms/install-azd) is installed.

2. Login to Azure:

    ```shell
    azd auth login
    ```

3. Provision the OpenAI account:

    ```shell
    azd provision
    ```

    It will prompt you to provide an `azd` environment name (like "agents-demos"), select a subscription from your Azure account, and select a location. Then it will provision the resources in your account.

4. Once the resources are provisioned, you should now see a local `.env` file with all the environment variables needed to run the scripts.
5. To delete the resources, run:

    ```shell
    azd down
    ```


## Using OpenAI.com models

1. Create a `.env` file by copying the `.env.sample` file and updating it with your OpenAI API key and desired model name.

    ```bash
    cp .env.sample .env
    ```

2. Update the `.env` file with your OpenAI API key and desired model name:

    ```bash
    API_HOST=openai
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL=gpt-4o-mini
    ```

## Using Ollama models

1. Install [Ollama](https://ollama.com/) and follow the instructions to set it up on your local machine.
2. Pull a model, for example:

    ```shell
    ollama pull llama3.1
    ```

3. Create a `.env` file by copying the `.env.sample` file and updating it with your Ollama endpoint and model name.

    ```bash
    cp .env.sample .env
    ```

4. Update the `.env` file with your Ollama endpoint and model name (any model you've pulled):

    ```bash
    API_HOST=ollama
    OLLAMA_ENDPOINT=http://localhost:11434/v1
    OLLAMA_MODEL=llama3.1
    ```

## Resources

* [Video series: Learn Python + AI (October 2025)](https://techcommunity.microsoft.com/blog/educatordeveloperblog/level-up-your-python--ai-skills-with-our-complete-series/4464546)
