import json
import os
from collections.abc import Callable
from typing import Any

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


# ---------------------------------------------------------------------------
# Tool implementation(s)
# ---------------------------------------------------------------------------
def search_database(search_query: str, price_filter: dict | None = None) -> dict[str, str]:
    """Search database for relevant products based on user query"""
    if not search_query:
        raise ValueError("search_query is required")
    if price_filter:
        if "comparison_operator" not in price_filter or "value" not in price_filter:
            raise ValueError("Both comparison_operator and value are required in price_filter")
        if price_filter["comparison_operator"] not in {">", "<", ">=", "<=", "="}:
            raise ValueError("Invalid comparison_operator in price_filter")
        if not isinstance(price_filter["value"], int | float):
            raise ValueError("Value in price_filter must be a number")
    return [{"id": "123", "name": "Example Product", "price": 19.99}]


tool_mapping: dict[str, Callable[..., Any]] = {
    "search_database": search_database,
}

tools = [
    {
        "type": "function",
        "name": "search_database",
        "description": "Search database for relevant products based on user query",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Query string to use for full text search, e.g. 'red shoes'",
                },
                "price_filter": {
                    "type": "object",
                    "description": "Filter search results based on price of the product",
                    "properties": {
                        "comparison_operator": {
                            "type": "string",
                            "description": "Operator to compare the column value, either '>', '<', '>=', '<=', '='",  # noqa
                        },
                        "value": {
                            "type": "number",
                            "description": "Value to compare against, e.g. 30",
                        },
                    },
                },
            },
            "required": ["search_query"],
        },
    }
]

messages: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a product search assistant."},
    {"role": "user", "content": "good options for climbing gear that can be used outside?"},
    {
        "type": "function_call",
        "id": "fc_abc123",
        "call_id": "call_abc123",
        "name": "search_database",
        "arguments": '{"search_query":"climbing gear outside"}',
    },
    {
        "type": "function_call_output",
        "call_id": "call_abc123",
        "output": "Search results for climbing gear that can be used outside: ...",
    },
    {"role": "user", "content": "are there any shoes less than $50?"},
    {
        "type": "function_call",
        "id": "fc_abc456",
        "call_id": "call_abc456",
        "name": "search_database",
        "arguments": '{"search_query":"tenis","price_filter":{"comparison_operator":"<","value":50}}',
    },
    {
        "type": "function_call_output",
        "call_id": "call_abc456",
        "output": "Search results for shoes cheaper than 50: ...",
    },
    {"role": "user", "content": "Find me a red shirt under $20."},
]

print(f"Model: {MODEL_NAME} on Host: {API_HOST}\n")

# First model response (may include tool call)
response = client.responses.create(
    model=MODEL_NAME,
    input=messages,
    tools=tools,
    tool_choice="auto",
    store=False,
)

tool_calls = [item for item in response.output if item.type == "function_call"]

# If no tool calls were requested, just print the answer.
if not tool_calls:
    print("Assistant:")
    print(response.output_text)
else:
    # Add the function call items from the response output
    messages.extend(response.output)

    # Process each requested tool sequentially (though usually one here)
    for tool_call in tool_calls:
        fn_name = tool_call.name
        raw_args = tool_call.arguments or "{}"
        print(f"Tool request: {fn_name}({raw_args})")

        target = tool_mapping.get(fn_name)
        if not target:
            tool_result: Any = f"ERROR: No implementation registered for tool '{fn_name}'"
        else:
            # Parse arguments safely
            try:
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                parsed_args = {}
                tool_result = "Warning: Malformed JSON arguments received; proceeding with empty args"
            else:
                try:
                    tool_result = target(**parsed_args)
                except Exception as e:  # safeguard tool execution
                    tool_result = f"Tool execution error in {fn_name}: {e}"

        # Serialize tool output (dict or str) as JSON string for the model
        try:
            tool_content = json.dumps(tool_result)
        except Exception:
            # Fallback to string conversion if something isn't JSON serializable
            tool_content = json.dumps({"result": str(tool_result)})

        messages.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": tool_content,
            }
        )

    # Follow-up model response after supplying tool outputs
    followup = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        tools=tools,
        store=False,
    )
    print("Assistant (final):")
    print(followup.output_text)
