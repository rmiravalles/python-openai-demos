import json
import os
from collections.abc import Callable
from typing import Any

import azure.identity
import openai
from dotenv import load_dotenv

# Setup del cliente OpenAI para usar Azure, OpenAI.com u Ollama (según variables de entorno)
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
# Implementación de la tool(s)
# ---------------------------------------------------------------------------
def search_database(search_query: str, price_filter: dict | None = None) -> dict[str, str]:
    """Busca productos relevantes en la base de datos usando el query del usuario.

    search_query: texto que quieres buscar (por ejemplo "playera roja").
    price_filter: objeto opcional con filtros de precio. Debe incluir:
      - comparison_operator: uno de ">", "<", ">=", "<=", "="
      - value: número límite para comparar.

    Regresa una lista con productos dummy (ejemplo) para mostrar el flujo de function calling.
    """
    if not search_query:
        raise ValueError("search_query es requerido")
    if price_filter:
        if "comparison_operator" not in price_filter or "value" not in price_filter:
            raise ValueError("Se requieren comparison_operator y value en price_filter")
        if price_filter["comparison_operator"] not in {">", "<", ">=", "<=", "="}:
            raise ValueError("comparison_operator inválido en price_filter")
        if not isinstance(price_filter["value"], int | float):
            raise ValueError("value en price_filter debe ser numérico")
    return [{"id": "123", "name": "Producto Ejemplo", "price": 19.99}]


tool_mapping: dict[str, Callable[..., Any]] = {
    "search_database": search_database,
}

tools = [
    {
        "type": "function",
        "name": "search_database",
        "description": "Busca en la base de datos productos relevantes según el query del usuario",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Texto (query) para búsqueda full text, ej: 'tenis rojos'",
                },
                "price_filter": {
                    "type": "object",
                    "description": "Filtra resultados según el precio del producto",
                    "properties": {
                        "comparison_operator": {
                            "type": "string",
                            "description": "Operador para comparar el valor de la columna: '>', '<', '>=', '<=', '='",  # noqa
                        },
                        "value": {
                            "type": "number",
                            "description": "Valor límite para comparar, ej: 30",
                        },
                    },
                },
            },
            "required": ["search_query"],
        },
    }
]

messages: list[dict[str, Any]] = [
    {"role": "system", "content": "Eres un assistant que ayuda a buscar productos."},
    {"role": "user", "content": "Búscame una camiseta roja que cueste menos de $20."},
]

print(f"Modelo: {MODEL_NAME} en Host: {API_HOST}\n")

# Primera respuesta del model (puede incluir una tool call)
response = client.responses.create(
    model=MODEL_NAME,
    input=messages,
    tools=tools,
    tool_choice="auto",
    store=False,
)

tool_calls = [item for item in response.output if item.type == "function_call"]

# Si el model no pidió ninguna tool call, solo imprime la respuesta.
if not tool_calls:
    print("Assistant:")
    print(response.output_text)
else:
    # Agrega los items de function_call del response output
    messages.extend(response.output)

    # Procesa cada tool pedida de forma secuencial (normalmente solo una aquí)
    for tool_call in tool_calls:
        fn_name = tool_call.name
        raw_args = tool_call.arguments or "{}"
        print(f"Tool request: {fn_name}({raw_args})")

        target = tool_mapping.get(fn_name)
        if not target:
            tool_result: Any = f"ERROR: No hay implementación registrada para la tool '{fn_name}'"
        else:
            # Parseo seguro de argumentos JSON
            try:
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                parsed_args = {}
                tool_result = "Warning: JSON arguments malformados; sigo con args vacíos"
            else:
                try:
                    tool_result = target(**parsed_args)
                except Exception as e:  # safeguard tool execution
                    tool_result = f"Error ejecutando la tool {fn_name}: {e}"

        # Serializa el output de la tool (dict o str) como JSON string para el model
        try:
            tool_content = json.dumps(tool_result)
        except Exception:
            # Fallback a string si algo no es serializable a JSON
            tool_content = json.dumps({"result": str(tool_result)})

        messages.append(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": tool_content,
            }
        )

    # Segunda respuesta del model después de darle los tool outputs
    followup = client.responses.create(
        model=MODEL_NAME,
        input=messages,
        tools=tools,
        store=False,
    )
    print("Assistant (final):")
    print(followup.output_text)
